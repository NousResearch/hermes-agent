"""Pure Git command target resolver.

This module intentionally performs no Git subprocess calls, network access, or
filesystem writes.  It only normalizes caller-provided command structure and the
logical cwd/target path.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import re
import shlex
from typing import Mapping, Sequence


_ASSIGNMENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=.*$")
_AMBIGUOUS_TOKENS = ("&&", "||", ";", "|", ">", "<", "$(", "`", "(", ")")
_REDIRECTION_CHARS = {">", "<"}


@dataclass(frozen=True)
class GitTargetResolution:
    canonical_path: str
    canonical_cwd: str
    normalized_argv: tuple[str, ...]
    raw_command: str | None
    is_git_command: bool
    git_subcommand: str | None
    parse_ambiguous: bool
    ambiguity_reason: str | None


def resolve(
    cwd: str | os.PathLike[str] | None = None,
    env_subset: Mapping[str, str] | None = None,
    command_argv: Sequence[str] | None = None,
    raw_command: str | None = None,
) -> GitTargetResolution:
    """Resolve the logical target path and visible Git command structure.

    Args:
        cwd: Explicit cwd to canonicalize.  If omitted, ``os.getcwd()`` is used.
        env_subset: Reserved caller-provided environment subset.  It is read-only
            and never mutated.
        command_argv: Structured argv.  When provided, it has priority over
            ``raw_command`` and must contain strings only.
        raw_command: Raw command string parsed with conservative ``shlex`` when
            no structured argv is supplied.
    """

    del env_subset  # Explicitly unused; kept for API symmetry and purity.

    canonical_cwd = _canonicalize_cwd(cwd)
    parse_ambiguous = False
    ambiguity_reason: str | None = None

    if command_argv is not None:
        argv = _validate_argv(command_argv)
    elif raw_command:
        raw_ambiguity = _raw_ambiguity_reason(raw_command)
        if raw_ambiguity is not None:
            parse_ambiguous = True
            ambiguity_reason = raw_ambiguity
            argv = _best_effort_split(raw_command)
        else:
            try:
                argv = tuple(shlex.split(raw_command, posix=True))
            except ValueError as exc:
                argv = ()
                parse_ambiguous = True
                ambiguity_reason = f"invalid shell quoting: {exc}"
    else:
        argv = ()

    if not argv and parse_ambiguous:
        return GitTargetResolution(
            canonical_path=canonical_cwd,
            canonical_cwd=canonical_cwd,
            normalized_argv=(),
            raw_command=raw_command,
            is_git_command=False,
            git_subcommand=None,
            parse_ambiguous=True,
            ambiguity_reason=ambiguity_reason,
        )

    effective_argv = _unwrap_supported_wrappers(argv)
    git_info = _parse_git_argv(effective_argv, canonical_cwd)

    if parse_ambiguous and not git_info.is_git_command:
        git_info = _AmbiguousGitInfo(
            is_git_command=_contains_plausible_git_invocation(raw_command or ""),
            git_subcommand=None,
            canonical_path=canonical_cwd,
        )

    return GitTargetResolution(
        canonical_path=git_info.canonical_path,
        canonical_cwd=canonical_cwd,
        normalized_argv=effective_argv,
        raw_command=raw_command,
        is_git_command=git_info.is_git_command,
        git_subcommand=git_info.git_subcommand,
        parse_ambiguous=parse_ambiguous,
        ambiguity_reason=ambiguity_reason,
    )


@dataclass(frozen=True)
class _AmbiguousGitInfo:
    is_git_command: bool
    git_subcommand: str | None
    canonical_path: str


def _canonicalize_cwd(cwd: str | os.PathLike[str] | None) -> str:
    source = Path(os.getcwd() if cwd is None else cwd).expanduser()
    return str(source.resolve(strict=False))


def _canonicalize_target(path: str, canonical_cwd: str) -> str:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = Path(canonical_cwd) / candidate
    return str(candidate.resolve(strict=False))


def _validate_argv(command_argv: Sequence[str]) -> tuple[str, ...]:
    if not isinstance(command_argv, (list, tuple)):
        raise TypeError("command_argv must be a list or tuple of strings")

    result: list[str] = []
    for index, item in enumerate(command_argv):
        if not isinstance(item, str):
            raise TypeError(f"command_argv[{index}] must be a string, got {type(item).__name__}")
        result.append(item)
    return tuple(result)


def _raw_ambiguity_reason(raw_command: str) -> str | None:
    if raw_command.count("\n") > 0 and _contains_plausible_multiple_commands(raw_command):
        return "raw_command contains multiple shell commands separated by newlines"

    for token in _AMBIGUOUS_TOKENS:
        if token in raw_command:
            return f"raw_command contains shell structure {token!r}"

    if any(char in raw_command for char in _REDIRECTION_CHARS):
        return "raw_command contains shell redirection"

    return None


def _contains_plausible_multiple_commands(raw_command: str) -> bool:
    commands = [line.strip() for line in raw_command.splitlines() if line.strip()]
    return len(commands) > 1


def _best_effort_split(raw_command: str) -> tuple[str, ...]:
    try:
        return tuple(shlex.split(raw_command, posix=True))
    except ValueError:
        return ()


def _unwrap_supported_wrappers(argv: tuple[str, ...]) -> tuple[str, ...]:
    index = 0

    while index < len(argv) and _is_assignment(argv[index]):
        index += 1

    if index < len(argv) and argv[index] == "command":
        index += 1

    while index < len(argv) and _is_assignment(argv[index]):
        index += 1

    if index < len(argv) and argv[index] == "sudo":
        index += 1
        if index < len(argv) and argv[index] == "-n":
            index += 1

    while index < len(argv) and _is_assignment(argv[index]):
        index += 1

    if index < len(argv) and argv[index] == "env":
        index += 1
        while index < len(argv) and _is_assignment(argv[index]):
            index += 1

    while index < len(argv) and _is_assignment(argv[index]):
        index += 1

    return argv[index:]


def _is_assignment(value: str) -> bool:
    return bool(_ASSIGNMENT_RE.match(value))


def _parse_git_argv(argv: tuple[str, ...], canonical_cwd: str) -> _AmbiguousGitInfo:
    if not argv or argv[0] != "git":
        return _AmbiguousGitInfo(False, None, canonical_cwd)

    canonical_path = canonical_cwd
    git_subcommand: str | None = None
    index = 1

    while index < len(argv):
        token = argv[index]

        if token == "-C":
            if index + 1 >= len(argv):
                return _AmbiguousGitInfo(True, None, canonical_path)
            canonical_path = _canonicalize_target(argv[index + 1], canonical_cwd)
            index += 2
            continue

        if token.startswith("-C") and token != "-C":
            canonical_path = _canonicalize_target(token[2:], canonical_cwd)
            index += 1
            continue

        if token in {"--git-dir", "--work-tree"}:
            if index + 1 >= len(argv):
                return _AmbiguousGitInfo(True, None, canonical_path)
            index += 2
            continue

        if token.startswith("--git-dir=") or token.startswith("--work-tree="):
            index += 1
            continue

        if token == "--":
            index += 1
            continue

        if token.startswith("-"):
            index += 1
            continue

        git_subcommand = token
        break

    return _AmbiguousGitInfo(True, git_subcommand, canonical_path)


def _contains_plausible_git_invocation(raw_command: str) -> bool:
    try:
        lexer = shlex.shlex(raw_command, posix=True, punctuation_chars=True)
        lexer.whitespace_split = True
        tokens = list(lexer)
    except ValueError:
        return "git" in raw_command.split()

    return "git" in tokens
