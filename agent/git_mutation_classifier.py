"""Pure Git command mutation classifier."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from agent.git_target_resolver import GitTargetResolution, resolve


Classification = str

_READ_ONLY_COMMANDS = {
    "status",
    "diff",
    "log",
    "show",
    "rev-parse",
    "ls-files",
    "merge-base",
    "grep",
    "blame",
}

_MUTATING_COMMANDS = {
    "add",
    "commit",
    "reset",
    "restore",
    "checkout",
    "switch",
    "clean",
    "stash",
    "push",
    "pull",
    "fetch",
    "merge",
    "rebase",
    "rm",
    "mv",
    "update-ref",
    "cherry-pick",
    "revert",
    "apply",
    "am",
    "gc",
    "repack",
}

_BRANCH_READ_FLAGS = {"--list", "--show-current", "--contains", "--merged", "--no-merged"}
_BRANCH_MUTATING_FLAGS = {"-d", "-D", "-m", "-M", "-c", "-C", "--delete", "--move", "--copy"}

_TAG_READ_FLAGS = {"--list", "-l", "--contains", "--merged", "--no-merged", "--points-at"}
_TAG_MUTATING_FLAGS = {"-d", "--delete", "-a", "--annotate", "-s", "--sign", "-m", "--message", "-f", "--force"}

_REMOTE_READ_SUBCOMMANDS = {"get-url", "show"}
_REMOTE_MUTATING_SUBCOMMANDS = {
    "add",
    "remove",
    "rm",
    "rename",
    "set-url",
    "prune",
    "update",
    "set-head",
    "set-branches",
}

_WORKTREE_READ_SUBCOMMANDS = {"list"}
_WORKTREE_MUTATING_SUBCOMMANDS = {"add", "remove", "move", "prune", "repair", "lock", "unlock"}

_SUBMODULE_READ_SUBCOMMANDS = {"status", "summary"}
_SUBMODULE_MUTATING_SUBCOMMANDS = {
    "add",
    "update",
    "deinit",
    "set-branch",
    "set-url",
    "sync",
    "absorbgitdirs",
}

_CONFIG_READ_FLAGS = {"--get", "--get-all", "--get-regexp", "--list", "-l"}
_CONFIG_MUTATING_FLAGS = {
    "--unset",
    "--unset-all",
    "--add",
    "--replace-all",
    "--rename-section",
    "--remove-section",
    "--edit",
    "-e",
}
_CONFIG_SCOPE_FLAGS = {
    "--global",
    "--system",
    "--local",
    "--worktree",
    "--includes",
    "--null",
    "-z",
    "--show-origin",
    "--show-scope",
    "--name-only",
    "--fixed-value",
}
_CONFIG_VALUE_FLAGS = {"--file", "-f", "--blob", "--type", "--default"}

_GIT_GLOBAL_VALUE_OPTIONS = {"-C", "--git-dir", "--work-tree"}
_GIT_GLOBAL_VALUE_PREFIXES = ("-C", "--git-dir=", "--work-tree=")


@dataclass(frozen=True)
class GitMutationClassification:
    is_git: bool
    classification: Classification
    subcommand: str | None
    reason: str
    normalized_argv: tuple[str, ...]
    parse_ambiguous: bool


def classify(
    command_argv: Sequence[str] | None = None,
    raw_command: str | None = None,
    resolved_target: GitTargetResolution | Any | None = None,
) -> GitMutationClassification:
    """Classify whether a Git command is read-only, mutating, unknown, or not Git."""

    target = resolved_target
    if target is None:
        target = resolve(command_argv=command_argv, raw_command=raw_command)

    is_git = bool(getattr(target, "is_git_command", False))
    subcommand = getattr(target, "git_subcommand", None)
    normalized_argv = tuple(getattr(target, "normalized_argv", ()) or ())
    parse_ambiguous = bool(getattr(target, "parse_ambiguous", False))

    if parse_ambiguous:
        return _result(is_git, "unknown", subcommand, "ambiguous command structure", normalized_argv, True)

    if not is_git:
        return _result(False, "non_git", None, "command is not Git", normalized_argv, False)

    if not subcommand:
        return _result(True, "unknown", None, "Git command has no subcommand", normalized_argv, False)

    args = _args_after_git_subcommand(normalized_argv, subcommand)
    classification, reason = _classify_subcommand(subcommand, args)
    return _result(True, classification, subcommand, reason, normalized_argv, False)


def _result(
    is_git: bool,
    classification: Classification,
    subcommand: str | None,
    reason: str,
    normalized_argv: tuple[str, ...],
    parse_ambiguous: bool,
) -> GitMutationClassification:
    return GitMutationClassification(
        is_git=is_git,
        classification=classification,
        subcommand=subcommand,
        reason=reason,
        normalized_argv=normalized_argv,
        parse_ambiguous=parse_ambiguous,
    )


def _classify_subcommand(subcommand: str, args: tuple[str, ...]) -> tuple[Classification, str]:
    if subcommand in _READ_ONLY_COMMANDS:
        return "read_only", f"git {subcommand} is read-only"
    if subcommand in _MUTATING_COMMANDS:
        return "mutating", f"git {subcommand} mutates repository or remotes"
    if subcommand == "branch":
        return _classify_branch(args)
    if subcommand == "tag":
        return _classify_tag(args)
    if subcommand == "config":
        return _classify_config(args)
    if subcommand == "remote":
        return _classify_remote(args)
    if subcommand == "worktree":
        return _classify_table_subcommand("worktree", args, _WORKTREE_READ_SUBCOMMANDS, _WORKTREE_MUTATING_SUBCOMMANDS)
    if subcommand == "submodule":
        if args and args[0] == "foreach":
            return "unknown", "git submodule foreach can run arbitrary commands"
        return _classify_table_subcommand("submodule", args, _SUBMODULE_READ_SUBCOMMANDS, _SUBMODULE_MUTATING_SUBCOMMANDS)
    return "unknown", f"git {subcommand} is not classified"


def _classify_branch(args: tuple[str, ...]) -> tuple[Classification, str]:
    read = _contains_flag(args, _BRANCH_READ_FLAGS)
    mutating = _contains_flag(args, _BRANCH_MUTATING_FLAGS)
    if read and mutating:
        return "unknown", "git branch combines read-only and mutating flags"
    if mutating:
        return "mutating", "git branch uses a mutating flag"
    if read:
        return "read_only", "git branch uses a read-only flag"
    if _positionals(args):
        return "mutating", "git branch with a positional branch name creates a branch"
    if _only_options(args):
        return "read_only", "git branch without a mutating operation is read-only"
    return "unknown", "git branch form is not classified"


def _classify_tag(args: tuple[str, ...]) -> tuple[Classification, str]:
    read = _contains_flag(args, _TAG_READ_FLAGS)
    mutating = _contains_flag(args, _TAG_MUTATING_FLAGS)
    if read and mutating:
        return "unknown", "git tag combines read-only and mutating flags"
    if mutating:
        return "mutating", "git tag uses a mutating flag"
    if read:
        return "read_only", "git tag uses a read-only flag"
    if _positionals(args):
        return "mutating", "git tag with a positional tag name creates a tag"
    if _only_options(args):
        return "read_only", "git tag without a mutating operation is read-only"
    return "unknown", "git tag form is not classified"


def _classify_config(args: tuple[str, ...]) -> tuple[Classification, str]:
    read_flags = _matching_flags(args, _CONFIG_READ_FLAGS)
    mutating_flags = _matching_flags(args, _CONFIG_MUTATING_FLAGS)

    if read_flags and mutating_flags:
        return "unknown", "git config combines read-only and mutating operations"
    if mutating_flags:
        return "mutating", "git config uses an explicit mutating operation"
    if read_flags:
        return _classify_config_read_operation(args, read_flags)

    positionals = _config_positionals_without_operational_flags(args)
    if len(positionals) == 1:
        return "read_only", "git config with one key reads a value"
    if len(positionals) == 2:
        return "mutating", "git config with key and value writes a value"
    return "unknown", "git config form is not classified"


def _classify_config_read_operation(args: tuple[str, ...], read_flags: set[str]) -> tuple[Classification, str]:
    positionals = _config_positionals_without_operational_flags(args)
    if "--list" in read_flags or "-l" in read_flags:
        if len(positionals) == 0:
            return "read_only", "git config list reads configuration"
        return "unknown", "git config list has extra positional arguments"
    if "--get-regexp" in read_flags:
        if len(positionals) <= 1:
            return "read_only", "git config get-regexp reads configuration"
        return "unknown", "git config get-regexp has extra positional arguments"
    if "--get" in read_flags or "--get-all" in read_flags:
        if len(positionals) == 1:
            return "read_only", "git config get reads one key"
        return "unknown", "git config get has invalid arity"
    return "unknown", "git config read-only operation is not classified"


def _classify_remote(args: tuple[str, ...]) -> tuple[Classification, str]:
    if not args:
        return "read_only", "git remote without a mutating subcommand is read-only"
    if args[0] in {"-v", "--verbose"} and len(args) == 1:
        return "read_only", "git remote verbose lists remotes"
    if args[0] in _REMOTE_READ_SUBCOMMANDS:
        return "read_only", f"git remote {args[0]} is read-only"
    if args[0] in _REMOTE_MUTATING_SUBCOMMANDS:
        return "mutating", f"git remote {args[0]} mutates remotes"
    return "unknown", "git remote subcommand is not classified"


def _classify_table_subcommand(
    family: str,
    args: tuple[str, ...],
    read_subcommands: set[str],
    mutating_subcommands: set[str],
) -> tuple[Classification, str]:
    if not args:
        return "unknown", f"git {family} has no subcommand"
    operation = args[0]
    if operation in read_subcommands:
        return "read_only", f"git {family} {operation} is read-only"
    if operation in mutating_subcommands:
        return "mutating", f"git {family} {operation} mutates state"
    return "unknown", f"git {family} {operation} is not classified"


def _args_after_git_subcommand(argv: tuple[str, ...], subcommand: str) -> tuple[str, ...]:
    if not argv or argv[0] != "git":
        return ()

    index = 1
    while index < len(argv):
        token = argv[index]
        if token == "--":
            index += 1
            continue
        if token in _GIT_GLOBAL_VALUE_OPTIONS:
            index += 2
            continue
        if token.startswith(_GIT_GLOBAL_VALUE_PREFIXES) and token not in {"-C"}:
            index += 1
            continue
        if token.startswith("-"):
            index += 1
            continue
        if token == subcommand:
            return argv[index + 1 :]
        index += 1
    return ()


def _contains_flag(args: tuple[str, ...], flags: set[str]) -> bool:
    return bool(_matching_flags(args, flags))


def _matching_flags(args: tuple[str, ...], flags: set[str]) -> set[str]:
    matches: set[str] = set()
    for arg in args:
        if arg in flags:
            matches.add(arg)
            continue
        for flag in flags:
            if arg.startswith(f"{flag}="):
                matches.add(flag)
    return matches


def _positionals(args: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(arg for arg in args if arg != "--" and not arg.startswith("-"))


def _only_options(args: tuple[str, ...]) -> bool:
    return all(arg == "--" or arg.startswith("-") for arg in args)


def _config_positionals_without_operational_flags(args: tuple[str, ...]) -> tuple[str, ...]:
    positionals: list[str] = []
    index = 0
    while index < len(args):
        token = args[index]
        if token == "--":
            positionals.extend(args[index + 1 :])
            break
        if token.startswith("--") and "=" in token:
            name, value = token.split("=", 1)
            if name in _CONFIG_READ_FLAGS:
                if value:
                    positionals.append(value)
                index += 1
                continue
            if name in _CONFIG_VALUE_FLAGS or name in _CONFIG_SCOPE_FLAGS or name in _CONFIG_MUTATING_FLAGS:
                index += 1
                continue
        if token in _CONFIG_READ_FLAGS or token in _CONFIG_MUTATING_FLAGS or token in _CONFIG_SCOPE_FLAGS:
            index += 1
            continue
        if token in _CONFIG_VALUE_FLAGS:
            index += 2
            continue
        if token.startswith("-"):
            index += 1
            continue
        positionals.append(token)
        index += 1
    return tuple(positionals)
