#!/usr/bin/env python3
"""CLI wrapper for scripts/ci/verification_bundle.py.

This command is read-only: it writes rendered output only to stdout. Redirect
stdout at the shell if a report file is needed.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

sys.dont_write_bytecode = True

SCRIPT_DIR = Path(__file__).resolve().parent
CI_DIR = SCRIPT_DIR / "ci"
if str(CI_DIR) not in sys.path:
    sys.path.insert(0, str(CI_DIR))

from verification_bundle import format_json, format_markdown, suggest_bundle  # noqa: E402


_GIT_REPOSITORY_ENV_NAMES = {
    "GIT_ALTERNATE_OBJECT_DIRECTORIES",
    "GIT_CEILING_DIRECTORIES",
    "GIT_COMMON_DIR",
    "GIT_DIR",
    "GIT_DISCOVERY_ACROSS_FILESYSTEM",
    "GIT_IMPLICIT_WORK_TREE",
    "GIT_INDEX_FILE",
    "GIT_NAMESPACE",
    "GIT_OBJECT_DIRECTORY",
    "GIT_PREFIX",
    "GIT_QUARANTINE_PATH",
    "GIT_REPLACE_REF_BASE",
    "GIT_SHALLOW_FILE",
    "GIT_GRAFT_FILE",
    "GIT_NO_REPLACE_OBJECTS",
    "GIT_WORK_TREE",
}


def _sanitized_git_environment() -> dict[str, str]:
    env = os.environ.copy()
    for name in list(env):
        if name in _GIT_REPOSITORY_ENV_NAMES or name == "GIT_CONFIG" or name.startswith("GIT_CONFIG_"):
            env.pop(name, None)
    env["GIT_OPTIONAL_LOCKS"] = "0"
    env["GIT_NO_REPLACE_OBJECTS"] = "1"
    return env


def _decode_git_utf8(data: bytes, label: str) -> str:
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise RuntimeError(f"{label} was not valid UTF-8") from exc
    if "\x00" in text:
        raise RuntimeError(f"{label} contained an invalid NUL byte")
    return text


def _run_git_bytes(args: list[str], *, cwd: Path, env: dict[str, str]) -> bytes:
    try:
        proc = subprocess.run(
            args,
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=30,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("git command timed out while collecting changed paths") from exc
    except OSError as exc:
        raise RuntimeError("unable to execute git while collecting changed paths") from exc
    if proc.returncode != 0:
        raise RuntimeError(f"git command failed while collecting changed paths (exit {proc.returncode})")
    return proc.stdout


def _validate_base_ref(base: str) -> str:
    value = str(base).strip()
    if not value:
        raise RuntimeError("base ref must not be empty")
    if value.startswith("-"):
        raise RuntimeError("base ref must not start with '-' or look like a git option")
    if any(ch.isspace() or ord(ch) < 32 for ch in value):
        raise RuntimeError("base ref must not contain whitespace or control characters")
    return value


def _paths_from_git(base: str) -> list[str]:
    safe_base = _validate_base_ref(base)
    start_dir = Path.cwd().resolve()
    git_env = _sanitized_git_environment()
    raw_top = _run_git_bytes(
        ["git", "-c", "core.fsmonitor=false", "rev-parse", "--show-toplevel"], cwd=start_dir, env=git_env
    )
    top_text = _decode_git_utf8(raw_top, "git repository root").strip()
    if not top_text:
        raise RuntimeError("git repository root was invalid")
    repo_root = Path(top_text).resolve()
    try:
        start_dir.relative_to(repo_root)
    except ValueError as exc:
        raise RuntimeError("git repository root does not contain the current directory") from exc
    raw_paths = _run_git_bytes(
        [
            "git",
            "-c",
            "core.fsmonitor=false",
            "diff",
            "--no-ext-diff",
            "--no-textconv",
            "--name-only",
            "--end-of-options",
            f"{safe_base}...HEAD",
        ],
        cwd=repo_root,
        env=git_env,
    )
    return _decode_git_utf8(raw_paths, "git changed paths").splitlines()


def _collect_paths(args: argparse.Namespace) -> list[str]:
    if args.paths is not None:
        return list(args.paths)
    if args.read_stdin:
        return sys.stdin.read().splitlines()
    if args.from_git:
        return _paths_from_git(args.base)
    return []


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Suggest Hermes verification commands from changed paths. Writes only to stdout."
    )
    parser.add_argument("--paths", nargs="*", help="Changed paths to classify")
    parser.add_argument("--stdin", dest="read_stdin", action="store_true", help="Read changed paths from stdin")
    parser.add_argument("--from-git", action="store_true", help="Use git diff --name-only BASE...HEAD")
    parser.add_argument("--base", default="origin/main", help="Base ref for --from-git (default: origin/main)")
    parser.add_argument("--format", choices=("markdown", "json"), default="markdown")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        paths = _collect_paths(args)
        bundle = suggest_bundle(paths)
        rendered = format_json(bundle) if args.format == "json" else format_markdown(bundle)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
