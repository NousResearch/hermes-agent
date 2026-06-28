#!/usr/bin/env python3
"""Fail CI if a PR includes local runtime/overlay paths.

This guard prevents accidental leakage of developer-local runtime state into
upstream pull requests. It checks only repo-relative path names from the PR diff;
it does not depend on any machine-local absolute paths or private manifests.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from collections.abc import Iterable, Sequence

FORBIDDEN_PREFIXES: tuple[str, ...] = (
    ".hermes/",
    "hermes-local-overlay/",
    "ops/plugins/config/",
)

FORBIDDEN_FILENAMES: frozenset[str] = frozenset(
    {
        "config-overlay.yaml",
        "local-only-files.txt",
    }
)


def normalize_repo_path(path: str) -> str:
    """Normalize a git path to a safe repo-relative POSIX-ish form."""
    normalized = path.replace("\\", "/").strip()
    while normalized.startswith("./"):
        normalized = normalized[2:]
    parts: list[str] = []
    for part in normalized.split("/"):
        if part in {"", "."}:
            continue
        if part == "..":
            # Diff paths should never traverse out of the repo. Keep this
            # fail-closed by returning the original-ish path with .. intact so
            # it is visible in the error output if it ever happens.
            return normalized
        parts.append(part)
    return "/".join(parts)


def is_forbidden_path(path: str) -> bool:
    """Return True if a repo-relative path is forbidden in an upstream PR."""
    normalized = normalize_repo_path(path)
    if normalized in {".hermes", "hermes-local-overlay", "ops/plugins/config"}:
        return True
    if any(normalized.startswith(prefix) for prefix in FORBIDDEN_PREFIXES):
        return True
    name = normalized.rsplit("/", 1)[-1]
    return name in FORBIDDEN_FILENAMES


def find_forbidden_paths(paths: Iterable[str]) -> list[str]:
    """Return normalized forbidden paths in deterministic order."""
    forbidden = {normalize_repo_path(path) for path in paths if is_forbidden_path(path)}
    return sorted(forbidden)


def pr_diff_paths(base_ref: str, head_ref: str) -> list[str]:
    """Read PR diff paths via git. Fail closed if diff calculation fails."""
    cmd = [
        "git",
        "diff",
        "--name-only",
        "-z",
        "--diff-filter=ACMRTUXB",
        f"{base_ref}...{head_ref}",
    ]
    try:
        proc = subprocess.run(cmd, check=True, capture_output=True)
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.decode("utf-8", errors="replace")
        stdout = exc.stdout.decode("utf-8", errors="replace")
        raise RuntimeError(
            "failed to calculate PR diff paths with "
            f"base={base_ref!r} head={head_ref!r}: {stderr or stdout or exc}"
        ) from exc
    raw = proc.stdout.decode("utf-8", errors="surrogateescape")
    return [path for path in raw.split("\0") if path]


def default_base_ref() -> str:
    base_ref = os.environ.get("GITHUB_BASE_REF") or "main"
    return f"origin/{base_ref}"


def emit_forbidden(paths: Sequence[str]) -> None:
    print("Forbidden local runtime/overlay paths found in this PR diff:", file=sys.stderr)
    for path in paths:
        print(f"::error file={path}::Forbidden local runtime/overlay path in PR diff", file=sys.stderr)
        print(f"- {path}", file=sys.stderr)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-ref", default=default_base_ref(), help="Git base ref for PR diff (default: origin/$GITHUB_BASE_REF or origin/main)")
    parser.add_argument("--head-ref", default="HEAD", help="Git head ref for PR diff (default: HEAD)")
    parser.add_argument(
        "--paths-from-stdin",
        action="store_true",
        help="Read newline-delimited paths from stdin instead of running git diff (used by tests).",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.paths_from_stdin:
        paths = [line.rstrip("\n") for line in sys.stdin if line.rstrip("\n")]
    else:
        try:
            paths = pr_diff_paths(args.base_ref, args.head_ref)
        except RuntimeError as exc:
            print(f"::error::{exc}", file=sys.stderr)
            return 2

    forbidden = find_forbidden_paths(paths)
    if forbidden:
        emit_forbidden(forbidden)
        return 1

    print("No forbidden local runtime/overlay paths found in PR diff.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
