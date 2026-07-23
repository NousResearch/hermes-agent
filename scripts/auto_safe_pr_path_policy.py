#!/usr/bin/env python3
"""Repository-owned path policy for the Auto Safe PR workflow.

This module is the single source of truth for which generated-file prefixes may
appear in an automated draft PR. User-controlled path/globs are intentionally
not accepted.

Fail closed: any changed path outside the allowlist, any broad glob, any
secret/key path, or any cache/build/db artifact causes non-zero exit.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Repository-owned fixed allowlist (prefixes only; trailing slash required).
# Only paths under these prefixes may be staged by Auto Safe PR.
# ---------------------------------------------------------------------------
ALLOWED_GENERATED_PREFIXES: tuple[str, ...] = (
    "website/static/api/",
    "website/docs/",
    "docs/",
)

# Explicit broad / catch-all forms that must always be rejected as inputs
# (defensive: workflow no longer accepts user add_paths, but tests cover these).
BROAD_GLOBS: frozenset[str] = frozenset(
    {
        "*",
        "**",
        "**/",
        "**/*",
        ".",
        "./",
        "/",
        "",
        "/*",
        "/**",
    }
)

# Secret / key path patterns (path segments or suffixes).
SECRET_PATH_RE = re.compile(
    r"(^|/)"
    r"("
    r"\.env(\.|$)|"
    r"secrets?/|"
    r"credentials?/|"
    r"id_rsa|"
    r"id_ed25519|"
    r"[^/]*\.(pem|key|p12|pfx|keychain)$"
    r")",
    re.IGNORECASE,
)

# Cache / build / database artifacts.
ARTIFACT_PATH_RE = re.compile(
    r"(^|/)"
    r"("
    r"node_modules|"
    r"dist|"
    r"build|"
    r"target|"
    r"__pycache__|"
    r"\.pytest_cache|"
    r"\.mypy_cache|"
    r"\.ruff_cache|"
    r"\.tox|"
    r"\.venv|"
    r"venv"
    r")(/|$)"
    r"|"
    r"\.(log|sqlite|sqlite3|db|dump|bak)$",
    re.IGNORECASE,
)


def normalize_path(path: str) -> str:
    """Normalize to a repo-relative POSIX path without leading ./."""
    p = path.strip().replace("\\", "/")
    while p.startswith("./"):
        p = p[2:]
    return p.lstrip("/")


def is_broad_glob(path: str) -> bool:
    """Return True for catch-all / repository-root / star-only globs."""
    p = normalize_path(path)
    if p in BROAD_GLOBS:
        return True
    # path/* or path/**
    if p.endswith("/*") or p.endswith("/**"):
        return True
    # lone * or ** segments at any depth that swallow the tree
    if p == "*" or p.startswith("**/") or "/**/" in f"/{p}/":
        return True
    # unanchored recursive glob
    if "**" in p and p.count("**") == 1 and p.replace("**", "").strip("/") == "":
        return True
    return False


def is_secret_path(path: str) -> bool:
    return bool(SECRET_PATH_RE.search(normalize_path(path)))


def is_artifact_path(path: str) -> bool:
    return bool(ARTIFACT_PATH_RE.search(normalize_path(path)))


def is_allowed_prefix(path: str) -> bool:
    """True if path is under a fixed repository allowlist prefix."""
    p = normalize_path(path)
    if not p or is_broad_glob(p):
        return False
    # Directory markers from git status (trailing slash) ok
    for prefix in ALLOWED_GENERATED_PREFIXES:
        if p == prefix.rstrip("/") or p.startswith(prefix):
            return True
    return False


def classify_path(path: str) -> str:
    """Return 'ok', 'broad', 'secret', 'artifact', or 'outside_allowlist'."""
    p = normalize_path(path)
    if is_broad_glob(p):
        return "broad"
    if is_secret_path(p):
        return "secret"
    if is_artifact_path(p):
        return "artifact"
    if not is_allowed_prefix(p):
        return "outside_allowlist"
    return "ok"


def validate_paths(paths: list[str]) -> list[tuple[str, str]]:
    """Return list of (path, reason) for all failing paths."""
    failures: list[tuple[str, str]] = []
    for raw in paths:
        if not raw or not str(raw).strip():
            continue
        reason = classify_path(raw)
        if reason != "ok":
            failures.append((normalize_path(raw), reason))
    return failures


def changed_paths_from_git(repo: Path | None = None) -> list[str]:
    """Return paths with unstaged/untracked/staged changes (porcelain)."""
    cmd = ["git", "status", "--porcelain=v1", "--untracked-files=all"]
    result = subprocess.run(
        cmd,
        cwd=str(repo) if repo else None,
        check=True,
        capture_output=True,
        text=True,
    )
    paths: list[str] = []
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        # porcelain: XY PATH or XY ORIG -> PATH
        body = line[3:] if len(line) > 3 else line
        if " -> " in body:
            body = body.split(" -> ", 1)[1]
        body = body.strip().strip('"')
        if body:
            paths.append(body)
    return paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--from-git",
        action="store_true",
        help="Read changed paths from git status --porcelain",
    )
    parser.add_argument(
        "--paths-file",
        type=Path,
        help="Optional file with one path per line to validate",
    )
    parser.add_argument(
        "--write-allowed",
        type=Path,
        help="Write newline-separated allowed changed paths here",
    )
    parser.add_argument(
        "--print-allowlist",
        action="store_true",
        help="Print the fixed allowlist and exit 0",
    )
    args = parser.parse_args(argv)

    if args.print_allowlist:
        for prefix in ALLOWED_GENERATED_PREFIXES:
            print(prefix)
        return 0

    paths: list[str] = []
    if args.from_git:
        paths.extend(changed_paths_from_git())
    if args.paths_file:
        paths.extend(
            line.strip()
            for line in args.paths_file.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#")
        )

    if not paths:
        print("No changed paths. Safe no-op (no draft PR needed).")
        if args.write_allowed:
            args.write_allowed.write_text("", encoding="utf-8")
        # Exit 0; workflow should skip PR creation when file empty.
        return 0

    failures = validate_paths(paths)
    if failures:
        print("Path policy failed (fail closed):", file=sys.stderr)
        for path, reason in failures:
            print(f"  - {path}: {reason}", file=sys.stderr)
        print(
            "Allowed prefixes: " + ", ".join(ALLOWED_GENERATED_PREFIXES),
            file=sys.stderr,
        )
        return 1

    allowed = [normalize_path(p) for p in paths]
    print(f"All {len(allowed)} changed path(s) within allowlist.")
    for p in allowed:
        print(f"  ok: {p}")
    if args.write_allowed:
        args.write_allowed.write_text("\n".join(allowed) + ("\n" if allowed else ""), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
