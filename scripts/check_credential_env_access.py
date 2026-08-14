#!/usr/bin/env python3
"""Block direct os.getenv()/os.environ access to credential-shaped env vars.

Context: the multiplexing gateway (``gateway.multiplex_profiles``) serves
many profiles from one process, each with its own ``.env``. Reading a
credential straight from ``os.environ`` instead of through
``agent.secret_scope.get_secret()`` bypasses the per-profile secret scope and
can leak profile A's key into profile B's turn (or crash loud-but-late via
``UnscopedSecretError`` only in multiplex mode, silently using a stale
process-env value otherwise). Three real instances of this were found and
fixed in ``agent/auxiliary_client.py``, ``agent/chat_completion_helpers.py``,
and ``agent/anthropic_adapter.py`` (OPENROUTER_API_KEY, OPENAI_API_KEY,
ANTHROPIC_TOKEN, etc., plus dynamic ``key_env``-style lookups) — this script
exists so the next one is caught before merge instead of by audit.

Two shapes are flagged, both scoped to ``agent/`` and ``gateway/`` (the two
directories that resolve provider/platform credentials):

1. Literal credential-shaped names: ``os.getenv("FOO_API_KEY")``,
   ``os.environ.get("FOO_TOKEN")``, ``os.environ["FOO_SECRET"]`` — any name
   ending in ``_API_KEY`` / ``_TOKEN`` / ``_SECRET`` / ``_PASSWORD``.
2. Dynamic env-var-name variables following the ``key_env`` convention seen
   in the fixed call sites: ``os.getenv(key_env)``, ``os.getenv(fb_key_env)``
   — a variable whose name ends in ``_env`` and contains key/token/secret/
   password, i.e. it *holds the name* of a credential env var.

See ``agent/secret_scope.py`` for the sanctioned resolution path and
``docs/design/multiplexing-gateway.md`` (Workstream A) for the design
rationale.

Exit codes:
  0 — no violations
  1 — violations found
  2 — script error

Usage:
  python scripts/check_credential_env_access.py            # staged files
  python scripts/check_credential_env_access.py --all       # full agent/+gateway/
  python scripts/check_credential_env_access.py --diff main # vs a git ref
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# The two directories that resolve provider/platform credentials. Kept
# tight and explicit rather than repo-wide: cron/ and plugins/platforms/
# also touch credentials and are worth the same treatment, but that's a
# follow-up, not bundled into this rule's first cut.
SCOPE_DIRS = ("agent", "gateway")

# Files inside SCOPE_DIRS that are the sanctioned exception: secret_scope.py
# IS the get_secret() bridge, and legitimately reads os.environ directly for
# genuinely-global vars and the multiplex-off fallback path.
SAFE_FILES = {
    "agent/secret_scope.py",
}

# Inline marker that exempts a single line from this check (e.g. a
# documented, reviewed case where the process-env read is intentional).
SUPPRESS_MARKER = re.compile(r"#\s*credential-lint\s*:\s*ok\b", re.IGNORECASE)

# Literal env-var name ends with one of these -> credential-shaped.
_CRED_NAME_SUFFIX = re.compile(r"_(?:API_KEY|TOKEN|SECRET|PASSWORD)$")

_LITERAL_ACCESS = re.compile(
    r"""\bos\.(?:getenv|environ\.get)\s*\(\s*["']([A-Z0-9_]+)["']"""
    r"""|\bos\.environ\s*\[\s*["']([A-Z0-9_]+)["']"""
)

_DYNAMIC_ACCESS = re.compile(
    r"""\bos\.(?:getenv|environ\.get)\s*\(\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*[,)]"""
    r"""|\bos\.environ\s*\[\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\]"""
)
_CRED_VAR_HINT = re.compile(r"(?i:key|token|secret|password)")


def _is_cred_literal(name: str) -> bool:
    return bool(_CRED_NAME_SUFFIX.search(name))


def _is_cred_var_name(name: str) -> bool:
    return name.lower().endswith("_env") and bool(_CRED_VAR_HINT.search(name))


def should_scan_file(path: Path) -> bool:
    """Return True if the file is a non-test .py file under SCOPE_DIRS."""
    if path.suffix != ".py":
        return False
    try:
        rel = path.relative_to(REPO_ROOT)
    except ValueError:
        return False
    if not rel.parts or rel.parts[0] not in SCOPE_DIRS:
        return False
    if rel.as_posix() in SAFE_FILES:
        return False
    return True


def iter_files(paths: list[Path]) -> list[Path]:
    files: list[Path] = []
    for p in paths:
        if p.is_file():
            if should_scan_file(p):
                files.append(p)
        elif p.is_dir():
            for f in sorted(p.rglob("*.py")):
                if should_scan_file(f):
                    files.append(f)
    return files


def scan_file(path: Path) -> list[tuple[int, str, str]]:
    """Return (line_number, line, reason) for each unsuppressed violation."""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []

    violations: list[tuple[int, str, str]] = []
    for i, line in enumerate(text.splitlines(), start=1):
        stripped = line.lstrip()
        if stripped.startswith("#"):
            continue
        if SUPPRESS_MARKER.search(line):
            continue

        for m in _LITERAL_ACCESS.finditer(line):
            name = m.group(1) or m.group(2)
            if name and _is_cred_literal(name):
                violations.append((
                    i, line.rstrip(),
                    f'credential-shaped env var "{name}" read directly via '
                    f"os.getenv/os.environ instead of get_secret({name!r})",
                ))

        for m in _DYNAMIC_ACCESS.finditer(line):
            var = m.group(1) or m.group(2)
            if var and _is_cred_var_name(var):
                violations.append((
                    i, line.rstrip(),
                    f'env-var-name variable "{var}" (looks like it names a '
                    f"credential) passed to os.getenv/os.environ instead of "
                    f"get_secret({var})",
                ))

    return violations


def get_staged_files() -> list[Path]:
    try:
        out = subprocess.check_output(
            ["git", "diff", "--cached", "--name-only", "--diff-filter=ACMR"],
            cwd=REPO_ROOT, stderr=subprocess.DEVNULL,
            text=True, encoding="utf-8", errors="replace",
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return []
    return [REPO_ROOT / f for f in out.splitlines() if f.strip()]


def get_diff_files(ref: str) -> list[Path]:
    try:
        out = subprocess.check_output(
            ["git", "diff", f"{ref}...HEAD", "--name-only", "--diff-filter=ACMR"],
            cwd=REPO_ROOT, stderr=subprocess.DEVNULL,
            text=True, encoding="utf-8", errors="replace",
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return []
    return [REPO_ROOT / f for f in out.splitlines() if f.strip()]


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Flag direct os.getenv/os.environ reads of credential-shaped "
        "env vars in agent/ and gateway/ — use agent.secret_scope.get_secret() instead."
    )
    p.add_argument("paths", nargs="*", type=Path,
                    help="Specific files/dirs to scan (default: staged changes).")
    p.add_argument("--all", action="store_true",
                    help="Scan the full agent/ and gateway/ trees.")
    p.add_argument("--diff", metavar="REF",
                    help="Scan files changed vs. a git ref (e.g. --diff main).")
    return p.parse_args(argv)


def main(argv: list[str]) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")

    args = parse_args(argv)

    if args.all:
        roots = [REPO_ROOT / d for d in SCOPE_DIRS]
        roots = [r for r in roots if r.exists()]
    elif args.diff:
        roots = get_diff_files(args.diff)
    elif args.paths:
        roots = [p.resolve() for p in args.paths]
    else:
        roots = get_staged_files()
        if not roots:
            print(
                "No staged files to check (use --all for a full scan, "
                "--diff REF to compare against a ref).",
                file=sys.stderr,
            )
            return 0

    files = iter_files(roots)
    if not files:
        print("✓ No agent/ or gateway/ files to scan.")
        return 0

    total = 0
    for path in files:
        for line_no, line, reason in scan_file(path):
            rel = path.relative_to(REPO_ROOT).as_posix()
            print(f"{rel}:{line_no}: {reason}")
            print(f"    {line.strip()}")
            total += 1

    if total:
        print(
            f"\n✗ {total} direct credential env-access violation(s) found "
            f"across {len(files)} file(s) scanned.",
            file=sys.stderr,
        )
        print(
            "  Fix: from agent.secret_scope import get_secret; use "
            "get_secret(NAME) instead of os.getenv/os.environ. If the read "
            "is intentional (reviewed, non-profile-scoped), suppress with "
            "`# credential-lint: ok` on the same line.",
            file=sys.stderr,
        )
        return 1

    print(f"✓ No credential env-access violations found ({len(files)} file(s) scanned).")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
