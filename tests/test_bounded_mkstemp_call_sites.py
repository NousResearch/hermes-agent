"""Persistent-dir temp files must go through ``utils.bounded_mkstemp``.

``tempfile.mkstemp`` retries candidate names up to ``TMP_MAX`` times
(2**31-1 on Windows) and its bpo-22107 workaround treats every
``PermissionError`` as a name collision whenever ``os.access(dir, os.W_OK)``
claims the directory is writable.  ``os.access`` cannot see ACL denials on
Windows, so ``mkstemp`` aimed at a directory the process may not write to
(HERMES_HOME under a sandboxed agent shell, a hardened service account)
busy-spins one CPU core for hours instead of raising.

System-temp usage (no ``dir=``) is out of scope: tempfile falls back
across several candidate directories for those, and an unwritable system
temp dir breaks the interpreter long before it breaks Hermes.  Anything
that passes ``dir=`` targets a persistent location and must use the
bounded helper instead.
"""

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Product code reachable from a running `hermes` process.  Skills under
# optional-skills/ run as standalone scripts and cannot import utils.
PRODUCT_DIRS = ("agent", "cron", "gateway", "hermes_cli", "tools", "plugins")

# Matches mkstemp calls that pass dir= (via `tempfile` or any alias ending
# in "tempfile", e.g. `_tempfile`), across line breaks.
_MKSTEMP_WITH_DIR = re.compile(
    r"tempfile\.mkstemp\((?:[^()]|\([^()]*\))*\bdir\s*=", re.S
)


def test_no_unbounded_mkstemp_into_persistent_dirs():
    files = [REPO_ROOT / "utils.py"]
    for d in PRODUCT_DIRS:
        files.extend(sorted((REPO_ROOT / d).rglob("*.py")))

    violations = [
        path.relative_to(REPO_ROOT).as_posix()
        for path in files
        if _MKSTEMP_WITH_DIR.search(
            path.read_text(encoding="utf-8", errors="replace")
        )
    ]

    assert not violations, (
        "tempfile.mkstemp(dir=...) targets a persistent directory and "
        "busy-spins for hours on an ACL-denied dir on Windows (bpo-22107); "
        f"use utils.bounded_mkstemp instead: {violations}"
    )
