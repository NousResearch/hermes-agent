"""Lint guard: no new raw ``os.environ.copy()`` spawn-env sites.

Every child-process env in the codebase must be built through
``tools.environments.local.build_subprocess_env`` (or its sibling
``hermes_subprocess_env`` / ``_sanitize_subprocess_env``, which the factory
wraps) so profile-home propagation and secret-scrubbing have a single owner.
History: ~11 commits over 6 months each fixed one more spawn site that missed
``HERMES_HOME`` or secret-scrub propagation.

This test greps the source tree for ``os.environ.copy()`` appearing within
``PROXIMITY_LINES`` lines of a spawn call (``Popen`` / ``subprocess.run`` /
``create_subprocess*`` / ``PtyProcess.spawn`` / ``execvpe``) and asserts every
hit is in the explicit allowlist below.  If you are adding a new spawn site:

* use ``build_subprocess_env(...)`` — with ``scrub_secrets=False,
  inherit_profile_home=False`` if you need today's exact-inherit behavior; or
* consciously add the file to ``ALLOWED_RAW_SPAWN_ENV_FILES`` with a comment
  explaining why the factory cannot be used.
"""

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# Directories that make up the shipped source tree.
SCAN_DIRS = ("agent", "hermes_cli", "tools", "gateway", "cron", "tui_gateway")
SCAN_ROOT_FILES = ("cli.py", "hermes_constants.py")

# How many lines around an `os.environ.copy()` we look for a spawn call.
PROXIMITY_LINES = 20

SPAWN_RE = re.compile(
    r"\bPopen\b|\bsubprocess\.run\b|\bcreate_subprocess|\bPtyProcess\.spawn\b"
    r"|\bexecvpe\b|\bptyprocess\.PtyProcess\b|\bspawn\("
)
COPY_RE = re.compile(r"\bos\.environ\.copy\(\)")

# ---------------------------------------------------------------------------
# ALLOWLIST — intentionally-raw sites.  Each entry is a relative posix path.
# Adding to this list is a conscious decision: document WHY inline here.
# ---------------------------------------------------------------------------
ALLOWED_RAW_SPAWN_ENV_FILES = {
    # THE owner module: _sanitize_subprocess_env / hermes_subprocess_env /
    # build_subprocess_env legitimately snapshot os.environ — everything else
    # delegates to them.
    "tools/environments/local.py",
    # Bang-shell (`!cmd` in the CLI) goes through build_subprocess_env(); the
    # only raw copy is the except-fallback for when the tools package itself
    # cannot be imported, so the user's typed command still runs.
    "hermes_cli/bang_shell.py",
}


def _iter_source_files():
    for d in SCAN_DIRS:
        root = REPO_ROOT / d
        if root.is_dir():
            yield from root.rglob("*.py")
    for f in SCAN_ROOT_FILES:
        p = REPO_ROOT / f
        if p.is_file():
            yield p


def _raw_spawn_env_sites():
    """Return [(relpath, lineno)] of os.environ.copy() near a spawn call."""
    sites = []
    for path in _iter_source_files():
        rel = path.relative_to(REPO_ROOT).as_posix()
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        for i, line in enumerate(lines):
            stripped = line.lstrip()
            if stripped.startswith("#"):
                continue  # comments referencing the old pattern are fine
            if not COPY_RE.search(line.split("#", 1)[0]):
                continue
            lo = max(0, i - PROXIMITY_LINES)
            hi = min(len(lines), i + PROXIMITY_LINES + 1)
            window = "\n".join(lines[lo:hi])
            if SPAWN_RE.search(window):
                sites.append((rel, i + 1))
    return sites


def test_no_new_raw_environ_copy_spawn_sites():
    sites = _raw_spawn_env_sites()
    offenders = [
        f"{rel}:{lineno}"
        for rel, lineno in sites
        if rel not in ALLOWED_RAW_SPAWN_ENV_FILES
    ]
    assert not offenders, (
        "New raw os.environ.copy() spawn-env site(s) found:\n  "
        + "\n  ".join(offenders)
        + "\nUse tools.environments.local.build_subprocess_env() instead "
        "(scrub_secrets=False, inherit_profile_home=False preserves exact "
        "legacy behavior), or consciously extend "
        "ALLOWED_RAW_SPAWN_ENV_FILES in this test with a justification."
    )


def test_allowlist_entries_still_exist():
    """Prune the allowlist when files are removed/renamed."""
    for rel in ALLOWED_RAW_SPAWN_ENV_FILES:
        assert (REPO_ROOT / rel).is_file(), (
            f"Allowlist entry {rel} no longer exists — remove it from "
            "ALLOWED_RAW_SPAWN_ENV_FILES."
        )
