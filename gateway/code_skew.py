"""Detect when the gateway is running stale code after a hot ``git pull``.

The gateway is a single long-lived process; its ``sys.modules`` is frozen at
boot. If the checkout is updated underneath it (a manual ``git pull``, or the
window before ``hermes update``'s graceful restart fires), a first-time lazy
import on a new code path can resolve a freshly-pulled consumer module against a
stale cached dependency -> ImportError (see
``tests/test_stale_utils_module_import.py`` for the exact failure).

We snapshot the checkout revision at gateway startup and compare on demand, so
risky callers (e.g. ``/model`` switching) can refuse with a clear "restart the
gateway" message instead of crashing on a cryptic import error.

If the revision can't be read (non-git install, IO error), the boot snapshot
stays ``None`` and skew detection no-ops — it never produces a false positive.

We also persist the boot fingerprint to ``HERMES_HOME`` so the next gateway
boot can detect that the checkout advanced while the previous gateway was
running, and purge stale ``__pycache__`` before any import picks up bytecode
compiled against the old source (see ``purge_stale_pycache``).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_boot_fingerprint: str | None = None
_last_boot_fingerprint_file = (
    Path(os.environ.get("HERMES_HOME", Path.home() / ".hermes"))
    / ".gateway_boot_fingerprint"
)

logger = logging.getLogger("gateway.code_skew")


def _fingerprint() -> str | None:
    """Current checkout fingerprint, reusing the CLI's git-rev reader.

    ``hermes_cli.main`` is always already imported in a gateway process (it's
    the entry point), so this import is free and avoids duplicating the
    worktree-aware ref resolution.
    """
    try:
        from hermes_cli.main import _read_git_revision_fingerprint

        return _read_git_revision_fingerprint(_PROJECT_ROOT)
    except Exception:
        return None


def record_boot_fingerprint() -> None:
    """Snapshot the checkout revision at gateway startup (idempotent).

    Also persists the fingerprint to ``HERMES_HOME`` so the next boot can
    detect drift and purge stale ``__pycache__`` before imports pick up
    bytecode compiled against the old source.
    """
    global _boot_fingerprint
    if _boot_fingerprint is None:
        _boot_fingerprint = _fingerprint()
    _persist_boot_fingerprint(_boot_fingerprint)


def _persist_boot_fingerprint(fingerprint: str | None) -> None:
    """Write the boot fingerprint to ``HERMES_HOME`` for next-boot comparison."""
    if fingerprint is None:
        return
    try:
        _last_boot_fingerprint_file.parent.mkdir(parents=True, exist_ok=True)
        _last_boot_fingerprint_file.write_text(fingerprint, encoding="utf-8")
    except Exception:
        logger.debug("Failed to persist boot fingerprint", exc_info=True)


def purge_stale_pycache() -> bool:
    """Purge stale ``__pycache__`` directories if the checkout advanced.

    At gateway boot, compares the persisted fingerprint from the *previous*
    gateway run with the current checkout. If they differ, Python's
    ``__pycache__`` may contain bytecode compiled against the old source
    (``.pyc`` files whose header mtime matches the old ``.py`` because git
    operations can preserve mtime). Deleting the ``__pycache__`` dirs forces
    a clean recompile on first import.

    Returns ``True`` if a purge was performed, ``False`` otherwise.
    """
    current = _fingerprint()
    if current is None:
        return False
    try:
        previous = _last_boot_fingerprint_file.read_text(encoding="utf-8").strip()
    except (FileNotFoundError, OSError):
        return False
    if not previous or previous == current:
        return False

    purged = 0
    for pycache in _PROJECT_ROOT.rglob("__pycache__"):
        # Skip .venv and node_modules to avoid slow/unnecessary I/O
        if ".venv" in pycache.parts or "node_modules" in pycache.parts:
            continue
        try:
            for pyc in pycache.glob("*.pyc"):
                pyc.unlink()
                purged += 1
        except OSError:
            pass
    if purged:
        logger.info(
            "Purged %d stale .pyc file(s) — checkout advanced from %s to %s",
            purged,
            _short(previous),
            _short(current),
        )
    return purged > 0


def _short(fingerprint: str) -> str:
    """Render a ``git:<ref>:<sha>`` fingerprint as a compact label."""
    sha = fingerprint.rsplit(":", 1)[-1]
    if sha and sha != "unresolved" and len(sha) > 10:
        return sha[:10]
    return sha or fingerprint


def detect_code_skew() -> tuple[str, str] | None:
    """Return ``(boot_rev, disk_rev)`` short labels if the checkout drifted
    since boot, else ``None``."""
    if _boot_fingerprint is None:
        return None
    current = _fingerprint()
    if current is None or current == _boot_fingerprint:
        return None
    return _short(_boot_fingerprint), _short(current)
