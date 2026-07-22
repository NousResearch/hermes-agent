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

We also persist the boot fingerprint to ``HERMES_HOME``. On the next boot,
supported pre-import gateway launchers can detect that the checkout advanced and
purge stale ``__pycache__`` before importing ``gateway.run``. The compatible
``python -m gateway.run`` path does not provide that pre-import guarantee.
"""

from __future__ import annotations

import logging
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_boot_fingerprint: str | None = None
logger = logging.getLogger("gateway.code_skew")


def _fingerprint_file() -> Path:
    """Return the path to the persisted boot fingerprint.

    Resolved at call time (not import time) via the canonical
    ``get_hermes_home()`` so profile overrides and context-local
    HERMES_HOME changes are respected.
    """
    from hermes_constants import get_hermes_home

    return get_hermes_home() / ".gateway_boot_fingerprint"


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

    Also persists the fingerprint to ``HERMES_HOME`` for supported launchers to
    compare before importing ``gateway.run`` on a later boot.
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
        fp_file = _fingerprint_file()
        fp_file.parent.mkdir(parents=True, exist_ok=True)
        fp_file.write_text(fingerprint, encoding="utf-8")
    except Exception:
        logger.debug("Failed to persist boot fingerprint", exc_info=True)


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
