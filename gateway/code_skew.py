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
"""

from __future__ import annotations

from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_boot_fingerprint: str | None = None


def _fingerprint() -> str | None:
    """Current checkout fingerprint, reusing the CLI's git-rev reader.

    Look the reader up in ``sys.modules`` first. Under ``python -m
    hermes_cli.main`` (how the systemd unit launches the gateway) the CLI
    module is registered as ``__main__``, NOT as ``hermes_cli.main`` — so a
    plain ``from hermes_cli.main import ...`` would re-execute the whole CLI
    module, re-running its module-level ``_apply_profile_override()`` a second
    time with the already-consumed ``--profile`` flag stripped. That second
    pass falls back to the sticky ``active_profile`` and redirects
    HERMES_HOME into a different profile, which then trips the duplicate
    gateway PID guard for that profile and crash-loops the service.
    """
    try:
        import sys

        module = sys.modules.get("hermes_cli.main") or sys.modules.get("__main__")
        reader = getattr(module, "_read_git_revision_fingerprint", None)
        if reader is None:
            from hermes_cli.main import _read_git_revision_fingerprint as reader

        return reader(_PROJECT_ROOT)
    except Exception:
        return None


def record_boot_fingerprint() -> None:
    """Snapshot the checkout revision at gateway startup (idempotent)."""
    global _boot_fingerprint
    if _boot_fingerprint is None:
        _boot_fingerprint = _fingerprint()


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
