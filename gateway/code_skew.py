"""Detect when the gateway is running stale code after a hot ``git pull``.

The gateway's ``sys.modules`` is frozen at boot.  If the checkout is updated
underneath it, a first-time lazy import can resolve a freshly-pulled module
against a stale cached dependency -> ImportError.  We snapshot the revision at
startup so risky callers (e.g. ``/model`` switching) can refuse with a clear
"restart the gateway" message.  If the revision can't be read (non-git install,
IO error) the boot snapshot stays ``None`` and detection no-ops — never a false positive.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_boot_fingerprint: str | None = None


def _fingerprint() -> str | None:
    """Current checkout fingerprint via the CLI's worktree-aware git-rev reader
    (``hermes_cli.main`` is always already imported in a gateway process)."""
    try:
        from hermes_cli.main import _read_git_revision_fingerprint

        return _read_git_revision_fingerprint(_PROJECT_ROOT)
    except Exception:
        return None


def record_boot_fingerprint() -> None:
    """Snapshot the checkout at gateway startup (idempotent).

    Two halves, ONE boot record. The ref half (``_boot_fingerprint``) is the cheap fast path it
    always was. The CONTENT half is the load-bearing one (ruling ``t_8fed34c8`` R2): the measured
    incident moved the WORKING TREE while ``HEAD`` stayed put, so a ref-only snapshot reports
    "no skew" for exactly the window that froze a mixed revision into a live gateway. It is
    recorded through :mod:`hermes_cli.tree_fingerprint` -- the same single record the dispatcher
    tick's consumer gate reads -- never as a second one.
    """
    global _boot_fingerprint
    if _boot_fingerprint is None:
        _boot_fingerprint = _fingerprint()
        try:
            from hermes_cli import tree_fingerprint

            tree_fingerprint.record_boot()
        except Exception:
            logger.debug("code-skew: could not record the boot content fingerprint", exc_info=True)


def _short_content(fingerprint: str) -> str:
    """Render a ``tree-content:<sha256>`` fingerprint as a compact label."""
    try:
        from hermes_cli.tree_fingerprint import short as _short_fp

        return _short_fp(fingerprint)
    except Exception:  # pragma: no cover - a tree without the module keeps the raw value
        return fingerprint


def detect_content_skew() -> tuple[str, str] | None:
    """``(boot_label, disk_label)`` when the tree's CONTENT moved since boot, else ``None``.

    The complement of :func:`detect_code_skew`, and the half a caller must not skip: the ref
    check is blind to a working-tree rewrite with ``HEAD`` unchanged. Fail-open by construction --
    no boot record, an unreadable tree, or an unchanged tree all return ``None``.
    """
    try:
        from hermes_cli import tree_fingerprint

        skew = tree_fingerprint.detect_skew()
    except Exception:
        return None
    if skew is None:
        return None
    return _short_content(skew[0]), _short_content(skew[1])


def _short(fingerprint: str) -> str:
    """Render a ``git:<ref>:<sha>`` fingerprint as a compact label."""
    sha = fingerprint.rsplit(":", 1)[-1]
    return sha[:10] if sha and sha != "unresolved" and len(sha) > 10 else (sha or fingerprint)


def current_code_sha() -> str | None:
    """Full SHA for the checkout currently on disk, or None when unresolved."""
    fingerprint = _fingerprint()
    if fingerprint is None:
        return None
    sha = fingerprint.rsplit(":", 1)[-1]
    return sha if sha and sha != "unresolved" else None


def detect_code_skew() -> tuple[str, str] | None:
    """``(boot_rev, disk_rev)`` short labels if the checkout drifted since boot, else ``None``."""
    current = _fingerprint() if _boot_fingerprint is not None else None
    if current is None or current == _boot_fingerprint:
        return None
    return _short(_boot_fingerprint), _short(current)
