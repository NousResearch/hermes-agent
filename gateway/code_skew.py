"""Detect when the gateway is running stale code after a hot ``git pull``.

The gateway's ``sys.modules`` is frozen at boot.  If the checkout is updated
underneath it, a first-time lazy import can resolve a freshly-pulled module
against a stale cached dependency -> ImportError.  We snapshot the revision at
startup so risky callers (e.g. ``/model`` switching) can refuse with a clear
"restart the gateway" message.  If the revision can't be read (non-git install,
IO error) the boot snapshot stays ``None`` and detection no-ops — never a false positive.
"""

from __future__ import annotations

import re
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_boot_fingerprint: str | None = None

# A resolved commit id inside a ``git:<ref>:<sha>`` fingerprint. Short and full
# object names both match; ``unresolved`` (no ref file, no packed-ref) does not.
_COMMIT_RE = re.compile(r"[0-9a-f]{7,64}")


def _fingerprint() -> str | None:
    """Current checkout fingerprint via the CLI's worktree-aware git-rev reader
    (``hermes_cli.main`` is always already imported in a gateway process)."""
    try:
        from hermes_cli.main import _read_git_revision_fingerprint

        return _read_git_revision_fingerprint(_PROJECT_ROOT)
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
    return sha[:10] if sha and sha != "unresolved" and len(sha) > 10 else (sha or fingerprint)


def _commit_id(fingerprint: str) -> str | None:
    """The commit id inside a ``git:<ref>:<sha>`` fingerprint, or None if unresolved."""
    sha = fingerprint.rsplit(":", 1)[-1].strip().lower()
    return sha if _COMMIT_RE.fullmatch(sha) else None


def detect_code_skew() -> tuple[str, str] | None:
    """``(boot_rev, disk_rev)`` short labels if the checkout drifted since boot, else ``None``.

    Identity is the COMMIT, not the ref name: switching branches at the same
    commit (a checkout between refs, a worktree of the same commit) leaves every
    file on disk byte-identical, so no lazily-imported module can be stale.
    Comparing raw fingerprints reported those as skew and blocked the model
    picker with a "restart required" banner until the next process restart.
    What is stored is unchanged, so the callers that key on the whole string —
    the ``__pycache__`` sweep stamp (``hermes_cli.main_web_build``) and the
    Termux bundled-skill sync key (``_termux_bundled_skills_fingerprint``) —
    behave exactly as before; only the safety decision narrows here.
    """
    current = _fingerprint() if _boot_fingerprint is not None else None
    if current is None or current == _boot_fingerprint:
        return None
    boot_commit = _commit_id(_boot_fingerprint)
    if boot_commit is not None and boot_commit == _commit_id(current):
        return None
    return _short(_boot_fingerprint), _short(current)
