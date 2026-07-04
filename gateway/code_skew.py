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

**Precision (2026-07-03):** the crash this guards against is an ImportError on a
first-time lazy import of a *Python module*. A checkout can advance without any
importable Python changing — a docs-only, locale-only, skill-only, YAML-only, or
test-only deploy moves the SHA but cannot cause a stale-module import crash. So
when the SHA drifts we look at *which files* changed and only report skew if a
**runtime Python module** (a ``*.py`` outside the test/docs trees) differs
between the boot and disk revisions. This kills the false-alarm class where every
docs/skill deploy refused a ``/model`` switch, while staying conservative: if the
file-level diff can't be computed for any reason, we fall back to the original
"any drift refuses" behavior — a needless refuse is a minor annoyance, a missed
skew is a crash.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_boot_fingerprint: str | None = None

# Path prefixes whose ``*.py`` files are NOT imported by the running gateway, so
# a change confined to them cannot cause a stale-module import crash. Kept
# deliberately small and conservative — anything not listed here that is a
# ``*.py`` file is treated as runtime code (fail toward refusing).
_NON_RUNTIME_PY_PREFIXES: tuple[str, ...] = ("tests/", "docs/")


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


def _sha(fingerprint: str) -> str | None:
    """Extract the raw commit sha from a ``git:<ref>:<sha>`` fingerprint.

    Returns ``None`` when the sha is missing or ``unresolved`` (a ref whose
    object we couldn't read) — those can't seed a ``git diff``.
    """
    sha = fingerprint.rsplit(":", 1)[-1].strip()
    if not sha or sha == "unresolved":
        return None
    return sha


def _is_runtime_python(path: str) -> bool:
    """True if ``path`` is a Python module the gateway could import at runtime.

    Only ``*.py`` files count (a docs/locale/YAML/skill change can't cause a
    stale-import crash), and files under the test/docs trees are excluded (they
    are never imported by the running gateway).
    """
    if not path.endswith(".py"):
        return False
    return not path.startswith(_NON_RUNTIME_PY_PREFIXES)


def _runtime_python_changed(boot_sha: str, disk_sha: str) -> bool | None:
    """Whether any runtime ``*.py`` module differs between two revisions.

    Returns ``True``/``False`` on a successful diff, or ``None`` if the diff
    could not be computed (git error, missing object) so the caller can fall
    back to the conservative "any drift is skew" behavior.
    """
    try:
        proc = subprocess.run(
            ["git", "-C", str(_PROJECT_ROOT), "diff", "--name-only",
             f"{boot_sha}..{disk_sha}"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    changed = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    # An empty diff between two distinct sha strings is unexpected (they compared
    # unequal upstream); treat "no files" as inconclusive so we stay conservative.
    if not changed:
        return None
    return any(_is_runtime_python(p) for p in changed)


def detect_code_skew() -> tuple[str, str] | None:
    """Return ``(boot_rev, disk_rev)`` short labels if the checkout drifted
    since boot *in a way that risks a stale-module import crash*, else ``None``.

    A SHA change alone is not enough: only a change to a runtime Python module
    can cause the ImportError this guards against. When the sha drifts we diff
    the two revisions and suppress the skew if no runtime ``*.py`` changed
    (docs/skill/locale/test/YAML-only deploys). If the file-level diff can't be
    computed we conservatively report the skew (a needless refuse beats a
    missed crash).
    """
    if _boot_fingerprint is None:
        return None
    current = _fingerprint()
    if current is None or current == _boot_fingerprint:
        return None

    boot_sha = _sha(_boot_fingerprint)
    disk_sha = _sha(current)
    if boot_sha is not None and disk_sha is not None:
        if boot_sha == disk_sha:
            # Same commit, only the ref/branch label differs — no code changed,
            # so there's nothing to diff and no import risk. Don't refuse.
            return None
        runtime_changed = _runtime_python_changed(boot_sha, disk_sha)
        if runtime_changed is False:
            # Drift confined to non-imported files (docs/skills/tests/locale/YAML).
            return None
        # runtime_changed is True (real risk) or None (couldn't tell) -> refuse.

    return _short(_boot_fingerprint), _short(current)
