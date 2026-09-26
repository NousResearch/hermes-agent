"""Best-effort cgroup v2 isolation for spawned language servers.

``attach()`` moves a freshly spawned server PID into a dedicated cgroup
(``<cgroup root>/hermes-lsp``) carrying its own ``memory.max``, so a large
workspace indexes inside a known memory budget instead of competing with the
gateway for the shared cgroup limit (a spike that used to be able to push the
whole gateway cgroup into the kernel OOM killer).

Every failure degrades to a no-op: the server keeps sharing the gateway's
cgroup exactly as before (pre-isolation behaviour) and one warning is logged
so the fallback is visible. Unavailable cases: non-Linux hosts, cgroup v1,
read-only or non-writable cgroup mounts (restricted containers), missing
permissions, or ``lsp.cgroup_isolate: false``.

``HERMES_LSP_CGROUP_ROOT`` overrides the cgroup root (tests point it at a temp
directory; the default is the standard cgroup v2 mount).
"""
from __future__ import annotations

import logging
import os
import pathlib
import sys

logger = logging.getLogger(__name__)

CGROUP_ROOT_ENV = "HERMES_LSP_CGROUP_ROOT"
DIR_NAME = "hermes-lsp"

# One warning per process: after the first fallback notice, repeats add noise, not information.
_warned = False


def cgroup_root() -> pathlib.Path:
    return pathlib.Path(os.environ.get(CGROUP_ROOT_ENV) or "/sys/fs/cgroup")


def _warn_once(reason: str) -> None:
    global _warned
    if _warned:
        return
    _warned = True
    logger.warning(
        "LSP cgroup isolation unavailable (%s); language servers share the gateway's "
        "cgroup with no independent memory cap. See lsp.md (cgroup_isolate) for details.",
        reason,
    )


def attach(pid: int, memory_max_mb: int = 0) -> bool:
    """Move ``pid`` into the dedicated LSP cgroup.

    Returns True iff the process was moved (``memory_max_mb`` may still be
    unapplied if the memory controller is unavailable — separation holds, only
    the cap is lost). Never raises: any OSError means "isolation unavailable"
    and the caller falls back to the shared cgroup.
    """
    if not sys.platform.startswith("linux"):
        _warn_once(f"platform {sys.platform}")
        return False
    root = cgroup_root()
    path = root / DIR_NAME
    try:
        path.mkdir(exist_ok=True)
        # Enable the memory controller for children when we own the tree. Best
        # effort: already enabled, or not permitted (non-root, no delegation)
        # -> the memory.max write below fails and we keep separation only.
        control = root / "cgroup.subtree_control"
        try:
            if control.exists() and "memory" not in control.read_text(encoding="utf-8"):
                control.write_text("+memory", encoding="utf-8")
        except OSError:
            pass
        # ponytail: leader PID only — LSP servers are long-lived singletons and
        # rarely fork; move the whole process group here if server helpers fork.
        (path / "cgroup.procs").write_text(str(pid), encoding="utf-8")
    except OSError as e:
        _warn_once(str(e))
        logger.debug("LSP cgroup attach failed for pid %s: %s", pid, e)
        return False
    if memory_max_mb > 0:
        try:
            (path / "memory.max").write_text(str(memory_max_mb * 1024 * 1024), encoding="utf-8")
        except OSError as e:
            # cgroup.procs write succeeded above, so the split already happened;
            # losing only the cap keeps the gateway safe from accounting surprises.
            logger.debug("LSP cgroup memory.max not applied (%s); separation only", e)
    return True


def reset_warning_for_tests() -> None:
    """Forget the one-shot warning so tests can assert the fallback notice."""
    global _warned
    _warned = False
