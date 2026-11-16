"""Opt-in per-profile self-modification guard (4e).

A profile opts in from its own ``config.yaml``::

    agent_write_guard: true
    # or
    agent_write_guard:
      enabled: true
      extra_paths: [/srv/kb, notes/locked]

For an opted-in profile, the agent's file tools and ``skill_manage`` refuse
writes to that profile's ``config.yaml``, ``SOUL.md``, ``cron/``,
``scripts/`` and ``.env*``, to every ``skills.external_dirs`` entry, and to
each ``extra_paths`` entry (relative entries resolve under the profile home).

Keyed on the **owning** profile's home (``get_hermes_home()`` under the turn's
override) — never the process-wide ``HERMES_WRITE_SAFE_ROOT``, which would
restrict every bot in the gateway. Profiles that do not opt in are unchanged.
Denials go to the ``data_access_audit`` trail.
"""
from __future__ import annotations

import logging
import os
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_cache: Dict[str, Tuple[Tuple[int, int], Any]] = {}

_PROTECTED_FILES = ("config.yaml", "SOUL.md")
_PROTECTED_DIRS = ("cron", "scripts")


def _real(path) -> str:
    return os.path.realpath(os.path.expanduser(str(path)))


def _under(path: str, root: str) -> bool:
    return path == root or path.startswith(root.rstrip(os.sep) + os.sep)


def _guard_setting(home: Path) -> Any:
    cfg = home / "config.yaml"
    try:
        st = cfg.stat()
    except OSError:
        return None
    key = str(cfg)
    fp = (st.st_mtime_ns, st.st_size)
    with _lock:
        hit = _cache.get(key)
        if hit is not None and hit[0] == fp:
            return hit[1]
    try:
        import yaml

        data = yaml.safe_load(cfg.read_text(encoding="utf-8")) or {}
    except Exception:
        logger.debug("agent_write_guard: unreadable %s", cfg, exc_info=True)
        data = {}
    setting = data.get("agent_write_guard") if isinstance(data, dict) else None
    with _lock:
        _cache[key] = (fp, setting)
    return setting


def _enabled_and_extra(setting: Any) -> Tuple[bool, List[str]]:
    if setting is True:
        return True, []
    if isinstance(setting, dict):
        enabled = setting.get("enabled", True)
        if isinstance(enabled, str):
            enabled = enabled.strip().lower() in {"true", "1", "yes", "on"}
        extra = setting.get("extra_paths") or []
        if isinstance(extra, str):
            extra = [extra]
        return bool(enabled), [str(p) for p in extra if str(p).strip()]
    return False, []


def protected_roots(home: Optional[Path] = None) -> Optional[List[Tuple[str, bool]]]:
    """``[(path, is_tree)]`` the guard protects, or None when not opted in."""
    from hermes_constants import get_hermes_home

    home = Path(home or get_hermes_home())
    enabled, extra = _enabled_and_extra(_guard_setting(home))
    if not enabled:
        return None
    home_real = _real(home)
    roots: List[Tuple[str, bool]] = [
        (os.path.join(home_real, name), False) for name in _PROTECTED_FILES
    ]
    roots += [(os.path.join(home_real, name), True) for name in _PROTECTED_DIRS]
    try:
        from agent.skill_utils import get_external_skills_dirs

        roots += [(_real(d), True) for d in get_external_skills_dirs()]
    except Exception:
        logger.debug("agent_write_guard: external_dirs lookup failed", exc_info=True)
    for entry in extra:
        p = Path(os.path.expanduser(entry))
        if not p.is_absolute():
            p = Path(home_real) / p
        roots.append((_real(p), True))
    return roots


def write_guard_denial(path) -> Optional[str]:
    """Return a denial if the opted-in owning profile protects *path*."""
    try:
        roots = protected_roots()
        if roots is None:
            return None
        from hermes_constants import get_hermes_home

        target = _real(path)
        home_real = _real(get_hermes_home())
        hit = any(_under(target, root) if tree else target == root for root, tree in roots)
        if not hit and os.path.dirname(target) == home_real:
            hit = os.path.basename(target).startswith(".env")
        if not hit:
            return None
        from agent.multiplex_file_guard import audit_denial

        audit_denial("write_guard_denied", str(path))
        return (
            f"Write denied: '{path}' is protected by this profile's "
            "agent_write_guard (its own config, persona, cron, scripts, "
            "secrets and externally owned skills). Ask an operator to change it."
        )
    except Exception:
        logger.debug("agent_write_guard check failed for %s", path, exc_info=True)
        return None
