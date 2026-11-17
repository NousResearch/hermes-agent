"""The boundary config: one strict reader, plus startup validation.

**Reader.** ``hermes_cli.config.read_raw_config`` returns ``{}`` for a
config.yaml that fails to parse — for a boundary that reads as "not
configured", i.e. every guard OFF, and one YAML typo during an admin's level
edit would open them all (and, with 4e off too, let the agent make it
permanent). Every boundary check — the Notion write guard, the Slack posting
policy (tool, cron delivery, gated ingress, ``send_message``), the 4e
self-modification guard, autonomy levels, the Zendesk / posting
preconditions — reads through ``read_boundary_config`` instead:

- parsed mapping → it (and remembered as the last known good for that path);
- parse error, non-mapping or empty file → the LAST KNOWN GOOD parse of that
  path in this process; with none, ``UnreadableConfig`` — a deny sentinel;
- missing file → "not configured" (``{}``) only if no good parse of that
  path was ever seen; otherwise the last known good.

``UnreadableConfig`` is a dict pre-filled with the most restrictive values
(write guard on, agent_write_guard on, empty post allowlist, autonomy
paused), and every check also tests ``is_unreadable`` and denies with a clear
message: writes refused, posting refused, write guard ON, autonomy ``off``.
Parse failures are logged at ERROR once per (path, file version).

**Validation.** The guards read config per call and fail closed; startup
validation only makes a broken block LOUD at gateway start: the write guard,
every ``autonomy.actions`` level, the veto settings, the posting lists.
"""
from __future__ import annotations

import copy
import logging
import threading
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class UnreadableConfig(dict):
    """Deny sentinel: config.yaml could not be parsed and no good parse is known."""

    def __init__(self, path: str = "", reason: str = ""):
        super().__init__({
            "notion": {"write_guard": {"enabled": True}},
            "agent_write_guard": True,
            "slack": {"post_allowed_channels": [], "post_gated_channels": []},
            "autonomy": {"paused": True, "actions": {}},
        })
        self.path = path
        self.reason = reason


def is_unreadable(cfg) -> bool:
    return isinstance(cfg, UnreadableConfig)


def unreadable_message(cfg) -> str:
    where = getattr(cfg, "path", "") or "config.yaml"
    return (f"{where} could not be parsed and no earlier good version is known — every "
            "boundary-guarded action is refused until an admin fixes it.")


_lock = threading.Lock()
_LAST_GOOD: dict = {}     # path -> dict
_VERSION: dict = {}       # path -> ((mtime_ns, size), dict | None)
_LOGGED: set = set()


def clear_cache() -> None:
    with _lock:
        _LAST_GOOD.clear()
        _VERSION.clear()
        _LOGGED.clear()


def _parse(text: str):
    try:
        from utils import fast_safe_load
        return fast_safe_load(text)
    except ImportError:  # pragma: no cover
        import yaml
        return yaml.safe_load(text)


def read_boundary_config(home: Optional[Path] = None) -> dict:
    """The active profile's (or ``home``'s) config.yaml, strictly. See module doc."""
    if home is None:
        from hermes_cli.config import get_config_path
        path = Path(get_config_path())
    else:
        path = Path(home) / "config.yaml"
    key = str(path)
    try:
        st = path.stat()
    except FileNotFoundError:
        with _lock:
            good = _LAST_GOOD.get(key)
        return copy.deepcopy(good) if good is not None else {}
    except OSError as e:
        return _fallback(key, f"stat failed: {e}", None)
    version = (st.st_mtime_ns, st.st_size)
    with _lock:
        hit = _VERSION.get(key)
    if hit is not None and hit[0] == version:
        if hit[1] is not None:
            return copy.deepcopy(hit[1])
        return _fallback(key, "parse failed", version)
    try:
        data = _parse(path.read_text(encoding="utf-8"))
        reason = "" if isinstance(data, dict) else f"top level is {type(data).__name__}, not a mapping"
    except Exception as e:
        data, reason = None, f"{type(e).__name__}: {e}"
    if reason:
        with _lock:
            _VERSION[key] = (version, None)
        return _fallback(key, reason, version)
    with _lock:
        _VERSION[key] = (version, copy.deepcopy(data))
        _LAST_GOOD[key] = copy.deepcopy(data)
    return data


def _fallback(key: str, reason: str, version) -> dict:
    with _lock:
        good = _LAST_GOOD.get(key)
        first = (key, version) not in _LOGGED
        _LOGGED.add((key, version))
    if first:
        logger.error("[boundary] %s could not be parsed (%s) — using %s", key, reason,
                     "the last good version" if good is not None else "DENY-ALL")
    if good is not None:
        return copy.deepcopy(good)
    return UnreadableConfig(key, reason)


def boundary_config_errors(cfg) -> list[str]:
    if is_unreadable(cfg):
        return [unreadable_message(cfg)]
    if not isinstance(cfg, dict):
        return []
    errors: list[str] = []
    from tools.notion_write_guard import validate
    errors += validate(cfg)

    from agent import autonomy
    block = cfg.get("autonomy") if isinstance(cfg.get("autonomy"), dict) else {}
    actions = block.get("actions") if isinstance(block.get("actions"), dict) else {}
    for action in actions:
        try:
            autonomy.resolve_level(str(action), cfg)
        except autonomy.AutonomyConfigError as e:
            errors.append(str(e))
    if block:
        try:
            autonomy.veto_settings(cfg)
            autonomy.veto_expiry_hours(cfg)
        except autonomy.AutonomyConfigError as e:
            errors.append(str(e))

    if isinstance(cfg.get("release_updates"), dict):
        from tools.release_update_post import settings
        rup_error = settings(cfg).config_error
        if rup_error:
            errors.append(f"release_updates: {rup_error}")

    from agent.slack_post_policy import post_policy
    policy = post_policy(cfg)
    if policy.error:
        errors.append(policy.error)
    elif policy.allowed is not None and policy.gated - policy.allowed:
        errors.append("slack.post_gated_channels lists channel(s) missing from "
                      f"slack.post_allowed_channels ({', '.join(sorted(policy.gated - policy.allowed))}); "
                      "posts there are refused.")
    return list(dict.fromkeys(errors))


def log_boundary_config_errors(profile_name: str, cfg: Optional[dict] = None) -> list[str]:
    """Log each error for ``profile_name`` (reads the active profile's
    config.yaml when ``cfg`` is None). Never raises."""
    try:
        if cfg is None:
            cfg = read_boundary_config()
        errors = boundary_config_errors(cfg)
    except Exception as e:  # pragma: no cover - defensive
        logger.error("[boundary] profile '%s': could not validate boundary config: %s", profile_name, e)
        return []
    for err in errors:
        logger.error("[boundary] profile '%s': %s", profile_name, err)
    return errors
