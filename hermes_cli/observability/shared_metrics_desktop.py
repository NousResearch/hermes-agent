"""Shared-metrics facts the Desktop app reports about itself: area use, friction, first-run steps,
dislike signals, and one daily report (Bot Mode / Sessions split, button presses per action).

The renderer dedupes on its side (a per-day set persisted only while collection is on); the backend
latches again here so a second window, a reconnect replay or a cleared localStorage cannot inflate a
row: feature use once per (profile, UTC day, area), friction and dislike capped per (profile, UTC day),
onboarding once per (profile, step, event) ever, the daily report once per (profile, usage day). Every value collapses onto the closed
sets in ``shared_metrics_contract``; raw client words are never recorded. All latches are claimed only
after the ``enabled()`` gate (inside the ``_emit`` builders), so a disabled profile writes nothing.
"""

from __future__ import annotations

import os
import re
import threading
from datetime import datetime, timezone
from typing import Any

from . import shared_metrics_contract as contract
from .shared_metrics_events import _emit

# Per-day ceiling for one (kind, detail) friction pair: a toast loop or a janky session still shows
# up as "a lot", but cannot flood the store.
FRICTION_DAILY_CAP = 50
ONBOARDING_LATCH_DIRNAME = "desktop_onboarding"
DAILY_ACTION_ROWS_MAX = 200
_DAY = re.compile(r"\d{4}-\d{2}-\d{2}")
_LATCH_LOCK = threading.Lock()
# (home, day, key) -> count; pruned to the current day so the process never grows it.
_daily: dict[tuple[str, str, str], int] = {}
# (home, usage day) already reported by a daily Desktop report in this process.
_reported_days: set[tuple[str, str]] = set()


def _utc_day() -> str:
    return datetime.now(timezone.utc).date().isoformat()


def _home() -> str:
    from hermes_constants import get_hermes_home

    return str(get_hermes_home())


def _word(value: Any) -> str:
    return value.strip().lower() if isinstance(value, str) else ""


def _claim_daily(key: str, *, cap: int = 1, day: str | None = None) -> bool:
    """True while ``key`` has been claimed fewer than ``cap`` times today in this profile."""
    today = _utc_day()
    slot = (_home(), day or today, key)
    with _LATCH_LOCK:
        for stale in [k for k in _daily if k[1] < today and k[1] != slot[1]]:
            del _daily[stale]
        if _daily.get(slot, 0) >= cap:
            return False
        _daily[slot] = _daily.get(slot, 0) + 1
    return True


def feature_area(area: Any) -> str:
    word = _word(area)
    if word in contract.DESKTOP_FEATURE_AREAS:
        return word
    return "settings_other" if word.startswith("settings_") else "other"


def friction_pair(kind: Any, detail: Any) -> tuple[str, str] | None:
    """``(kind, detail)`` with detail checked against ITS kind's set; unknown kinds record nothing."""
    kind_word = _word(kind)
    details = contract.DESKTOP_FRICTION_DETAILS.get(kind_word)
    if details is None:
        return None
    detail_word = _word(detail)
    return kind_word, detail_word if detail_word in details else "other"


def _feature_use_fields(*, area: Any) -> dict[str, str] | None:
    value = feature_area(area)
    return {"area": value} if _claim_daily(f"area:{value}") else None


def _friction_fields(*, kind: Any, detail: Any) -> dict[str, str] | None:
    pair = friction_pair(kind, detail)
    if pair is None or not _claim_daily(f"friction:{pair[0]}:{pair[1]}", cap=FRICTION_DAILY_CAP):
        return None
    return {"detail": pair[1], "kind": pair[0]}


def _claim_onboarding(step: str, event: str) -> bool:
    """Once per (step, event) per profile, across processes: an O_EXCL latch file (≤ ~51 of them)."""
    from hermes_constants import get_hermes_home

    directory = get_hermes_home() / "telemetry" / "shared_metrics" / ONBOARDING_LATCH_DIRNAME
    directory.mkdir(parents=True, exist_ok=True)
    try:
        os.close(os.open(directory / f"{step}.{event}", os.O_CREAT | os.O_EXCL | os.O_WRONLY))
    except FileExistsError:
        return False
    return True


def _onboarding_fields(*, step: Any, event: Any) -> dict[str, str] | None:
    step_word, event_word = _word(step), _word(event)
    if step_word not in contract.DESKTOP_ONBOARDING_STEPS or event_word not in contract.DESKTOP_ONBOARDING_EVENTS:
        return None
    return {"event": event_word, "step": step_word} if _claim_onboarding(step_word, event_word) else None


def _count(value: Any) -> int:
    return int(value) if isinstance(value, (int, float)) and not isinstance(value, bool) and value >= 0 else 0


def action_id(action: Any) -> str:
    """Action ids are case-sensitive registry ids (``view.toggleSidebar``): only strip them."""
    word = action.strip() if isinstance(action, str) else ""
    return word if word in contract.DESKTOP_ACTION_IDS else "other"


def _setting_leaves(config: dict, prefix: str = "") -> dict[str, Any]:
    leaves: dict[str, Any] = {}
    for key, value in config.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict) and value:
            leaves.update(_setting_leaves(value, path))
        else:
            leaves[path] = value
    return leaves


def _setting_direction(key: str) -> tuple[str, str]:
    """``(setting, direction)`` for a config key the Desktop settings page just saved: the key only if
    it is a DEFAULT_CONFIG leaf (the settings schema), the direction from comparing the saved value to
    the default here in the backend, so the value never leaves this process."""
    from hermes_cli.config import DEFAULT_CONFIG, load_config

    defaults = _setting_leaves(DEFAULT_CONFIG)
    if key not in defaults or len(key) > contract.DESKTOP_SETTING_KEY_MAX_LENGTH:
        return "other", "none"
    current: Any = load_config()
    for part in key.split("."):
        current = current.get(part) if isinstance(current, dict) else None
    return key, "to_default" if current == defaults[key] else "away_from_default"


def _dislike_fields(*, signal: Any, target: Any, setting: Any) -> dict[str, str] | None:
    signal_word = _word(signal)
    targets = contract.DESKTOP_DISLIKE_TARGETS.get(signal_word)
    if targets is None:
        return None
    if signal_word == "setting_off_default":
        key = setting.strip() if isinstance(setting, str) else ""
        setting_word, direction = _setting_direction(key)
        target_word = "setting"
    else:
        raw = target.strip() if isinstance(target, str) else ""
        target_word = raw if signal_word == "rage_click" and raw in targets else _word(raw)
        target_word = target_word if target_word in targets else "other"
        setting_word, direction = "none", "none"
    if not _claim_daily(f"dislike:{signal_word}", cap=FRICTION_DAILY_CAP):
        return None
    return {"direction": direction, "setting": setting_word, "signal": signal_word, "target": target_word}


def _mode_rows(modes: Any, bot_count: Any) -> list[tuple[str, dict[str, str]]]:
    rows: dict[str, tuple[str, dict[str, str]]] = {}
    for entry in modes if isinstance(modes, list) else []:
        mode = _word(entry.get("mode")) if isinstance(entry, dict) else ""
        active_ms, messages = (_count(entry.get("active_ms")), _count(entry.get("messages_sent"))) if mode else (0, 0)
        # Only modes actually used that day get a row.
        if mode not in contract.DESKTOP_MODES or mode in rows or not (active_ms or messages):
            continue
        rows[mode] = (contract.DESKTOP_MODE_USE_MARK, {
            "active_minutes_bucket": contract.desktop_active_minutes_bucket(active_ms),
            "bot_count_bucket": contract.size_bucket(_count(bot_count)),
            "messages_sent_bucket": contract.size_bucket(messages),
            "mode": mode,
        })
    return list(rows.values())


def _action_rows(actions: Any) -> list[tuple[str, dict[str, str]]]:
    totals: dict[tuple[str, str], int] = {}
    for entry in (actions if isinstance(actions, list) else [])[:DAILY_ACTION_ROWS_MAX]:
        if not isinstance(entry, dict) or _word(entry.get("via")) not in contract.DESKTOP_ACTION_VIAS:
            continue
        key = (action_id(entry.get("action")), _word(entry.get("via")))
        totals[key] = totals.get(key, 0) + _count(entry.get("count"))
    return [
        (contract.DESKTOP_ACTION_USE_MARK, {"action": action, "count_bucket": contract.size_bucket(count), "via": via})
        for (action, via), count in sorted(totals.items()) if count
    ]


def record_desktop_daily(*, day: Any, modes: Any, actions: Any, bot_count: Any) -> bool:
    """Record one finished Desktop day: a mode_use row per mode used (with the bot count) and an
    action_use row per (action, via). True once the day is settled (saved, already reported, or
    collection off) so the client drops its copy; False keeps it for the next attach."""
    try:
        from .relay_shared_metrics import enabled
        from .shared_metrics_events import emit_saved

        if not enabled():
            return True
        usage_day = day if isinstance(day, str) and _DAY.fullmatch(day) else None
        if usage_day is None:
            return True
        marks = _mode_rows(modes, bot_count) + _action_rows(actions)
        if not marks:
            return True
        with _LATCH_LOCK:
            latch = (_home(), usage_day)
            if latch in _reported_days:
                return True
            _reported_days.add(latch)
        settled = emit_saved(marks) == len(marks)
        if not settled:
            with _LATCH_LOCK:
                _reported_days.discard(latch)
        return settled
    except Exception:
        return False


def record_desktop_feature_use(*, area: Any) -> None:
    _emit(contract.DESKTOP_FEATURE_USE_MARK, _feature_use_fields, area=area)


def record_desktop_friction(*, kind: Any, detail: Any) -> None:
    _emit(contract.DESKTOP_FRICTION_MARK, _friction_fields, kind=kind, detail=detail)


def record_desktop_onboarding(*, step: Any, event: Any) -> None:
    _emit(contract.DESKTOP_ONBOARDING_MARK, _onboarding_fields, step=step, event=event)


def record_desktop_dislike(*, signal: Any, target: Any, setting: Any = None) -> None:
    _emit(contract.DESKTOP_DISLIKE_MARK, _dislike_fields, signal=signal, target=target, setting=setting)
