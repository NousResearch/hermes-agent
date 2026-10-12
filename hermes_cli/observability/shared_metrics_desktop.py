"""Shared-metrics facts the Desktop app reports about itself: area use, friction, first-run steps,
dislike signals, and one daily report (Bot Mode / Sessions split, button presses per action).

The renderer dedupes on its side (per-profile state persisted only while collection is on); the
backend latches again so a second window, a reconnect replay, a backend restart or a cleared
localStorage cannot inflate a row. Feature use (once per profile, UTC day, area) and the daily report
(once per profile and usage day, counted in the usage day's period) latch durably in the profile's
shared-metrics store, in the same transaction as the row; onboarding latches once per (profile, step,
event) ever in a file; friction and dislike are capped per (profile, UTC day) in this process. Every
value collapses onto the closed sets in ``shared_metrics_contract``; raw client words are never
recorded. Every latch is claimed only after the ``enabled()`` gate, so a disabled profile writes nothing.
"""

from __future__ import annotations

import logging
import os
import platform
import re
import shutil
import threading
from datetime import date, datetime, timedelta, timezone
from typing import Any

from . import shared_metrics_contract as contract
from .shared_metrics_events import _emit

logger = logging.getLogger(__name__)

# Per-day ceiling for one (kind, detail) friction pair: a toast loop or a janky session still shows
# up as "a lot", but cannot flood the store.
FRICTION_DAILY_CAP = 50
ONBOARDING_LATCH_DIRNAME = "desktop_onboarding"
# Room for every distinct (action, via) the renderer can send once it collapses unknown ids to `other`.
DAILY_ACTION_ROWS_MAX = len(contract.DESKTOP_ACTION_IDS) * len(contract.DESKTOP_ACTION_VIAS)
DAILY_REPORT_STATE_KEY = "desktop_daily_reported"
# The renderer holds a finished day for at most 7 days; anything older (or in the future) is settled
# unrecorded, which keeps the durable per-day latch bounded.
DAILY_REPORT_WINDOW_DAYS = 8
_DAY = re.compile(r"\d{4}-\d{2}-\d{2}")
_LATCH_LOCK = threading.Lock()
# (home, day, key) -> count; pruned to the current day so the process never grows it.
_daily: dict[tuple[str, str, str], int] = {}


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


def _store_and_resource() -> tuple[Any, dict[str, str]]:
    """The active profile's store and client resource (the subscriber's), for rows that latch
    durably or belong to a past period and so cannot ride the relay's current-day marks."""
    from hermes_cli.config import detect_install_method
    from hermes_cli.version_info import get_version_info

    from .shared_metrics import SharedMetricsStore

    return SharedMetricsStore(), contract.client_resource(
        get_version_info().base_version, os_name=platform.system(), architecture=platform.machine(),
        install_method=detect_install_method(),
    )


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


def purge_onboarding_latches() -> None:
    """Collection turned off: drop the per-profile onboarding latches like the renderer drops its copy."""
    from hermes_constants import get_hermes_home

    shutil.rmtree(get_hermes_home() / "telemetry" / "shared_metrics" / ONBOARDING_LATCH_DIRNAME, ignore_errors=True)


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


def _daily_rollup(usage_day: str, oldest: str, marks: list[tuple[str, dict[str, str]]], resource: dict[str, str]):
    """``update_rollup_state`` step: the usage day's rows, unless that day was already reported."""
    def update(state: dict[str, Any] | None) -> tuple[dict[str, Any], list[tuple[str, dict, dict, str]]]:
        days = [d for d in (state or {}).get("days", []) if isinstance(d, str) and d >= oldest]
        if usage_day in days:
            return {"days": days}, []
        return {"days": sorted([*days, usage_day])}, [(mark, data, resource, usage_day) for mark, data in marks]
    return update


def record_desktop_daily(*, day: Any, modes: Any, actions: Any, bot_count: Any) -> bool:
    """Record one finished Desktop day, in that day's period: a mode_use row per mode used (with the
    bot count) and an action_use row per (action, via). True once the day is settled (saved, already
    reported, empty or out of range) so the client drops its copy; False keeps it for the next attach,
    including when this profile's collection is off (the client's own gate decides what it keeps)."""
    try:
        from .relay_shared_metrics import enabled

        if not enabled():
            return False
        today = _utc_day()
        oldest = (date.fromisoformat(today) - timedelta(days=DAILY_REPORT_WINDOW_DAYS)).isoformat()
        usage_day = day if isinstance(day, str) and _DAY.fullmatch(day) and oldest <= day <= today else None
        marks = _mode_rows(modes, bot_count) + _action_rows(actions) if usage_day else []
        if marks:
            store, resource = _store_and_resource()
            store.update_rollup_state(DAILY_REPORT_STATE_KEY, _daily_rollup(usage_day, oldest, marks, resource))
        return True
    except Exception:
        logger.debug("Desktop daily report not recorded", exc_info=True)
        return False


def record_desktop_feature_use(*, area: Any) -> None:
    """Once per (profile, UTC day, area), latched in the store with the row (never raises)."""
    try:
        from .relay_shared_metrics import enabled

        if enabled():
            store, resource = _store_and_resource()
            store.record_counter_once_per_day(contract.DESKTOP_FEATURE_USE_METRIC, {"area": feature_area(area)}, resource)
    except Exception:
        logger.debug("Desktop feature use not recorded", exc_info=True)


def record_desktop_friction(*, kind: Any, detail: Any) -> None:
    _emit(contract.DESKTOP_FRICTION_MARK, _friction_fields, kind=kind, detail=detail)


def record_desktop_onboarding(*, step: Any, event: Any) -> None:
    _emit(contract.DESKTOP_ONBOARDING_MARK, _onboarding_fields, step=step, event=event)


def record_desktop_dislike(*, signal: Any, target: Any, setting: Any = None) -> None:
    _emit(contract.DESKTOP_DISLIKE_MARK, _dislike_fields, signal=signal, target=target, setting=setting)
