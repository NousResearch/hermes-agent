"""Blackbox telemetry plugin hooks."""

from __future__ import annotations

import json
import time
import uuid
import logging
from pathlib import Path
from threading import Lock
from typing import Any

_PREVIEW_CHARS = 2000


def _preview(value: Any) -> str:
    """Compact string preview of a tool arg/result for the side table.

    The store scrubs secrets and truncates again before persisting; this just
    coerces dict/list/other into a bounded string so the hook stays cheap.
    """
    if value is None:
        return ""
    if isinstance(value, str):
        text = value
    else:
        try:
            text = json.dumps(value, ensure_ascii=False, default=str)
        except Exception:
            text = str(value)
    return text[:_PREVIEW_CHARS]

import yaml

from hermes_constants import get_hermes_home
from plugins.blackbox.card import render_card
from plugins.blackbox.cost import compute_turn_cost
from plugins.blackbox.record import TurnRecord
from plugins.blackbox import routing

logger = logging.getLogger(__name__)


_DEFAULTS = {
    "enabled": False,
    "alerts_enabled": True,
    "cost_alert_threshold_usd": 1.00,
    "always_card": False,
    "store_text": True,
    "record_subagents": True,
    "retention_days": 30,
}

_lock = Lock()
_sessions: dict[str, dict[str, Any]] = {}


def _turn_id() -> str:
    return "turn_" + uuid.uuid4().hex


def _profile_name() -> str:
    try:
        from hermes_cli.profiles import get_active_profile_name

        return get_active_profile_name() or "default"
    except Exception:
        return "default"


def _config() -> dict[str, Any] | None:
    try:
        path = Path(get_hermes_home()) / "config.yaml"
        if not path.exists():
            return None
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        block = data.get("blackbox")
        if not isinstance(block, dict):
            return None
        cfg = dict(_DEFAULTS)
        cfg.update(block)
        if not bool(cfg.get("enabled")):
            return None
        return cfg
    except Exception:
        return None


def _int_value(value: Any) -> int:
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


def _int_or_none_value(value: Any) -> int | None:
    """Like _int_value but preserves None (absent last-call split → SQL NULL)."""
    if value is None:
        return None
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return None


def _float_value(value: Any) -> float:
    try:
        return float(value or 0)
    except (TypeError, ValueError):
        return 0.0


def _session_state(session_id: str) -> dict[str, Any]:
    with _lock:
        return _sessions.setdefault(
            session_id or "", {"ts_start": time.time(), "tools": [], "tool_calls": []}
        )


def _on_session_start(session_id: str = "", **_: Any) -> None:
    try:
        if _config() is None:
            return
        with _lock:
            _sessions[session_id or ""] = {
                "ts_start": time.time(),
                "tools": [],
                "tool_calls": [],
            }
    except Exception:
        return


def _on_post_tool_call(
    tool_name: str = "",
    name: str = "",
    session_id: str = "",
    args: Any = None,
    result: Any = None,
    **_: Any,
) -> None:
    try:
        cfg = _config()
        if cfg is None:
            return
        tool = tool_name or name
        if not tool:
            return
        state = _session_state(session_id)
        with _lock:
            state.setdefault("tools", []).append(str(tool))
            if bool(cfg.get("store_text", True)):
                state.setdefault("tool_calls", []).append(
                    {
                        "name": str(tool),
                        "args_preview": _preview(args),
                        "result_preview": _preview(result),
                    }
                )
    except Exception:
        return


def _comp_get(usage: dict[str, Any], key: str) -> int | None:
    """Read one bucket from the final-call request composition (or None).

    ``usage["last_composition"]`` is the char/4 fixed-vs-non-fixed breakdown of
    the last API call (see agent.model_metadata.compose_request_breakdown).
    Returns None when composition is absent (old turns / capture failed) so the
    column stays NULL and renderers fall back instead of showing a fake 0.
    """
    comp = usage.get("last_composition")
    if not isinstance(comp, dict):
        return None
    val = comp.get(key)
    if val is None:
        return None
    try:
        return int(val)
    except (TypeError, ValueError):
        return None


def _comp_calls_json(usage: dict[str, Any]) -> str | None:
    """Serialize per-call composition/output history to a compact JSON blob."""
    calls = usage.get("composition_calls")
    if not calls:
        return None
    try:
        cleaned = [c for c in calls if isinstance(c, dict)]
        if not cleaned:
            return None
        return json.dumps(cleaned, separators=(",", ":"))
    except Exception:
        return None


def _build_record(
    *,
    session_id: str,
    interrupted: bool,
    model: str,
    platform: str,
    provider: str,
    user_message: str,
    final_response: str,
    turn_usage: dict[str, Any] | None,
    cfg: dict[str, Any],
    kwargs: dict[str, Any],
) -> TurnRecord | None:
    usage = turn_usage or {}
    is_subagent = bool(usage.get("is_subagent"))
    if is_subagent and not bool(cfg.get("record_subagents", True)):
        return None

    now = time.time()
    state = _session_state(session_id)
    with _lock:
        state = _sessions.pop(session_id or "", state)
    ts_start = _float_value(state.get("ts_start")) or now - _float_value(usage.get("latency_s"))
    ts_end = now
    tool_calls = list(state.get("tool_calls") or [])

    cost_usd, cost_status, cost_perclass = compute_turn_cost(
        model,
        provider,
        kwargs.get("base_url") or usage.get("base_url"),
        usage.get("calls") or [],
    )

    store_text = bool(cfg.get("store_text", True))
    chat_id = (
        kwargs.get("chat_id")
        or usage.get("chat_id")
        or usage.get("parent_chat_id")
        or ""
    )
    chat_name = (
        kwargs.get("chat_name")
        or usage.get("chat_name")
        or usage.get("parent_chat_name")
        or ""
    )

    return TurnRecord(
        turn_id=_turn_id(),
        parent_turn_id=usage.get("parent_turn_id"),
        is_subagent=is_subagent,
        ts_start=ts_start,
        ts_end=ts_end,
        profile=str(kwargs.get("profile") or _profile_name()),
        provider=str(provider or ""),
        model=str(model or ""),
        platform=str(platform or usage.get("parent_platform") or ""),
        chat_id=str(chat_id or ""),
        chat_name=str(chat_name or ""),
        api_calls=_int_value(usage.get("api_calls")),
        tools=list(state.get("tools") or []),
        input_tokens=_int_value(usage.get("input_tokens")),
        output_tokens=_int_value(usage.get("output_tokens")),
        cache_read_tokens=_int_value(usage.get("cache_read_tokens")),
        cache_write_tokens=_int_value(usage.get("cache_write_tokens")),
        reasoning_tokens=_int_value(usage.get("reasoning_tokens")),
        context_used=_int_value(usage.get("context_used")),
        context_length=_int_value(usage.get("context_length")),
        last_cache_read_tokens=_int_or_none_value(usage.get("last_cache_read_tokens")),
        last_cache_write_tokens=_int_or_none_value(usage.get("last_cache_write_tokens")),
        last_uncached_tokens=_int_or_none_value(usage.get("last_uncached_tokens")),
        comp_sys_tokens=_comp_get(usage, "sys_tokens"),
        comp_tool_schema_tokens=_comp_get(usage, "tool_schema_tokens"),
        comp_history_tokens=_comp_get(usage, "history_tokens"),
        comp_history_message_count=_comp_get(usage, "history_message_count"),
        comp_tool_result_tokens=_comp_get(usage, "tool_result_tokens"),
        comp_tool_arg_tokens=_comp_get(usage, "tool_arg_tokens"),
        comp_tool_result_count=_comp_get(usage, "tool_result_count"),
        comp_skills_tokens=_comp_get(usage, "skills_tokens"),
        comp_skills_count=_comp_get(usage, "skills_count"),
        comp_framing_tokens=_comp_get(usage, "framing_tokens"),
        comp_calls_json=_comp_calls_json(usage),
        cost_usd=cost_usd,
        cost_status=cost_status,
        cost_uncached_usd=cost_perclass.get("uncached"),
        cost_cache_read_usd=cost_perclass.get("cache_read"),
        cost_cache_write_usd=cost_perclass.get("cache_write"),
        cost_output_usd=cost_perclass.get("output"),
        interrupted=bool(interrupted),
        user_text=str(user_message or "") if store_text else "",
        final_text=str(final_response or "") if store_text else "",
        tool_calls=tool_calls if store_text else [],
    )


def _on_session_end(
    session_id: str = "",
    completed: bool = True,
    interrupted: bool = False,
    model: str = "",
    platform: str = "",
    provider: str = "",
    user_message: str = "",
    final_response: str = "",
    turn_usage: dict[str, Any] | None = None,
    **kwargs: Any,
) -> None:
    try:
        cfg = _config()
        if cfg is None:
            # Disabled / no config block: ensure we never leak a session entry
            # that on_session_start/post_tool_call may have created.
            try:
                with _lock:
                    _sessions.pop(session_id or "", None)
            except Exception:
                pass
            return
        record = _build_record(
            session_id=session_id,
            interrupted=interrupted,
            model=model,
            platform=platform,
            provider=provider,
            user_message=user_message,
            final_response=final_response,
            turn_usage=turn_usage,
            cfg=cfg,
            kwargs=kwargs,
        )
        if record is None:
            return

        from plugins.blackbox import store

        store.insert_turn(record)

        # Retention sweep — prune turns older than retention_days. sweep() is
        # self-throttling (a 'last_sweep_date' sentinel makes it a no-op after
        # the first call each UTC day), so calling it every turn costs one
        # indexed SELECT/day and keeps the store from growing unbounded. Guard
        # it so a sweep failure never blocks recording/alerting.
        try:
            store.sweep(int(cfg.get("retention_days", 30) or 30))
        except Exception:
            logger.warning("blackbox retention sweep failed", exc_info=True)

        threshold = float(cfg.get("cost_alert_threshold_usd", 1.0) or 1.0)
        # The turn is always recorded above (visible to /cost and /context);
        # alerts_enabled only gates the proactive card PUSH to the channel.
        if not bool(cfg.get("alerts_enabled", True)):
            return
        should_alert = (
            bool(cfg.get("always_card"))
            or (record.cost_usd is not None and record.cost_usd >= threshold and not record.interrupted)
        )
        if not should_alert:
            return
        if record.interrupted and not bool(cfg.get("always_card")):
            return
        if store.mark_alerted(record.turn_id):
            record.alerted = True
            routing.send_card(
                render_card(record, threshold),
                record.platform,
                record.chat_id,
                record.profile,
            )
    except Exception:
        # Telemetry must never break a user turn, but a silent failure defeats
        # the purpose of telemetry — log it (the gateway loop still proceeds).
        logger.warning("blackbox on_session_end failed", exc_info=True)
        return


def register(ctx) -> None:
    ctx.register_hook("on_session_start", _on_session_start)
    ctx.register_hook("post_tool_call", _on_post_tool_call)
    ctx.register_hook("on_session_end", _on_session_end)
    # The slash command lives in commands.py; the loader only calls this
    # package-level register(), so delegate explicitly or /cost never wires in.
    try:
        from plugins.blackbox import commands

        commands.register(ctx)
    except Exception:
        logger.warning("blackbox: failed to register /cost command", exc_info=True)
