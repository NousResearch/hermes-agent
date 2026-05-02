from __future__ import annotations

from datetime import datetime
from typing import Callable, Optional

from .profile import DEFAULT_RUNTIME_PROFILE
from .signal_policy import DEFAULT_SIGNAL_POLICY, render_signal_text
from .store import DEFAULT_EXECUTION_STATE, OlinStateStore, TradingStateStore

DEFAULT_RUNTIME_POLICY = {
    "pending_signal_timeout_sec": 300,
    "pending_signal_max_attempts": 3,
    "pending_retry_cooldown_sec": 60,
    "candidate_freshness_limit_sec": 0,
}


def _now_str(now: Optional[datetime] = None) -> str:
    return (now or datetime.now()).strftime("%Y-%m-%d %H:%M:%S")


def _age_seconds(ts: Optional[str], now: Optional[datetime] = None) -> Optional[int]:
    if not ts:
        return None
    try:
        current = now or datetime.now()
        return int((current - datetime.strptime(str(ts), "%Y-%m-%d %H:%M:%S")).total_seconds())
    except Exception:
        return None


def load_runtime_policy() -> dict:
    return dict(DEFAULT_RUNTIME_POLICY)


def _runtime_profile(store: TradingStateStore):
    return getattr(store, "profile", DEFAULT_RUNTIME_PROFILE)


def _profile_trade_unit(store: TradingStateStore) -> int:
    return int(getattr(_runtime_profile(store), "trade_unit", DEFAULT_RUNTIME_PROFILE.trade_unit) or DEFAULT_RUNTIME_PROFILE.trade_unit)


def _profile_max_trades(store: TradingStateStore) -> int:
    return int(getattr(_runtime_profile(store), "max_trades", DEFAULT_RUNTIME_PROFILE.max_trades) or DEFAULT_RUNTIME_PROFILE.max_trades)


def _clear_active_signal_if_matches(execution_state: dict, signal_id: str) -> dict:
    state = dict(execution_state or {})
    active_signal = state.get("active_signal") or {}
    if str(active_signal.get("signal_id") or "").strip() == str(signal_id or "").strip():
        state["active_signal"] = None
    return state


def _pending_to_active_signal(pending: dict) -> dict:
    return {
        "signal_id": pending.get("signal_id"),
        "signal_key": pending.get("signal_key"),
        "trade_date": pending.get("trade_date"),
        "action": pending.get("action"),
        "sequence": int(pending.get("sequence", 0) or 0),
        "status": pending.get("status"),
        "created_at": pending.get("created_at"),
        "last_attempt_at": pending.get("last_attempt_at"),
        "last_error": pending.get("last_error"),
    }


def recover_signal_runtime(
    store: TradingStateStore,
    *,
    trade_date: Optional[str] = None,
    now: Optional[datetime] = None,
) -> dict:
    runtime_now = now or datetime.now()
    policy = load_runtime_policy()
    pending = store.load_pending_signal() or {}
    execution_state = store.load_execution_state()
    repairs: list[str] = []
    result = {
        "checked_at": _now_str(runtime_now),
        "policy": policy,
        "pending_status": pending.get("status") if pending else None,
        "repairs": repairs,
    }

    active_signal = execution_state.get("active_signal") or None
    if not pending:
        if active_signal:
            execution_state["active_signal"] = None
            store.save_execution_state(execution_state)
            repairs.append("cleared_orphan_active_signal")
        result["pending_status"] = None
        return result

    pending = dict(pending)
    signal_id = str(pending.get("signal_id") or "").strip()
    pending_status = str(pending.get("status") or "").strip().lower()

    if active_signal:
        if str(active_signal.get("signal_id") or "").strip() != signal_id:
            execution_state["active_signal"] = _pending_to_active_signal(pending)
            store.save_execution_state(execution_state)
            repairs.append("repaired_active_signal_mismatch")
    else:
        execution_state["active_signal"] = _pending_to_active_signal(pending)
        store.save_execution_state(execution_state)
        repairs.append("restored_missing_active_signal")

    pending_trade_date = str(pending.get("trade_date") or "").strip()
    if trade_date and pending_trade_date and pending_trade_date != trade_date:
        pending["status"] = "expired_cross_day"
        pending["last_attempt_status"] = "expired_cross_day"
        pending["escalated"] = True
        pending["escalation_reason"] = "cross_day_cleanup"
        pending["escalated_at"] = _now_str(runtime_now)
        pending["age_seconds"] = _age_seconds(pending.get("created_at"), runtime_now)
        store.save_pending_signal(pending)
        execution_state = _clear_active_signal_if_matches(execution_state, signal_id)
        store.save_execution_state(execution_state)
        store.append_signal_send_history(
            "expired_cross_day",
            {
                "signal_id": signal_id,
                "signal_key": pending.get("signal_key"),
                "trade_date": pending_trade_date,
                "current_trade_date": trade_date,
            },
        )
        repairs.append("expired_cross_day_pending_signal")
        result["pending_status"] = pending["status"]
        return result

    pending_age = _age_seconds(pending.get("created_at"), runtime_now)
    result["pending_age_seconds"] = pending_age
    max_attempts = int(policy.get("pending_signal_max_attempts", 3) or 3)
    timeout_sec = int(policy.get("pending_signal_timeout_sec", 300) or 300)

    if pending_status in ("pending", "failed") and pending_age is not None and pending_age > timeout_sec:
        pending["status"] = "timed_out"
        pending["last_attempt_status"] = "timed_out"
        pending["escalated"] = True
        pending["timed_out"] = True
        pending["escalation_reason"] = "pending_timeout"
        pending["escalated_at"] = _now_str(runtime_now)
        pending["age_seconds"] = pending_age
        pending["timeout_sec"] = timeout_sec
        store.save_pending_signal(pending)
        execution_state = _clear_active_signal_if_matches(execution_state, signal_id)
        store.save_execution_state(execution_state)
        store.append_signal_send_history(
            "timed_out",
            {
                "signal_id": signal_id,
                "signal_key": pending.get("signal_key"),
                "attempts": pending.get("attempts", 0),
                "age_seconds": pending_age,
                "timeout_sec": timeout_sec,
            },
        )
        repairs.append("timed_out_pending_signal")
        result["pending_status"] = pending["status"]
        return result

    attempts = int(pending.get("attempts", 0) or 0)
    non_retryable_failed = pending_status == "failed" and pending.get("last_error_retryable") is False
    if non_retryable_failed:
        pending["status"] = "failed_exhausted"
        pending["last_attempt_status"] = "failed_exhausted"
        pending["escalated"] = True
        pending["escalation_reason"] = "non_retryable_dispatch_error"
        pending["escalated_at"] = _now_str(runtime_now)
        pending["max_attempts"] = max_attempts
        store.save_pending_signal(pending)
        execution_state = _clear_active_signal_if_matches(execution_state, signal_id)
        store.save_execution_state(execution_state)
        store.append_signal_send_history(
            "failed_exhausted",
            {
                "signal_id": signal_id,
                "signal_key": pending.get("signal_key"),
                "attempts": attempts,
                "max_attempts": max_attempts,
                "error_class": pending.get("last_error_class"),
                "escalation_reason": "non_retryable_dispatch_error",
            },
        )
        repairs.append("non_retryable_failed_pending_signal")
        result["pending_status"] = pending["status"]
        return result

    if pending_status == "failed" and attempts >= max_attempts:
        pending["status"] = "failed_exhausted"
        pending["last_attempt_status"] = "failed_exhausted"
        pending["escalated"] = True
        pending["escalation_reason"] = "max_attempts_exhausted"
        pending["escalated_at"] = _now_str(runtime_now)
        pending["max_attempts"] = max_attempts
        store.save_pending_signal(pending)
        execution_state = _clear_active_signal_if_matches(execution_state, signal_id)
        store.save_execution_state(execution_state)
        store.append_signal_send_history(
            "failed_exhausted",
            {
                "signal_id": signal_id,
                "signal_key": pending.get("signal_key"),
                "attempts": attempts,
                "max_attempts": max_attempts,
            },
        )
        repairs.append("exhausted_failed_pending_signal")
        result["pending_status"] = pending["status"]
        return result

    if pending_status == "failed" and attempts < max_attempts:
        pending["status"] = "pending"
        pending["last_attempt_status"] = "retry_scheduled"
        pending["retry_scheduled_at"] = _now_str(runtime_now)
        store.save_pending_signal(pending)
        refreshed_state = _normalize_execution_state(store.load_execution_state(), trade_date or pending_trade_date)
        active_signal = dict((refreshed_state or {}).get("active_signal") or {})
        if active_signal and str(active_signal.get("signal_id") or "").strip() == signal_id:
            active_signal["status"] = "pending"
            active_signal["last_attempt_at"] = pending.get("last_attempt_at")
            active_signal["last_error"] = pending.get("last_error")
            active_signal["last_error_class"] = pending.get("last_error_class")
            refreshed_state["active_signal"] = active_signal
            store.save_execution_state(refreshed_state)
        repairs.append("requeued_retryable_failed_pending_signal")
        result["pending_status"] = pending["status"]
        return result

    if pending_status in ("sent", "timed_out", "failed_exhausted", "expired_cross_day"):
        repaired_state = _clear_active_signal_if_matches(execution_state, signal_id)
        if repaired_state.get("active_signal") != execution_state.get("active_signal"):
            store.save_execution_state(repaired_state)
            repairs.append("cleared_terminal_active_signal")

    result["pending_status"] = pending_status or None
    return result



def _default_dispatch_pending_signal(payload: dict) -> dict:
    from gateway.config import Platform, load_gateway_config
    from gateway.session_context import get_session_env
    from model_tools import _run_async
    from tools.send_message_tool import _send_to_platform

    config = load_gateway_config()
    platform_name = str(payload.get("channel") or "feishu").strip().lower()
    if platform_name != "feishu":
        raise ValueError(f"Unsupported dispatch channel: {platform_name}")

    platform = Platform.FEISHU
    pconfig = config.platforms.get(platform)
    if not pconfig or not pconfig.enabled:
        raise RuntimeError("Feishu platform is not configured or not enabled")

    thread_id = str(payload.get("thread_id") or "").strip() or None
    chat_id = str(payload.get("chat_id") or "").strip()
    if not chat_id:
        session_platform = get_session_env("HERMES_SESSION_PLATFORM", "").strip().lower()
        if session_platform == "feishu":
            chat_id = get_session_env("HERMES_SESSION_CHAT_ID", "").strip()
            thread_id = thread_id or (get_session_env("HERMES_SESSION_THREAD_ID", "").strip() or None)
    if not chat_id:
        home = config.get_home_channel(platform)
        if home:
            chat_id = str(home.chat_id).strip()
    if not chat_id:
        raise RuntimeError("No Feishu chat_id available from session context or home channel")

    result = _run_async(
        _send_to_platform(
            platform,
            pconfig,
            chat_id,
            str(payload.get("message") or ""),
            thread_id=thread_id,
            media_files=None,
        )
    )
    if not isinstance(result, dict):
        raise RuntimeError(f"Unexpected dispatch result type: {type(result).__name__}")
    if result.get("error") and "success" not in result:
        return {"success": False, "error": result.get("error"), **result}
    return {"chat_id": chat_id, "thread_id": thread_id, **result}


def _mark_dispatch_failed(store: TradingStateStore, pending: dict, error: str) -> dict:
    failed_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    attempts = int(pending.get("attempts", 0) or 0) + 1
    failed = dict(pending)
    failed.update(
        {
            "status": "failed",
            "attempts": attempts,
            "last_attempt_at": failed_at,
            "last_attempt_status": "failed",
            "last_error": error,
        }
    )
    store.save_pending_signal(failed)

    trade_date = str(failed.get("trade_date") or "")
    if trade_date:
        execution_state = _normalize_execution_state(store.load_execution_state(), trade_date)
        active_signal = execution_state.get("active_signal") or {}
        if active_signal.get("signal_id") == failed.get("signal_id"):
            active_signal.update(
                {
                    "status": "failed",
                    "last_attempt_at": failed_at,
                    "last_error": error,
                }
            )
            execution_state["active_signal"] = active_signal
            execution_state["last_signal_id"] = failed.get("signal_id")
            execution_state["last_signal_action"] = failed.get("action")
            execution_state["last_signal_status"] = "failed"
            execution_state["last_signal_at"] = failed_at
            store.save_execution_state(execution_state)

    store.record_dispatch_event(
        "failed",
        {
            "signal_id": failed.get("signal_id"),
            "signal_key": failed.get("signal_key"),
            "trade_date": trade_date,
            "action": failed.get("action"),
            "sequence": int(failed.get("sequence", 0) or 0),
            "attempts": attempts,
            "error": error,
        },
    )
    store.append_signal_send_history(
        "failed",
        {
            "signal_id": failed.get("signal_id"),
            "signal_key": failed.get("signal_key"),
            "trade_date": trade_date,
            "action": failed.get("action"),
            "sequence": int(failed.get("sequence", 0) or 0),
            "attempts": attempts,
            "error": error,
        },
    )
    return failed


def _normalize_execution_state(state: Optional[dict], effective_trade_date: str) -> dict:
    state = dict(state or {})
    if state.get("trade_date") != effective_trade_date:
        reset = dict(DEFAULT_EXECUTION_STATE)
        reset["trade_date"] = effective_trade_date
        return reset
    merged = dict(DEFAULT_EXECUTION_STATE)
    merged["trade_date"] = effective_trade_date
    merged.update(state)
    merged["actions"] = list(merged.get("actions") or [])
    if merged.get("active_signal") in ({}, []):
        merged["active_signal"] = None
    return merged


def _candidate_signal_key(candidate: dict) -> str:
    signal_key = str(candidate.get("signal_key") or "").strip()
    if signal_key:
        return signal_key
    action = str(candidate.get("action") or candidate.get("next_action") or "").strip()
    sequence = int(candidate.get("sequence", 0) or 0)
    if action and sequence > 0:
        return f"{action}_{sequence}"
    return ""


def _freshness_ok(candidate: dict, policy: dict, now: Optional[datetime] = None) -> tuple[bool, Optional[int]]:
    limit_sec = int(policy.get("candidate_freshness_limit_sec", 0) or 0)
    if limit_sec <= 0:
        return True, None
    age = _age_seconds(candidate.get("signal_time"), now)
    if age is None:
        return True, None
    return age <= limit_sec, age


def _retry_cooldown_remaining_sec(pending: dict, policy: dict, now: Optional[datetime] = None) -> int:
    cooldown_sec = int(policy.get("pending_retry_cooldown_sec", 0) or 0)
    if cooldown_sec <= 0:
        return 0
    last_attempt_at = pending.get("last_attempt_at")
    age = _age_seconds(last_attempt_at, now)
    if age is None:
        return 0
    return max(cooldown_sec - age, 0)


def arbitrate_candidate(
    store: TradingStateStore,
    candidate: dict,
    *,
    effective_trade_date: str,
    now: Optional[datetime] = None,
) -> dict:
    runtime_now = now or datetime.now()
    policy = load_runtime_policy()
    next_action = str(candidate.get("next_action") or candidate.get("action") or "hold").strip().lower()
    execution_state = _normalize_execution_state(store.load_execution_state(), effective_trade_date)
    push_state = store.load_push_state()
    pending = store.load_pending_signal() or {}
    signal_key = _candidate_signal_key(candidate)
    action = str(candidate.get("action") or next_action).strip().lower()
    sequence = int(candidate.get("sequence", 0) or 0)

    result = {
        "decision": "stage_new_pending",
        "reason": "candidate_allowed",
        "candidate": dict(candidate),
        "pending": {},
        "policy": policy,
        "cooldown_remaining_sec": 0,
    }
    if next_action in ("", "hold") or sequence <= 0:
        result["decision"] = "blocked"
        result["reason"] = "hold_candidate"
        return result

    freshness_ok, candidate_age = _freshness_ok(candidate, policy, runtime_now)
    if not freshness_ok:
        result["decision"] = "blocked"
        result["reason"] = "candidate_stale"
        result["candidate_age_sec"] = candidate_age
        return result

    pending_status = str(pending.get("status") or "").strip().lower()
    pending_trade_date = str(pending.get("trade_date") or "").strip()
    pending_signal_key = str(pending.get("signal_key") or "").strip()
    if pending and pending_trade_date == effective_trade_date:
        if pending_status == "pending":
            if pending_signal_key == signal_key:
                result["decision"] = "blocked"
                result["reason"] = "duplicate_pending_signal"
            else:
                result["decision"] = "blocked"
                result["reason"] = "another_pending_signal_active"
            result["pending"] = dict(pending)
            return result
        if pending_status == "failed" and pending_signal_key == signal_key and bool(pending.get("last_error_retryable", True)):
            remaining = _retry_cooldown_remaining_sec(pending, policy, runtime_now)
            if remaining > 0:
                result["decision"] = "blocked"
                result["reason"] = "retry_cooldown_active"
                result["pending"] = dict(pending)
                result["cooldown_remaining_sec"] = remaining
                return result
            result["decision"] = "reuse_failed_pending"
            result["reason"] = "retry_ready_reuse_failed_pending"
            result["pending"] = dict(pending)
            return result

    if (
        str(push_state.get("last_pushed_trade_date") or "").strip() == effective_trade_date
        and str(push_state.get("last_pushed_signal") or "").strip() == signal_key
    ):
        result["decision"] = "blocked"
        result["reason"] = "duplicate_sent_signal"
        return result

    for action_item in list(execution_state.get("actions") or []):
        if (
            str(action_item.get("trade_date") or "").strip() == effective_trade_date
            and str(action_item.get("action") or "").strip().lower() == action
            and int(action_item.get("sequence", 0) or 0) == sequence
        ):
            result["decision"] = "blocked"
            result["reason"] = "duplicate_execution_signal"
            return result

    return result


def build_execution_suggestion(store: OlinStateStore | TradingStateStore, tech_data: dict, effective_trade_date: str) -> dict:
    state = _normalize_execution_state(store.load_execution_state(), effective_trade_date)
    trade_unit = _profile_trade_unit(store)
    max_trades = _profile_max_trades(store)
    active_signal = state.get("active_signal") or {}
    if active_signal and active_signal.get("status") == "pending":
        return {
            "signal": tech_data.get("summary_signal", "hold"),
            "trade_unit": trade_unit,
            "max_trades": max_trades,
            "action": "hold",
            "sequence": 0,
            "text": DEFAULT_SIGNAL_POLICY.active_signal_hold_text,
            "trade_date": effective_trade_date,
            "reason": "active_signal_exists",
            "state_snapshot": state,
            "execution_state": state,
            "next_action": "hold",
        }

    signal = tech_data.get("summary_signal", "hold")
    score = tech_data.get("score", {}).get("total", 50)

    candidate = {
        "signal": signal,
        "trade_unit": trade_unit,
        "max_trades": max_trades,
        "action": "hold",
        "sequence": 0,
        "text": DEFAULT_SIGNAL_POLICY.no_action_text,
        "trade_date": effective_trade_date,
        "reason": "",
        "state_snapshot": state,
        "execution_state": state,
        "next_action": "hold",
    }

    if signal == "sell" and score <= DEFAULT_SIGNAL_POLICY.sell_score_threshold and state.get("sell_count", 0) < max_trades:
        next_seq = state.get("sell_count", 0) + 1
        candidate.update(
            {
                "action": "sell",
                "next_action": "sell",
                "sequence": next_seq,
                "text": render_signal_text(action="sell", sequence=next_seq, trade_unit=trade_unit),
                "reason": f"signal={signal}, score={score}",
            }
        )
    elif signal == "buy" and score >= DEFAULT_SIGNAL_POLICY.buy_score_threshold and state.get("buy_count", 0) < max_trades:
        next_seq = state.get("buy_count", 0) + 1
        candidate.update(
            {
                "action": "buy",
                "next_action": "buy",
                "sequence": next_seq,
                "text": render_signal_text(action="buy", sequence=next_seq, trade_unit=trade_unit),
                "reason": f"signal={signal}, score={score}",
            }
        )

    return candidate


def stage_pending_signal(
    store: TradingStateStore,
    suggestion: dict,
    execution_state: dict,
    trade_date: str,
    now: datetime,
) -> dict:
    pending_existing = store.load_pending_signal()
    if pending_existing:
        pending_trade_date = str(pending_existing.get("trade_date") or "")
        if pending_trade_date != trade_date:
            store.clear_pending_signal()
            pending_existing = {}
    if pending_existing and pending_existing.get("status") in ("pending", "failed"):
        return pending_existing

    action = suggestion.get("next_action") or suggestion.get("action") or "hold"
    sequence = int(suggestion.get("sequence", 0) or 0)
    if action not in {"buy", "sell"} or sequence <= 0:
        return {}
    signal_key = f"{action}_{sequence}"
    signal_id = f"{signal_key}_{now.strftime('%Y%m%d_%H%M%S')}"
    pending_signal = {
        "signal_id": signal_id,
        "signal_key": signal_key,
        "status": "pending",
        "created_at": now.strftime("%Y-%m-%d %H:%M:%S"),
        "trade_date": trade_date,
        "action": action,
        "sequence": sequence,
        "text": suggestion.get("text", ""),
        "signal": suggestion.get("signal", action),
        "trade_unit": int(suggestion.get("trade_unit", _profile_trade_unit(store)) or _profile_trade_unit(store)),
        "attempts": 0,
        "last_attempt_at": None,
        "last_attempt_status": None,
        "last_error": None,
        "sent_at": None,
        "state": execution_state,
        "candidate_snapshot": suggestion,
    }
    store.save_pending_signal(pending_signal)

    next_state = _normalize_execution_state(execution_state, trade_date)
    next_state["active_signal"] = {
        "signal_id": signal_id,
        "signal_key": signal_key,
        "action": action,
        "sequence": sequence,
        "status": "pending",
        "created_at": pending_signal["created_at"],
    }
    store.save_execution_state(next_state)
    return pending_signal


def dispatch_ledger_sent_event(delivery_result: Optional[dict] = None) -> str:
    result = delivery_result or {}
    return "dry_run_sent" if bool(result.get("dry_run")) else "sent"


def _reconcile_already_sent_signal(
    store: TradingStateStore,
    pending: dict,
    *,
    sent_at: Optional[str] = None,
    channel: Optional[str] = None,
    delivery_result: Optional[dict] = None,
    fallback_trade_date: Optional[str] = None,
) -> dict:
    reconciled = dict(pending)
    signal_id = str(reconciled.get("signal_id") or "").strip()
    trade_date = str(reconciled.get("trade_date") or fallback_trade_date or "").strip()
    action = str(reconciled.get("action") or "").strip()
    sequence = int(reconciled.get("sequence", 0) or 0)
    if not sent_at:
        sent_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    persisted_pending = store.load_pending_signal()
    execution_state_raw = store.load_execution_state()
    active_signal_raw = (execution_state_raw or {}).get("active_signal") or {}
    actions_raw = list((execution_state_raw or {}).get("actions") or [])
    existing = next((item for item in actions_raw if str(item.get("signal_id") or "").strip() == signal_id), None)
    if not trade_date:
        trade_date = str(
            active_signal_raw.get("trade_date")
            or (existing or {}).get("trade_date")
            or (execution_state_raw or {}).get("trade_date")
            or ""
        ).strip()

    reconciled.update(
        {
            "status": "sent",
            "sent_at": sent_at,
            "dispatch_channel": channel,
            "delivery_result": delivery_result or {},
            "last_attempt_status": "sent",
        }
    )
    if trade_date:
        reconciled["trade_date"] = trade_date

    if str(persisted_pending.get("signal_id") or "").strip() == signal_id:
        store.clear_pending_signal()

    if trade_date:
        execution_state = _normalize_execution_state(execution_state_raw, trade_date)
        active_signal = execution_state.get("active_signal") or {}
        if active_signal.get("signal_id") == signal_id:
            execution_state["active_signal"] = None
        actions = list(execution_state.get("actions") or [])
        existing = next((item for item in actions if item.get("signal_id") == signal_id), None)
        if existing is None and action and sequence > 0:
            actions.append(
                {
                    "signal_id": signal_id,
                    "signal_key": reconciled.get("signal_key"),
                    "trade_date": trade_date,
                    "action": action,
                    "sequence": sequence,
                    "status": "sent",
                    "sent_at": sent_at,
                    "channel": channel,
                    "text": reconciled.get("text", ""),
                }
            )
        elif existing is not None:
            existing.update(
                {
                    "signal_key": reconciled.get("signal_key"),
                    "trade_date": trade_date,
                    "action": action,
                    "sequence": sequence,
                    "status": "sent",
                    "sent_at": sent_at,
                    "channel": channel,
                    "text": reconciled.get("text", ""),
                }
            )
        execution_state.update(
            {
                "trade_date": trade_date,
                "actions": actions,
                "last_signal_id": signal_id,
                "last_signal_action": action,
                "last_signal_status": "sent",
                "last_signal_at": sent_at,
            }
        )
        if action == "sell":
            execution_state["sell_count"] = max(int(execution_state.get("sell_count", 0) or 0), sequence)
        elif action == "buy":
            execution_state["buy_count"] = max(int(execution_state.get("buy_count", 0) or 0), sequence)
        store.save_execution_state(execution_state)

    return reconciled


def deliver_pending_signal(
    store: TradingStateStore,
    pending: dict,
    *,
    dispatch_fn: Optional[Callable[[dict], dict]] = None,
    channel: str = "feishu",
    chat_id: Optional[str] = None,
    thread_id: Optional[str] = None,
) -> dict:
    pending = dict(pending or {})
    signal_id = str(pending.get("signal_id") or "").strip()
    message = str(pending.get("text") or "").strip()
    if not signal_id:
        raise ValueError("pending signal missing signal_id")
    if not message:
        raise ValueError("pending signal missing text")

    status = str(pending.get("status") or "pending").strip().lower()
    if status == "sent":
        return pending
    if status != "pending":
        return pending

    trade_date = str(pending.get("trade_date") or "").strip()
    action = str(pending.get("action") or "").strip().lower()
    sequence = int(pending.get("sequence", 0) or 0)

    push_state = store.load_push_state()
    if str(push_state.get("last_pushed_signal_id") or "").strip() == signal_id:
        return _reconcile_already_sent_signal(
            store,
            pending,
            sent_at=push_state.get("last_pushed_at"),
            channel=push_state.get("last_dispatch_channel"),
            delivery_result=push_state.get("last_delivery_result") or {},
            fallback_trade_date=str(push_state.get("last_pushed_trade_date") or "").strip() or None,
        )

    execution_state = store.load_execution_state()
    actions = list((execution_state or {}).get("actions") or [])
    if any(str(item.get("signal_id") or "").strip() == signal_id for item in actions):
        existing = next((item for item in actions if str(item.get("signal_id") or "").strip() == signal_id), {})
        return _reconcile_already_sent_signal(
            store,
            pending,
            sent_at=existing.get("sent_at"),
            channel=existing.get("channel"),
            delivery_result={},
        )

    if not trade_date or action not in {"buy", "sell"} or sequence <= 0:
        return _mark_dispatch_failed(store, pending, "invalid pending signal payload")

    persisted_pending = store.load_pending_signal()
    persisted_signal_id = str(persisted_pending.get("signal_id") or "").strip()
    persisted_status = str(persisted_pending.get("status") or "").strip().lower()
    if persisted_signal_id and persisted_signal_id != signal_id:
        return pending
    if persisted_pending and persisted_status != "pending":
        return dict(persisted_pending)

    dispatch_payload = {
        "signal_id": signal_id,
        "signal_key": pending.get("signal_key"),
        "trade_date": pending.get("trade_date"),
        "action": pending.get("action"),
        "sequence": pending.get("sequence"),
        "channel": channel,
        "chat_id": chat_id,
        "thread_id": thread_id,
        "message": message,
    }
    dispatcher = dispatch_fn or _default_dispatch_pending_signal

    try:
        delivery_result = dispatcher(dispatch_payload)
    except Exception as exc:
        return _mark_dispatch_failed(store, pending, str(exc))

    if not isinstance(delivery_result, dict):
        return _mark_dispatch_failed(store, pending, f"Unexpected dispatch result type: {type(delivery_result).__name__}")
    if not delivery_result.get("success"):
        return _mark_dispatch_failed(store, pending, str(delivery_result.get("error") or "dispatch failed"))

    effective_channel = str(
        delivery_result.get("channel")
        or delivery_result.get("platform")
        or channel
    )
    return confirm_dispatch_sent(
        store,
        pending,
        channel=effective_channel,
        delivery_result=delivery_result,
    )


def confirm_dispatch_sent(
    store: TradingStateStore,
    pending: dict,
    *,
    channel: str,
    delivery_result: Optional[dict] = None,
) -> dict:
    pending = dict(pending)
    sent_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    signal_id = pending.get("signal_id")
    trade_date = str(pending.get("trade_date") or "")
    action = str(pending.get("action") or "").strip()
    sequence = int(pending.get("sequence", 0) or 0)
    if not trade_date:
        raise ValueError("pending signal missing trade_date")
    pending.update(
        {
            "status": "sent",
            "sent_at": sent_at,
            "dispatch_channel": channel,
            "delivery_result": delivery_result or {},
            "last_attempt_status": "sent",
        }
    )

    store.record_dispatch_event(
        dispatch_ledger_sent_event(delivery_result),
        {
            "signal_id": signal_id,
            "signal_key": pending.get("signal_key"),
            "trade_date": trade_date,
            "action": action,
            "sequence": sequence,
            "attempts": pending.get("attempts", 0),
            "channel": channel,
            "delivery_result": delivery_result or {},
        },
    )

    push_state = store.load_push_state()
    push_state.update(
        {
            "last_pushed_signal": pending.get("signal_key"),
            "last_pushed_signal_id": signal_id,
            "last_pushed_trade_date": trade_date,
            "last_pushed_at": sent_at,
            "last_pushed_text": pending.get("text", ""),
            "last_dispatch_channel": channel,
            "last_delivery_result": delivery_result or {},
        }
    )
    store.save_push_state(push_state)

    execution_state = _normalize_execution_state(store.load_execution_state(), trade_date)
    actions = list(execution_state.get("actions") or [])
    existing = next((item for item in actions if item.get("signal_id") == signal_id), None)
    if existing is None and trade_date and action and sequence > 0:
        actions.append(
            {
                "signal_id": signal_id,
                "signal_key": pending.get("signal_key"),
                "trade_date": trade_date,
                "action": action,
                "sequence": sequence,
                "status": "sent",
                "sent_at": sent_at,
                "channel": channel,
                "text": pending.get("text", ""),
            }
        )
    elif existing is not None:
        existing.update(
            {
                "signal_key": pending.get("signal_key"),
                "trade_date": trade_date,
                "action": action,
                "sequence": sequence,
                "status": "sent",
                "sent_at": sent_at,
                "channel": channel,
                "text": pending.get("text", ""),
            }
        )
    active_signal = execution_state.get("active_signal") or {}
    if active_signal.get("signal_id") == signal_id:
        execution_state["active_signal"] = None
    execution_state.update(
        {
            "trade_date": trade_date,
            "actions": actions,
            "last_signal_id": signal_id,
            "last_signal_action": action,
            "last_signal_status": "sent",
            "last_signal_at": sent_at,
        }
    )
    if action == "sell":
        execution_state["sell_count"] = max(int(execution_state.get("sell_count", 0) or 0), sequence)
    elif action == "buy":
        execution_state["buy_count"] = max(int(execution_state.get("buy_count", 0) or 0), sequence)
    store.save_execution_state(execution_state)

    store.save_pending_signal(pending)
    store.clear_pending_signal()
    store.append_signal_send_history(
        "sent",
        {
            "signal_id": signal_id,
            "signal_key": pending.get("signal_key"),
            "trade_date": trade_date,
            "action": action,
            "sequence": sequence,
            "attempts": pending.get("attempts", 0),
            "sent_at": sent_at,
            "channel": channel,
            "delivery_result": delivery_result or {},
        },
    )
    return pending


def run_runtime_cycle(
    store: TradingStateStore,
    *,
    tech_data: dict,
    effective_trade_date: str,
    now: Optional[datetime] = None,
    dispatch: bool = False,
    dispatch_fn: Optional[Callable[[dict], dict]] = None,
    channel: str = "feishu",
    chat_id: Optional[str] = None,
    thread_id: Optional[str] = None,
) -> dict:
    """Run one minimal Hermes-native trading runtime cycle."""

    cycle_now = now or datetime.now()
    pre_recovery_pending = store.load_pending_signal() or {}
    recovery = recover_signal_runtime(
        store,
        trade_date=effective_trade_date,
        now=cycle_now,
    )
    blocked_statuses = {"timed_out", "failed_exhausted"}
    terminal_recovery_repairs = {
        "timed_out_pending_signal",
        "non_retryable_failed_pending_signal",
        "exhausted_failed_pending_signal",
    }
    current_pending = store.load_pending_signal() or {}
    current_execution_state = store.load_execution_state()
    current_push_state = store.load_push_state()
    current_signal = str(tech_data.get("summary_signal") or "hold").strip().lower()
    recovery_repairs = set(recovery.get("repairs") or [])

    if recovery.get("pending_status") in blocked_statuses:
        if recovery_repairs & terminal_recovery_repairs:
            suggestion = {
                "next_action": "hold",
                "sequence": 0,
                "signal": "hold",
                "reason": "recovery_blocked",
                "recovery": recovery,
                "execution_state": current_execution_state,
                "state_snapshot": current_execution_state,
            }
            pending = current_pending
            result = dict(pending)
        else:
            store.clear_pending_signal()
            current_pending = {}
            recovery["pending_status"] = None
            recovery_repairs = set(recovery.get("repairs") or [])
            recovery_repairs.add("cleared_terminal_pending_signal")
            recovery["repairs"] = list(recovery_repairs)
            current_execution_state = store.load_execution_state()
            current_push_state = store.load_push_state()
            suggestion = build_execution_suggestion(store, tech_data, effective_trade_date)
            arbitration = arbitrate_candidate(
                store,
                suggestion,
                effective_trade_date=effective_trade_date,
                now=cycle_now,
            )
            decision = str(arbitration.get("decision") or "").strip().lower()
            execution_state = suggestion.get("execution_state") or store.load_execution_state()

            if decision == "blocked":
                pending = store.load_pending_signal()
                suggestion = {
                    **suggestion,
                    "next_action": "hold",
                    "action": "hold",
                    "sequence": 0,
                    "reason": str(arbitration.get("reason") or "blocked"),
                    "arbitration": arbitration,
                }
                result = dict(pending)
            elif decision == "reuse_failed_pending":
                pending = dict(arbitration.get("pending") or store.load_pending_signal() or {})
                suggestion = {
                    **suggestion,
                    "reason": str(arbitration.get("reason") or suggestion.get("reason") or ""),
                    "arbitration": arbitration,
                }
                result = dict(pending)
                if dispatch and pending:
                    result = deliver_pending_signal(
                        store,
                        pending,
                        dispatch_fn=dispatch_fn,
                        channel=channel,
                        chat_id=chat_id,
                        thread_id=thread_id,
                    )
            else:
                pending = stage_pending_signal(
                    store,
                    suggestion,
                    execution_state,
                    effective_trade_date,
                    cycle_now,
                )
                suggestion = {
                    **suggestion,
                    "arbitration": arbitration,
                }
                result = dict(pending)
                if dispatch and pending:
                    result = deliver_pending_signal(
                        store,
                        pending,
                        dispatch_fn=dispatch_fn,
                        channel=channel,
                        chat_id=chat_id,
                        thread_id=thread_id,
                    )
    elif current_pending and "requeued_retryable_failed_pending_signal" in recovery_repairs:
        pending = dict(current_pending)
        pending_action = str(pending.get("action") or "hold").strip().lower()
        pending_sequence = int(pending.get("sequence", 0) or 0)
        suggestion = {
            "signal": pending.get("signal", pending_action),
            "trade_unit": int(pending.get("trade_unit", store.profile.trade_unit) or store.profile.trade_unit),
            "max_trades": int(store.profile.max_trades),
            "action": pending_action,
            "sequence": pending_sequence,
            "text": pending.get("text", ""),
            "trade_date": effective_trade_date,
            "reason": "retry_ready_reuse_failed_pending",
            "state_snapshot": current_execution_state,
            "execution_state": current_execution_state,
            "next_action": pending_action,
            "recovery": recovery,
        }
        result = dict(pre_recovery_pending or pending)
        if dispatch and pending:
            result = deliver_pending_signal(
                store,
                pending,
                dispatch_fn=dispatch_fn,
                channel=channel,
                chat_id=chat_id,
                thread_id=thread_id,
            )
    elif (
        not current_pending
        and current_signal in {"buy", "sell"}
        and str(current_push_state.get("last_pushed_trade_date") or "").strip() == effective_trade_date
        and str(current_push_state.get("last_pushed_signal") or "").strip().startswith(f"{current_signal}_")
    ):
        suggestion = {
            "signal": current_signal,
            "trade_unit": int(store.profile.trade_unit),
            "max_trades": int(store.profile.max_trades),
            "action": "hold",
            "sequence": 0,
            "text": DEFAULT_SIGNAL_POLICY.no_action_text,
            "trade_date": effective_trade_date,
            "reason": "duplicate_sent_signal",
            "state_snapshot": current_execution_state,
            "execution_state": current_execution_state,
            "next_action": "hold",
            "recovery": recovery,
        }
        pending = {}
        result = {}
    else:
        suggestion = build_execution_suggestion(store, tech_data, effective_trade_date)
        arbitration = arbitrate_candidate(
            store,
            suggestion,
            effective_trade_date=effective_trade_date,
            now=cycle_now,
        )
        decision = str(arbitration.get("decision") or "").strip().lower()
        execution_state = suggestion.get("execution_state") or store.load_execution_state()

        if decision == "blocked":
            pending = store.load_pending_signal()
            suggestion = {
                **suggestion,
                "next_action": "hold",
                "action": "hold",
                "sequence": 0,
                "reason": str(arbitration.get("reason") or "blocked"),
                "arbitration": arbitration,
            }
            result = dict(pending)
        elif decision == "reuse_failed_pending":
            pending = dict(arbitration.get("pending") or store.load_pending_signal() or {})
            suggestion = {
                **suggestion,
                "reason": str(arbitration.get("reason") or suggestion.get("reason") or ""),
                "arbitration": arbitration,
            }
            result = dict(pending)
            if dispatch and pending:
                result = deliver_pending_signal(
                    store,
                    pending,
                    dispatch_fn=dispatch_fn,
                    channel=channel,
                    chat_id=chat_id,
                    thread_id=thread_id,
                )
        else:
            pending = stage_pending_signal(
                store,
                suggestion,
                execution_state,
                effective_trade_date,
                cycle_now,
            )
            suggestion = {
                **suggestion,
                "arbitration": arbitration,
            }
            result = dict(pending)
            if dispatch and pending:
                result = deliver_pending_signal(
                    store,
                    pending,
                    dispatch_fn=dispatch_fn,
                    channel=channel,
                    chat_id=chat_id,
                    thread_id=thread_id,
                )
    current_pending = store.load_pending_signal()
    return {
        "trade_date": effective_trade_date,
        "now": cycle_now.strftime("%Y-%m-%d %H:%M:%S"),
        "recovery": recovery,
        "suggestion": suggestion,
        "pending": current_pending,
        "result": result,
        "execution_state": store.load_execution_state(),
        "push_state": store.load_push_state(),
    }
