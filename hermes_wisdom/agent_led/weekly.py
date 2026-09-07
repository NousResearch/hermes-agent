"""Weekly agent-led review job.

Piggybacks on the same housekeeping tick that runs the curator (see
``gateway/run.py``) and ``hermes wisdom review-week`` for manual runs. The
job gathers evidence, asks the agent for recommendations, and hands the
resulting events to a delivery callback. A failed delivery never marks a
skill as suggested; signed-out profiles produce nothing and never replay.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

from ..store import WisdomStore
from .agent import ModelCall, review_candidates
from .evidence import build_evidence
from .history import SuggestionHistory, history_path
from .notify import DeliveryLedger, RecommendationEvent, deliver, share_candidate_event
from .policy import AgentLedPolicy, load_policy
from .schemas import SchemaRejected

logger = logging.getLogger(__name__)

STATE_FILE = "agent_led_review_state.json"
LEDGER_FILE = "agent_led_delivery.json"


def _state_path() -> Path:
    return history_path().parent / STATE_FILE


def _ledger_path() -> Path:
    return history_path().parent / LEDGER_FILE


def load_state(path: Path | None = None) -> dict[str, Any]:
    target = path or _state_path()
    try:
        raw = json.loads(target.read_text(encoding="utf-8"))
        return raw if isinstance(raw, dict) else {}
    except (OSError, ValueError):
        return {}


def save_state(state: dict[str, Any], path: Path | None = None) -> None:
    target = path or _state_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, target)


def is_due(policy: AgentLedPolicy, *, state: dict[str, Any] | None = None, now: datetime | None = None) -> bool:
    if not policy.enabled:
        return False
    current = now or datetime.now(timezone.utc)
    data = state if state is not None else load_state()
    last = data.get("last_run_at")
    if not isinstance(last, str):
        return True
    try:
        last_dt = datetime.fromisoformat(last)
    except ValueError:
        return True
    if last_dt.tzinfo is None:
        last_dt = last_dt.replace(tzinfo=timezone.utc)
    return current - last_dt >= timedelta(hours=policy.review_interval_hours)


def _signed_in(store: WisdomStore) -> bool:
    try:
        return store.active_org_id() is not None
    except Exception:
        return False


def _supporting_signals(store: WisdomStore) -> dict[str, list[str]]:
    """Old deterministic triggers become advisory signals for the agent."""
    signals: dict[str, list[str]] = {}
    try:
        for event in store.local_events(kind="wisdom.candidate"):
            payload = event.get("payload") or {}
            name = str(payload.get("skill_name") or "")
            qualification = str(event.get("qualification") or payload.get("qualification") or "")
            if name and qualification:
                signals.setdefault(name, []).append(qualification)
    except Exception as exc:  # pragma: no cover - advisory only
        logger.debug("Supporting signals unavailable: %s", type(exc).__name__)
    return signals


def _safety(service: Any, rec_skill_name: str, content_hash: str) -> tuple[bool | None, list[str]]:
    """Map the existing professionalism review onto the fixed check lines."""
    if service is None:
        return None, []
    try:
        review = service.finish_candidate_professionalism_review(
            skill_id=rec_skill_name, content_hash=content_hash
        )
    except Exception:
        return None, []
    failed: list[str] = []
    mapping = {
        "profanity_or_abuse": "No profanity or abusive language",
        "hate_or_harassment": "No hate or harassment",
        "sexual_or_graphic_language": "No sexual or graphic content",
    }
    for check in (review or {}).get("checks") or []:
        if isinstance(check, dict) and check.get("status") == "advisory":
            line = mapping.get(str(check.get("key")))
            if line:
                failed.append(line)
    # A language review alone cannot assert that credential checks succeeded.
    return False if failed else None, failed


def run_weekly_review(
    *,
    store: WisdomStore | None = None,
    policy: AgentLedPolicy | None = None,
    history: SuggestionHistory | None = None,
    ledger: DeliveryLedger | None = None,
    sender: Callable[[RecommendationEvent], Any] | None = None,
    model_call: ModelCall | None = None,
    service: Any = None,
    organization: dict[str, Any] | None = None,
    memory: dict[str, Any] | None = None,
    recipient_id: str = "owner",
    now: datetime | None = None,
    skills_root: Path | None = None,
    state_path: Path | None = None,
    force: bool = False,
    sleep: Callable[[float], None] | None = None,
) -> dict[str, Any]:
    """Run one review pass and return a structured log of what happened."""
    current = now or datetime.now(timezone.utc)
    state = store or WisdomStore()
    effective = policy or load_policy(client=getattr(service, "client", None))
    log: dict[str, Any] = {
        "ran": False,
        "policy": effective.as_dict(),
        "window_days": effective.window_days,
        "considered": [],
        "excluded": {},
        "selected": [],
        "delivery": [],
        "skipped_reason": None,
    }
    if not effective.enabled:
        log["skipped_reason"] = "agent_led_disabled"
        return log
    if not effective.notification_defaults.get("skill_ready_to_share", False) or effective.max_candidates == 0:
        log["skipped_reason"] = "organization_notifications_disabled"
        return log
    if not _signed_in(state):
        # Signed-out users get nothing; no deferred replay is queued.
        log["skipped_reason"] = "signed_out"
        return log
    run_state = load_state(state_path)
    if not force and not is_due(effective, state=run_state, now=current):
        log["skipped_reason"] = "not_due"
        return log

    ledger_obj = ledger or DeliveryLedger(_ledger_path())
    hist = history or SuggestionHistory()
    org = organization
    if org is None and service is not None:
        try:
            org = {"id": state.active_org_id(), "name": service.organization_display_name()}
        except Exception:
            org = {"id": state.active_org_id()}
    org = org or {"id": state.active_org_id()}

    evidence = build_evidence(
        store=state,
        policy=effective,
        history=hist,
        organization=org,
        at=current,
        skills_root=skills_root,
        supporting_signals=_supporting_signals(state),
    )
    log["ran"] = True
    log["window"] = {"start": evidence.window_start, "end": evidence.window_end}
    log["considered"] = [c.skill_name for c in evidence.candidates]
    log["excluded"] = evidence.excluded
    run_state["last_run_at"] = current.isoformat()
    save_state(run_state, state_path)

    if not evidence.candidates:
        log["skipped_reason"] = "no_candidates"
        logger.info("wisdom agent-led review: window=%s considered=0 excluded=%s", log["window"], evidence.excluded)
        return log

    payload = {**evidence.as_dict(), "max_candidates": effective.max_candidates, "memory": memory or {}}
    try:
        result = review_candidates(payload, model_call=model_call)
    except SchemaRejected as exc:
        log["skipped_reason"] = f"agent_output_rejected: {exc}"
        logger.warning("wisdom agent-led review: agent output rejected: %s", exc)
        return log
    except Exception as exc:  # noqa: BLE001 - background job must not raise
        log["skipped_reason"] = f"agent_call_failed: {type(exc).__name__}"
        logger.warning("wisdom agent-led review: agent call failed: %s", type(exc).__name__)
        return log

    by_name = {c.skill_name: c for c in evidence.candidates}
    org_id = str(org.get("id") or "")
    for rec in result.recommendations[: effective.max_candidates]:
        source = by_name.get(rec.skill_name)
        if source is None or source.content_hash != rec.content_hash:
            log["delivery"].append({"skill": rec.skill_name, "delivered": False, "reason": "unknown_or_stale_candidate"})
            continue
        # Real evidence always comes from the ledger, never from the agent.
        rec.evidence.invocation_count = source.invocation_count
        rec.evidence.days_used = source.days_used
        rec.evidence.window_days = source.window_days
        checks_passed, failed = _safety(service, rec.skill_name, rec.content_hash)
        event = share_candidate_event(
            rec,
            organization_id=org_id,
            recipient_id=recipient_id,
            checks_passed=checks_passed,
            failed_checks=failed,
            ttl_hours=effective.recommendation_ttl_hours,
            at=current,
        )
        log["selected"].append(rec.skill_name)
        if sender is None:
            ledger_obj.record(event, state="pending")
            log["delivery"].append({"skill": rec.skill_name, "delivered": False, "reason": "no_sender", "dedup_key": event.dedup_key})
            continue
        outcome = deliver(
            event,
            sender=sender,
            ledger=ledger_obj,
            retries=effective.delivery_retries,
            backoff_seconds=effective.delivery_backoff_seconds,
            now=current,
            **({"sleep": sleep} if sleep else {}),
        )
        log["delivery"].append({"skill": rec.skill_name, **outcome})
        if outcome.get("delivered") and not outcome.get("duplicate"):
            hist.record_suggested(rec.skill_name, rec.content_hash, at=current)
    logger.info(
        "wisdom agent-led review: window=%s considered=%d excluded=%s selected=%s delivery=%s",
        log["window"],
        len(log["considered"]),
        {k: len(v) for k, v in evidence.excluded.items() if v},
        log["selected"],
        [(d.get("skill"), d.get("delivered")) for d in log["delivery"]],
    )
    return log


def maybe_run_weekly_review(**kwargs: Any) -> dict[str, Any] | None:
    """Best-effort tick hook; never raises."""
    try:
        return run_weekly_review(**kwargs)
    except Exception as exc:  # noqa: BLE001
        logger.debug("maybe_run_weekly_review failed: %s", exc, exc_info=True)
        return None
