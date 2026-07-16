"""Notification / attention budget governor.

Gates PROACTIVE agent-initiated messages (cron deliveries, webhook pushes,
goal-status notices, Omi nudges) against a hard daily attention budget with
per-category thresholds that self-tune from user dismiss/act feedback. Live
user-facing replies are never routed through here — only producers that
explicitly tag a send as ``proactive`` call ``should_deliver``.

Design invariants:
  - FAIL OPEN. A bug in scoring must never silence a message: every entry
    point wraps its body and returns ALLOW on any unexpected error.
  - IMPORT-LIGHT. This module is imported lazily on the delivery hot path,
    so it opens no database and reads no config at import time.
  - IDEMPOTENT. A repeated ``candidate_id`` within a day returns the prior
    decision instead of double-spending the budget.

Scoring: ``score = p_act * value_hint - attention_cost`` where ``p_act`` is a
per-category exponential-moving act rate, ``value_hint`` is the producer's
estimate of importance, and ``attention_cost`` grows with today's send count.
"""

import logging
import os
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, Optional

from hermes_cli.config import load_config
from hermes_constants import get_hermes_home
from hermes_state import SessionDB

logger = logging.getLogger(__name__)


def _open_db() -> SessionDB:
    """Open the active-profile state DB.

    Resolves the path via get_hermes_home() at call time rather than the
    import-frozen DEFAULT_DB_PATH so profile switches and per-test HERMES_HOME
    overrides are honored.
    """
    return SessionDB(get_hermes_home() / "state.db")


@dataclass
class BudgetDecision:
    """Outcome of a governor evaluation."""

    allow: bool
    reason: str
    score: float
    threshold: float
    category: str
    ledger_id: Optional[str] = None


def _day_key() -> str:
    """Local-TZ day bucket, e.g. '2026-07-16'. Tests pin TZ=UTC."""
    return time.strftime("%Y-%m-%d")


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _governor_disabled_env() -> bool:
    """Env kill-switch mirrors the FLAG_RESOLUTION pattern in delivery.py."""
    env = os.getenv("HERMES_NOTIFICATIONS_DISABLED")
    if env is not None:
        return env.strip().lower() in ("1", "true", "yes", "on")
    return False


def _config() -> Dict[str, Any]:
    cfg = load_config()
    section = cfg.get("notifications", {}) if isinstance(cfg, dict) else {}
    return section if isinstance(section, dict) else {}


def _category_config(notif_cfg: Dict[str, Any], category: str) -> Dict[str, Any]:
    """Merge per-category overrides over the base notifications config."""
    merged = dict(notif_cfg)
    overrides = notif_cfg.get("categories", {})
    if isinstance(overrides, dict):
        cat_override = overrides.get(category)
        if isinstance(cat_override, dict):
            merged.update(cat_override)
    return merged


def should_deliver(
    category: str,
    *,
    value_hint: Optional[float] = None,
    candidate_id: Optional[str] = None,
    platform: Optional[str] = None,
    chat_id: Optional[str] = None,
) -> BudgetDecision:
    """Decide whether a proactive notification in *category* may be delivered.

    Returns a :class:`BudgetDecision`. On ALLOW the send is counted against
    today's budget; on suppress a ``deferred`` ledger row is retained for the
    digest. Any internal error results in ALLOW (fail open).
    """
    try:
        return _should_deliver_impl(
            category,
            value_hint=value_hint,
            candidate_id=candidate_id,
            platform=platform,
            chat_id=chat_id,
        )
    except Exception as exc:  # fail open — never block a message on our bug
        logger.debug(
            "notification_budget.should_deliver failed for %s: %s",
            category,
            exc,
            exc_info=True,
        )
        return BudgetDecision(
            allow=True,
            reason="governor-error-fail-open",
            score=0.0,
            threshold=0.0,
            category=category,
        )


def _should_deliver_impl(
    category: str,
    *,
    value_hint: Optional[float],
    candidate_id: Optional[str],
    platform: Optional[str],
    chat_id: Optional[str],
) -> BudgetDecision:
    notif_cfg = _config()

    if not notif_cfg.get("enabled", True) or _governor_disabled_env():
        return BudgetDecision(
            allow=True,
            reason="governor-disabled",
            score=0.0,
            threshold=0.0,
            category=category,
        )

    cat_cfg = _category_config(notif_cfg, category)
    daily_cap = int(cat_cfg.get("daily_cap", 3))
    daily_ceiling = int(cat_cfg.get("daily_ceiling", 5))
    base_threshold = float(cat_cfg.get("base_threshold", 0.5))
    escalation_threshold = float(cat_cfg.get("escalation_threshold", 0.8))
    threshold_min = float(cat_cfg.get("threshold_min", 0.1))
    threshold_max = float(cat_cfg.get("threshold_max", 0.95))
    default_value_hint = float(cat_cfg.get("default_value_hint", 0.5))

    day = _day_key()
    db = _open_db()

    # Idempotency: a repeated candidate must not double-spend the budget.
    if candidate_id:
        prior = db.find_notification_by_candidate(candidate_id, day)
        if prior is not None:
            return BudgetDecision(
                allow=(prior["decision"] == "allowed"),
                reason="idempotent-replay",
                score=float(prior["score"]),
                threshold=float(prior["threshold_used"]),
                category=category,
                ledger_id=prior["id"],
            )

    stats = db.get_category_stats(category)
    if stats is None:
        # Cold start: seed p_act optimistically (1.0) so a brand-new category's
        # first genuinely-valuable notification is judged on the producer's
        # value_hint alone (score == value_hint) rather than being auto-
        # suppressed before any feedback exists. Dismissals pull p_act down
        # from there via the EWMA.
        threshold = _clamp(base_threshold, threshold_min, threshold_max)
        p_act = 1.0
    else:
        threshold = _clamp(float(stats["threshold"]), threshold_min, threshold_max)
        p_act = float(stats["p_act_ewma"])

    value = default_value_hint if value_hint is None else float(value_hint)
    sent_today = db.count_notifications_today(day, decision="allowed")
    # `score` is the message's INTRINSIC importance (p_act * value). Quantity
    # is governed separately by the cap/ceiling counters below — folding the
    # send count into `score` would make the escalation tier unreachable once
    # enough messages had been sent. `attention_cost` is retained purely for
    # ledger observability (how loaded the day was when this fired).
    # daily_ceiling <= 0 means "no hard ceiling" (unbounded), NOT "suppress
    # everything" — guard both the cost divisor and the ceiling comparison so a
    # 0/negative config doesn't silently gag every proactive message.
    has_ceiling = daily_ceiling > 0
    attention_cost = min(1.0, sent_today / daily_ceiling) if has_ceiling else 0.0
    score = _clamp(p_act * value, 0.0, 1.0)

    if has_ceiling and sent_today >= daily_ceiling:
        decision, allow, reason = "deferred", False, "hard-ceiling-reached"
    elif score < threshold:
        decision, allow, reason = "deferred", False, "below-threshold"
    elif daily_cap > 0 and sent_today >= daily_cap and score < escalation_threshold:
        # Between the soft cap and the hard ceiling only high-value messages
        # (score >= escalation_threshold) may spend from the budget.
        decision, allow, reason = "deferred", False, "over-soft-cap-low-score"
    else:
        decision, allow, reason = "allowed", True, "under-budget"

    ledger_id = uuid.uuid4().hex
    db.record_notification({
        "id": ledger_id,
        "candidate_id": candidate_id,
        "category": category,
        "score": score,
        "p_act": p_act,
        "value_hint": value,
        "attention_cost": attention_cost,
        "threshold_used": threshold,
        "decision": decision,
        "platform": platform,
        "chat_id": chat_id,
        "day_key": day,
        "created_at": time.time(),
    })

    if allow:
        sent_count = int(stats["sent_count"]) + 1 if stats else 1
        db.upsert_category_stats(
            category,
            threshold=threshold,
            p_act_ewma=p_act,
            sent_count=sent_count,
            updated_at=time.time(),
        )

    logger.info(
        "notification budget: %s cat=%s score=%.2f thr=%.2f sent_today=%d (%s)",
        decision,
        category,
        score,
        threshold,
        sent_today,
        reason,
    )
    return BudgetDecision(
        allow=allow,
        reason=reason,
        score=score,
        threshold=threshold,
        category=category,
        ledger_id=ledger_id,
    )


def record_feedback(
    category: str, signal: str, *, ledger_id: Optional[str] = None
) -> None:
    """Update the learned threshold for *category* from user feedback.

    ``signal`` is ``"act"`` (user engaged → lower the bar) or ``"dismiss"``
    (user ignored → raise the bar). Fails silently on any error.
    """
    try:
        if signal not in ("act", "dismiss"):
            logger.debug("notification_budget: ignoring unknown signal %r", signal)
            return

        notif_cfg = _config()
        cat_cfg = _category_config(notif_cfg, category)
        dismiss_step = float(cat_cfg.get("dismiss_step", 0.1))
        act_step = float(cat_cfg.get("act_step", 0.05))
        threshold_min = float(cat_cfg.get("threshold_min", 0.1))
        threshold_max = float(cat_cfg.get("threshold_max", 0.95))
        base_threshold = float(cat_cfg.get("base_threshold", 0.5))
        alpha = float(cat_cfg.get("p_act_ewma_alpha", 0.3))

        db = _open_db()
        stats = db.get_category_stats(category)
        threshold = float(stats["threshold"]) if stats else base_threshold
        p_act = float(stats["p_act_ewma"]) if stats else base_threshold
        act_count = int(stats["act_count"]) if stats else 0
        dismiss_count = int(stats["dismiss_count"]) if stats else 0

        if signal == "act":
            threshold = _clamp(threshold - act_step, threshold_min, threshold_max)
            p_act = (1 - alpha) * p_act + alpha * 1.0
            act_count += 1
        else:  # dismiss
            threshold = _clamp(threshold + dismiss_step, threshold_min, threshold_max)
            p_act = (1 - alpha) * p_act + alpha * 0.0
            dismiss_count += 1

        db.upsert_category_stats(
            category,
            threshold=threshold,
            p_act_ewma=p_act,
            act_count=act_count,
            dismiss_count=dismiss_count,
            updated_at=time.time(),
        )

        if ledger_id:
            db.set_notification_feedback(ledger_id, signal)

        logger.info(
            "notification feedback: cat=%s signal=%s -> thr=%.2f p_act=%.2f",
            category,
            signal,
            threshold,
            p_act,
        )
    except Exception as exc:
        logger.debug(
            "notification_budget.record_feedback failed for %s/%s: %s",
            category,
            signal,
            exc,
            exc_info=True,
        )


def budget_status(day_key: Optional[str] = None) -> Dict[str, Any]:
    """Return today's budget usage + per-category thresholds for the CLI.

    Shape: ``{"day": ..., "allowed": int, "cap": int, "ceiling": int,
    "categories": {cat: stats}, "deferred": [ledger rows]}``. Returns an
    ``error`` key on failure.
    """
    try:
        notif_cfg = _config()
        day = day_key or _day_key()
        db = _open_db()
        allowed = db.count_notifications_today(day, decision="allowed")
        deferred = db.list_deferred_today(day)
        categories: Dict[str, Any] = {}
        for row in deferred:
            cat = row["category"]
            if cat not in categories:
                stats = db.get_category_stats(cat)
                categories[cat] = stats or {}
        return {
            "day": day,
            "allowed": allowed,
            "cap": int(notif_cfg.get("daily_cap", 3)),
            "ceiling": int(notif_cfg.get("daily_ceiling", 5)),
            "enabled": bool(notif_cfg.get("enabled", True)),
            "categories": categories,
            "deferred": deferred,
        }
    except Exception as exc:
        logger.debug("notification_budget.budget_status failed: %s", exc, exc_info=True)
        return {"error": str(exc)}
