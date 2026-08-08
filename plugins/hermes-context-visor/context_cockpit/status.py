"""Status classification for the Context Cockpit.

Operator-facing ribbons use plain language. Internal chips stay short codes
for CSS / JSON proof.
"""

from __future__ import annotations

from typing import Any, Dict

from .controls import build_action_controls
from .metrics import HYGIENE_PCT, LCM_ACT_RATIO, LCM_SOON_RATIO


# Display ribbons (what O reads). Keep stable for CSS maps in web.py.
RIBBON_STATES = (
    "HERMES OFFLINE",
    "OLD NUMBERS",
    "MODEL CHANGED",
    "SHRINKING NOW",
    "CAN'T SHRINK YET",
    "SHRINK QUEUED",
    "MEMORY LINE HIT",
    "JUST SHRANK",
    "MEMORY UNKNOWN",
    "COST WARNING",
    "GETTING FULL",
    "QUIET",
    "ALL GOOD",
)


def _fmt_tokens(n: int | None) -> str:
    if n is None:
        return "—"
    n = int(n)
    if n >= 1_000_000:
        return f"{n/1_000_000:.2f}M"
    if n >= 1_000:
        return f"{n/1_000:.1f}k"
    return str(n)


def _fmt_usd(n: float | None) -> str:
    if n is None:
        return "—"
    if n < 0.01:
        return f"${n:.4f}"
    return f"${n:.2f}"


def _fmt_age_short(seconds: float | int | None) -> str:
    if seconds is None:
        return "unknown"
    n = max(0, float(seconds))
    if n >= 3600:
        return f"{n / 3600:.1f}h"
    if n >= 60:
        return f"{n / 60:.1f}m"
    return f"{n:.0f}s"


def _float_or_none(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def plain_noop_reason(reason: str) -> str:
    """Translate engine no-op codes into plain language for O."""
    raw = (reason or "").strip()
    low = raw.lower()
    if not low:
        return "Hermes looked, then decided not to shrink the chat yet."
    if "fresh_tail" in low or "fresh tail" in low:
        return (
            "Almost everything left is recent chat that Hermes protects, "
            "so there isn't enough older chat to shrink yet."
        )
    if "no eligible" in low or "backlog" in low:
        return (
            "There isn't enough older chat outside the protected recent "
            "messages to make shrinking useful yet."
        )
    if "leaf" in low or "chunk" in low or "floor" in low:
        return "There isn't enough older chat piled up yet to make shrinking worthwhile."
    if "no_active_session" in low or "no active" in low:
        return "Hermes does not have an active chat session bound for live auto-shrink status."
    return f"Hermes decided not to shrink yet. Technical note: {raw}."


def plain_live_status(status: str) -> str:
    """Translate live compression status codes for O."""
    low = (status or "").strip().lower()
    return {
        "running": "Shrinking the chat right now",
        "pending": "Ready to shrink on the next chance",
        "noop": "Checked, but did not shrink",
        "idle": "Idle — not shrinking right now",
        "": "No live update yet",
    }.get(low, status or "Unknown")


def classify_lcm_state(metrics: Dict[str, Any]) -> Dict[str, Any]:
    lcm = metrics.get("lcm") or {}
    fill = float(lcm.get("fill_of_lcm") or 0.0)
    threshold_tokens = int(lcm.get("threshold_tokens") or 0)
    last_status = str(lcm.get("last_compression_status") or "").strip().lower()
    noop_reason = str(lcm.get("last_compression_noop_reason") or "").strip()
    turns_since = lcm.get("turns_since_leaf")
    turns_since_i = int(turns_since) if turns_since is not None else None
    last_leaf_at = _float_or_none(lcm.get("last_leaf_compaction_at"))
    last_api_at = _float_or_none(lcm.get("last_api_call_at"))
    compactions = int(lcm.get("compressions") or 0)

    if not lcm.get("loaded"):
        return {
            "label": "Memory status unknown",
            "chip": "unknown",
            "ribbon": "MEMORY UNKNOWN",
            "summary": "This cockpit cannot read LCM / auto-shrink status for this chat yet.",
            "detail": "Open Hermes Desktop on personal-ops, or send one chat turn and refresh.",
            "active_proven": False,
            "live_status_plain": plain_live_status(last_status),
            "noop_reason_plain": "",
        }

    if threshold_tokens <= 0:
        return {
            "label": "Auto-shrink line unknown",
            "chip": "unknown",
            "ribbon": "MEMORY UNKNOWN",
            "summary": "The auto-shrink line is not set yet, so pressure cannot be judged.",
            "detail": "Wait for Hermes to finish loading this session, then refresh.",
            "active_proven": False,
            "live_status_plain": plain_live_status(last_status),
            "noop_reason_plain": "",
        }

    if last_status == "running":
        return {
            "label": "Shrinking now",
            "chip": "running",
            "ribbon": "SHRINKING NOW",
            "summary": "Hermes is shrinking older chat right now to free space.",
            "detail": "Give it a moment, then refresh. You usually do not need to do anything.",
            "active_proven": True,
            "live_status_plain": plain_live_status(last_status),
            "noop_reason_plain": "",
        }

    if last_status == "pending":
        return {
            "label": "Shrink queued",
            "chip": "pending",
            "ribbon": "SHRINK QUEUED",
            "summary": "Chat crossed the auto-shrink line and Hermes plans to shrink soon.",
            "detail": "Send or finish the next turn and let Hermes try. Check again if it stays stuck.",
            "active_proven": False,
            "live_status_plain": plain_live_status(last_status),
            "noop_reason_plain": "",
        }

    if last_status == "noop":
        reason_plain = plain_noop_reason(noop_reason)
        pre_tail = lcm.get("pre_tail_message_count")
        fresh_tail = lcm.get("fresh_tail_count")
        total = lcm.get("total_message_count")
        bits = []
        if total is not None:
            bits.append(f"{int(total)} messages in view")
        if fresh_tail is not None:
            bits.append(f"{int(fresh_tail)} recent (protected)")
        if pre_tail is not None:
            bits.append(f"{int(pre_tail)} older (shrinkable)")
        detail = reason_plain
        if bits:
            detail += " Right now: " + "; ".join(bits) + "."
        return {
            "label": "Can't shrink yet",
            "chip": "noop",
            "ribbon": "CAN'T SHRINK YET",
            "summary": reason_plain,
            "detail": detail,
            "active_proven": False,
            "live_status_plain": plain_live_status(last_status),
            "noop_reason_plain": reason_plain,
        }

    recent_compaction = False
    if last_leaf_at is not None and last_api_at is not None and compactions > 0:
        recent_compaction = last_api_at >= last_leaf_at and (last_api_at - last_leaf_at) <= 120
    if compactions > 0 and recent_compaction and (turns_since_i is None or turns_since_i <= 1):
        ago_seconds = 0.0 if last_api_at is None or last_leaf_at is None else (last_api_at - last_leaf_at)
        ago = _fmt_age_short(ago_seconds)
        return {
            "label": "Just shrank",
            "chip": "recent",
            "ribbon": "JUST SHRANK",
            "summary": f"Hermes already shrank older chat recently ({ago} before the last reply).",
            "detail": "No action needed unless the chat fills up again.",
            "active_proven": False,
            "live_status_plain": plain_live_status(last_status),
            "noop_reason_plain": "",
        }

    if fill >= 1.0:
        return {
            "label": "Memory line hit",
            "chip": "threshold",
            "ribbon": "MEMORY LINE HIT",
            "summary": (
                f"This chat is at {_fmt_tokens(int(metrics.get('prompt_tokens') or 0))} tokens, "
                f"at or above the auto-shrink line of {_fmt_tokens(threshold_tokens)}."
            ),
            "detail": (
                "That only means the line was crossed. It does not prove Hermes is shrinking "
                "right this second."
            ),
            "active_proven": False,
            "live_status_plain": plain_live_status(last_status),
            "noop_reason_plain": "",
        }

    if fill >= LCM_ACT_RATIO:
        return {
            "label": "Getting full",
            "chip": "near",
            "ribbon": "GETTING FULL",
            "summary": f"About {fill * 100:.0f}% of the way to the auto-shrink line.",
            "detail": "You are close. Hermes may shrink soon, or you can compress after this answer.",
            "active_proven": False,
            "live_status_plain": plain_live_status(last_status),
            "noop_reason_plain": "",
        }

    if fill >= LCM_SOON_RATIO:
        return {
            "label": "Watching",
            "chip": "armed",
            "ribbon": "QUIET",
            "summary": f"About {fill * 100:.0f}% of the way to the auto-shrink line.",
            "detail": "Still comfortable. No action needed.",
            "active_proven": False,
            "live_status_plain": plain_live_status(last_status),
            "noop_reason_plain": "",
        }

    return {
        "label": "Plenty of room",
        "chip": "below",
        "ribbon": "ALL GOOD",
        "summary": "Chat memory is below the auto-shrink line.",
        "detail": "No memory pressure right now.",
        "active_proven": False,
        "live_status_plain": plain_live_status(last_status),
        "noop_reason_plain": "",
    }


def classify_status(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Return ribbon + next-action classification from a metrics snapshot."""
    freshness = metrics.get("freshness") or "unknown"
    pct = float(metrics.get("prompt_pct") or 0.0)
    lcm = metrics.get("lcm") or {}
    cost = metrics.get("cost") or {}
    fill = float(lcm.get("fill_of_lcm") or 0.0)
    thresh_tok = int(lcm.get("threshold_tokens") or 0)
    compressions = int(lcm.get("compressions") or 0)
    msg_count = int(metrics.get("message_count") or 0)
    model_alert = metrics.get("model_alert")
    est_usd = cost.get("estimated_usd")
    burn = cost.get("burn") or {}
    lcm_state = classify_lcm_state(metrics)

    if freshness == "offline":
        return _result(
            "HERMES OFFLINE",
            "critical",
            "The cockpit is open, but Hermes Desktop does not look running for personal-ops yet.",
            "If you just opened /visor, wait a few seconds. Otherwise open Hermes Desktop on personal-ops.",
            "/status",
            dim_gauges=True,
        )
    if freshness == "stale":
        return _result(
            "OLD NUMBERS",
            "critical",
            "These numbers look old — they may not match what Hermes is doing right now.",
            "Send one chat message in Hermes Desktop, wait for refresh, and check again.",
            "/status",
            dim_gauges=True,
        )

    if model_alert:
        return _result(
            "MODEL CHANGED",
            "critical",
            str(model_alert),
            "Check which model is active before the next heavy turn — cost and quality may have changed.",
            "/model",
            dim_gauges=False,
        )

    if pct >= HYGIENE_PCT:
        return _result(
            "GETTING FULL",
            "critical",
            f"Chat context is at {pct:.0f}% of the model window — getting tight.",
            "After this answer, run /compress in Hermes Desktop to free space.",
            "/compress here 4",
        )

    if lcm_state["ribbon"] == "SHRINKING NOW":
        return _result(
            "SHRINKING NOW",
            "critical",
            lcm_state["summary"],
            "Wait and refresh. Hermes is already shrinking older chat (recent messages stay).",
            "/lcm status",
        )

    if lcm_state["ribbon"] == "SHRINK QUEUED":
        return _result(
            "SHRINK QUEUED",
            "critical",
            lcm_state["summary"],
            "Let the next chat turn finish. If nothing changes, ask Hermes for /lcm status.",
            "/lcm status",
        )

    if lcm_state["ribbon"] == "CAN'T SHRINK YET":
        return _result(
            "CAN'T SHRINK YET",
            "critical",
            lcm_state["summary"],
            "You usually do not need to force anything. Keep chatting, or ask Hermes for /lcm status if it stays stuck.",
            "/lcm status",
        )

    if lcm_state["ribbon"] == "MEMORY LINE HIT":
        return _result(
            "MEMORY LINE HIT",
            "critical",
            lcm_state["summary"],
            "Watch the next turns. If the chat stays stuck full, ask Hermes for /lcm status.",
            "/lcm status",
        )

    if lcm_state["ribbon"] == "JUST SHRANK":
        return _result(
            "JUST SHRANK",
            "info",
            lcm_state["summary"],
            "No action needed unless the chat fills up again.",
            "/lcm status",
        )

    if lcm_state["ribbon"] == "MEMORY UNKNOWN":
        return _result(
            "MEMORY UNKNOWN",
            "warn",
            lcm_state["summary"],
            "Send one Desktop turn and refresh, or ask Hermes for /lcm status.",
            "/lcm status",
        )

    included = False
    status = (cost.get("cost_status") or "").lower()
    mode = (cost.get("billing_mode") or "").lower()
    if status == "included" or "subscription" in mode:
        included = True
    usd_per_call = burn.get("usd_per_call_recent")
    if (
        not included
        and est_usd is not None
        and float(est_usd) >= 2.0
        and usd_per_call is not None
        and float(usd_per_call) >= 0.15
    ):
        return _result(
            "COST WARNING",
            "warn",
            f"This session is about {_fmt_usd(float(est_usd))}; recent burn {_fmt_usd(float(usd_per_call))} per call.",
            "Consider compressing or switching to a cheaper model before the next heavy turn.",
            "/compress --preview",
        )

    already_compacted = compressions >= 1 and msg_count <= 120
    if already_compacted and fill < LCM_ACT_RATIO:
        return _result(
            "ALL GOOD",
            "ok",
            "Older chat was already auto-shrunk; what remains is mostly fixed overhead.",
            "Do nothing. Manual compress will not shrink the fixed tool/system parts.",
            "/usage",
        )

    if thresh_tok > 0 and fill >= LCM_ACT_RATIO:
        return _result(
            "GETTING FULL",
            "warn",
            f"About {fill*100:.0f}% of the way to the auto-shrink line.",
            "After this answer, run /compress in Hermes Desktop, or wait for auto-shrink.",
            "/compress here 4",
        )

    if thresh_tok > 0 and fill >= LCM_SOON_RATIO:
        return _result(
            "QUIET",
            "info",
            f"About {fill*100:.0f}% of the way to the auto-shrink line.",
            "Optional: preview /compress, or just keep going.",
            "/compress --preview",
        )

    if freshness == "quiet":
        return _result(
            "QUIET",
            "info",
            "No live Hermes process seen yet, but the session files look recent.",
            "Confirm Hermes Desktop is open on personal-ops if you expect live numbers.",
            "/status",
        )

    if freshness == "idle":
        return _result(
            "QUIET",
            "info",
            "Hermes Desktop is open, but no new chat activity has been seen for a while.",
            "If a turn is still running, wait — numbers catch up when the reply finishes. Otherwise do nothing.",
            "/compress --preview",
        )

    return _result(
        "ALL GOOD",
        "ok",
        "Chat has room · auto-shrink is armed · cost looks steady · model is stable",
        "Do nothing.",
        "/compress --preview",
    )


def _result(
    ribbon: str,
    severity: str,
    summary: str,
    next_action: str,
    command: str,
    *,
    dim_gauges: bool = False,
) -> Dict[str, Any]:
    color = {
        "ok": "green",
        "info": "yellow",
        "warn": "yellow",
        "critical": "red",
    }.get(severity, "white")
    severity_label = {
        "ok": "good",
        "info": "info",
        "warn": "watch",
        "critical": "needs attention",
    }.get(severity, severity)
    return {
        "ribbon": ribbon,
        "severity": severity,
        "severity_label": severity_label,
        "color": color,
        "summary": summary,
        "next_action": next_action,
        "command": command,
        "dim_gauges": dim_gauges,
        "useful_commands": [
            "/compress --preview",
            "/compress here 4",
            "/lcm status",
            "/usage",
            "/model",
        ],
    }


def build_status_payload(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """JSON proof surface: metrics + classification."""
    status = classify_status(metrics)
    lcm_state = classify_lcm_state(metrics)
    return {
        "ok": True,
        "read_only": True,
        "ribbon": status["ribbon"],
        "severity": status["severity"],
        "severity_label": status.get("severity_label"),
        "summary": status["summary"],
        "next_action": status["next_action"],
        "command": status["command"],
        "dim_gauges": status["dim_gauges"],
        "lcm_state": lcm_state,
        "controls": build_action_controls(status),
        "metrics": metrics,
        "status": status,
    }
