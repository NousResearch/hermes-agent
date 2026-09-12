"""Content-free audit trace for protected pre-delivery turns.

Only bounded enums, counters, booleans and a random per-turn identifier are
emitted. Prompts, responses, evidence, tool arguments, session identifiers and
policy reasons must never enter this channel.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
import uuid
from typing import Any

_LOGGER = logging.getLogger("hermes.clinical_delivery_trace")
_ALLOWED_EVENTS = {"scope", "repair", "decision", "delivery"}
_ALLOWED_OUTCOMES = {
    "protected", "ordinary", "allow", "replace", "block",
    "released", "delivered", "failed", "suppressed", "interrupted",
}
_ALLOWED_REASON_CODES = {
    "missing_ledger", "missing_body", "irrelevant_evidence", "polarity_mismatch",
    "query_echo", "unsupported_specifics", "unsupported_claims", "missing_evidence",
    "missing_state", "safe_degradation", "policy_block",
}


def _append_trace_record(payload: dict[str, Any]) -> None:
    """Append one private JSONL record; tracing must never break delivery."""
    try:
        hermes_home = Path(os.environ.get("HERMES_HOME", "~/.hermes")).expanduser()
        log_dir = hermes_home / "logs"
        log_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        path = log_dir / "clinical_delivery_trace.jsonl"
        encoded = (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode()
        fd = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
        try:
            os.write(fd, encoded)
        finally:
            os.close(fd)
    except Exception:
        _LOGGER.debug("clinical trace sink unavailable", exc_info=True)


def begin_trace(agent: Any, *, protected: bool, platform: str) -> dict[str, Any] | None:
    if not protected:
        agent._clinical_delivery_trace = None
        return None
    trace = {
        "schema": 1,
        "trace_id": uuid.uuid4().hex,
        "platform": str(platform or "unknown")[:32],
        "protected": True,
        "repair_count": 0,
        "decision": "",
    }
    agent._clinical_delivery_trace = trace
    emit(agent, "scope", outcome="protected")
    return snapshot(agent)


def emit(
    agent: Any,
    event: str,
    *,
    outcome: str = "",
    repair_count: int | None = None,
    reason_codes: list[str] | tuple[str, ...] | None = None,
) -> None:
    trace = getattr(agent, "_clinical_delivery_trace", None)
    if not isinstance(trace, dict) or event not in _ALLOWED_EVENTS:
        return
    if repair_count is not None:
        trace["repair_count"] = max(0, int(repair_count))
    if event == "decision" and outcome in {"allow", "replace", "block"}:
        trace["decision"] = outcome
    payload = {
        "schema": 1,
        "trace_id": trace["trace_id"],
        "event": event,
        "platform": trace["platform"],
        "protected": trace["protected"],
        "repair_count": trace["repair_count"],
        "decision": trace["decision"],
        "outcome": outcome if outcome in _ALLOWED_OUTCOMES else "",
        "reason_codes": sorted({
            code for code in (reason_codes or ()) if code in _ALLOWED_REASON_CODES
        }),
    }
    _LOGGER.info("clinical_delivery_trace %s", json.dumps(payload, sort_keys=True, separators=(",", ":")))
    _append_trace_record(payload)


def snapshot(agent: Any) -> dict[str, Any] | None:
    trace = getattr(agent, "_clinical_delivery_trace", None)
    if not isinstance(trace, dict) or not trace.get("protected"):
        return None
    return {
        "schema": 1,
        "trace_id": trace["trace_id"],
        "platform": trace["platform"],
        "protected": True,
        "repair_count": trace["repair_count"],
        "decision": trace["decision"],
    }


def delivery(trace: Any, *, outcome: str) -> None:
    if not isinstance(trace, dict) or not trace.get("protected") or outcome not in _ALLOWED_OUTCOMES:
        return
    payload = {
        "schema": 1,
        "trace_id": str(trace.get("trace_id") or "")[:32],
        "event": "delivery",
        "platform": str(trace.get("platform") or "unknown")[:32],
        "protected": True,
        "repair_count": max(0, int(trace.get("repair_count") or 0)),
        "decision": str(trace.get("decision") or "") if trace.get("decision") in {"allow", "replace", "block"} else "",
        "outcome": outcome,
    }
    _LOGGER.info("clinical_delivery_trace %s", json.dumps(payload, sort_keys=True, separators=(",", ":")))
    _append_trace_record(payload)
