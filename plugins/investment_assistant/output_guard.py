"""LLM-backed output guard for investment assistant workflow replies."""

from __future__ import annotations

import json
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

_OUTPUT_GUARD_BY_SESSION: dict[str, dict[str, Any]] = {}
_OUTPUT_GUARD_AUDIT: list[dict[str, Any]] = []

_SYSTEM_PROMPT = """\
You are a strict output guard for an investment assistant.

Decide whether the assistant response is a faithful user-facing explanation
of the provided workflow artifact. The assistant may rephrase and explain
terms in plain language, but it must not add facts.

Return JSON only:
{"allowed": true|false, "reasons": ["short reason"], "confidence": 0.0-1.0}

Mark allowed=false if the response:
- introduces symbols, sectors, weights, percentages, orders, option legs, or
  risk claims not present in the artifact brief or contract;
- claims current holdings were read when the artifact says they were not;
- gives buy/sell/order instructions without a construction_plan artifact;
- exposes internal implementation field names instead of user-facing language;
- contradicts warnings or scope limits in the artifact.
"""


def remember_output_guard(result: dict[str, Any], session_key: str | None = None) -> None:
    """Store one workflow answer contract for the next final LLM response."""
    contract = result.get("answer_contract")
    fallback_response = result.get("fallback_response") or result.get("display_response")
    key = session_key or os.getenv("HERMES_SESSION_ID") or ""
    if not key:
        return
    if not isinstance(contract, dict) or contract.get("mode") != "artifact_only":
        return
    if isinstance(fallback_response, str) and fallback_response.strip():
        _OUTPUT_GUARD_BY_SESSION[key] = {
            "contract": contract,
            "fallback_response": fallback_response,
            "agent_brief": result.get("agent_brief") or "",
            "data_keys": sorted((result.get("data") or {}).keys()),
            "state": result.get("state") or "",
        }


def transform_llm_output(response_text: str, session_id: str = "", **_: Any) -> str | None:
    """Allow safe rephrasing, but fall back if the judge sees artifact escape."""
    if not session_id:
        return None
    guard = _OUTPUT_GUARD_BY_SESSION.pop(session_id, None)
    if not guard:
        return None

    allowed, reasons, judge = _judge_agent_response(response_text, guard)
    _record_output_guard_audit(session_id, allowed, reasons, judge)
    if allowed:
        return None
    return guard["fallback_response"]


def output_guard_audit() -> list[dict[str, Any]]:
    """Return a copy of recent output-guard decisions for tests/debugging."""
    return list(_OUTPUT_GUARD_AUDIT)


def _judge_agent_response(response_text: str, guard: dict[str, Any]) -> tuple[bool, list[str], str]:
    if not (response_text or "").strip():
        return False, ["empty_response"], "deterministic"
    try:
        return _llm_judge_agent_response(response_text, guard)
    except Exception as exc:
        logger.info("investment assistant output judge unavailable: %s", exc)
        return False, [f"judge_unavailable:{type(exc).__name__}"], "fallback"


def _llm_judge_agent_response(response_text: str, guard: dict[str, Any]) -> tuple[bool, list[str], str]:
    from agent.auxiliary_client import call_llm

    payload = {
        "assistant_response": response_text,
        "workflow_contract": guard.get("contract") or {},
        "workflow_brief": guard.get("agent_brief") or guard.get("fallback_response") or "",
        "workflow_state": guard.get("state") or "",
        "data_keys": guard.get("data_keys") or [],
        "fallback_response": guard.get("fallback_response") or "",
    }
    response = call_llm(
        task="approval",
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ],
        temperature=None,
        max_tokens=300,
        timeout=30,
    )
    content = response.choices[0].message.content or ""
    verdict = _parse_json_object(content)
    allowed = _parse_allowed(verdict.get("allowed"))
    reasons = verdict.get("reasons")
    if isinstance(reasons, str):
        reasons = [reasons]
    if not isinstance(reasons, list):
        reasons = []
    if not allowed and not reasons:
        reasons = ["judge_rejected"]
    return allowed, [str(reason) for reason in reasons], "llm"


def _parse_allowed(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "yes", "allow", "allowed"}
    return False


def _parse_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        if stripped.lower().startswith("json"):
            stripped = stripped[4:].strip()
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start == -1 or end == -1 or end < start:
            raise
        parsed = json.loads(stripped[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("judge response must be a JSON object")
    return parsed


def _record_output_guard_audit(
    session_id: str,
    allowed: bool,
    reasons: list[str],
    judge: str,
) -> None:
    _OUTPUT_GUARD_AUDIT.append(
        {
            "session_id": session_id,
            "allowed": allowed,
            "reasons": reasons,
            "judge": judge,
        },
    )
    del _OUTPUT_GUARD_AUDIT[:-50]
