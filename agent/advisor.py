"""Evidence-based Advisor final audit gate.

The Advisor is a non-executing reviewer: it never calls tools and never mutates
agent history beyond the final assistant text chosen by the Commander path.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import logging
import re
import time
from typing import Any, Dict, Iterable, List, Optional

logger = logging.getLogger(__name__)

_VERDICTS = {"PASS", "CHANGES_REQUIRED", "BLOCK", "UNAVAILABLE"}
_SECRET_VALUE_PATTERNS = (
    re.compile(r"(?i)(sk-[A-Za-z0-9_\-]{20,})"),
    re.compile(r"(?i)(sk-or-v1-[A-Za-z0-9_\-]{20,})"),
    re.compile(r"(?i)(xox[baprs]-[A-Za-z0-9_\-]{20,})"),
    re.compile(r"(?i)(gh[pousr]_[A-Za-z0-9_]{20,})"),
    re.compile(r"(?i)(hf_[A-Za-z0-9_\-]{20,})"),
    re.compile(r"(?i)(api[_-]?key|token|secret|password)(\s*[:=]\s*)\S+"),
)


@dataclass
class AdvisorConfig:
    enabled: bool = False
    mode: str = "observe"  # off | observe | final_gate | full (v0: full aliases final_gate)
    provider: str = "auto"
    model: str = ""
    base_url: str = ""
    api_key: str = ""
    max_calls_per_turn: int = 2
    max_repair_iterations: int = 1
    max_input_chars: int = 64000
    timeout_seconds: float = 90.0
    max_output_tokens: int = 1200
    failure_policy: str = "fail_open"  # fail_open | fail_closed
    phases: Dict[str, Any] = field(default_factory=lambda: {"final": "actionful_only"})
    block_on: List[str] = field(default_factory=lambda: [
        "secret_exposure",
        "irreversible_unapproved_action",
        "critical_false_claim",
        "user_constraint_violation",
    ])
    receipt: Dict[str, Any] = field(default_factory=lambda: {
        "persist": True,
    })
    extra_body: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, raw: Any) -> "AdvisorConfig":
        if not isinstance(raw, dict):
            return cls()

        def _bool(value: Any, default: bool = False) -> bool:
            if isinstance(value, bool):
                return value
            if value is None:
                return default
            return str(value).strip().lower() in {"1", "true", "yes", "on"}

        def _int(value: Any, default: int, minimum: int = 0) -> int:
            try:
                if isinstance(value, bool):
                    raise ValueError
                return max(minimum, int(value))
            except (TypeError, ValueError):
                return default

        def _float(value: Any, default: float, minimum: float = 0.0) -> float:
            try:
                if isinstance(value, bool):
                    raise ValueError
                return max(minimum, float(value))
            except (TypeError, ValueError):
                return default

        def _choice(value: Any, default: str, allowed: set[str]) -> str:
            chosen = str(value or default).strip().lower()
            return chosen if chosen in allowed else default

        phases_raw = raw.get("phases")
        phases: Dict[str, Any] = dict(phases_raw) if isinstance(phases_raw, dict) else {}
        receipt_raw = raw.get("receipt")
        receipt: Dict[str, Any] = dict(receipt_raw) if isinstance(receipt_raw, dict) else {}
        extra_body_raw = raw.get("extra_body")
        extra_body: Dict[str, Any] = dict(extra_body_raw) if isinstance(extra_body_raw, dict) else {}
        block_on_raw = raw.get("block_on")
        block_on = block_on_raw if isinstance(block_on_raw, list) else None
        mode = str(raw.get("mode", "observe") or "observe").strip().lower()
        if mode not in {"off", "observe", "final_gate", "full"}:
            mode = "observe"
        default_block_on = [
            "secret_exposure",
            "irreversible_unapproved_action",
            "critical_false_claim",
            "user_constraint_violation",
        ]
        return cls(
            enabled=_bool(raw.get("enabled"), False),
            mode=mode,
            provider=str(raw.get("provider", "auto") or "auto").strip() or "auto",
            model=str(raw.get("model", "") or "").strip(),
            base_url=str(raw.get("base_url", "") or "").strip(),
            api_key=str(raw.get("api_key", "") or "").strip(),
            max_calls_per_turn=_int(raw.get("max_calls_per_turn"), 2, 0),
            max_repair_iterations=_int(raw.get("max_repair_iterations"), 1, 0),
            max_input_chars=_int(raw.get("max_input_chars"), 64000, 4000),
            timeout_seconds=_float(raw.get("timeout_seconds", raw.get("timeout")), 90.0, 1.0),
            max_output_tokens=_int(raw.get("max_output_tokens"), 1200, 256),
            failure_policy=_choice(raw.get("failure_policy"), "fail_open", {"fail_open", "fail_closed"}),
            phases={"final": phases.get("final", "actionful_only"), **phases},
            block_on=[str(x) for x in block_on] if block_on is not None else default_block_on,
            receipt={"persist": _bool(receipt.get("persist"), True)},
            extra_body=extra_body,
        )


@dataclass
class AdvisorGateDecision:
    final_response: str
    receipt: Optional[Dict[str, Any]] = None
    response_changed: bool = False
    blocked: bool = False
    repaired: bool = False
    unavailable: bool = False
    turn_exit_reason: Optional[str] = None


def _sanitize_text(value: Any, limit: int = 4000) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        try:
            value = json.dumps(value, ensure_ascii=False, default=str)
        except Exception:
            value = str(value)
    text = value.replace("\x00", "")
    for pattern in _SECRET_VALUE_PATTERNS:
        if pattern.pattern.endswith("\\S+"):
            text = pattern.sub(lambda m: f"{m.group(1)}{m.group(2)}[redacted]", text)
        else:
            text = pattern.sub("[redacted]", text)
    if len(text) > limit:
        return text[: max(0, limit - 20)] + "…[truncated]"
    return text


def _message_text(msg: Dict[str, Any], limit: int = 1200) -> str:
    content = msg.get("content")
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                parts.append(str(part.get("text", "")))
            elif isinstance(part, dict) and part.get("type") in {"image", "image_url", "input_image"}:
                parts.append("[image]")
        content = "\n".join(parts)
    return _sanitize_text(content, limit)


def _iter_tool_calls(messages: Iterable[Dict[str, Any]]) -> Iterable[Dict[str, Any]]:
    for msg in messages:
        if not isinstance(msg, dict) or msg.get("role") != "assistant":
            continue
        for tool_call in msg.get("tool_calls") or []:
            if not isinstance(tool_call, dict):
                continue
            fn = tool_call.get("function") or {}
            yield {
                "tool_call_id": tool_call.get("id", ""),
                "name": fn.get("name") or tool_call.get("name") or "unknown",
                "arguments": _sanitize_text(fn.get("arguments", ""), 800),
            }


def _iter_tool_results(messages: Iterable[Dict[str, Any]]) -> Iterable[Dict[str, Any]]:
    for msg in messages:
        if not isinstance(msg, dict) or msg.get("role") != "tool":
            continue
        yield {
            "tool_call_id": msg.get("tool_call_id", ""),
            "name": msg.get("name") or msg.get("tool_name") or "tool",
            "content": _message_text(msg, 1200),
        }


def _tail_current_evidence(messages: List[Dict[str, Any]], max_messages: int = 60) -> List[Dict[str, str]]:
    tail = [m for m in messages[-max_messages:] if isinstance(m, dict)]
    out: List[Dict[str, str]] = []
    for msg in tail:
        role = str(msg.get("role") or "")
        if role == "assistant" and msg.get("tool_calls"):
            names = [tc.get("function", {}).get("name", "unknown") for tc in msg.get("tool_calls") or [] if isinstance(tc, dict)]
            out.append({"role": "assistant", "content": "tool_calls: " + ", ".join(names)})
        elif role in {"user", "assistant", "tool"}:
            out.append({"role": role, "content": _message_text(msg, 900)})
    return out


def _current_turn_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Return the live turn slice from the last user message onward."""
    if not messages:
        return []
    for idx in range(len(messages) - 1, -1, -1):
        msg = messages[idx]
        if isinstance(msg, dict) and msg.get("role") == "user":
            return [m for m in messages[idx:] if isinstance(m, dict)]
    return [m for m in messages if isinstance(m, dict)]


def _has_actionable_work(messages: List[Dict[str, Any]], api_call_count: int) -> bool:
    if api_call_count > 1:
        return True
    for msg in _current_turn_messages(messages):
        if not isinstance(msg, dict):
            continue
        if msg.get("role") == "tool":
            return True
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            return True
    return False


def _load_config_from_agent(agent: Any) -> AdvisorConfig:
    raw = getattr(agent, "_advisor_config", None)
    if raw is None:
        try:
            from hermes_cli.config import load_config
            cfg = load_config() or {}
            raw = cfg.get("advisor", {}) if isinstance(cfg, dict) else {}
        except Exception:
            raw = {}
    return AdvisorConfig.from_mapping(raw)


def _should_run(cfg: AdvisorConfig, messages: List[Dict[str, Any]], api_call_count: int) -> bool:
    if not cfg.enabled or cfg.mode == "off":
        return False
    final_phase = str((cfg.phases or {}).get("final", "actionful_only") or "actionful_only").lower()
    if final_phase in {"off", "disabled", "never"}:
        return False
    if final_phase in {"always", "all"}:
        return True
    return _has_actionable_work(messages, api_call_count)


def _main_runtime(agent: Any) -> Dict[str, Any]:
    return {
        "provider": getattr(agent, "provider", "") or "",
        "model": getattr(agent, "model", "") or "",
        "base_url": getattr(agent, "base_url", "") or "",
        "api_key": getattr(agent, "api_key", "") or "",
        "api_mode": getattr(agent, "api_mode", "") or "",
    }


def _call_aux_llm(
    *,
    cfg: AdvisorConfig,
    messages: List[Dict[str, str]],
    agent: Any,
    max_tokens: Optional[int] = None,
) -> str:
    from agent.auxiliary_client import call_llm

    provider = cfg.provider if cfg.provider and cfg.provider != "auto" else ""
    response = call_llm(
        task="advisor",
        provider=provider,
        model=cfg.model or "",
        base_url=cfg.base_url or "",
        api_key=cfg.api_key or "",
        main_runtime=_main_runtime(agent),
        messages=messages,
        temperature=0,
        max_tokens=max_tokens or cfg.max_output_tokens,
        timeout=cfg.timeout_seconds,
        extra_body=cfg.extra_body,
    )
    content = response.choices[0].message.content
    return content if isinstance(content, str) else str(content or "")


_ADVISOR_SYSTEM = """You are Hermes Advisor, a non-executing final audit gate.
You never call tools. You audit whether the drafted final answer is justified by
provided evidence and user constraints. Return STRICT JSON only, no markdown.
Schema:
{
  "verdict": "PASS" | "CHANGES_REQUIRED" | "BLOCK",
  "findings": [
    {
      "finding_id": "short-id",
      "severity": "low" | "medium" | "high" | "critical",
      "category": "verification" | "scope" | "safety" | "reporting" | "delegation" | "exception",
      "message": "specific issue",
      "evidence_quote": "quote from payload",
      "recommended_action": "what Commander must change",
      "acceptance_check": "how to know it is fixed"
    }
  ],
  "summary": "one sentence"
}
Use BLOCK only for secret exposure, irreversible unapproved action, critical false claim,
or direct user-constraint violation. Use CHANGES_REQUIRED for ordinary missing caveats,
unsupported completion claims, or unverified subagent/tool assertions.
"""

_REPAIR_SYSTEM = """You are the Hermes Commander repairing your own final answer.
You do not call tools. Rewrite the final answer so it satisfies the Advisor
findings using only the evidence provided. Do not claim unverified checks. If
something is unverified or failed, say so plainly. Output the final answer only.
"""


def _extract_json_object(text: str) -> Dict[str, Any]:
    text = (text or "").strip()
    if not text:
        return {"verdict": "UNAVAILABLE", "findings": [], "summary": "empty advisor response"}
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.S)
        if not match:
            return {"verdict": "UNAVAILABLE", "findings": [], "summary": text[:200]}
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            return {"verdict": "UNAVAILABLE", "findings": [], "summary": text[:200]}
    if not isinstance(data, dict):
        return {"verdict": "UNAVAILABLE", "findings": [], "summary": "advisor JSON was not an object"}
    verdict = str(data.get("verdict") or "UNAVAILABLE").strip().upper()
    if verdict not in _VERDICTS:
        verdict = "UNAVAILABLE"
    findings = data.get("findings")
    if not isinstance(findings, list):
        findings = []
    data["verdict"] = verdict
    data["findings"] = findings[:10]
    return data


def _build_payload(
    *,
    agent: Any,
    cfg: AdvisorConfig,
    messages: List[Dict[str, Any]],
    final_response: str,
    api_call_count: int,
    original_user_message: Any,
    turn_exit_reason: str,
) -> Dict[str, Any]:
    turn_messages = _current_turn_messages(messages)
    actions = list(_iter_tool_calls(turn_messages))[-30:]
    tool_results = list(_iter_tool_results(turn_messages))[-30:]
    payload = {
        "phase": "final",
        "session_id": getattr(agent, "session_id", "") or "",
        "turn_exit_reason": str(turn_exit_reason or ""),
        "api_call_count": api_call_count,
        "user_message": _sanitize_text(original_user_message, 4000),
        "actions_taken": actions,
        "tests_or_checks": [
            item for item in actions
            if item.get("name") in {"terminal", "execute_code", "process", "browser_console", "browser_snapshot"}
            or any(word in item.get("arguments", "").lower() for word in ("pytest", "test", "health", "status", "verify"))
        ][-20:],
        "tool_result_evidence": tool_results,
        "known_unresolved": [],
        "final_answer_draft": _sanitize_text(final_response, 12000),
        "flow_summary": (
            f"api_calls={api_call_count}; tool_calls={len(actions)}; "
            f"tool_results={len(tool_results)}; model={getattr(agent, 'model', '')}"
        ),
        "evidence_tail": _tail_current_evidence(turn_messages),
        "block_on": list(cfg.block_on),
    }
    encoded = json.dumps(payload, ensure_ascii=False, default=str)
    if len(encoded) <= cfg.max_input_chars:
        return payload
    # Preserve the decisive fields, compress evidence first.
    payload["evidence_tail"] = payload["evidence_tail"][-20:]
    payload["tool_result_evidence"] = payload["tool_result_evidence"][-12:]
    payload["actions_taken"] = payload["actions_taken"][-20:]
    payload["final_answer_draft"] = _sanitize_text(final_response, 8000)
    encoded = json.dumps(payload, ensure_ascii=False, default=str)
    if len(encoded) > cfg.max_input_chars:
        payload["tool_result_evidence"] = [
            {**item, "content": _sanitize_text(item.get("content", ""), 500)}
            for item in payload["tool_result_evidence"][-8:]
        ]
        payload["evidence_tail"] = [
            {**item, "content": _sanitize_text(item.get("content", ""), 400)}
            for item in payload["evidence_tail"][-12:]
        ]
        payload["final_answer_draft"] = _sanitize_text(final_response, 5000)
    return payload


def _block_response(audit: Dict[str, Any]) -> str:
    findings = audit.get("findings") or []
    if findings and isinstance(findings[0], dict):
        reason = findings[0].get("message") or findings[0].get("recommended_action") or audit.get("summary")
    else:
        reason = audit.get("summary") or "Advisor blocked the draft final answer."
    return (
        "I can’t safely deliver the drafted answer as-is.\n\n"
        f"Advisor block reason: {_sanitize_text(reason, 1200)}\n\n"
        "I have not taken additional action after this block. Please review the issue or ask me to continue with a corrected scope."
    )


def _repair_response(
    *,
    cfg: AdvisorConfig,
    agent: Any,
    payload: Dict[str, Any],
    audit: Dict[str, Any],
) -> Optional[str]:
    repair_payload = {
        "user_message": payload.get("user_message", ""),
        "final_answer_draft": payload.get("final_answer_draft", ""),
        "findings": audit.get("findings", []),
        "evidence_tail": payload.get("evidence_tail", []),
        "tool_result_evidence": payload.get("tool_result_evidence", []),
    }
    text = _call_aux_llm(
        cfg=cfg,
        agent=agent,
        max_tokens=max(1200, cfg.max_output_tokens),
        messages=[
            {"role": "system", "content": _REPAIR_SYSTEM},
            {"role": "user", "content": json.dumps(repair_payload, ensure_ascii=False, default=str)},
        ],
    )
    text = text.strip()
    return text or None


def _update_last_assistant_message(messages: List[Dict[str, Any]], final_response: str) -> None:
    for msg in reversed(messages):
        if not isinstance(msg, dict) or msg.get("role") != "assistant":
            continue
        if msg.get("tool_calls"):
            continue
        content = msg.get("content")
        if isinstance(content, str) or content is None:
            msg["content"] = final_response
            return
        if isinstance(content, list):
            for part in reversed(content):
                if isinstance(part, dict) and part.get("type") == "text":
                    part["text"] = final_response
                    return
            content.append({"type": "text", "text": final_response})
            return
        msg["content"] = final_response
        return
    messages.append({"role": "assistant", "content": final_response})


def _persist_receipt(agent: Any, receipt: Dict[str, Any]) -> None:
    try:
        from hermes_constants import get_hermes_home
        path = get_hermes_home() / "logs" / "advisor" / f"{getattr(agent, 'session_id', '') or 'unknown'}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(receipt, ensure_ascii=False, default=str) + "\n")
    except Exception:
        logger.debug("advisor receipt persistence failed", exc_info=True)


def run_final_advisor_gate(
    agent: Any,
    *,
    messages: List[Dict[str, Any]],
    final_response: str,
    api_call_count: int,
    original_user_message: Any,
    turn_exit_reason: str,
) -> AdvisorGateDecision:
    cfg = _load_config_from_agent(agent)
    if not final_response or not _should_run(cfg, messages, api_call_count):
        return AdvisorGateDecision(final_response=final_response)

    payload = _build_payload(
        agent=agent,
        cfg=cfg,
        messages=messages,
        final_response=final_response,
        api_call_count=api_call_count,
        original_user_message=original_user_message,
        turn_exit_reason=turn_exit_reason,
    )
    receipt: Dict[str, Any] = {
        "type": "advisor_final_audit",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "session_id": getattr(agent, "session_id", "") or "",
        "mode": cfg.mode,
        "phase": "final",
        "enforced": cfg.mode in {"final_gate", "full"},
        "verdict": "UNAVAILABLE",
        "findings": [],
        "response_changed": False,
    }

    try:
        advisor_text = _call_aux_llm(
            cfg=cfg,
            agent=agent,
            messages=[
                {"role": "system", "content": _ADVISOR_SYSTEM},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False, default=str)},
            ],
        )
        audit = _extract_json_object(advisor_text)
    except Exception as exc:
        audit = {
            "verdict": "UNAVAILABLE",
            "findings": [],
            "summary": f"advisor unavailable: {type(exc).__name__}: {exc}",
        }
        logger.warning("Advisor final audit unavailable: %s", exc)

    receipt.update({
        "verdict": audit.get("verdict", "UNAVAILABLE"),
        "findings": audit.get("findings", []),
        "summary": audit.get("summary", ""),
    })

    verdict = receipt["verdict"]
    enforced = receipt["enforced"]
    if enforced and verdict == "UNAVAILABLE" and cfg.failure_policy == "fail_closed":
        new_response = _block_response({
            "summary": audit.get("summary") or "Advisor was unavailable and failure_policy=fail_closed.",
            "findings": audit.get("findings", []),
        })
        receipt.update({
            "response_changed": new_response != final_response,
            "blocked": True,
            "repaired": False,
            "turn_exit_reason": "advisor_unavailable_fail_closed",
        })
        _update_last_assistant_message(messages, new_response)
        if cfg.receipt.get("persist", True):
            _persist_receipt(agent, receipt)
        return AdvisorGateDecision(
            final_response=new_response,
            receipt=receipt,
            response_changed=True,
            blocked=True,
            unavailable=True,
            turn_exit_reason="advisor_unavailable_fail_closed",
        )
    if not enforced or verdict in {"PASS", "UNAVAILABLE"}:
        if cfg.receipt.get("persist", True):
            _persist_receipt(agent, receipt)
        return AdvisorGateDecision(
            final_response=final_response,
            receipt=receipt,
            unavailable=(verdict == "UNAVAILABLE"),
        )

    new_response = final_response
    blocked = False
    repaired = False
    turn_reason = None

    if verdict == "BLOCK":
        new_response = _block_response(audit)
        blocked = True
        turn_reason = "advisor_blocked"
    elif verdict == "CHANGES_REQUIRED":
        repairs = max(0, min(cfg.max_repair_iterations, cfg.max_calls_per_turn - 1 if cfg.max_calls_per_turn else 0))
        if repairs <= 0:
            new_response = _block_response({
                "summary": "Advisor requested changes but repair iterations are disabled.",
                "findings": audit.get("findings", []),
            })
            blocked = True
            turn_reason = "advisor_changes_required_unrepaired"
        else:
            try:
                repaired_response = _repair_response(cfg=cfg, agent=agent, payload=payload, audit=audit)
                if repaired_response:
                    new_response = repaired_response
                    repaired = True
                    turn_reason = "advisor_repaired"
                else:
                    new_response = _block_response({
                        "summary": "Advisor requested changes but the repair response was empty.",
                        "findings": audit.get("findings", []),
                    })
                    blocked = True
                    turn_reason = "advisor_repair_empty"
            except Exception as exc:
                logger.warning("Advisor repair failed: %s", exc)
                new_response = _block_response({
                    "summary": f"Advisor requested changes but repair failed: {type(exc).__name__}: {exc}",
                    "findings": audit.get("findings", []),
                })
                blocked = True
                turn_reason = "advisor_repair_failed"

    response_changed = new_response != final_response
    receipt.update({
        "response_changed": response_changed,
        "blocked": blocked,
        "repaired": repaired,
        "turn_exit_reason": turn_reason,
    })
    if response_changed:
        _update_last_assistant_message(messages, new_response)
    if cfg.receipt.get("persist", True):
        _persist_receipt(agent, receipt)
    return AdvisorGateDecision(
        final_response=new_response,
        receipt=receipt,
        response_changed=response_changed,
        blocked=blocked,
        repaired=repaired,
        turn_exit_reason=turn_reason,
    )


__all__ = [
    "AdvisorConfig",
    "AdvisorGateDecision",
    "run_final_advisor_gate",
]
