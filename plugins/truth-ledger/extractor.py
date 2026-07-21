from __future__ import annotations

import asyncio
import json
import random
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Mapping

from agent.plugin_llm import PluginLlmTextInput

from .redaction import redact_text, sanitize_payload
from .schemas import load_schema, validate_document

_FACT_SCHEMA_NAME = "truth-ledger.fact-candidates.v1"
_FACT_JSON_SCHEMA = load_schema("fact-candidates.v1")
_REASON_SCHEMA_MISMATCH = "schema_mismatch"
_REASON_EXTRACTION_FAILED = "extraction_failed"
_KEY_ALIASES = {
    "response.default_conciseness": "response.style",
    "proposals.presentation_order": "proposal.presentation_order",
    "proposals.delivery_format": "proposal.delivery_format",
    "rollout.review_requirement": "rollout.independent_exact_commit_review_required",
    "merge.approval_requirement": "rollout.merge_requires_explicit_approval",
    "default_profile_enablement.approval_requirement": (
        "rollout.default_profile_change_requires_explicit_approval"
    ),
}
_KIND_BY_KEY = {
    "response.style": "preference",
    "timezone": "preference",
}
_VALUE_ALIASES_BY_KEY = {
    "proposal.presentation_order": {
        "options_first": "options first",
    },
    "proposal.delivery_format": {
        "google_docs_with_links": "Google Docs with links",
    },
}
_BOOLEAN_REQUIREMENT_KEYS = {
    "rollout.independent_exact_commit_review_required",
    "rollout.merge_requires_explicit_approval",
    "rollout.default_profile_change_requires_explicit_approval",
}
_RESPONSE_STYLE_VALUE_ALIASES = {
    "concise by default": "concise",
    "detailed by default": "detailed",
}


def _normalize_fact_value(key: str, value: Any) -> Any:
    if isinstance(value, str):
        normalized_text = value.strip().lower()
        aliases = _VALUE_ALIASES_BY_KEY.get(key, {})
        if normalized_text in aliases:
            return aliases[normalized_text]
        if key in _BOOLEAN_REQUIREMENT_KEYS and (
            ("required" in normalized_text and "not required" not in normalized_text)
            or "without explicit approval" in normalized_text
        ):
            return True
    if key != "response.style":
        return value
    if isinstance(value, str):
        return _RESPONSE_STYLE_VALUE_ALIASES.get(value.strip().lower(), value)
    if isinstance(value, Mapping):
        verbosity = str(value.get("verbosity") or "").strip().lower()
        context = str(value.get("context") or "").strip().lower()
        if verbosity == "detailed" and context in {"engineering", "engineering topics"}:
            return "detailed"
    return value


@dataclass(frozen=True)
class ExtractorSettings:
    timeout_seconds: float = 30.0
    max_attempts: int = 6
    base_delay_ms: int = 500
    max_delay_ms: int = 60_000
    prompt_version: int = 4
    override_mode: str = "off"  # off | explicit
    provider_override: str | None = None
    model_override: str | None = None


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _redact_error_text(value: Any) -> str:
    redacted, _ = redact_text(str(value))
    cleaned = redacted.replace("conversation_history", "[redacted-field]")
    return cleaned[:1024] if cleaned else "extraction failure"


def _source_ref_from_envelope(envelope: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "profile": str(envelope.get("profile") or "unknown"),
        "session_id": str(envelope.get("session_id") or "unknown"),
        "turn_id": str(envelope.get("turn_id") or ""),
    }


def _dead_letter(
    *,
    envelope: Mapping[str, Any],
    reason_code: str,
    last_error: Any,
) -> dict[str, Any]:
    payload = {
        "schema_name": "truth-ledger.dead-letter.v1",
        "schema_version": 1,
        "occurred_at": _now_iso(),
        "reason_code": reason_code,
        "source_ref": _source_ref_from_envelope(envelope),
        "attempt_count": _safe_int(envelope.get("attempt_count"), 0),
        "last_error": _redact_error_text(last_error),
    }
    validate_document("dead-letter.v1", payload)
    return payload


def _transient_http(exc: BaseException) -> bool:
    status_code = getattr(exc, "status_code", None)
    try:
        return status_code is not None and int(status_code) >= 500
    except (TypeError, ValueError):
        return False


def _retry_delay_ms(*, attempt_count: int, settings: ExtractorSettings, rng: random.Random) -> int:
    attempt = max(attempt_count, 0)
    delay = min(settings.max_delay_ms, settings.base_delay_ms * (2 ** attempt))
    return int(rng.uniform(0, max(delay, 0)))


def _extraction_kwargs(settings: ExtractorSettings) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    if settings.override_mode == "explicit" and settings.provider_override and settings.model_override:
        kwargs["provider"] = settings.provider_override
        kwargs["model"] = settings.model_override
    return kwargs


def _build_input_blocks(envelope: Mapping[str, Any]) -> list[PluginLlmTextInput]:
    raw_origin = envelope.get("origin")
    origin: Mapping[str, Any] = raw_origin if isinstance(raw_origin, Mapping) else {}
    user_message = ""
    output = envelope.get("output")
    if isinstance(envelope.get("input"), Mapping):
        user_message = str(envelope["input"].get("user_message") or "")
    assistant_response = ""
    if isinstance(output, Mapping):
        assistant_response = str(output.get("assistant_response") or "")

    minimal_payload = {
        "profile": envelope.get("profile"),
        "session_id": envelope.get("session_id"),
        "turn_id": envelope.get("turn_id"),
        "origin": {
            "platform": origin.get("platform"),
            "conversation_id": origin.get("conversation_id"),
            "thread_id": origin.get("thread_id"),
            "speaker_id": origin.get("speaker_id"),
        },
        "input": {"user_message": user_message},
        "output": {"assistant_response": assistant_response},
    }
    sanitized = sanitize_payload(minimal_payload)
    return [PluginLlmTextInput(text=json.dumps(sanitized, ensure_ascii=False, separators=(",", ":")))]


def _normalize_facts(
    document: Mapping[str, Any],
    *,
    envelope: Mapping[str, Any],
) -> list[dict[str, Any]]:
    facts = document.get("facts")
    if not isinstance(facts, list):
        raise ValueError("schema validation failed: facts must be a list")

    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    raw_origin = envelope.get("origin")
    origin: Mapping[str, Any] = raw_origin if isinstance(raw_origin, Mapping) else {}
    platform = str(origin.get("platform") or "").strip()
    speaker_id = str(origin.get("speaker_id") or "").strip()
    for item in facts:
        fact = item["fact"]
        evidence = item["evidence"]
        scope = str(fact["scope"])
        subject = str(fact["subject"])
        if scope == "user" and platform and speaker_id:
            subject = f"platform-user:{platform}:{speaker_id}"
        raw_key = str(fact["key"]).strip()
        key = _KEY_ALIASES.get(raw_key, raw_key)
        candidate = {
            "scope": scope,
            "kind": _KIND_BY_KEY.get(key, fact["kind"]),
            "subject": subject,
            "key": key,
            "value": _normalize_fact_value(key, fact.get("value")),
            "operation": item["operation"],
            "evidence_type": evidence["type"],
            "confidence": item.get("confidence"),
        }
        fingerprint = json.dumps(candidate, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        out.append(candidate)
    return out


async def extract_candidates(
    *,
    ctx: Any,
    envelope: Mapping[str, Any],
    settings: ExtractorSettings | None = None,
    rng: random.Random | None = None,
) -> dict[str, Any]:
    cfg = settings or ExtractorSettings()
    rand = rng or random.Random()
    attempt_count = _safe_int(envelope.get("attempt_count"), 0)

    instructions = (
        "Extract only durable atomic facts for truth-ledger.fact-candidates.v1. "
        "Return JSON matching schema_name truth-ledger.fact-candidates.v1 with facts array. "
        "Use canonical key response.style only for an unqualified global response preference; "
        "a response.style value must be exactly concise or detailed. "
        "For a context-specific response preference, preserve the context in a dotted key; "
        "for example, use response.style.engineering_review for engineering reviews and "
        "response.style.slack_progress for Slack progress updates, with a concise or detailed value. "
        "Emit separate atomic facts when one message states preferences for multiple contexts. "
        "For proposal workflows, use proposal.presentation_order with value options first and "
        "proposal.delivery_format with value Google Docs with links. "
        "For rollout constraints, use rollout.independent_exact_commit_review_required, "
        "rollout.merge_requires_explicit_approval, and "
        "rollout.default_profile_change_requires_explicit_approval with boolean values. "
        "Use canonical key timezone for timezone preferences. "
        "User-scoped subjects are derived from trusted origin metadata after extraction. "
        "Conversational gratitude, acknowledgements, pleasantries, and transient remarks are not durable facts. "
        "If no admissible durable fact exists, return an empty facts array. "
        "Never include secrets, raw conversation history, or tool output dumps."
    )

    try:
        result = await asyncio.wait_for(
            asyncio.to_thread(
                ctx.llm.complete_structured,
                instructions=instructions,
                input=_build_input_blocks(envelope),
                json_schema=_FACT_JSON_SCHEMA,
                json_mode=False,
                schema_name=_FACT_SCHEMA_NAME,
                timeout=cfg.timeout_seconds,
                purpose="truth-ledger.extract",
                **_extraction_kwargs(cfg),
            ),
            timeout=cfg.timeout_seconds,
        )
    except asyncio.TimeoutError:
        if attempt_count < cfg.max_attempts:
            delay_ms = _retry_delay_ms(attempt_count=attempt_count, settings=cfg, rng=rand)
            return {
                "status": "retry",
                "reason": "timeout",
                "attempt_count": attempt_count + 1,
                "retry_delay_ms": delay_ms,
            }
        return {
            "status": "dead_letter",
            "reason": "permanent_failure",
            "dead_letter": _dead_letter(
                envelope=envelope,
                reason_code=_REASON_EXTRACTION_FAILED,
                last_error="extraction timeout retry budget exhausted",
            ),
        }
    except Exception as exc:
        transient = _transient_http(exc)
        if transient and attempt_count < cfg.max_attempts:
            delay_ms = _retry_delay_ms(attempt_count=attempt_count, settings=cfg, rng=rand)
            return {
                "status": "retry",
                "reason": "upstream_5xx",
                "attempt_count": attempt_count + 1,
                "retry_delay_ms": delay_ms,
            }
        return {
            "status": "dead_letter",
            "reason": "permanent_failure",
            "dead_letter": _dead_letter(
                envelope=envelope,
                reason_code=_REASON_EXTRACTION_FAILED,
                last_error=exc,
            ),
        }

    parsed = getattr(result, "parsed", None)
    if not isinstance(parsed, Mapping):
        return {
            "status": "dead_letter",
            "reason": "schema_mismatch",
            "dead_letter": _dead_letter(
                envelope=envelope,
                reason_code=_REASON_SCHEMA_MISMATCH,
                last_error="schema validation failed: parsed payload is not an object",
            ),
        }

    try:
        validate_document("fact-candidates.v1", parsed)
        normalized = _normalize_facts(parsed, envelope=envelope)
    except Exception as exc:
        return {
            "status": "dead_letter",
            "reason": "schema_mismatch",
            "dead_letter": _dead_letter(
                envelope=envelope,
                reason_code=_REASON_SCHEMA_MISMATCH,
                last_error=f"schema validation failed: {exc}",
            ),
        }

    if not normalized:
        return {
            "status": "none",
            "facts": [],
            "reason": "none",
            "extraction": {
                "schema_name": _FACT_SCHEMA_NAME,
                "provider": getattr(result, "provider", "unknown"),
                "model": getattr(result, "model", "unknown"),
                "prompt_version": cfg.prompt_version,
            },
        }

    return {
        "status": "ok",
        "facts": normalized,
        "reason": None,
        "extraction": {
            "schema_name": _FACT_SCHEMA_NAME,
            "provider": getattr(result, "provider", "unknown"),
            "model": getattr(result, "model", "unknown"),
            "prompt_version": cfg.prompt_version,
        },
    }
