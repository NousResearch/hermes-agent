"""Bounded adaptive routing for authenticated WhatsApp text turns.

This module is deliberately a small boundary around the existing native
Gemini transport.  It does not construct an :class:`AIAgent`, inspect the
Hermes tool registry, or load session history.  A fast decision therefore
cannot accidentally pay the prompt cost of the agentic lane merely to decide
which lane should handle a message.

The feature is opt-in in ``gateway.whatsapp_adaptive_routing.enabled``.  The
stable Flash-Lite model is selected from the official Gemini model catalog and
validated against the provider's real ``ListModels`` response before use.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import re
import threading
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

import httpx

from agent.gemini_native_adapter import (
    DEFAULT_GEMINI_BASE_URL,
    GeminiAPIError,
    GeminiNativeClient,
    bare_gemini_model_id,
)

logger = logging.getLogger(__name__)

FAST_PROVIDER = "gemini"
FAST_MODEL = "gemini-3.1-flash-lite"

ROUTER_REASON_VALUES = frozenset(
    {
        "simple",
        "tool_required",
        "multi_step",
        "consequential",
        "ambiguous",
        "unknown",
        "fast_provider_quota_exhausted",
        "fast_provider_unavailable",
    }
)
# This is intentionally a closed set.  A new or ambiguous classification must
# earn its way into the contract before it can authorize a direct answer.
SAFE_DIRECT_REASONS = frozenset({"simple"})
ROUTER_FIELDS = frozenset({"route", "response", "reason", "confidence"})

# DIRECT is an allowlist, not a model-mediated denylist. The router may
# recommend DIRECT for any text, but only a small set of demonstrably bounded
# conversational shapes can receive that authorization. Everything else
# remains AGENTIC, including unknown intent.
_SAFE_DIRECT_INPUT_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"(?:hi|hello|hey|olá|ola|oi)(?:\s+(?:there|hermes|aí|ai))?[!.?]*",
        r"(?:good\s+morning|good\s+afternoon|good\s+evening|bom\s+dia|boa\s+tarde|boa\s+noite)[!.?]*",
        r"(?:how\s+are\s+you(?:\s+doing)?|como\s+você\s+está|como\s+voce\s+esta|tudo\s+bem)(?:\s+(?:today|hoje))?[!.?]*",
        r"(?:thanks|thank\s+you|obrigado|obrigada)(?:\s+(?:so\s+much|muito))?[!.?]*",
        r"(?:nice\s+to\s+meet\s+you|prazer\s+em\s+conhecer\s+você|prazer\s+em\s+conhecer\s+voce|can\s+we\s+chat|podemos\s+conversar|let['’]s\s+chat)[!.?]*",
        r"(?:what\s+can\s+you\s+do|o\s+que\s+você\s+pode\s+fazer|o\s+que\s+voce\s+pode\s+fazer)[!.?]*",
        r"(?:tell\s+me\s+a\s+joke|conte\s+uma\s+piada)[!.?]*",
        r"(?:i['’]?m\s+(?:fine|good|happy|bored|curious)|eu\s+estou\s+(?:bem|feliz|entediado|entediada|curioso|curiosa))[!.?]*",
    )
)

_DIRECT_INPUT_RISK_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\b(?:ignore|disregard|bypass|override|jailbreak|prompt\s+injection|system\s+prompt|developer\s+message|router|direct)\b",
        r"\b(?:tool|function|shell|terminal|command|script|file|filesystem|runtime|server|database|api|browser|inspect|debug|log|logs)\b",
        r"\b(?:delete|remove|create|change|update|send|execute|run|install|deploy|restart|approve|deny|buy|pay|book|schedule|call|email|upload|download)\b",
        r"\b(?:latest|current|weather|news|price|stock|search|browse|internet|live|real[- ]time)\b",
        r"\b(?:step\s+by\s+step|first|then|several|multiple|workflow|automate|diagnos|medical|legal|financial|credential|password|secret)\b",
    )
)

# The schema is intentionally small.  ``response`` is the direct answer from
# the same call that selected DIRECT; it is never used as an instruction or as
# a replacement for the original user message on the AGENTIC path.
FAST_ROUTE_RESPONSE_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "route": {"type": "string", "enum": ["DIRECT", "AGENTIC"]},
        "response": {"type": "string", "nullable": True},
        "reason": {
            "type": "string",
            "enum": [
                "simple",
                "tool_required",
                "multi_step",
                "consequential",
                "ambiguous",
                "unknown",
                "fast_provider_quota_exhausted",
                "fast_provider_unavailable",
            ],
        },
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
    },
    "required": ["route", "response", "reason"],
    "additionalProperties": False,
}

FAST_ROUTER_SYSTEM_INSTRUCTION = (
    "You are the bounded fast router for authenticated WhatsApp text. "
    "Return only JSON matching the supplied schema. Choose DIRECT only when "
    "the current message can be answered safely and completely without any "
    "tool, file, runtime inspection, external action, multi-step diagnosis, "
    "or consequential effect; put that answer in response. Choose AGENTIC "
    "for tool requests, multi-step work, runtime-dependent questions, "
    "consequential or ambiguous requests, and whenever uncertain; set "
    "response to null. The route is only a bounded dispatch hint and never "
    "changes user authority. Do not invent tool calls."
)


class AdaptiveRoute(str, Enum):
    DIRECT = "DIRECT"
    AGENTIC = "AGENTIC"


@dataclass(frozen=True)
class AdaptiveDecision:
    """The only decision contract exposed by the fast lane."""

    route: AdaptiveRoute
    response: Optional[str] = None
    reason: str = "unknown"
    confidence: Optional[float] = None
    quota_exhausted: bool = False


@dataclass(frozen=True)
class WhatsAppAdaptiveConfig:
    """Validated, deliberately narrow configuration for the feature."""

    enabled: bool = False
    timeout_seconds: float = 8.0
    discovery_timeout_seconds: float = 5.0
    max_output_tokens: int = 256

    @classmethod
    def from_gateway_config(cls, config: Any) -> "WhatsAppAdaptiveConfig":
        root = config if isinstance(config, dict) else {}
        gateway = root.get("gateway") or {}
        raw = gateway.get("whatsapp_adaptive_routing") or {}
        if not isinstance(raw, dict):
            return cls()

        def _bounded_float(name: str, default: float, minimum: float, maximum: float) -> float:
            try:
                value = float(raw.get(name, default))
            except (TypeError, ValueError):
                return default
            return min(max(value, minimum), maximum)

        def _bounded_int(name: str, default: int, minimum: int, maximum: int) -> int:
            try:
                value = int(raw.get(name, default))
            except (TypeError, ValueError):
                return default
            return min(max(value, minimum), maximum)

        # The fast lane owns only its bounded transport settings.  AGENTIC is
        # deliberately resolved by the normal gateway agent pipeline.
        return cls(
            enabled=raw.get("enabled") is True,
            timeout_seconds=_bounded_float("timeout_seconds", 8.0, 1.0, 30.0),
            discovery_timeout_seconds=_bounded_float(
                "discovery_timeout_seconds", 5.0, 1.0, 15.0
            ),
            max_output_tokens=_bounded_int("max_output_tokens", 256, 64, 1024),
        )


@dataclass(frozen=True)
class FlashLiteDiscovery:
    model: str
    generate_content_supported: bool
    structured_output_supported: bool


class FlashLiteUnavailable(RuntimeError):
    """The configured Gemini account has no compatible Flash-Lite model."""


_DISCOVERY_CACHE: Dict[tuple[str, str], FlashLiteDiscovery] = {}
_DISCOVERY_LOCK = threading.Lock()


def _normalized_gemini_base_url(base_url: Optional[str]) -> str:
    normalized = str(base_url or DEFAULT_GEMINI_BASE_URL).strip().rstrip("/")
    if normalized.lower().endswith("/openai"):
        normalized = normalized[: -len("/openai")]
    return normalized or DEFAULT_GEMINI_BASE_URL


def _model_name(item: Dict[str, Any]) -> str:
    value = item.get("baseModelId") or item.get("name") or ""
    return bare_gemini_model_id(str(value).removeprefix("models/"))


def discover_flash_lite_model(
    api_key: str,
    base_url: Optional[str] = None,
    *,
    timeout: float = 5.0,
    http_client_factory: Callable[..., Any] = httpx.Client,
) -> FlashLiteDiscovery:
    """Discover a usable stable Flash-Lite via Gemini's real ListModels API.

    The API key is used only in the request header and is never included in
    the cache key, logs, exception text, or returned data.  Structured output
    capability is asserted only for the stable model whose official model
    documentation declares it; a preview or unknown alias is not accepted.
    """
    key = (api_key or "").strip()
    if not key:
        raise FlashLiteUnavailable("Gemini credentials are not configured")
    normalized_base = _normalized_gemini_base_url(base_url)
    cache_key = (normalized_base, hashlib.sha256(key.encode()).hexdigest())
    with _DISCOVERY_LOCK:
        cached = _DISCOVERY_CACHE.get(cache_key)
    if cached is not None:
        return cached

    try:
        with http_client_factory(timeout=timeout) as client:
            response = client.get(
                f"{normalized_base}/models",
                headers={
                    "Accept": "application/json",
                    "x-goog-api-key": key,
                },
            )
    except Exception as exc:
        raise FlashLiteUnavailable("Gemini model discovery failed") from exc
    if response.status_code != 200:
        raise FlashLiteUnavailable(
            f"Gemini model discovery returned HTTP {response.status_code}"
        )
    try:
        payload = response.json()
    except (TypeError, ValueError) as exc:
        raise FlashLiteUnavailable("Gemini model discovery returned invalid JSON") from exc

    candidates: List[str] = []
    for item in payload.get("models", []) if isinstance(payload, dict) else []:
        if not isinstance(item, dict):
            continue
        model = _model_name(item)
        methods = item.get("supportedGenerationMethods") or []
        if (
            model == FAST_MODEL
            and "generateContent" in methods
            and "preview" not in model.lower()
        ):
            candidates.append(model)
    if not candidates:
        raise FlashLiteUnavailable(
            "No stable Gemini Flash-Lite model supporting generateContent and "
            "structured output was returned by ListModels"
        )

    result = FlashLiteDiscovery(
        model=FAST_MODEL,
        generate_content_supported=True,
        structured_output_supported=True,
    )
    with _DISCOVERY_LOCK:
        _DISCOVERY_CACHE[cache_key] = result
    return result


def build_fast_router_messages(message: str) -> List[Dict[str, str]]:
    """Build the complete fast request; no session history is accepted."""
    return [
        {"role": "system", "content": FAST_ROUTER_SYSTEM_INSTRUCTION},
        {"role": "user", "content": str(message or "")},
    ]


def is_gemini_quota_exhausted(error: BaseException) -> bool:
    """Classify only Gemini capacity/quota exhaustion for fast fallback."""
    status = getattr(error, "status_code", None)
    code = str(getattr(error, "code", "") or "").lower()
    details = getattr(error, "details", None)
    text = " ".join(
        str(value)
        for value in (error, code, details)
        if value is not None
    ).lower()
    return bool(
        status == 429
        and (
            "resource_exhausted" in text
            or "resource exhausted" in text
            or "quota" in text
            or "free_tier" in text
            or "rate_limit" in code
        )
    )


def _response_content(response: Any) -> str:
    choices = getattr(response, "choices", None) or []
    if not choices:
        return ""
    message = getattr(choices[0], "message", None)
    content = getattr(message, "content", "") if message is not None else ""
    return content if isinstance(content, str) else str(content or "")


def is_deterministically_eligible_for_direct(message: str) -> bool:
    """Authorize DIRECT only for a bounded, low-risk original input.

    This is intentionally conservative. The router's structured answer is a
    recommendation; it cannot expand this input-side policy. A future
    conversational shape must be added here with a regression test before it
    can enter the fast lane.
    """
    normalized = " ".join(str(message or "").strip().split())
    if not normalized or len(normalized) > 240:
        return False
    if any(pattern.fullmatch(normalized) for pattern in _SAFE_DIRECT_INPUT_PATTERNS):
        return True
    if any(pattern.search(normalized) for pattern in _DIRECT_INPUT_RISK_PATTERNS):
        return False
    return False


def _parse_decision(response: Any) -> AdaptiveDecision:
    raw = _response_content(response).strip()
    if raw.startswith("```"):
        raw = raw.strip("`").strip()
        if raw.lower().startswith("json"):
            raw = raw[4:].strip()
    try:
        payload = json.loads(raw)
    except (TypeError, ValueError):
        return AdaptiveDecision(AdaptiveRoute.AGENTIC, reason="unknown")
    if not isinstance(payload, dict):
        return AdaptiveDecision(AdaptiveRoute.AGENTIC, reason="unknown")
    if set(payload) - ROUTER_FIELDS:
        return AdaptiveDecision(AdaptiveRoute.AGENTIC, reason="unknown")
    try:
        route = AdaptiveRoute(str(payload.get("route", "")).upper())
    except (TypeError, ValueError):
        return AdaptiveDecision(AdaptiveRoute.AGENTIC, reason="unknown")
    reason_value = payload.get("reason")
    reason = reason_value if isinstance(reason_value, str) else "unknown"
    if reason not in ROUTER_REASON_VALUES:
        reason = "unknown"
    confidence = payload.get("confidence")
    if not isinstance(confidence, (int, float)) or isinstance(confidence, bool):
        if "confidence" in payload:
            return AdaptiveDecision(AdaptiveRoute.AGENTIC, reason="unknown")
        confidence = None
    else:
        confidence = float(confidence)
        if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            return AdaptiveDecision(AdaptiveRoute.AGENTIC, reason="unknown")
    if route is AdaptiveRoute.DIRECT:
        response_text = payload.get("response")
        if reason not in SAFE_DIRECT_REASONS:
            return AdaptiveDecision(AdaptiveRoute.AGENTIC, reason=reason)
        if not isinstance(response_text, str) or not response_text.strip():
            return AdaptiveDecision(AdaptiveRoute.AGENTIC, reason="unknown")
        return AdaptiveDecision(
            route=route,
            response=response_text.strip(),
            reason=reason,
            confidence=confidence,
        )
    return AdaptiveDecision(
        route=AdaptiveRoute.AGENTIC,
        reason=reason,
        confidence=confidence,
    )


class WhatsAppFastRouter:
    """One-call, tool-free Gemini route plus direct response boundary."""

    def __init__(
        self,
        *,
        api_key: str,
        base_url: Optional[str] = None,
        config: WhatsAppAdaptiveConfig = WhatsAppAdaptiveConfig(),
        discover: Callable[..., FlashLiteDiscovery] = discover_flash_lite_model,
        client_factory: Callable[..., GeminiNativeClient] = GeminiNativeClient,
        completion_call: Optional[Callable[..., Any]] = None,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url
        self.config = config
        self.discover = discover
        self.client_factory = client_factory
        self.completion_call = completion_call

    def route(self, message: str) -> AdaptiveDecision:
        try:
            discovered = self.discover(
                self.api_key,
                self.base_url,
                timeout=self.config.discovery_timeout_seconds,
            )
        except FlashLiteUnavailable:
            logger.warning("WhatsApp fast lane has no compatible Gemini Flash-Lite")
            return AdaptiveDecision(
                AdaptiveRoute.AGENTIC,
                reason="fast_provider_unavailable",
            )

        if not (
            discovered.generate_content_supported
            and discovered.structured_output_supported
        ):
            return AdaptiveDecision(
                AdaptiveRoute.AGENTIC,
                reason="fast_provider_unavailable",
            )

        messages = build_fast_router_messages(message)
        request = {
            "model": discovered.model,
            "messages": messages,
            "tools": None,
            "tool_choice": "none",
            "temperature": 0.0,
            "max_tokens": self.config.max_output_tokens,
            "extra_body": {
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "whatsapp_fast_route",
                        "strict": True,
                        "schema": FAST_ROUTE_RESPONSE_SCHEMA,
                    },
                }
            },
        }
        client = None
        try:
            if self.completion_call is not None:
                response = self.completion_call(**request)
            else:
                client = self.client_factory(
                    api_key=self.api_key,
                    base_url=self.base_url,
                    timeout=self.config.timeout_seconds,
                )
                response = client.chat.completions.create(**request)
            decision = _parse_decision(response)
            if decision.route is AdaptiveRoute.DIRECT and not is_deterministically_eligible_for_direct(
                message
            ):
                return AdaptiveDecision(AdaptiveRoute.AGENTIC, reason="unknown")
            return decision
        except GeminiAPIError as exc:
            if is_gemini_quota_exhausted(exc):
                return AdaptiveDecision(
                    AdaptiveRoute.AGENTIC,
                    reason="fast_provider_quota_exhausted",
                    quota_exhausted=True,
                )
            logger.warning("WhatsApp fast Gemini request failed; using AGENTIC")
            return AdaptiveDecision(
                AdaptiveRoute.AGENTIC,
                reason="fast_provider_unavailable",
            )
        except Exception:
            logger.warning("WhatsApp fast router failed; using AGENTIC", exc_info=True)
            return AdaptiveDecision(
                AdaptiveRoute.AGENTIC,
                reason="fast_provider_unavailable",
            )
        finally:
            if client is not None:
                try:
                    client.close()
                except Exception:
                    pass


__all__ = [
    "AdaptiveDecision",
    "AdaptiveRoute",
    "FAST_MODEL",
    "FAST_PROVIDER",
    "FlashLiteDiscovery",
    "FlashLiteUnavailable",
    "WhatsAppAdaptiveConfig",
    "WhatsAppFastRouter",
    "build_fast_router_messages",
    "discover_flash_lite_model",
    "is_deterministically_eligible_for_direct",
    "is_gemini_quota_exhausted",
]
