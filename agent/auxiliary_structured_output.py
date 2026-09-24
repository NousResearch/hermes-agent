"""Structured-output (``response_format``) capability for auxiliary requests.

Two sources decide whether an aux request may carry a ``response_format`` type up front:

* the provider profile's ``unsupported_response_formats`` (DeepSeek's native API implements only
  ``json_object`` — https://api-docs.deepseek.com/guides/json_mode — and answers ``json_schema`` with
  HTTP 400 "This response_format type is unavailable now"), also consulted when a ``custom`` route's
  base_url points at a profiled provider's host, and
* a process-level memo of (endpoint, model, type) triples that already rejected the type once; the
  recovery ladder records the triple when its retry without the field succeeded.

Either way the field is dropped before the first request instead of burning a guaranteed-fail
round-trip per call (#83390, #105191, #113064). Dropping — not downgrading to ``json_object`` — is the
same end state the rejection retry already produces: ``json_object`` needs the prompt to mention JSON
and some relays return empty content under it, so callers already tolerate prompt compliance.

The memo carries the model because capability is per model on aggregators (openrouter.ai, the Nous
Portal, api.openai.com host dozens of models with different structured-output support), and it is
fed only by rejections that name the *capability* — not by schema-validation 400s from providers that
do implement ``json_schema`` ("Invalid schema for response_format 'json_schema': additionalProperties
must be false"), which the ladder still retries once but which say nothing about the next schema.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

# (route key, model, response_format type) triples a provider rejected in this process.
_REJECTED_ROUTES: set[tuple[str, str, str]] = set()

# Rejections that describe the route/model's capability rather than this request's schema.
_CAPABILITY_REJECTION_MARKERS = (
    "unavailable", "does not support", "doesn't support", "not supported", "unsupported parameter",
    "unsupported_parameter", "unknown parameter", "unrecognized request argument", "unrecognized parameter",
    "extra inputs are not permitted",
)


def _route_key(provider: Optional[str], base_url: Optional[str]) -> str:
    """Endpoint host:port when known (a base_url override turns a named provider into ``custom``; local
    servers differ by port), else the provider name."""
    return (urlparse(base_url or "").netloc or "").lower() or str(provider or "").strip().lower()


def _response_format_type(request_kwargs: Dict[str, Any]) -> Optional[str]:
    extra_body = request_kwargs.get("extra_body")
    response_format = (extra_body or {}).get("response_format") if isinstance(extra_body, dict) else None
    if response_format is None:
        response_format = request_kwargs.get("response_format")
    return response_format.get("type") if isinstance(response_format, dict) else None


def _profile_unsupported_formats(provider: Optional[str], base_url: Optional[str]) -> tuple:
    """The provider profile's declared unsupported types; a ``custom`` route whose base_url is a profiled
    provider's own host (``api.deepseek.com``) gets that provider's profile."""
    try:
        from providers import get_provider_profile
        name = str(provider or "").strip().lower()
        if name == "custom" and base_url:
            from agent.model_metadata import _infer_provider_from_url
            name = _infer_provider_from_url(base_url) or name
        profile = get_provider_profile(name)
    except Exception:
        return ()
    return tuple(getattr(profile, "unsupported_response_formats", ()) or ()) if profile is not None else ()


def is_capability_rejection(error: Optional[BaseException]) -> bool:
    """Whether a structured-output rejection speaks to the route/model's capability (memoisable) rather
    than to this request's schema (retry once, remember nothing)."""
    err_lower = str(error or "").lower()
    if "invalid schema" in err_lower:
        return False
    return any(marker in err_lower for marker in _CAPABILITY_REJECTION_MARKERS)


def remember_structured_output_rejection(
    provider: Optional[str], base_url: Optional[str], rejected_kwargs: Dict[str, Any], error: BaseException,
) -> None:
    """Record that this route's ``rejected_kwargs["model"]`` rejected the ``response_format`` type carried
    by *rejected_kwargs* — only when *error* names the capability, never for a schema-validation 400."""
    format_type = _response_format_type(rejected_kwargs)
    if format_type and is_capability_rejection(error):
        _REJECTED_ROUTES.add((_route_key(provider, base_url), str(rejected_kwargs.get("model") or ""), format_type))


def without_unsupported_response_format(
    extra_body: Dict[str, Any], provider: Optional[str], base_url: Optional[str], model: Optional[str],
    task: Optional[str] = None,
) -> Dict[str, Any]:
    """*extra_body* minus a ``response_format`` whose type this route+model is known to reject; unchanged
    otherwise."""
    response_format = extra_body.get("response_format")
    format_type = response_format.get("type") if isinstance(response_format, dict) else None
    if not format_type:
        return extra_body
    known_unsupported = (
        format_type in _profile_unsupported_formats(provider, base_url)
        or (_route_key(provider, base_url), str(model or ""), format_type) in _REJECTED_ROUTES
    )
    if not known_unsupported:
        return extra_body
    logger.info(
        "Auxiliary %s: %s (%s) does not accept response_format %s; sending without it "
        "(schema enforcement degrades to prompt compliance)",
        task or "call", _route_key(provider, base_url) or "provider", model or "model", format_type,
    )
    return {k: v for k, v in extra_body.items() if k != "response_format"}
