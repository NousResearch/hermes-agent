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

import json
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


def _response_format_type(request_kwargs: dict[str, Any]) -> Optional[str]:
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


# Body keys that carry routing METADATA rather than a reason -- an opaque identity or timestamp whose
# value cannot tell the caller anything. A rejection whose payload holds nothing but these has named no
# reason. opencode-go (opencode.ai/zen/go) answers HTTP 400 with exactly ``{"model":
# "deepseek-v4.1-flash"}`` -- or an empty body -- for a deepseek-v4.1-flash request carrying
# ``response_format`` (probed 2026-10-03: 23 of 24 fresh sessions; every one of them returned 200 for
# the same request without the field).
#
# Deliberately NOT in this set: ``code``/``error_code``/``status``/``type``/``success``/``ok``. Those
# carry a value the caller can READ, and a relay that answers ``{"error_code": "missing_session_id"}`` or
# ``{"type": "rate_limit_error"}`` has named its reason -- treating such a payload as "names nothing"
# would swallow exactly the failures the rest of the classifier exists to surface.
_ECHO_BODY_KEYS = frozenset({
    "model", "id", "object", "created", "request_id", "requestid", "request-id", "trace_id", "trace",
    "log_id", "span_id",
})
# Body keys that WOULD carry a human-readable reason, when the payload has one.
_DIAGNOSTIC_BODY_KEYS = frozenset({
    "message", "msg", "detail", "details", "error", "errors", "reason", "description", "error_description",
})


def _error_body_text(error: Optional[BaseException]) -> str:
    """The rejection's raw response body when the HTTP client exposes one, else the exception text."""
    response = getattr(error, "response", None)
    if response is not None:
        for attr in ("text", "content"):
            try:
                value = getattr(response, attr, None)
            except Exception:
                value = None
            if isinstance(value, bytes):
                try:
                    value = value.decode("utf-8", "replace")
                except Exception:
                    value = None
            if isinstance(value, str) and value.strip():
                return value
    return str(error or "")


def _payload_says_nothing(text: str) -> bool:
    """Whether a rejection payload names no reason: empty, or JSON holding only routing echoes."""
    text = (text or "").strip()
    # The OpenAI SDK prefixes its own status line. With a parsed body that is "Error code: 400 - {body}";
    # with an EMPTY (or closed-stream) body it is the bare "Error code: 400" -- no separator, nothing
    # after it. Strip either form; the bare form leaves no text at all, which is the silent case.
    if text.lower().startswith("error code:"):
        _, sep, rest = text.partition("-")
        text = rest.strip() if sep else ""
    if not text:
        return True
    try:
        body = json.loads(text)
    except Exception:
        return False  # unparsed prose may well name the problem — never assume it does not
    if body is None:
        return True  # a raw ``null`` body names nothing either
    if not isinstance(body, dict) or not body:
        return False
    for key, value in body.items():
        name = str(key).strip().lower()
        if name in _DIAGNOSTIC_BODY_KEYS:
            if isinstance(value, str):
                if value.strip():
                    return False
            elif value:
                return False
            continue
        if name in _ECHO_BODY_KEYS:
            continue
        # An unrecognised key with content is assumed to be a diagnostic.
        if isinstance(value, str):
            if value.strip():
                return False
        elif value:
            return False
    return True


def is_diagnostic_free_rejection(error: Optional[BaseException], statuses: tuple = (400, 422)) -> bool:
    """Whether *error* is an HTTP rejection whose payload names no reason at all.

    Relays that spread one model across several upstreams can refuse a parameter on some of them and
    answer with an empty body or bare routing echoes; there is no text to pattern-match. Callers treat
    this as a last-resort signal: the request already carries the field, and the deciding evidence is
    the retry — the ladder re-raises the narrowed error unchanged when the stripped retry fails too.
    """
    # The status may ride on the exception (``openai.APIStatusError.status_code``) or only on the
    # response it wraps; a silent 401/429/500 must not slip through because the first lookup missed.
    status = getattr(error, "status_code", None)
    if status is None:
        status = getattr(getattr(error, "response", None), "status_code", None)
    if status is not None and status not in statuses:
        return False
    return _payload_says_nothing(_error_body_text(error))


def is_capability_rejection(error: Optional[BaseException]) -> bool:
    """Whether a structured-output rejection speaks to the route/model's capability (memoisable) rather
    than to this request's schema (retry once, remember nothing)."""
    err_lower = str(error or "").lower()
    if "invalid schema" in err_lower:
        return False
    if any(marker in err_lower for marker in _CAPABILITY_REJECTION_MARKERS):
        return True
    # Nothing to read at all: the caller only records *after* the retry without ``response_format``
    # succeeded, so a diagnostic-free rejection is evidence the route refused the field, not this
    # schema. Memoising it is what keeps the next structured-output call from paying the same 400.
    return is_diagnostic_free_rejection(error)


def remember_structured_output_rejection(
    provider: Optional[str], base_url: Optional[str], rejected_kwargs: dict[str, Any], error: BaseException,
) -> None:
    """Record that this route's ``rejected_kwargs["model"]`` rejected the ``response_format`` type carried
    by *rejected_kwargs* — only when *error* names the capability, never for a schema-validation 400."""
    format_type = _response_format_type(rejected_kwargs)
    if format_type and is_capability_rejection(error):
        _REJECTED_ROUTES.add((_route_key(provider, base_url), str(rejected_kwargs.get("model") or ""), format_type))


def without_unsupported_response_format(
    extra_body: dict[str, Any], provider: Optional[str], base_url: Optional[str], model: Optional[str],
    task: Optional[str] = None,
) -> dict[str, Any]:
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
