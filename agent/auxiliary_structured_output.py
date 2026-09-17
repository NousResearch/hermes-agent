"""Structured-output (``response_format``) capability for auxiliary requests.

Two sources decide whether an aux request may carry a ``response_format`` type up front:

* the provider profile's ``unsupported_response_formats`` (DeepSeek's native API implements only
  ``json_object`` — https://api-docs.deepseek.com/guides/json_mode — and answers ``json_schema`` with
  HTTP 400 "This response_format type is unavailable now"), and
* a process-level memo of routes that already rejected a type once; the recovery ladder records the
  route when its retry without the field succeeded.

Either way the field is dropped before the first request instead of burning a guaranteed-fail
round-trip per call (#83390, #105191, #113064). Dropping — not downgrading to ``json_object`` — is the
same end state the rejection retry already produces: ``json_object`` needs the prompt to mention JSON
and some relays return empty content under it, so callers already tolerate prompt compliance.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

# (route key, response_format type) pairs a provider rejected in this process.
_REJECTED_ROUTES: set[tuple[str, str]] = set()


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


def _profile_unsupported_formats(provider: Optional[str]) -> tuple:
    try:
        from providers import get_provider_profile
        profile = get_provider_profile(str(provider or "").strip().lower())
    except Exception:
        return ()
    return tuple(getattr(profile, "unsupported_response_formats", ()) or ()) if profile is not None else ()


def remember_structured_output_rejection(
    provider: Optional[str], base_url: Optional[str], rejected_kwargs: Dict[str, Any],
) -> None:
    """Record that this route rejected the ``response_format`` type carried by *rejected_kwargs*."""
    format_type = _response_format_type(rejected_kwargs)
    if format_type:
        _REJECTED_ROUTES.add((_route_key(provider, base_url), format_type))


def without_unsupported_response_format(
    extra_body: Dict[str, Any], provider: Optional[str], base_url: Optional[str], task: Optional[str] = None,
) -> Dict[str, Any]:
    """*extra_body* minus a ``response_format`` whose type this route is known to reject; unchanged otherwise."""
    response_format = extra_body.get("response_format")
    format_type = response_format.get("type") if isinstance(response_format, dict) else None
    if not format_type:
        return extra_body
    known_unsupported = (
        format_type in _profile_unsupported_formats(provider)
        or (_route_key(provider, base_url), format_type) in _REJECTED_ROUTES
    )
    if not known_unsupported:
        return extra_body
    logger.info(
        "Auxiliary %s: %s does not accept response_format %s; sending without it "
        "(schema enforcement degrades to prompt compliance)",
        task or "call", _route_key(provider, base_url) or "provider", format_type,
    )
    return {k: v for k, v in extra_body.items() if k != "response_format"}
