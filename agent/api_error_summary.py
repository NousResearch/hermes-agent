"""Provider API error summarising for ``AIAgent``.

Entitlement-failure detection, xAI subscription decoration, structured-detail coercion, and log-safe
redaction of provider error payloads.
Extracted from ``run_agent.py``; every method resolves through ``AIAgent``'s MRO unchanged.
"""
import json
import re
from typing import Any, Dict, Optional

from agent.redact import redact_sensitive_text

# Offline DNS failures are wrapped in a generic "Connection error" by SDKs — inspect the chain.
_NETWORK_RESOLUTION_MARKERS = (
    "temporary failure in name resolution",
    "name or service not known",
    "nodename nor servname provided, or not known",
    "getaddrinfo failed",
    "no address associated with hostname",
    "network is unreachable",
)
_XAI_ENTITLEMENT_HINT = (
    " — xAI rejected this OAuth account. NOTE: X Premium+ does NOT "
    "include xAI API access — only standalone SuperGrok subscribers "
    "can use this provider. Other possible causes: no Grok "
    "subscription, your tier doesn't include this model, or your "
    "quota is exhausted. Check https://grok.com/?_s=usage to see "
    "which, or run `/model` to switch providers."
)
_ERROR_DETAIL_KEYS = ("message", "detail", "error", "code", "type")


def _is_xai_entitlement_text(lower: str) -> bool:
    """xAI's permission-denied body text for an unsubscribed / under-tiered / exhausted account."""
    return (
        "do not have an active grok subscription" in lower
        or ("out of available resources" in lower and "grok" in lower)
        or ("does not have permission" in lower and "grok" in lower)
    )


def _http_prefix(error: Exception) -> str:
    status_code = getattr(error, "status_code", None)
    return f"HTTP {status_code}: " if status_code else ""


class ApiErrorSummaryMixin:
    """Provider error -> user/log-safe summary (see module docstring)."""

    @staticmethod
    def _is_entitlement_failure(
        error_context: Optional[Dict[str, Any]], status_code: Optional[int]
    ) -> bool:
        """Detect subscription/entitlement 401/403s that masquerade as auth failures.

        Refreshing a token cannot fix an unsubscribed account, so callers surface the error instead of looping
        the pool. xAI returns the same permission-denied text for BOTH cases; a ``[WKE=unauthenticated:...]``
        suffix (or "access token could not be validated") means stale token → return False so the refresh path
        runs.

        Disambiguator for xAI (#29344): the same ``code`` text ("The caller does not have permission to
        execute the specified operation") is returned for BOTH an unsubscribed account AND a stale OAuth
        access token.  xAI ships an explicit signal in the ``error`` field that tells the two apart: a
        ``[WKE=unauthenticated:...]`` suffix (and/or the ``OAuth2 access token could not be validated``
        phrasing) means the credentials failed validation — that's recoverable by refreshing the token, NOT
        by surfacing an entitlement message. When either signal is present we return False eagerly so the
        credential-pool refresh path runs, letting long-running TUI sessions recover from stale tokens
        without an exit/reopen cycle.
        """
        if status_code not in {401, 403, None}:
            return False
        if not isinstance(error_context, dict):
            return False
        # Single lowercase haystack over every field shape (message/reason and raw code/error).
        haystack = " ".join(
            str(error_context.get(k) or "").lower() for k in ("message", "reason", "code", "error")
        )
        if not haystack.strip():
            return False
        if "[wke=unauthenticated:" in haystack or "oauth2 access token could not be validated" in haystack:
            return False
        return _is_xai_entitlement_text(haystack)

    @staticmethod
    def _decorate_xai_entitlement_error(detail: str) -> str:
        """Append a neutral hint when xAI's OAuth surface returns the permission-denied 403.

        xAI's ``/v1/responses`` uses one body for several causes (no subscription, tier lacks the model, quota
        exhausted). The least obvious: X Premium+ does NOT include API access — only SuperGrok does. Lead with
        that, keep the raw text, point at https://grok.com/?_s=usage. Idempotent: a substring unique to the
        hint marks prior decoration.
        """
        if not detail or not _is_xai_entitlement_text(detail.lower()):
            return detail
        if "X Premium+ does NOT include" in detail:
            return detail
        return f"{detail}{_XAI_ENTITLEMENT_HINT}"

    @staticmethod
    def _coerce_api_error_detail(value: Any) -> str:
        """Return a display-safe string for structured provider error fields."""
        if isinstance(value, str):
            return value
        if isinstance(value, dict):
            for key in _ERROR_DETAIL_KEYS:
                nested = value.get(key)
                if isinstance(nested, str) and nested.strip():
                    return nested
            for key in _ERROR_DETAIL_KEYS:
                if key in value:
                    nested_detail = ApiErrorSummaryMixin._coerce_api_error_detail(value[key])
                    if nested_detail:
                        return nested_detail
            try:
                return json.dumps(value, ensure_ascii=False, sort_keys=True)
            except TypeError:
                return str(value)
        if isinstance(value, (list, tuple)):
            parts = [ApiErrorSummaryMixin._coerce_api_error_detail(item) for item in value]
            return "; ".join(part for part in parts if part)
        if value is None:
            return ""
        return str(value)

    @staticmethod
    def _summarize_api_error(error: Exception) -> str:
        """Extract a human-readable one-liner from an API error.

        Cloudflare HTML pages → ``<title>``; network/DNS failures (even SDK-wrapped) → offline hint; else
        truncated str(error).
        """
        raw = str(error)

        current: Optional[BaseException] = error
        seen: set[int] = set()
        while current is not None and id(current) not in seen:
            seen.add(id(current))
            if any(marker in str(current).lower() for marker in _NETWORK_RESOLUTION_MARKERS):
                return (
                    "Hermes can't reach the model provider. You may be offline. "
                    "Check your internet connection and try again."
                )
            current = current.__cause__ or current.__context__

        if isinstance(error, ValueError) and "expected ident at line" in raw.lower():
            return f"Malformed provider streaming response: {raw[:300]}"

        prefix = _http_prefix(error)
        # Cloudflare / proxy HTML pages: grab the <title> (and Ray ID) for a clean summary
        if "<!DOCTYPE" in raw or "<html" in raw:
            m = re.search(r"<title[^>]*>([^<]+)</title>", raw, re.IGNORECASE)
            title = m.group(1).strip() if m else "HTML error page (title not found)"
            ray = re.search(r"Cloudflare Ray ID:\s*<strong[^>]*>([^<]+)</strong>", raw)
            parts = [prefix[:-2]] if prefix else []
            parts.append(title)
            if ray:
                parts.append(f"Ray {ray.group(1).strip()}")
            return " — ".join(parts)

        # GeminiAPIError already composes a clean one-liner with guidance; don't re-extract the raw body.
        if type(error).__name__ == "GeminiAPIError":
            return redact_sensitive_text(raw[:1000])

        # JSON body errors from OpenAI/Anthropic SDKs
        body = getattr(error, "body", None)
        if isinstance(body, dict):
            msg = body.get("error", {}).get("message") if isinstance(body.get("error"), dict) else body.get("message")
            if msg:
                msg = ApiErrorSummaryMixin._coerce_api_error_detail(msg)
                return ApiErrorSummaryMixin._decorate_xai_entitlement_error(f"{prefix}{msg[:300]}")

        # SDK may leave body empty while httpx has the payload. Redact: the body is attacker-influenced
        # and may echo Authorization / x-api-key / request JSON.
        # Redact before returning: the raw provider/proxy error body is attacker-influenced and may echo
        # Authorization / x-api-key / request JSON, which would otherwise leak into final_response + logs
        # (this path widens exposure vs the old empty-body "HTTP 400" string). See #36109.
        response = getattr(error, "response", None)
        if response is not None:
            try:
                snippet = (getattr(response, "text", None) or "").strip()
            except Exception:
                snippet = ""
            if snippet:
                try:
                    payload = json.loads(snippet)
                except (json.JSONDecodeError, TypeError):
                    payload = None
                if isinstance(payload, dict):
                    err = payload.get("error")
                    if isinstance(err, dict) and err.get("message"):
                        return redact_sensitive_text(f"{prefix}{str(err['message'])[:300]}")
                    if payload.get("message"):
                        return redact_sensitive_text(f"{prefix}{str(payload['message'])[:300]}")
                return redact_sensitive_text(f"{prefix}{snippet[:300]}")

        # Fallback: truncate the raw string but give more room than 200 chars
        return ApiErrorSummaryMixin._decorate_xai_entitlement_error(f"{prefix}{raw[:500]}")

    def _mask_api_key_for_logs(self, key: Any) -> Optional[str]:
        # Azure Foundry Entra ID bearer providers are callables — never invoke them in log
        # paths; identify the auth surface instead.
        if callable(key) and not isinstance(key, str):
            return "<entra-id-bearer>"
        if not key:
            return None
        if len(key) <= 12:
            return "***"
        return f"{key[:8]}...{key[-4:]}"

    def _clean_error_message(self, error_msg: str) -> str:
        """Clean up error messages for user display, removing HTML content and truncating."""
        if not error_msg:
            return "Unknown error"
        # HTML content is common with CloudFlare and gateway error pages
        if error_msg.strip().startswith('<!DOCTYPE html') or '<html' in error_msg:
            return "Service temporarily unavailable (HTML error page returned)"
        cleaned = ' '.join(error_msg.split())
        if len(cleaned) > 150:
            cleaned = cleaned[:150] + "..."
        return cleaned
