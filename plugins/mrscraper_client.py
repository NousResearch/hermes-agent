"""Shared HTTP client for the bundled MrScraper integrations."""

from __future__ import annotations

import json
from typing import Any, Dict, Mapping, Optional
from urllib.parse import quote, quote_plus

import requests

from agent.secret_scope import get_secret

PRIMARY_API_URL = "https://api.app.mrscraper.com"
SERP_API_URL = "https://sync.scraper.mrscraper.com"
RENDERED_API_URL = "https://api.mrscraper.com"
DEFAULT_HTTP_TIMEOUT = 120
MAX_ERROR_BODY_CHARS = 500


class MrScraperError(RuntimeError):
    """Base error for configuration, validation, and transport failures."""


class MrScraperAPIError(MrScraperError):
    """An upstream MrScraper request returned an HTTP or transport error."""

    def __init__(self, message: str, *, status_code: Optional[int] = None) -> None:
        super().__init__(message)
        self.status_code = status_code


def get_mrscraper_token() -> str:
    """Resolve the API token from the active Hermes secret scope."""
    token = (get_secret("MRSCRAPER_API_TOKEN", "") or "").strip()
    if not token:
        raise MrScraperError(
            "MRSCRAPER_API_TOKEN is required. Configure it with `hermes tools` "
            "or in the active profile's ~/.hermes/.env."
        )
    return token


def is_mrscraper_available() -> bool:
    """Return whether the active profile has a MrScraper token."""
    return bool((get_secret("MRSCRAPER_API_TOKEN", "") or "").strip())


def redact_token(text: Any, token: str) -> str:
    """Remove exact and URL-encoded representations of *token* from text."""
    safe = str(text)
    if not token:
        return safe
    for value in {token, quote(token, safe=""), quote_plus(token)}:
        if value:
            safe = safe.replace(value, "[REDACTED]")
    return safe


def compact_optional(mapping: Mapping[str, Any]) -> Dict[str, Any]:
    """Omit only ``None`` values, preserving false, zero, and empty arrays."""
    return {key: value for key, value in mapping.items() if value is not None}


class MrScraperClient:
    """Small requests-based client covering all three MrScraper API origins."""

    def __init__(self, token: Optional[str] = None) -> None:
        self.token = (token or get_mrscraper_token()).strip()

    def _request(
        self,
        method: str,
        url: str,
        *,
        headers: Mapping[str, str],
        params: Optional[Mapping[str, Any]] = None,
        body: Optional[Mapping[str, Any]] = None,
        timeout: int = DEFAULT_HTTP_TIMEOUT,
        force_text: bool = False,
    ) -> Any:
        try:
            response = requests.request(
                method,
                url,
                headers=dict(headers),
                params=dict(params) if params is not None else None,
                json=dict(body) if body is not None else None,
                timeout=timeout,
            )
        except requests.RequestException as exc:
            safe = redact_token(exc, self.token)
            raise MrScraperAPIError(
                f"MrScraper request failed: {type(exc).__name__}: {safe}"
            ) from exc

        if not response.ok:
            body_text = redact_token(response.text or "", self.token)
            if len(body_text) > MAX_ERROR_BODY_CHARS:
                body_text = body_text[:MAX_ERROR_BODY_CHARS] + "… [truncated]"
            detail = f": {body_text}" if body_text else ""
            raise MrScraperAPIError(
                f"MrScraper API returned HTTP {response.status_code}{detail}",
                status_code=response.status_code,
            )

        if force_text:
            return response.text

        content_type = (response.headers.get("Content-Type") or "").lower()
        if "json" in content_type:
            try:
                return response.json()
            except (ValueError, json.JSONDecodeError) as exc:
                raise MrScraperAPIError(
                    "MrScraper returned invalid JSON for a JSON response"
                ) from exc

        # Some API gateways omit Content-Type while still returning JSON.
        try:
            return response.json()
        except (ValueError, json.JSONDecodeError):
            return response.text

    def primary_get(
        self, path: str, *, params: Optional[Mapping[str, Any]] = None
    ) -> Any:
        return self._request(
            "GET",
            f"{PRIMARY_API_URL}{path}",
            headers={
                "Accept": "application/json",
                "Content-Type": "application/json",
                "x-api-token": self.token,
            },
            params=params,
        )

    def primary_post(self, path: str, body: Mapping[str, Any]) -> Any:
        return self._request(
            "POST",
            f"{PRIMARY_API_URL}{path}",
            headers={
                "Accept": "application/json",
                "Content-Type": "application/json",
                "x-api-token": self.token,
            },
            body=body,
        )

    def serp_search(self, body: Mapping[str, Any], *, html: bool = False) -> Any:
        return self._request(
            "POST",
            f"{SERP_API_URL}/api/google/serp/v2/sync",
            headers={
                "Accept": "application/json",
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.token}",
            },
            body=body,
            force_text=html,
        )

    def fetch_rendered(
        self,
        *,
        params: Mapping[str, Any],
        body: Mapping[str, Any],
        timeout: int,
    ) -> Any:
        query = {"token": self.token, "browserRendering": "true", **params}
        return self._request(
            "POST",
            f"{RENDERED_API_URL}/",
            headers={
                "Accept": "application/json, text/html, text/plain",
                "Content-Type": "application/json",
            },
            params=query,
            body=body,
            timeout=max(1, min(timeout + 30, 600)),
        )


def encoded_path_segment(value: str) -> str:
    """Encode one user-controlled value as exactly one URL path segment."""
    return quote(value, safe="")
