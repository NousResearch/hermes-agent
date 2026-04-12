from __future__ import annotations

from typing import Any, Mapping

from openai import AsyncOpenAI, OpenAI

_GEMINI_BASE_MARKER = "generativelanguage.googleapis.com"
_GEMINI_API_KEY_PREFIX = "AIza"


def _is_gemini_base_url(base_url: str | None) -> bool:
    return _GEMINI_BASE_MARKER in str(base_url or "").lower()


def _use_gemini_api_key_header(api_key: str | None, base_url: str | None) -> bool:
    key = str(api_key or "").strip()
    return _is_gemini_base_url(base_url) and key.startswith(_GEMINI_API_KEY_PREFIX)


class GeminiAPIKeyOpenAI(OpenAI):
    """OpenAI client variant that uses x-goog-api-key instead of Authorization."""

    @property
    def auth_headers(self) -> dict[str, str]:
        return {}


class GeminiAPIKeyAsyncOpenAI(AsyncOpenAI):
    """Async OpenAI client variant that uses x-goog-api-key instead of Authorization."""

    @property
    def auth_headers(self) -> dict[str, str]:
        return {}


def _normalize_default_headers(
    headers: Mapping[str, str] | None,
    *,
    api_key: str | None,
    base_url: str | None,
) -> dict[str, str]:
    normalized = dict(headers or {})
    if _use_gemini_api_key_header(api_key, base_url):
        normalized.pop("Authorization", None)
        normalized["x-goog-api-key"] = str(api_key or "").strip()
    return normalized


def create_openai_client(**client_kwargs: Any) -> OpenAI:
    kwargs = dict(client_kwargs)
    api_key = str(kwargs.get("api_key", "") or "")
    base_url = str(kwargs.get("base_url", "") or "")
    default_headers = _normalize_default_headers(
        kwargs.get("default_headers"),
        api_key=api_key,
        base_url=base_url,
    )
    if default_headers:
        kwargs["default_headers"] = default_headers
    else:
        kwargs.pop("default_headers", None)

    if _use_gemini_api_key_header(api_key, base_url):
        kwargs["api_key"] = ""
        return GeminiAPIKeyOpenAI(**kwargs)
    return OpenAI(**kwargs)


def create_async_openai_client(**client_kwargs: Any) -> AsyncOpenAI:
    kwargs = dict(client_kwargs)
    api_key = str(kwargs.get("api_key", "") or "")
    base_url = str(kwargs.get("base_url", "") or "")
    default_headers = _normalize_default_headers(
        kwargs.get("default_headers"),
        api_key=api_key,
        base_url=base_url,
    )
    if default_headers:
        kwargs["default_headers"] = default_headers
    else:
        kwargs.pop("default_headers", None)

    if _use_gemini_api_key_header(api_key, base_url):
        kwargs["api_key"] = ""
        return GeminiAPIKeyAsyncOpenAI(**kwargs)
    return AsyncOpenAI(**kwargs)
