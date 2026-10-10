"""Exa web search + content extraction via the ``exa-py`` SDK (lazy-installed).

Env: ``EXA_API_KEY`` (https://exa.ai). Both methods are sync — Exa's SDK is
sync-only; the dispatcher threads extract when the caller is async.
"""

from __future__ import annotations

import logging
import json
from typing import Any, Dict, List, Literal
import requests
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from agent.web_acquisition_errors import WebAttributionInvalidError, WebInvalidResponseError, WebPageFetchError, WebPageNotFoundError

from plugins.web._common import (
    BaseWebSearchProvider, cached_sdk_client, document, keyless_extract, keyless_search, keyless_variant_schema,
    page_error, provider_env, run_extract, run_search, search_ok, use_keyless, web_hit,
)

logger = logging.getLogger(__name__)

_MISSING_KEY = "EXA_API_KEY environment variable not set. Get your API key at https://exa.ai"


class _ExaResult(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)
    url: str = Field(min_length=1)
    title: str | None = None
    highlights: list[str] | None = None
    text: str | None = None
    id: str | None = None


class _ExaResponse(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)
    results: list[_ExaResult]


class _ExaPageError(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)
    tag: str | None = None
    httpStatusCode: int | None = Field(default=None, ge=100, le=599)


class _ExaPageStatus(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)
    id: str = Field(min_length=1)
    status: Literal["success", "error"]
    error: _ExaPageError | None = None


class _ExaContentsResponse(_ExaResponse):
    statuses: list[_ExaPageStatus] = Field(default_factory=list)


def _get_exa_client() -> Any:
    def _factory(api_key: str) -> Any:
        from exa_py import Exa  # deliberately lazy

        class _ValidatedExa(Exa):
            def request(
                self, endpoint: str, data: dict[str, Any] | str | None = None,
                method: str = "POST", params: dict[str, Any] | None = None,
                headers: dict[str, str] | None = None,
            ) -> dict[str, Any] | requests.Response:
                if endpoint not in {"/search", "/contents"}:
                    return super().request(endpoint, data, method, params, headers)
                from exa_py.api import ExaJSONEncoder
                payload = data if isinstance(data, str) else json.dumps(data, cls=ExaJSONEncoder) if data is not None else None
                streaming = bool(isinstance(data, dict) and data.get("stream") or params and params.get("stream") == "true")
                response = requests.request(method, self.base_url + endpoint, data=payload,
                                            params=params, headers={**self.headers, **(headers or {})}, stream=streaming)
                response.raise_for_status()
                if streaming:
                    return response
                raw = response.json()
                try:
                    _ExaResponse.model_validate(raw)
                except ValidationError as exc:
                    raise WebInvalidResponseError(cause=exc) from exc
                return raw

        client = _ValidatedExa(api_key=api_key)
        client.headers["x-exa-integration"] = "hermes-agent"
        return client

    return cached_sdk_client("_exa_client", "EXA_API_KEY", _MISSING_KEY, "exa", _factory)


class ExaWebSearchProvider(BaseWebSearchProvider):
    """Exa search + extract provider."""

    NAME = "exa"
    DISPLAY_NAME = "Exa"
    KEY_ENV = "EXA_API_KEY"
    EXTRACT = True
    KEYLESS = True
    EXTRACT_MAX_CHARACTERS = True

    def search(self, query: str, limit: int = 5) -> dict[str, Any]:
        def _body() -> dict[str, Any]:
            if use_keyless("exa", provider_env("EXA_API_KEY")):
                return keyless_search("Exa", "exa", query, limit, logger)
            logger.info("Exa search: '%s' (limit=%d)", query, limit)
            response = _get_exa_client().search(query, num_results=limit, contents={"highlights": True})
            return search_ok([
                web_hit(r.url or "", r.title or "", " ".join(r.highlights or []), i + 1)
                for i, r in enumerate(response.results or [])
            ])

        return run_search("Exa", logger, _body, sdk=True)

    def extract(self, urls: list[str], **kwargs: Any) -> list[dict[str, Any]]:
        def _body() -> list[dict[str, Any]]:
            if use_keyless("exa", provider_env("EXA_API_KEY")):
                return keyless_extract("Exa", "exa", urls, logger, **kwargs)
            logger.info("Exa extract: %d URL(s)", len(urls))
            max_chars = kwargs.get("max_chars")
            text = True if max_chars is None else {"maxCharacters": max_chars}
            raw = _get_exa_client().request("/contents", {"urls": urls, "text": text})
            try:
                response = _ExaContentsResponse.model_validate(raw)
            except ValidationError as exc:
                raise WebInvalidResponseError(cause=exc) from exc
            rows = [document(r.url, r.title or "", r.text or "", source_url=r.id,
                             coverage="full" if max_chars is None else "partial") for r in response.results]
            seen: set[str] = set()
            for status in response.statuses:
                if status.id not in urls or status.id in seen:
                    raise WebAttributionInvalidError()
                seen.add(status.id)
                if status.status != "error":
                    continue
                code = status.error.httpStatusCode if status.error is not None else None
                code = code if code is not None and code >= 400 else None
                failure = WebPageNotFoundError(code) if code in {404, 410} else WebPageFetchError(code)
                rows.append(page_error(status.id, failure.diagnostic, failure=failure))
            return rows

        return run_extract("Exa", logger, urls, _body, sdk=True)

    def get_setup_schema(self) -> dict[str, Any]:
        return keyless_variant_schema(
            "Exa", "EXA_API_KEY", "https://exa.ai",
            free_tag="Semantic + neural web search with content extraction on Exa's anonymous free tier. Rate-limited under burst load.",
            paid_tag="Semantic + neural web search with content extraction via the Exa SDK. Unthrottled, guaranteed service.",
        )
