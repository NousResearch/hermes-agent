"""Yandex Cloud AI Studio web search (``yandex-ai-studio-sdk``; lazy-installed).

Env: ``YC_API_KEY`` + ``YC_FOLDER_ID`` (https://yandex.cloud/ru/docs/search-api/).
Both are required — Yandex has no keyless/free tier, unlike most other providers here.
Search-only: Yandex's Search API returns snippets/passages, not full page content, so
this provider does not implement ``extract``.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

from plugins.web._common import (
    BaseWebSearchProvider, lazy_ensure, provider_env, run_search, search_fail, search_ok, web_hit,
)

logger = logging.getLogger(__name__)

_MISSING_KEY = (
    "YC_API_KEY / YC_FOLDER_ID environment variables not set. Get credentials at "
    "https://yandex.cloud/ru/docs/search-api/"
)

_sdk_cache: tuple[str, str, Any] | None = None


def _get_sdk(api_key: str, folder_id: str) -> Any:
    global _sdk_cache
    if _sdk_cache is not None and _sdk_cache[0] == api_key and _sdk_cache[1] == folder_id:
        return _sdk_cache[2]
    lazy_ensure("yandex-ai-studio-sdk")
    from yandex_ai_studio_sdk import AIStudio  # deliberately lazy

    import os
    os.environ["YC_API_KEY"] = api_key
    os.environ["YC_FOLDER_ID"] = folder_id
    sdk = AIStudio()
    _sdk_cache = (api_key, folder_id, sdk)
    return sdk


class YandexWebSearchProvider(BaseWebSearchProvider):
    """Yandex Cloud AI Studio search provider (search only, no extract)."""

    NAME = "yandex"
    DISPLAY_NAME = "Yandex"
    KEY_ENV = "YC_API_KEY"
    EXTRACT = False
    KEYLESS = False

    def is_available(self) -> bool:
        return bool(provider_env("YC_API_KEY")) and bool(provider_env("YC_FOLDER_ID"))

    def search(self, query: str, limit: int = 5) -> dict[str, Any]:
        def _body() -> dict[str, Any]:
            api_key = provider_env("YC_API_KEY")
            folder_id = provider_env("YC_FOLDER_ID")
            if not api_key or not folder_id:
                return search_fail(_MISSING_KEY)
            logger.info("Yandex search: '%s' (limit=%d)", query, limit)
            sdk = _get_sdk(api_key, folder_id)
            search = sdk.search_api.web(
                search_type="RU", family_mode="NONE", fix_typo_mode="ON",
                sort_mode="BY_RELEVANCE", sort_order="DESC", group_mode="DEEP",
                groups_on_page=min(max(1, int(limit)), 20), docs_in_group=1,
                max_passages=4, localization="RU",
            )
            result = search.run(query, page=0)
            hits = []
            for i, doc in enumerate(result.docs or []):
                content = " ".join(p for p in (doc.passages or []) if p)
                hits.append(web_hit(doc.url or "", doc.title or doc.url or "", content, i + 1))
            return search_ok(hits)

        return run_search("Yandex", logger, _body, sdk=True)

    def get_setup_schema(self) -> dict[str, Any]:
        return {
            "name": "Yandex",
            "badge": "key required",
            "tag": "Yandex Cloud AI Studio web search. Requires a Yandex Cloud API key and folder ID.",
            "env_vars": [
                {"key": "YC_API_KEY", "prompt": "Yandex Cloud API key", "url": "https://yandex.cloud/ru/docs/search-api/"},
                {"key": "YC_FOLDER_ID", "prompt": "Yandex Cloud folder ID", "url": "https://yandex.cloud/ru/docs/search-api/"},
            ],
        }
