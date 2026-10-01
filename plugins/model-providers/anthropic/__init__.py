"""Native Anthropic provider profile and provider-owned catalogue protocol."""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Callable

from hermes_cli.urllib_security import open_credentialed_url
from providers import register_provider
from providers.base import ProviderProfile
from providers.model_normalizers import strip_matching_prefix

logger = logging.getLogger(__name__)
_MODELS_MAX_PAGES = 20


def _models_url(base_url: str | None, *, after_id: str | None = None) -> str:
    endpoint = str(base_url or "https://api.anthropic.com").strip().rstrip("/")
    url = endpoint + ("/models" if endpoint.endswith("/v1") else "/v1/models")
    params = {"limit": "1000"}
    if after_id:
        params["after_id"] = after_id
    return url + ("&" if "?" in url else "?") + urllib.parse.urlencode(params)


def _next_cursor(page: Any, seen: set[str]) -> str | None:
    if not isinstance(page, dict) or page.get("has_more") is not True:
        return None
    cursor = page.get("last_id")
    if not isinstance(cursor, str) or not cursor or cursor in seen:
        return None
    seen.add(cursor)
    return cursor


class AnthropicProfile(ProviderProfile):
    """Anthropic's catalogue protocol is provider-specific; account resolution is not."""

    def normalize_model_id(self, model: str, *, known_ids=()) -> str:
        bare = strip_matching_prefix(self, model, excluded_prefixes=("claude-oauth",))
        return bare if "/" in bare else bare.replace(".", "-")

    def fetch_catalog_models(
        self,
        *,
        api_key: str,
        base_url: str | None = None,
        timeout: float = 8.0,
        oauth: bool = False,
        request_json: Callable[..., Any] | None = None,
    ) -> list[str] | None:
        """Fetch all cursor pages with one request policy for key and OAuth catalogue access.

        The caller determines account credentials and whether the token is OAuth. An optional
        guarded request callback supports the application's established HTTP/test boundary.
        """
        if not api_key:
            return None

        if request_json is None:
            def request_json(url: str, *, timeout: float, headers: dict[str, str]):
                req = urllib.request.Request(url, headers=headers)
                with open_credentialed_url(req, timeout=timeout) as response:
                    return json.loads(response.read().decode("utf-8"))

        headers = {"anthropic-version": "2023-06-01", "Accept": "application/json"}
        if oauth:
            from agent.anthropic_adapter import _COMMON_BETAS, _OAUTH_ONLY_BETAS, _CONTEXT_1M_BETA
            headers["Authorization"] = f"Bearer {api_key}"
            headers["anthropic-beta"] = ",".join(_COMMON_BETAS + _OAUTH_ONLY_BETAS)
        else:
            headers["x-api-key"] = api_key

        def fetch_page(cursor: str | None):
            url = _models_url(base_url, after_id=cursor)
            try:
                return request_json(url, timeout=timeout, headers=headers)
            except urllib.error.HTTPError as error:
                if not (oauth and cursor is None and error.code == 400):
                    raise
                try:
                    body = error.read().decode(errors="ignore").lower()
                except Exception:
                    body = ""
                if "long context beta" not in body or "not yet available" not in body:
                    raise
                headers["anthropic-beta"] = ",".join(
                    [beta for beta in _COMMON_BETAS if beta != _CONTEXT_1M_BETA]
                    + list(_OAUTH_ONLY_BETAS)
                )
                return request_json(url, timeout=timeout, headers=headers)

        try:
            models: list[str] = []
            seen: set[str] = set()
            cursor: str | None = None
            for _ in range(_MODELS_MAX_PAGES):
                payload = fetch_page(cursor)
                if not isinstance(payload, dict):
                    return None
                models.extend(
                    item["id"] for item in payload.get("data", [])
                    if isinstance(item, dict) and isinstance(item.get("id"), str) and item["id"]
                )
                cursor = _next_cursor(payload, seen)
                if cursor is None:
                    break
            return list(dict.fromkeys(models))
        except Exception as exc:
            logger.debug("fetch_catalog_models(anthropic): %s", exc)
            return None

    def fetch_models(
        self, *, api_key: str | None = None, base_url: str | None = None, timeout: float = 8.0
    ) -> list[str] | None:
        return self.fetch_catalog_models(api_key=api_key or "", base_url=base_url, timeout=timeout)


anthropic = AnthropicProfile(
    name="anthropic", aliases=("claude", "claude-oauth", "claude-code"),
    display_name="Anthropic",
    description="Anthropic (Claude models via API key or Claude Code)",
    api_mode="anthropic_messages",
    env_vars=("ANTHROPIC_API_KEY", "ANTHROPIC_TOKEN", "CLAUDE_CODE_OAUTH_TOKEN"),
    base_url="https://api.anthropic.com", base_url_env_var="ANTHROPIC_BASE_URL",
    auth_type="api_key", default_aux_model="claude-haiku-4-5-20251001",
)

register_provider(anthropic)
