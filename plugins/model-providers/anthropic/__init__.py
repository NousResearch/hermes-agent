"""Native Anthropic provider profile."""

import json
import logging
import urllib.parse
import urllib.request

from hermes_cli.urllib_security import open_credentialed_url
from providers import register_provider
from providers.base import ProviderProfile

logger = logging.getLogger(__name__)


class AnthropicProfile(ProviderProfile):
    """Native Anthropic — uses x-api-key header, not Bearer."""

    def fetch_models(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 8.0,
    ) -> list[str] | None:
        """Anthropic uses x-api-key header and anthropic-version.

        The endpoint is cursor-paginated (default page size 20, smaller than
        the live catalog), so request the max page size and follow
        ``has_more``/``last_id`` — an unpaginated read silently drops models.
        """
        if not api_key:
            return None
        endpoint = (base_url or "https://api.anthropic.com").strip().rstrip("/")
        if not endpoint.endswith("/v1"):
            endpoint += "/v1"
        try:
            models: list[str] = []
            after_id: str | None = None
            seen_cursors: set[str] = set()
            for _page in range(20):
                params = {"limit": "1000"}
                if after_id:
                    params["after_id"] = after_id
                url = endpoint + "/models?" + urllib.parse.urlencode(params)
                req = urllib.request.Request(url)
                req.add_header("x-api-key", api_key)
                req.add_header("anthropic-version", "2023-06-01")
                req.add_header("Accept", "application/json")
                with open_credentialed_url(req, timeout=timeout) as resp:
                    data = json.loads(resp.read().decode())
                models.extend(
                    m["id"]
                    for m in data.get("data", [])
                    if isinstance(m, dict) and "id" in m
                )
                if data.get("has_more") is not True:
                    break
                last_id = data.get("last_id")
                if not isinstance(last_id, str) or not last_id or last_id in seen_cursors:
                    break
                seen_cursors.add(last_id)
                after_id = last_id
            return list(dict.fromkeys(models))
        except Exception as exc:
            logger.debug("fetch_models(anthropic): %s", exc)
            return None


anthropic = AnthropicProfile(
    name="anthropic",
    aliases=("claude", "claude-oauth", "claude-code"),
    api_mode="anthropic_messages",
    env_vars=("ANTHROPIC_API_KEY", "ANTHROPIC_TOKEN", "CLAUDE_CODE_OAUTH_TOKEN"),
    base_url="https://api.anthropic.com",
    auth_type="api_key",
    default_aux_model="claude-haiku-4-5-20251001",
)

register_provider(anthropic)
