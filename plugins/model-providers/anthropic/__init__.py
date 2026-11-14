"""Native Anthropic provider profile."""

import json
import logging
import urllib.request

from hermes_cli.urllib_security import open_credentialed_url
from providers import register_provider
from providers.base import ProviderProfile

logger = logging.getLogger(__name__)


class AnthropicProfile(ProviderProfile):
    """Native Anthropic — uses x-api-key header, not Bearer."""

    def fetch_models(
        self, *, api_key: str | None = None, base_url: str | None = None, timeout: float = 8.0
    ) -> list[str] | None:
        """Anthropic uses x-api-key header and anthropic-version. ``/v1/models`` is cursor-paginated
        (default page 20, smaller than the live catalog), so follow ``has_more``/``last_id``."""
        if not api_key:
            return None
        from hermes_cli.models import _ANTHROPIC_MODELS_MAX_PAGES, _anthropic_models_url, _anthropic_next_cursor

        def _page(after_id: str | None):
            req = urllib.request.Request(_anthropic_models_url(base_url, after_id=after_id))
            for k, v in (("x-api-key", api_key), ("anthropic-version", "2023-06-01"), ("Accept", "application/json")):
                req.add_header(k, v)
            with open_credentialed_url(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode())

        try:
            models: list[str] = []
            seen_cursors: set[str] = set()
            cursor: str | None = None
            for _ in range(_ANTHROPIC_MODELS_MAX_PAGES):
                data = _page(cursor)
                models.extend(m["id"] for m in data.get("data", []) if isinstance(m, dict) and "id" in m)
                cursor = _anthropic_next_cursor(data, seen_cursors)
                if cursor is None:
                    break
            return list(dict.fromkeys(models))
        except Exception as exc:
            logger.debug("fetch_models(anthropic): %s", exc)
            return None


anthropic = AnthropicProfile(
    name="anthropic", aliases=("claude", "claude-oauth", "claude-code"), api_mode="anthropic_messages",
    env_vars=("ANTHROPIC_API_KEY", "ANTHROPIC_TOKEN", "CLAUDE_CODE_OAUTH_TOKEN"),
    base_url="https://api.anthropic.com", auth_type="api_key", default_aux_model="claude-haiku-4-5-20251001",
)

register_provider(anthropic)
