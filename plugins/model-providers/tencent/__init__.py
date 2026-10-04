"""Tencent Cloud provider profiles (LKEAP Token Plan).

Names match the auth-registry keys exactly so ``model.provider: tencent-tokenplan``
resolves at runtime; without a registered profile the live/curated merge in
``hermes_cli.models`` never runs for this provider (review on #126898).
"""

import json
import logging
import urllib.request

from hermes_cli.urllib_security import open_credentialed_url
from providers import register_provider
from providers.base import ProviderProfile

logger = logging.getLogger(__name__)


class TencentTokenPlanProfile(ProviderProfile):
    """LKEAP Token Plan — anthropic_messages transport; the catalog endpoint speaks the
    Anthropic protocol (x-api-key + anthropic-version, cursor-paginated /v1/models)."""

    def fetch_models(
        self, *, api_key: str | None = None, base_url: str | None = None, timeout: float = 8.0
    ) -> list[str] | None:
        if not api_key:
            return None
        from hermes_cli.models import _ANTHROPIC_MODELS_MAX_PAGES, _anthropic_models_url, _anthropic_next_cursor

        endpoint = (base_url or "").strip() or self.base_url

        def _page(after_id: str | None):
            req = urllib.request.Request(_anthropic_models_url(endpoint, after_id=after_id))
            for k, v in (("x-api-key", api_key), ("anthropic-version", "2023-06-01"),
                         ("Accept", "application/json")):
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
            logger.debug("fetch_models(tencent-tokenplan): %s", exc)
            return None


tencent_tokenplan = TencentTokenPlanProfile(
    name="tencent-tokenplan", aliases=("tokenplan", "tencent-lkeap"),
    api_mode="anthropic_messages",
    display_name="Tencent TokenPlan",
    description="Tencent Cloud LKEAP Token Plan (anthropic-compatible tier)",
    env_vars=("TOKENPLAN_API_KEY",),
    base_url="https://api.lkeap.cloud.tencent.com/plan/anthropic", auth_type="api_key",
)

register_provider(tencent_tokenplan)
