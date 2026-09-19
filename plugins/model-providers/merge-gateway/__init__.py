"""Merge Gateway provider profile."""

from __future__ import annotations

import json
import logging
from typing import Any
from urllib.parse import urlencode
import urllib.request

from providers import register_provider
from providers.base import ProviderProfile, _profile_user_agent

logger = logging.getLogger(__name__)

MERGE_GATEWAY_BASE_URL = "https://api-gateway.merge.dev/v1/openai"
MERGE_GATEWAY_MODELS_URL = "https://api-gateway.merge.dev/v1/models"


def _models_url_for_base(base_url: str | None) -> str:
    """Return the catalog URL corresponding to an inference base URL."""
    normalized = str(base_url or "").strip().rstrip("/")
    if not normalized or normalized == MERGE_GATEWAY_BASE_URL:
        return MERGE_GATEWAY_MODELS_URL
    if normalized.endswith("/openai"):
        normalized = normalized[: -len("/openai")]
    return normalized.rstrip("/") + "/models"


def _supports_agent_tools(item: dict[str, Any]) -> bool:
    """Whether at least one available Gateway vendor route supports tools."""
    if item.get("availability_status") == "unavailable":
        return False
    vendors = item.get("vendors")
    if not isinstance(vendors, dict):
        return False
    for route in vendors.values():
        if not isinstance(route, dict):
            continue
        if route.get("availability_status") == "unavailable":
            continue
        capabilities = route.get("capabilities")
        if (
            isinstance(capabilities, dict)
            and capabilities.get("supports_tool_calling") is True
        ):
            return True
    return False


class MergeGatewayProfile(ProviderProfile):
    """OpenAI-compatible Merge Gateway with its canonical model catalog."""

    def fetch_models(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 8.0,
    ) -> list[str] | None:
        """Fetch every tool-capable model from Gateway's paginated catalog."""
        if not api_key:
            return None

        catalog_url = _models_url_for_base(base_url)
        cursor: str | None = None
        seen_cursors: set[str] = set()
        model_ids: list[str] = []
        seen_models: set[str] = set()

        try:
            while True:
                query = {"limit": 500}
                if cursor:
                    query["cursor"] = cursor
                req = urllib.request.Request(
                    f"{catalog_url}?{urlencode(query)}",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Accept": "application/json",
                        "User-Agent": _profile_user_agent(),
                    },
                )

                from hermes_cli.urllib_security import open_credentialed_url

                with open_credentialed_url(req, timeout=timeout) as resp:
                    payload = json.loads(resp.read().decode())

                items = payload.get("data", []) if isinstance(payload, dict) else []
                for item in items:
                    if not isinstance(item, dict) or not _supports_agent_tools(item):
                        continue
                    model_id = item.get("model")
                    if not isinstance(model_id, str) or not model_id.strip():
                        continue
                    model_id = model_id.strip()
                    key = model_id.lower()
                    if key not in seen_models:
                        seen_models.add(key)
                        model_ids.append(model_id)

                if not isinstance(payload, dict) or not payload.get("has_more"):
                    break
                next_cursor = payload.get("next_cursor")
                if not isinstance(next_cursor, str) or not next_cursor:
                    break
                if next_cursor in seen_cursors:
                    break
                seen_cursors.add(next_cursor)
                cursor = next_cursor
        except (OSError, TimeoutError, UnicodeError, ValueError) as exc:
            logger.debug("fetch_models(merge-gateway): %s", exc)
            return None

        return model_ids


merge_gateway = MergeGatewayProfile(
    name="merge-gateway",
    display_name="Merge Gateway",
    description="Merge Gateway — multi-provider routing, governance, and observability",
    signup_url="https://gateway.merge.dev/api-keys",
    env_vars=("MERGE_GATEWAY_API_KEY",),
    base_url=MERGE_GATEWAY_BASE_URL,
    models_url=MERGE_GATEWAY_MODELS_URL,
    auth_type="api_key",
    api_mode="chat_completions",
)

register_provider(merge_gateway)
