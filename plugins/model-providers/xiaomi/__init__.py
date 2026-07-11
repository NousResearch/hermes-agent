"""Xiaomi MiMo provider profile.

Supports MiMo's native ``web_search`` tool when enabled via config::

    # config.yaml
    providers:
      xiaomi:
        web_search: true
        web_search_max_keyword: 3
        web_search_force: false
        web_search_location: ""   # e.g. "China", "US"
"""

from __future__ import annotations

import logging
from typing import Any

from providers import register_provider
from providers.base import ProviderProfile

logger = logging.getLogger(__name__)


class XiaomiProfile(ProviderProfile):
    """Xiaomi MiMo — optional native web_search tool injection."""

    def prepare_tools(
        self, tools: list[dict[str, Any]], *, model: str | None = None
    ) -> list[dict[str, Any]]:
        """Append MiMo's native ``web_search`` tool when configured.

        MiMo's web_search is an API-level tool that lets the model
        autonomously decide when to search, with citation support.
        It coexists with Hermes's own function tools — MiMo supports
        hybrid tool calling.
        """
        ws_cfg = self._web_search_config()
        if not ws_cfg:
            return tools

        ws_tool: dict[str, Any] = {
            "type": "web_search",
            "max_keyword": ws_cfg.get("max_keyword", 3),
            "force_search": ws_cfg.get("force", False),
        }
        location = ws_cfg.get("location", "")
        if location:
            ws_tool["user_location"] = {
                "type": "approximate",
                "country": location,
            }

        # Avoid duplicate injection if prepare_tools is called twice.
        for t in tools:
            if isinstance(t, dict) and t.get("type") == "web_search":
                return tools

        logger.debug("Xiaomi: injecting native web_search tool (force=%s, max_keyword=%s)",
                      ws_tool["force_search"], ws_tool["max_keyword"])
        return tools + [ws_tool]

    # ── Config access ──────────────────────────────────────────

    @staticmethod
    def _web_search_config() -> dict[str, Any] | None:
        """Read ``providers.xiaomi.web_search`` from config.yaml.

        Returns the config dict when web_search is truthy, else None.
        Catches all import/read errors so the profile never crashes
        the request path.
        """
        try:
            from hermes_cli.config import load_config_readonly
            cfg = load_config_readonly()
            providers = cfg.get("providers") or {}
            xiaomi_cfg = providers.get("xiaomi") or {}
            ws = xiaomi_cfg.get("web_search")
            if not ws:
                return None
            if ws is True:
                return {"max_keyword": 3, "force": False, "location": ""}
            if isinstance(ws, dict):
                return ws
            return None
        except Exception:
            return None


xiaomi = XiaomiProfile(
    name="xiaomi",
    aliases=("mimo", "xiaomi-mimo"),
    env_vars=("XIAOMI_API_KEY",),
    base_url="https://api.xiaomimimo.com/v1",
    supports_health_check=False,  # /v1/models returns 401 even with valid key
    supports_vision=True,  # mimo-v2-omni is vision-capable
    supports_vision_tool_messages=False,  # rejects list-type tool content (400 "text is not set")
)

register_provider(xiaomi)
