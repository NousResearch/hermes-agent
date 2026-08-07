"""Selection marker for DeepSeek's in-turn native Responses web search."""

from __future__ import annotations

from typing import Any, Dict

from agent.web_search_provider import WebSearchProvider, get_provider_env


class DeepSeekWebSearchProvider(WebSearchProvider):
    """Expose ``web.search_backend: deepseek`` to the provider registry.

    Search execution happens inside an active, capability-enabled DeepSeek
    Responses request. A local dispatch means the active route cannot provide
    that native capability, so return an actionable error rather than silently
    falling back.

    This backend is intentionally opt-in because a server-side search turn can
    consume substantially more tokens than a normal model turn::

        model:
          provider: deepseek
          default: deepseek-v4-flash
        web:
          search_backend: deepseek
    """

    @property
    def name(self) -> str:
        return "deepseek"

    @property
    def display_name(self) -> str:
        return "DeepSeek Native Web Search"

    def is_available(self) -> bool:
        return bool(get_provider_env("DEEPSEEK_API_KEY"))

    def supports_search(self) -> bool:
        return True

    def supports_extract(self) -> bool:
        return False

    def supports_auto_detection(self) -> bool:
        # This provider is a selection marker for an in-turn server-side tool,
        # not a local search implementation. A DEEPSEEK_API_KEY alone must not
        # replace the user's normal client-side web backend.
        return False

    def search(self, query: str, limit: int = 5) -> Dict[str, Any]:
        del query, limit
        from hermes_cli.providers import deepseek_native_web_search_models

        enabled_models = ", ".join(deepseek_native_web_search_models()) or "none"
        return {
            "success": False,
            "error": (
                "DeepSeek native web search requires an active DeepSeek model "
                "with both Responses API and native web-search capabilities "
                f"enabled (currently: {enabled_models}). Select a client web "
                "backend such as Firecrawl or Tavily for other models."
            ),
        }

    def get_setup_schema(self) -> Dict[str, Any]:
        return {
            "env_vars": [
                {
                    "name": "DEEPSEEK_API_KEY",
                    "prompt": "DeepSeek API key",
                    "secret": True,
                    "required": True,
                }
            ]
        }
