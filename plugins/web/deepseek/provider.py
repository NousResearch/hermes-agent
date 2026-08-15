"""Selection marker for DeepSeek's in-turn native Responses web search."""

from __future__ import annotations

from typing import Any, Dict

from agent.web_search_provider import WebSearchProvider


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
        # This is a selection marker for the active DeepSeek inference route,
        # not an independently dispatched search client. It is available only
        # after explicit user opt-in; this keeps it out of registry fallback
        # while allowing the legacy web dispatcher to honor the more specific
        # ``web.search_backend`` key instead of silently choosing another
        # installed backend.
        try:
            from hermes_cli.config import load_config_readonly

            config = load_config_readonly()
            web = config.get("web") if isinstance(config, dict) else None
            if not isinstance(web, dict):
                return False
            configured = web.get("search_backend") or web.get("backend") or ""
            return str(configured).strip().lower() == self.name
        except Exception:
            return False

    def supports_search(self) -> bool:
        return True

    def supports_extract(self) -> bool:
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
            "name": self.display_name,
            "badge": "native · search only",
            "tag": (
                "Uses the active capability-enabled DeepSeek inference route; "
                "configure DeepSeek under Models first."
            ),
            # Inference credentials are owned by the model-provider setup and
            # may come from its credential pool rather than a singleton env
            # key. Selecting native search must not prompt for or duplicate
            # that credential in the web-provider flow.
            "env_vars": [],
        }
