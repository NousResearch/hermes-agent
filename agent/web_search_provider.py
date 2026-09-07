"""Web Search Provider ABC.

The single plugin-facing surface every web provider (brave-free, ddgs, searxng,
exa, parallel, tavily, keenable, firecrawl) implements; registered via
``PluginContext.register_web_search_provider()`` and selected by
``web.search_backend`` / ``web.extract_backend`` / ``web.backend``.

Response shapes (legacy contract, the tool wrapper does not translate)::

    search:  {"success": True, "data": {"web": [{"title", "url", "description", "position"}, ...]}}
    extract: {"success": True, "data": [{"url", "title", "content", "raw_content", "metadata"}, ...]}
    failure: {"success": False, "error": str}
"""

from __future__ import annotations

import abc
import os
from typing import Any, Dict, List

from agent.provider_base import ProviderBase


def get_provider_env(name: str) -> str:
    """Config-aware env lookup (``os.environ`` first, then ``~/.hermes/.env``) so
    credentials set through the config layer are visible in gateway sessions /
    delegate children / subprocess runs. Stripped value, or ``""`` when unset.

    Falls back to a bare ``os.getenv`` when the config module is unavailable (stripped installs, early
    import contexts). See #40190.
    """
    try:
        from hermes_cli.config import get_env_value

        val = get_env_value(name)
    except Exception:  # noqa: BLE001 — config layer optional here
        val = None
    if val is None:
        val = os.getenv(name, "")
    return (val or "").strip()


class WebSearchProvider(ProviderBase):
    """Abstract base class for a web search/extract backend: implement :meth:`is_available`
    and at least one of :meth:`search` / :meth:`extract`; the ``supports_*`` flags route each capability."""

    @abc.abstractmethod
    def is_available(self) -> bool:
        """True when this provider can service calls. Cheap check only (env var, importable
        dep, instance URL) — NO network; runs at tool registration and on every ``hermes tools`` paint."""

    def supports_search(self) -> bool:
        """True if this provider implements :meth:`search`."""
        return True

    def is_keyless_available(self) -> bool:
        """True when this provider can serve calls WITHOUT credentials (public anonymous
        free tiers such as Exa / Parallel MCP); used only when NO provider is configured or
        keyed. Must never make :meth:`is_available` True, or the legacy preference walk would
        route keyed users onto a higher-priority backend's free tier. Cheap, no network."""
        return False

    def supports_extract(self) -> bool:
        """True if this provider implements :meth:`extract` (sync or ``async def`` —
        the dispatcher awaits coroutine functions)."""
        return False

    def search(self, query: str, limit: int = 5) -> Dict[str, Any]:
        """Execute a web search. Callers gate on :meth:`supports_search`."""
        raise NotImplementedError(
            f"{self.name} does not support search (override supports_search)"
        )

    def extract(self, urls: List[str], **kwargs: Any) -> Any:
        """Extract content from URLs (callers gate on :meth:`supports_extract`); may be ``async def``.
        Returns ``[{"url", "title", "content", "raw_content", "metadata"?, "error"?}, ...]`` (``error``
        only on per-URL failure). Ignore unknown ``kwargs`` (``format``, ``include_raw``, ``max_chars``)."""
        raise NotImplementedError(
            f"{self.name} does not support extract (override supports_extract)"
        )


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.
from typing import Optional  # noqa: F401,E402
# ---- END PLUGIN-COMPAT ----
