"""
Web Search Provider ABC
=======================

Defines the pluggable-backend interface for web search and content extraction.
Providers register instances via ``PluginContext.register_web_search_provider()``;
the active one (selected via ``web.search_backend`` / ``web.extract_backend`` /
``web.backend`` in ``config.yaml``) services every ``web_search`` /
``web_extract`` tool call.

Providers live in ``<repo>/plugins/web/<name>/`` (built-in, auto-loaded as
``kind: backend``) or ``~/.hermes/plugins/web/<name>/`` (user, opt-in via
``plugins.enabled``).

This ABC is the plugin-facing surface. The legacy
:mod:`tools.web_providers.base` module retains its own ABCs for in-tree
consumers that haven't migrated yet; over time those will all flow through
this provider.

Response shape (mirrors the legacy contract in ``tools/web_providers/base.py``
so the tool wrapper does not have to translate):

Search results::

    {
        "success": True,
        "data": {
            "web": [
                {"title": str, "url": str, "description": str, "position": int},
                ...
            ]
        }
    }

Extract results::

    {
        "success": True,
        "data": [
            {"url": str, "title": str, "content": str,
             "raw_content": str, "metadata": dict},
            ...
        ]
    }

On failure (either capability)::

    {"success": False, "error": str}
"""

from __future__ import annotations

import abc
from typing import Any, Dict, List


# ---------------------------------------------------------------------------
# ABC
# ---------------------------------------------------------------------------


class WebSearchProvider(abc.ABC):
    """Abstract base class for a web search/extract backend.

    Subclasses must implement :meth:`is_available` and at least one of
    :meth:`search` / :meth:`extract`. The :meth:`supports_search` and
    :meth:`supports_extract` capability flags let the registry route each
    tool call to the right provider, and let multi-capability providers
    (SearXNG, Firecrawl, Tavily, …) advertise both.
    """

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Stable short identifier used in ``web.search_backend`` /
        ``web.extract_backend`` / ``web.backend`` config keys.

        Lowercase, no spaces; hyphens permitted to preserve existing
        user-visible names. Examples: ``brave-free``, ``ddgs``,
        ``searxng``, ``firecrawl``.
        """

    @property
    def display_name(self) -> str:
        """Human-readable label shown in ``hermes tools``. Defaults to ``name``."""
        return self.name

    @abc.abstractmethod
    def is_available(self) -> bool:
        """Return True when this provider can service calls.

        Typically a cheap check (env var present, optional Python dep
        importable, instance URL set). Must NOT make network calls — this
        runs at tool-registration time and on every ``hermes tools`` paint.
        """

    def supports_search(self) -> bool:
        """Return True if this provider implements :meth:`search`."""
        return True

    def supports_extract(self) -> bool:
        """Return True if this provider implements :meth:`extract`."""
        return False

    def search(self, query: str, limit: int = 5) -> Dict[str, Any]:
        """Execute a web search.

        Override when :meth:`supports_search` returns True. The default
        raises NotImplementedError; callers should gate on
        :meth:`supports_search` before calling.
        """
        raise NotImplementedError(
            f"{self.name} does not support search (override supports_search)"
        )

    def extract(self, urls: List[str], **kwargs: Any) -> Dict[str, Any]:
        """Extract content from one or more URLs.

        Override when :meth:`supports_extract` returns True. The default
        raises NotImplementedError; callers should gate on
        :meth:`supports_extract` before calling.

        ``kwargs`` may carry forward-compat fields (e.g. ``include_raw``,
        ``max_chars``) — implementations should ignore unknown keys.
        """
        raise NotImplementedError(
            f"{self.name} does not support extract (override supports_extract)"
        )

    def get_setup_schema(self) -> Dict[str, Any]:
        """Return provider metadata for the ``hermes tools`` picker.

        Used by ``hermes_cli/tools_config.py`` to inject this provider as a
        row in the Web Search / Web Extract picker. Shape::

            {
                "name": "Brave Search (Free)",
                "badge": "free",
                "tag": "No paid tier needed — uses Brave's free API.",
                "env_vars": [
                    {"key": "BRAVE_SEARCH_API_KEY",
                     "prompt": "Brave Search API key",
                     "url": "https://brave.com/search/api/"},
                ],
            }

        Default: minimal entry derived from ``display_name``. Override to
        expose API key prompts, badges, and instance URL fields.
        """
        return {
            "name": self.display_name,
            "badge": "",
            "tag": "",
            "env_vars": [],
        }
