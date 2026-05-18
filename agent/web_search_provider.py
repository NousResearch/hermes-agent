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

This ABC is the SINGLE plugin-facing surface for web providers — every
provider in the tree (brave-free, ddgs, searxng, exa, parallel, tavily,
firecrawl) implements it. The legacy in-tree ``tools.web_providers.base``
ABCs were deleted in PR #25182 along with the per-vendor inline helpers
in ``tools/web_tools.py``; the response-shape contract documented below
is preserved bit-for-bit so the tool wrapper does not have to translate.

Response shape (preserved from the legacy contract):

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
    """Abstract base class for a web search/extract/crawl backend.

    Subclasses must implement :meth:`is_available` and at least one of
    :meth:`search` / :meth:`extract` / :meth:`crawl`. The
    :meth:`supports_search` / :meth:`supports_extract` / :meth:`supports_crawl`
    capability flags let the registry route each tool call to the right
    provider, and let multi-capability providers (Firecrawl, Tavily, Exa,
    …) advertise multiple capabilities from a single class.
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
        """Return True if this provider implements :meth:`extract`.

        Both sync and async :meth:`extract` implementations are valid — the
        dispatcher detects coroutine functions via
        :func:`inspect.iscoroutinefunction` and awaits as needed. Sync
        implementations that perform blocking I/O (HTTP, SDK calls) should
        ideally wrap in :func:`asyncio.to_thread` at the call site; small
        providers can keep their sync shape and let the dispatcher handle
        threading.
        """
        return False

    def supports_crawl(self) -> bool:
        """Return True if this provider implements :meth:`crawl`.

        Crawl differs from extract in that the agent provides a *seed URL*
        and the provider walks linked pages on its own — useful for
        documentation sites where the agent doesn't know all relevant
        URLs upfront. Tavily is the only built-in backend that natively
        crawls today; Firecrawl provides a similar capability that we
        don't currently surface as a tool.

        Providers that don't crawl should leave this as False; the
        dispatcher in :func:`tools.web_tools.web_crawl_tool` will fall
        back to its auxiliary-model summarization path.
        """
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

    def extract(self, urls: List[str], **kwargs: Any) -> Any:
        """Extract content from one or more URLs.

        Override when :meth:`supports_extract` returns True. The default
        raises NotImplementedError; callers should gate on
        :meth:`supports_extract` before calling.

        Return shape: a list of result dicts matching what the legacy
        :func:`tools.web_tools.web_extract_tool` post-processing pipeline
        expects::

            [
                {
                    "url": str,
                    "title": str,
                    "content": str,
                    "raw_content": str,
                    "metadata": dict,           # optional
                    "error": str,               # optional, only on per-URL failure
                },
                ...
            ]

        Implementations MAY be ``async def`` — the dispatcher detects
        coroutines via :func:`inspect.iscoroutinefunction` and awaits.

        ``kwargs`` may carry forward-compat fields (``format``, ``include_raw``,
        ``max_chars``) — implementations should ignore unknown keys.
        """
        raise NotImplementedError(
            f"{self.name} does not support extract (override supports_extract)"
        )

    def crawl(self, url: str, **kwargs: Any) -> Any:
        """Crawl a seed URL and return results.

        Override when :meth:`supports_crawl` returns True. The default
        raises NotImplementedError; callers should gate on
        :meth:`supports_crawl` before calling.

        Return shape: ``{"results": [{"url": str, "title": str,
        "content": str, ...}, ...]}`` matching what
        :func:`tools.web_tools.web_crawl_tool` post-processing expects.

        Implementations MAY be ``async def``.

        ``kwargs`` may carry forward-compat fields (e.g. ``max_depth``,
        ``include_domains``) — implementations should ignore unknown keys.
        """
        raise NotImplementedError(
            f"{self.name} does not support crawl (override supports_crawl)"
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
