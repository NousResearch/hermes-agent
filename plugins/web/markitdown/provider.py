"""MarkItDown extract provider — plugin form.

Subclasses :class:`agent.web_search_provider.WebSearchProvider`. Uses
Microsoft's MarkItDown library to locally convert HTML pages to clean
markdown. No API key, no cloud dependency — purely self-hosted.

Extract-only — ``supports_search()`` returns False. Pair with any search
provider (brave-free, ddgs, searxng, etc.) for ``web_search``.

Config keys this provider responds to::

    web:
      extract_backend: "markitdown"  # explicit per-capability
      backend: "markitdown"          # shared fallback

No env vars required. Ensure the ``markitdown`` Python package is installed:

    pip install markitdown

or via the bundled lazy-install mechanism (``hermes tools`` will prompt).
"""

from __future__ import annotations

import io
import logging
from typing import Any, Dict, List, Optional

import httpx

from agent.web_search_provider import WebSearchProvider

logger = logging.getLogger(__name__)

# Timeout per-URL fetch (seconds).
_FETCH_TIMEOUT = 30


def _markitdown_package_available() -> bool:
    """Return True when the ``markitdown`` package is importable.

    Cheap — Python caches the import after the first call. No network I/O.
    """
    try:
        import markitdown  # noqa: F401
        return True
    except ImportError:
        return False


class MarkItDownExtractProvider(WebSearchProvider):
    """Extract web page content to markdown via Microsoft's MarkItDown.

    Fetches each URL with httpx, converts the HTML response to markdown
    using the local ``markitdown`` library, and returns the result.
    """

    @property
    def name(self) -> str:
        return "markitdown"

    @property
    def display_name(self) -> str:
        return "MarkItDown"

    def is_available(self) -> bool:
        """Return True when the ``markitdown`` package is importable."""
        return _markitdown_package_available()

    def supports_search(self) -> bool:
        return False

    def supports_extract(self) -> bool:
        return True

    def extract(self, urls: List[str], **kwargs: Any) -> List[Dict[str, Any]]:
        """Fetch URLs and convert to markdown using MarkItDown.

        Args:
            urls: List of URL strings to extract content from.
            **kwargs: Forward-compat extras (``format``, etc.) — ignored.

        Returns:
            List of result dicts, one per URL, in the same order::

                [
                    {
                        "url": str,
                        "title": str,
                        "content": str,       # markdown output
                        "raw_content": str,   # same as content
                        "metadata": dict,
                        "error": str,         # only on per-URL failure
                    },
                    ...
                ]
        """
        from markitdown import MarkItDown

        md = MarkItDown()
        results: List[Dict[str, Any]] = []

        for url in urls:
            result: Dict[str, Any] = {"url": url, "title": "", "content": "", "raw_content": "", "metadata": {}}
            try:
                # Fetch the URL with httpx.
                resp = httpx.get(url, timeout=_FETCH_TIMEOUT, follow_redirects=True)
                resp.raise_for_status()

                # Convert the response content to markdown.
                converted = md.convert(io.BytesIO(resp.content))

                title = converted.title or ""
                text_content = converted.text_content or ""

                result["title"] = title
                result["content"] = text_content
                result["raw_content"] = text_content

                logger.info(
                    "MarkItDown extracted %s (%d chars) from %s",
                    title[:60] if title else "(no title)",
                    len(text_content),
                    url,
                )

            except httpx.TimeoutException as exc:
                msg = f"Timeout fetching {url}: {exc}"
                logger.warning("MarkItDown %s", msg)
                result["error"] = msg

            except httpx.HTTPStatusError as exc:
                msg = f"HTTP {exc.response.status_code} fetching {url}"
                logger.warning("MarkItDown %s", msg)
                result["error"] = msg

            except httpx.RequestError as exc:
                msg = f"Request failed for {url}: {exc}"
                logger.warning("MarkItDown %s", msg)
                result["error"] = msg

            except Exception as exc:  # noqa: BLE001
                msg = f"MarkItDown conversion failed for {url}: {exc}"
                logger.warning("%s", msg)
                result["error"] = msg

            results.append(result)

        return results

    def get_setup_schema(self) -> Dict[str, Any]:
        return {
            "name": "MarkItDown",
            "badge": "free · self-hosted",
            "tag": (
                "Local HTML-to-markdown via Microsoft's MarkItDown library. "
                "No API key needed. Extract-only — pair with any search provider."
            ),
            "env_vars": [],
            "post_setup": "markitdown",
        }
