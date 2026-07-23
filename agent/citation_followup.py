"""Citation follow-up support — store, detect, and resolve citation requests.

When Hermes generates a response using web search results, citations are
currently static text. This module enables users to follow up with questions
about specific cited sources (e.g. "tell me more about source [3]").

Normal flow
-----------
1. A turn produces a response citing URLs [1], [2], [3].
2. The caller (e.g. turn finalizer) calls ``CitationStore.store_citations()``
   with the citation metadata so it's available for the next user message.
3. On the next turn, ``CitationStore.detect_followup()`` checks whether the
   user is asking about a specific citation index.
4. If yes, ``CitationStore.resolve_source()`` fetches the full page content
   via the web_extract tool and returns it as enriched context.

Usage
-----
    store = CitationStore()
    citations = [{"url": "https://...", "title": "..."}, ...]
    store.store_citations(citations)

    idx = store.detect_followup("tell me more about source [2]")
    if idx is not None:
        content = await store.resolve_source(idx)
        # inject content into context
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Pattern matching citation references in user messages.
# Matches:
#   "source [3]"  "source 3"  "[3]"  "cite [1]"  "citation 2"
#   "source number 4"  "the first source" (via int-word mapping below)
# The reference number is captured as group 1.
CITATION_FOLLOWUP_RE = re.compile(
    r"""
    (?:
        (?:source|citation|cite|reference|ref|link)\s*
        (?:
            \[?(\d+)\]?          # "source 3" / "source [3]" / "cite 1"
            |
            (?:number\s+)?(\d+)  # "source number 4"
        )
    )
    |
    \[(\d+)\]                    # bare "[3]" not preceded by a known keyword
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Ordinal-to-int mapping for "the first source", "second citation", etc.
_ORDINAL_MAP: Dict[str, int] = {
    "first": 1,
    "second": 2,
    "third": 3,
    "fourth": 4,
    "fifth": 5,
    "sixth": 6,
    "seventh": 7,
    "eighth": 8,
    "ninth": 9,
    "tenth": 10,
}

_ORDINAL_RE = re.compile(
    r"(?:the\s+)?(" + "|".join(_ORDINAL_MAP) + r")\s+(?:source|citation|cite|reference|ref|link)",
    re.IGNORECASE,
)


class CitationStore:
    """Store cited URLs from the last assistant turn and detect follow-ups.

    Thread-safe for the single-agent case: all access happens on the same
    turn-processing thread. The agent creates one ``CitationStore`` instance
    (stored as ``agent._citation_store``) and reuses it across turns.
    """

    def __init__(self) -> None:
        # Ordered list of citation entries: each entry is a dict with
        # ``url``, ``title``, and optionally ``description``.
        self._citations: List[Dict[str, str]] = []
        # The query / prompt that produced these citations (diagnostic only).
        self._last_query: str = ""

    # -- Public API -----------------------------------------------------------

    def store_citations(
        self,
        citations: List[Dict[str, str]],
        query: str = "",
    ) -> None:
        """Store citation metadata from the last assistant turn.

        Accepts the citation format returned by Hermes web search tools::

            [
                {
                    "title": "Example Page",
                    "url": "https://example.com/article",
                    "description": "...",
                    "position": 1,
                },
                ...
            ]

        The positional index from the web search result is preserved as
        ``position`` if present; otherwise the list index (1-based) is used.

        Args:
            citations: List of citation dicts. Each must have at least a
                ``url`` key. ``title`` is optional but recommended.
            query: The search query that generated these citations (optional).
        """
        self._citations = []
        self._last_query = query

        for i, c in enumerate(citations or []):
            if not isinstance(c, dict):
                continue
            url = (c.get("url") or "").strip()
            if not url:
                continue
            self._citations.append({
                "url": url,
                "title": (c.get("title") or "").strip(),
                "description": (c.get("description") or "").strip(),
                "position": c.get("position", i + 1),
            })
        logger.debug(
            "CitationStore: stored %d citations (query=%r)",
            len(self._citations),
            query[:80] if query else "",
        )

    def detect_followup(self, user_message: str) -> Optional[int]:
        """Check if *user_message* is asking about a specific citation.

        Returns the 0-based index into :attr:`_citations`, or ``None`` if no
        citation reference is detected.

        Recognised patterns:
        - ``"source [N]"`` / ``"source N"``
        - ``"[N]"`` (bare bracket reference)
        - ``"citation N"`` / ``"cite N"`` / ``"reference N"`` / ``"ref N"`` / ``"link N"``
        - ``"the first source"``, ``"second citation"`` (ordinals up to tenth)
        """
        if not self._citations:
            return None

        message = user_message.strip()
        if not message:
            return None

        # 1. Try ordinal patterns first ("the first source", "second citation").
        ordinal_match = _ORDINAL_RE.search(message)
        if ordinal_match:
            ordinal_word = ordinal_match.group(1).lower()
            idx = _ORDINAL_MAP[ordinal_word] - 1  # to 0-based
            if 0 <= idx < len(self._citations):
                return idx

        # 2. Try numbered patterns.
        match = CITATION_FOLLOWUP_RE.search(message)
        if match:
            # Groups: the digit may be in group 1, 2, or 3.
            num_str = match.group(1) or match.group(2) or match.group(3)
            if num_str:
                idx = int(num_str) - 1  # to 0-based
                if 0 <= idx < len(self._citations):
                    return idx

        return None

    async def resolve_source(self, index: int) -> Optional[str]:
        """Fetch the full content of the citation at *index*.

        Uses the ``web_extract`` tool (via the module's async helper) to
        retrieve the page content and returns it as a markdown-formatted
        string suitable for injection into the model context.

        Args:
            index: 0-based index into the stored citations.

        Returns:
            A markdown-formatted string with the source title, URL, and
            fetched content, or ``None`` if *index* is out of range or the
            fetch fails.
        """
        if index < 0 or index >= len(self._citations):
            return None

        entry = self._citations[index]
        url = entry["url"]
        title = entry.get("title") or url

        logger.debug("CitationStore: resolving source %d (%s)", index + 1, url)

        try:
            content = await _async_web_extract(url)
        except Exception as exc:
            logger.warning("CitationStore: web_extract failed for %s: %s", url, exc)
            content = None

        if content:
            header = f"## Source [{index + 1}]: {title}"
            if url:
                header += f"\nURL: {url}"
            return f"{header}\n\n{content}"

        # Fallback: return just the metadata.
        header = f"## Source [{index + 1}]: {title}"
        desc = entry.get("description", "")
        if desc:
            header += f"\n\nDescription: {desc}"
        header += f"\nURL: {url}"
        if content is None:
            header += "\n\n*Full content could not be fetched.*"
        return header

    @property
    def citations(self) -> List[Dict[str, str]]:
        """Read-only access to the stored citations."""
        return list(self._citations)

    @property
    def citation_count(self) -> int:
        """Number of stored citations."""
        return len(self._citations)

    def clear(self) -> None:
        """Clear all stored citations."""
        self._citations = []
        self._last_query = ""

    def get_source_summary(self) -> str:
        """Return a compact text summary of all stored sources.

        Suitable for including in the system prompt so the model is aware of
        available citation follow-up sources on the next turn.
        """
        if not self._citations:
            return ""
        lines = [
            "Available cited sources (the user may ask about any of these):",
        ]
        for i, c in enumerate(self._citations):
            title = c.get("title", "")
            desc = c.get("description", "")
            if title:
                lines.append(f"  [{i + 1}] {title}")
            else:
                lines.append(f"  [{i + 1}] {c['url']}")
            if desc and len(desc) < 200:
                lines.append(f"      {desc}")
        return "\n".join(lines)


# -- Module-level utilities --------------------------------------------------


async def _async_web_extract(url: str) -> Optional[str]:
    """Fetch the full text content of *url* using web_extract tool.

    Runs the synchronous ``web_extract_tool`` in a thread pool executor so
    this module can be used from async agent contexts.
    """
    try:
        from tools.web_tools import web_extract_tool as _sync_extract
    except ImportError:
        logger.error("CitationStore: cannot import web_extract_tool")
        return None

    loop = asyncio.get_running_loop()
    try:
        result = await loop.run_in_executor(
            None,
            lambda: _sync_extract([url]),
        )
    except Exception as exc:
        logger.warning("CitationStore: web_extract raised %s: %s", type(exc).__name__, exc)
        return None

    # web_extract_tool returns a JSON string.
    try:
        data = json.loads(result) if isinstance(result, str) else result
    except (json.JSONDecodeError, TypeError):
        logger.warning("CitationStore: web_extract returned non-JSON result")
        return None

    if not isinstance(data, dict):
        logger.warning("CitationStore: web_extract returned unexpected type: %s", type(data).__name__)
        return None

    # The response has either a "results" list or a "data" dict.
    results = data.get("results") or data.get("data", {}).get("results") or []
    if not results and data.get("success") is False:
        error = data.get("error", "unknown error")
        logger.warning("CitationStore: web_extract failed: %s", error)
        return None

    if isinstance(results, list) and results:
        content = results[0].get("content", "")
        if content:
            return content[:15000]  # cap at 15K chars

    # Fallback: return the raw text if present.
    return data.get("content") or data.get("text") or None
