"""Camofox local-browser web search + extraction provider.

This provider bridges the generic ``web_search`` / ``web_extract`` tools to the
local Camofox REST API already used by Hermes' ``browser_*`` tools. Search uses
DuckDuckGo's lightweight HTML endpoint through Camofox and reads normalized
result data from the page DOM, with an accessibility-snapshot parser fallback
for older Camofox servers. Extraction navigates each URL in Camofox and returns
readable ``document.body.innerText`` content, falling back to an explicitly
labeled accessibility snapshot when DOM evaluation is unavailable.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List
from urllib.parse import parse_qs, quote_plus, unquote, urlparse

from agent.web_search_provider import WebSearchProvider
from tools.camofox_client import (
    camofox_tab_evaluate,
    camofox_tab_navigate,
    camofox_tab_snapshot,
    camofox_temporary_tab,
    get_camofox_url,
)

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT = 45
_SEARCH_TIMEOUT = 60
_MAX_EXTRACT_CHARS = 100_000

_DDG_RESULT_URL_RE = re.compile(r"/url:\s+([^\s]+duckduckgo\.com/l/\?[^\s]+)")
_HEADING_RE = re.compile(r"-\s+heading\s+[\"'](.*?)[\"']\s+\[level=2\]")
_LINK_TEXT_RE = re.compile(r"-\s+link\s+[\"'](.+?)[\"']")
_TEXT_RE = re.compile(r"-\s+text:\s+(.+)")
_DDG_SEARCH_EVALUATE = r"""
() => Array.from(document.querySelectorAll('.result, .web-result')).map((row) => {
  const link = row.querySelector('.result__a, a.result__a, h2 a, a[href]');
  const snippet = row.querySelector('.result__snippet, .result__body, .snippet');
  const href = link?.href || '';
  return {
    title: (link?.textContent || '').replace(/\s+/g, ' ').trim(),
    url: href,
    description: (snippet?.textContent || '').replace(/\s+/g, ' ').trim(),
  };
}).filter((item) => item.title || item.url)
""".strip()
_EXTRACT_READABLE_TEXT_EVALUATE = r"""
() => ({
  url: window.location.href,
  title: document.title || '',
  content: (document.body?.innerText || '').replace(/[ \t]+\n/g, '\n').replace(/\n{3,}/g, '\n\n').trim(),
})
""".strip()


def _decode_duckduckgo_url(raw_url: str) -> str:
    """Resolve a DuckDuckGo redirect URL to its target when possible."""
    value = raw_url.strip().strip('"').strip("'")
    if value.startswith("//"):
        value = f"https:{value}"
    parsed = urlparse(value)
    qs = parse_qs(parsed.query)
    target = qs.get("uddg", [""])[0]
    if target:
        return unquote(target)
    return value


def _clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _looks_like_display_url(text: str) -> bool:
    lowered = text.lower().strip()
    return (
        lowered.startswith("http://")
        or lowered.startswith("https://")
        or lowered.startswith("www.")
        or "/" in lowered and "." in lowered and " " not in lowered[:80]
    )


def parse_duckduckgo_snapshot(snapshot: str, limit: int) -> List[Dict[str, Any]]:
    """Parse Camofox's DuckDuckGo HTML accessibility snapshot into results."""
    lines = snapshot.splitlines()
    results: List[Dict[str, Any]] = []
    current_title = ""
    block_texts: List[str] = []

    def flush(url: str) -> None:
        nonlocal current_title, block_texts
        if not url or len(results) >= limit:
            return
        title = _clean_text(current_title) or url
        description = ""
        for candidate in block_texts:
            cleaned = _clean_text(candidate)
            if not cleaned or cleaned == title or _looks_like_display_url(cleaned):
                continue
            description = cleaned
            break
        results.append(
            {
                "title": title,
                "url": url,
                "description": description,
                "position": len(results) + 1,
            }
        )

    i = 0
    while i < len(lines) and len(results) < limit:
        line = lines[i]
        heading_match = _HEADING_RE.search(line)
        if not heading_match:
            i += 1
            continue

        current_title = heading_match.group(1)
        block_texts = []
        block_lines: List[str] = []
        i += 1
        while i < len(lines) and not _HEADING_RE.search(lines[i]):
            block_lines.append(lines[i])
            i += 1

        url = ""
        for block_line in block_lines:
            url_match = _DDG_RESULT_URL_RE.search(block_line)
            if url_match and not url:
                url = _decode_duckduckgo_url(url_match.group(1))
                continue
            link_match = _LINK_TEXT_RE.search(block_line)
            if link_match:
                block_texts.append(link_match.group(1))
                continue
            text_match = _TEXT_RE.search(block_line)
            if text_match:
                block_texts.append(text_match.group(1))

        flush(url)

    return results


def _normalize_search_items(items: Any, limit: int) -> List[Dict[str, Any]]:
    """Normalize DOM-evaluated search results to the web_search contract."""
    if not isinstance(items, list):
        return []

    results: List[Dict[str, Any]] = []
    for item in items:
        if len(results) >= limit:
            break
        if not isinstance(item, dict):
            continue
        url = _decode_duckduckgo_url(str(item.get("url") or ""))
        if not url:
            continue
        title = _clean_text(str(item.get("title") or "")) or url
        description = _clean_text(str(item.get("description") or ""))
        results.append(
            {
                "title": title,
                "url": url,
                "description": description,
                "position": len(results) + 1,
            }
        )
    return results


def _looks_like_unsupported_evaluate_error(exc: Exception) -> bool:
    text = str(exc).lower()
    missing_evaluate = "evaluate" in text and (
        "404" in text or "not found" in text or "unsupported" in text
    )
    return missing_evaluate or "old server" in text


def _extract_result_payload(data: Dict[str, Any], requested_url: str) -> Dict[str, str]:
    payload = data.get("result")
    if not isinstance(payload, dict):
        payload = {}
    source_url = _clean_text(str(payload.get("url") or requested_url)) or requested_url
    title = _clean_text(str(payload.get("title") or "")) or source_url
    content = str(payload.get("content") or "")[:_MAX_EXTRACT_CHARS]
    return {"url": source_url, "title": title, "content": content}


def _snapshot_result_payload(data: Dict[str, Any], requested_url: str) -> Dict[str, str]:
    source_url = str(data.get("url") or requested_url)
    snapshot = str(data.get("snapshot", ""))[:_MAX_EXTRACT_CHARS]
    title = _title_from_snapshot(snapshot, source_url)
    return {"url": source_url, "title": title, "content": snapshot}


def _format_extract_result(payload: Dict[str, str], content_type: str) -> Dict[str, Any]:
    return {
        "url": payload["url"],
        "title": payload["title"],
        "content": payload["content"],
        "raw_content": payload["content"],
        "metadata": {
            "sourceURL": payload["url"],
            "title": payload["title"],
            "backend": "camofox",
            "content_type": content_type,
        },
    }


def _title_from_snapshot(snapshot: str, fallback: str) -> str:
    for line in snapshot.splitlines()[:30]:
        heading_match = re.search(r"-\s+heading\s+[\"'](.+?)[\"']", line)
        if heading_match:
            return _clean_text(heading_match.group(1))
    return fallback


def _evaluate_or_snapshot(user_id: str, tab_id: str, expression: str) -> tuple[str, Dict[str, Any]]:
    """Evaluate JS, falling back to an accessibility snapshot on old servers."""
    try:
        return "evaluate", camofox_tab_evaluate(user_id, tab_id, expression, timeout=_DEFAULT_TIMEOUT)
    except Exception as exc:  # noqa: BLE001
        if not _looks_like_unsupported_evaluate_error(exc):
            raise
        return "snapshot", camofox_tab_snapshot(user_id, tab_id, timeout=_DEFAULT_TIMEOUT)


class CamofoxWebSearchProvider(WebSearchProvider):
    """Search and extract using the local Camofox browser server."""

    @property
    def name(self) -> str:
        return "camofox"

    @property
    def display_name(self) -> str:
        return "Camofox"

    def is_available(self) -> bool:
        """Return True when a Camofox server URL is configured."""
        return bool(get_camofox_url())

    def supports_search(self) -> bool:
        return True

    def supports_extract(self) -> bool:
        return True

    def search(self, query: str, limit: int = 5) -> Dict[str, Any]:
        """Execute search through Camofox and DuckDuckGo HTML results."""
        if not self.is_available():
            return {"success": False, "error": "CAMOFOX_URL is not set"}

        from tools.interrupt import is_interrupted

        if is_interrupted():
            return {"success": False, "error": "Interrupted"}

        safe_limit = max(1, min(int(limit or 5), 20))
        try:
            with camofox_temporary_tab("web_search") as (user_id, tab_id):
                search_url = f"https://html.duckduckgo.com/html/?q={quote_plus(query)}"
                camofox_tab_navigate(user_id, tab_id, search_url, timeout=_SEARCH_TIMEOUT)
                result_type, data = _evaluate_or_snapshot(user_id, tab_id, _DDG_SEARCH_EVALUATE)
                if result_type == "evaluate":
                    results = _normalize_search_items(data.get("result"), safe_limit)
                else:
                    results = parse_duckduckgo_snapshot(str(data.get("snapshot", "")), safe_limit)
            return {"success": True, "data": {"web": results}}
        except Exception as exc:  # noqa: BLE001
            logger.warning("Camofox search failed: %s", exc)
            return {"success": False, "error": f"Camofox search failed: {exc}"}

    def extract(self, urls: List[str], **kwargs: Any) -> List[Dict[str, Any]]:
        """Navigate to each URL in Camofox and return readable page text.

        Modern Camofox servers expose ``/evaluate``; this provider uses it to
        read ``document.body.innerText`` so ``web_extract`` receives page text.
        Older servers without evaluate support fall back to an accessibility
        snapshot and mark that explicitly in metadata.content_type.
        """
        if not self.is_available():
            return [
                {"url": url, "title": "", "content": "", "raw_content": "", "error": "CAMOFOX_URL is not set"}
                for url in urls
            ]

        from tools.interrupt import is_interrupted

        results: List[Dict[str, Any]] = []
        for url in urls:
            if is_interrupted():
                results.append({"url": url, "title": "", "content": "", "raw_content": "", "error": "Interrupted"})
                continue

            try:
                with camofox_temporary_tab("web_extract", url=url) as (user_id, tab_id):
                    result_type, data = _evaluate_or_snapshot(user_id, tab_id, _EXTRACT_READABLE_TEXT_EVALUATE)
                    if result_type == "evaluate":
                        payload = _extract_result_payload(data, url)
                        results.append(_format_extract_result(payload, "readable_text"))
                    else:
                        payload = _snapshot_result_payload(data, url)
                        results.append(_format_extract_result(payload, "accessibility_snapshot"))
            except Exception as exc:  # noqa: BLE001
                logger.warning("Camofox extract failed for %s: %s", url, exc)
                results.append(
                    {
                        "url": url,
                        "title": "",
                        "content": "",
                        "raw_content": "",
                        "error": f"Camofox extract failed: {exc}",
                    }
                )
        return results

    def get_setup_schema(self) -> Dict[str, Any]:
        return {
            "name": "Camofox",
            "badge": "local · no key",
            "tag": "Search and extract with the configured local Camofox browser server.",
            "post_setup": "camofox",
            "env_vars": [
                {
                    "key": "CAMOFOX_URL",
                    "prompt": "Camofox server URL",
                    "default": "http://localhost:9377",
                    "url": "https://github.com/jo-inc/camofox-browser",
                },
            ],
        }
