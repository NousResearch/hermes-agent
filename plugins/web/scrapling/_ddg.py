"""DDG Lite result parsing — pure stdlib, shared by the worker and the provider.

The module deliberately imports nothing from the Hermes runtime: the sidecar
worker runs in a different Python environment (the Scrapling venv), so this
file must be importable from both sides. It also has no Scrapling dependency,
so the provider can reuse the parser in its own process.
"""

from __future__ import annotations

import html as _html
import re
import urllib.parse

# Attribute order is not stable across DDG's markup, so tags are matched
# first and their attributes parsed afterwards.
_ANCHOR_RE = re.compile(r"<a\b([^>]*)>(.*?)</a>", re.DOTALL | re.IGNORECASE)
_CELL_RE = re.compile(r"<td\b([^>]*)>(.*?)</td>", re.DOTALL | re.IGNORECASE)
_ATTR_RE = re.compile(r"""\b(href|class)\s*=\s*(['"])(.*?)\2""", re.DOTALL | re.IGNORECASE)
_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")


def _attrs(tag_attrs: str) -> dict[str, str]:
    """``href``/``class`` values from a raw tag-attribute string."""
    out: dict[str, str] = {}
    for match in _ATTR_RE.finditer(tag_attrs or ""):
        out[match.group(1).lower()] = match.group(3)
    return out


def _has_class(value: str, wanted: str) -> bool:
    return wanted in (value or "").split()


def _norm(text: str) -> str:
    """Collapse whitespace runs (source markup wraps snippets across lines)."""
    return _WS_RE.sub(" ", text or "").strip()


def _text_of(markup: str) -> str:
    return _norm(_html.unescape(_TAG_RE.sub("", markup or "")))


def unwrap_uddg(href: str) -> str:
    """DDG wraps result links as ``//duckduckgo.com/l/?uddg=<encoded>``."""
    if not href:
        return ""
    if href.startswith("//"):
        href = "https:" + href
    query = urllib.parse.urlparse(href).query
    wrapped = urllib.parse.parse_qs(query).get("uddg")
    return wrapped[0] if wrapped else href


def parse_ddg_lite_results(html: str) -> list[dict]:
    """Parse DDG Lite result rows into ``{url, title, description, position}``.

    Each result is three sibling table cells: the anchor (``.result-link``),
    the snippet (``.result-snippet``) and the display host (``.link-text``).
    A result without a snippet is skipped — a title alone carries no
    description.
    """
    if not html:
        return []
    anchors: list[tuple[str, str]] = []
    for match in _ANCHOR_RE.finditer(html):
        attrs = _attrs(match.group(1))
        if not _has_class(attrs.get("class", ""), "result-link"):
            continue
        href = _html.unescape(attrs.get("href", ""))
        if href:
            anchors.append((href, _text_of(match.group(2))))

    snippets: list[str] = []
    for match in _CELL_RE.finditer(html):
        attrs = _attrs(match.group(1))
        if _has_class(attrs.get("class", ""), "result-snippet"):
            snippets.append(_text_of(match.group(2)))

    rows: list[dict] = []
    for index, (href, title) in enumerate(anchors):
        if index >= len(snippets):
            break
        description = snippets[index]
        if not title or not description:
            continue
        rows.append({
            "url": unwrap_uddg(href),
            "title": title,
            "description": description,
            "position": len(rows) + 1,
        })
    return rows
