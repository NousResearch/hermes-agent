"""RSS 2.0 / Atom parsing — stdlib only, no network."""

from __future__ import annotations

import re
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any
from xml.etree import ElementTree as ET

_ATOM_NS = {"atom": "http://www.w3.org/2005/Atom"}
_DC_NS = {"dc": "http://purl.org/dc/elements/1.1/"}


def _strip_bom(text: str) -> str:
    return text.lstrip("\ufeff").strip()


def _parse_datetime(raw: str) -> datetime | None:
    value = (raw or "").strip()
    if not value:
        return None
    try:
        dt = parsedate_to_datetime(value)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except (TypeError, ValueError, IndexError):
        pass
    for fmt in (
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
    ):
        try:
            dt = datetime.strptime(value[:19], fmt[: len(value[:19]) + 2])
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)
        except ValueError:
            continue
    iso = value.replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(iso)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except ValueError:
        return None


def _text(node: ET.Element | None) -> str:
    if node is None:
        return ""
    return re.sub(r"\s+", " ", "".join(node.itertext())).strip()


def _link_rss(item: ET.Element) -> str:
    for child in item:
        tag = child.tag.split("}")[-1] if "}" in child.tag else child.tag
        if tag == "link" and (child.text or "").strip():
            return (child.text or "").strip()
        if tag == "guid" and (child.text or "").strip().startswith("http"):
            return (child.text or "").strip()
    return ""


def _link_atom(entry: ET.Element) -> str:
    for link in entry.findall("atom:link", _ATOM_NS):
        href = link.get("href") or ""
        rel = (link.get("rel") or "alternate").lower()
        if href and rel in {"alternate", ""}:
            return href
    for link in entry.findall("link"):
        href = link.get("href") or ""
        if href:
            return href
    return ""


def parse_feed_xml(xml_text: str) -> list[dict[str, Any]]:
    """Parse RSS 2.0 or Atom feed body into normalized entries."""
    cleaned = _strip_bom(xml_text)
    if not cleaned:
        return []
    root = ET.fromstring(cleaned)
    tag = root.tag.split("}")[-1] if "}" in root.tag else root.tag

    if tag == "feed":
        return _parse_atom(root)
    if tag == "rss":
        channel = root.find("channel")
        if channel is None:
            return []
        return _parse_rss_channel(channel)
    return []


def _parse_rss_channel(channel: ET.Element) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in channel.findall("item"):
        title = _text(item.find("title"))
        link = _link_rss(item)
        pub = _text(item.find("pubDate")) or _text(item.find("date"))
        if not pub:
            dc_date = item.find("dc:date", _DC_NS)
            if dc_date is not None:
                pub = _text(dc_date)
        published = _parse_datetime(pub)
        summary = _text(item.find("description"))
        rows.append(
            {
                "title": title,
                "url": link,
                "published_at": published.isoformat() if published else "",
                "published_dt": published,
                "summary": summary[:500],
            }
        )
    return rows


def _parse_atom(feed: ET.Element) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for entry in feed.findall("atom:entry", _ATOM_NS) or feed.findall("entry"):
        title = _text(entry.find("atom:title", _ATOM_NS) or entry.find("title"))
        link = _link_atom(entry)
        pub_el = (
            entry.find("atom:published", _ATOM_NS)
            or entry.find("atom:updated", _ATOM_NS)
            or entry.find("published")
            or entry.find("updated")
        )
        pub = _text(pub_el)
        published = _parse_datetime(pub)
        summary_el = (
            entry.find("atom:summary", _ATOM_NS)
            or entry.find("atom:content", _ATOM_NS)
            or entry.find("summary")
            or entry.find("content")
        )
        summary = _text(summary_el)
        rows.append(
            {
                "title": title,
                "url": link,
                "published_at": published.isoformat() if published else "",
                "published_dt": published,
                "summary": summary[:500],
            }
        )
    return rows


def filter_entries_since(
    entries: list[dict[str, Any]],
    *,
    since: datetime,
    max_items: int = 20,
) -> list[dict[str, Any]]:
    """Keep entries at or after ``since`` (UTC); include undated at end."""
    since_utc = since.astimezone(timezone.utc)
    dated: list[tuple[datetime, dict[str, Any]]] = []
    undated: list[dict[str, Any]] = []
    for row in entries:
        dt = row.get("published_dt")
        if isinstance(dt, datetime):
            if dt >= since_utc:
                dated.append((dt, row))
        else:
            undated.append(row)
    dated.sort(key=lambda pair: pair[0], reverse=True)
    out = [row for _, row in dated[:max_items]]
    if len(out) < max_items:
        out.extend(undated[: max(0, max_items - len(out))])
    return out
