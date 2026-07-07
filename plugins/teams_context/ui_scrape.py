"""No-Graph Teams channel scrape helpers."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from html.parser import HTMLParser
from typing import Any

from plugins.teams_context.models import TeamsChatMessage, parse_graph_datetime, strip_html
from plugins.teams_context.store import TeamsContextStore


class TeamsUIScrapeError(RuntimeError):
    pass


@dataclass
class ParsedTeamsUIMessage:
    source_label: str
    message_id: str
    sender_name: str | None
    timestamp: datetime | None
    text: str
    html: str | None = None
    web_url: str | None = None
    channel_title: str | None = None
    thread_hint: str | None = None
    platform_id: str | None = None

    def to_store_message(self) -> TeamsChatMessage:
        raw = {
            "source": "teams_ui",
            "source_type": "channel",
            "source_label": self.source_label,
            "channel_title": self.channel_title,
            "thread_hint": self.thread_hint,
            "platform_id": self.platform_id,
        }
        return TeamsChatMessage(
            tenant_id=None,
            chat_id=self.source_label,
            message_id=self.message_id,
            sender_name=self.sender_name,
            created_at=self.timestamp,
            updated_at=self.timestamp,
            text=self.text,
            html=self.html,
            web_url=self.web_url,
            raw=raw,
        )


class _Node:
    def __init__(self, tag: str, attrs: dict[str, str], parent: "_Node | None" = None) -> None:
        self.tag = tag.lower()
        self.attrs = attrs
        self.parent = parent
        self.children: list[_Node] = []
        self.text_parts: list[str] = []

    def text(self) -> str:
        parts = list(self.text_parts)
        for child in self.children:
            parts.append(child.text())
        return _clean_text(" ".join(parts))

    def find_all(self, pred) -> list["_Node"]:
        found: list[_Node] = []
        if pred(self):
            found.append(self)
        for child in self.children:
            found.extend(child.find_all(pred))
        return found

    def attr_text(self) -> str:
        return " ".join(str(value) for value in self.attrs.values())


class _TeamsHTMLParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.root = _Node("document", {})
        self.stack = [self.root]

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        node = _Node(tag, {k.lower(): v or "" for k, v in attrs}, self.stack[-1])
        self.stack[-1].children.append(node)
        if tag.lower() not in {"br", "img", "input", "meta", "link"}:
            self.stack.append(node)

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        for idx in range(len(self.stack) - 1, 0, -1):
            if self.stack[idx].tag == tag:
                del self.stack[idx:]
                return

    def handle_data(self, data: str) -> None:
        if data.strip():
            self.stack[-1].text_parts.append(data)


def parse_teams_channel_html(
    html: str,
    *,
    label: str,
    since_days: int | None = None,
) -> list[ParsedTeamsUIMessage]:
    parser = _TeamsHTMLParser()
    parser.feed(html or "")
    cutoff = None
    if since_days is not None:
        cutoff = datetime.now(timezone.utc) - timedelta(days=max(0, int(since_days)))
    channel_title = _discover_channel_title(parser.root)
    containers = [
        node
        for node in parser.root.find_all(_looks_like_message_container)
        if not _has_message_ancestor(node)
    ]
    messages: list[ParsedTeamsUIMessage] = []
    seen: set[str] = set()
    for node in containers:
        parsed = _message_from_node(node, label=label, channel_title=channel_title)
        if parsed is None:
            continue
        if cutoff and parsed.timestamp and parsed.timestamp < cutoff:
            continue
        if parsed.message_id in seen:
            continue
        seen.add(parsed.message_id)
        messages.append(parsed)
    return messages


def scrape_open_teams_tab(
    *,
    label: str,
    max_scrolls: int,
    since_days: int,
    cdp_url: str = "http://127.0.0.1:9222",
) -> list[ParsedTeamsUIMessage]:
    try:
        from playwright.sync_api import sync_playwright
    except Exception as exc:
        raise TeamsUIScrapeError(
            "Teams UI scrape requires Playwright and a Chrome DevTools endpoint. "
            "Install Playwright or run with a Hermes environment that includes it."
        ) from exc

    with sync_playwright() as pw:
        try:
            browser = pw.chromium.connect_over_cdp(cdp_url)
        except Exception as exc:
            raise TeamsUIScrapeError(
                "Could not connect to Chrome DevTools at http://127.0.0.1:9222. "
                "Start Chrome with remote debugging enabled, log into Teams, and open the target channel."
            ) from exc
        pages = [page for context in browser.contexts for page in context.pages]
        teams_pages = [page for page in pages if "teams.microsoft.com" in (page.url or "")]
        if not teams_pages:
            raise TeamsUIScrapeError(
                "No Microsoft Teams tab found. Open the target Teams channel in Chrome first."
            )
        page = teams_pages[0]
        for _ in range(max(0, int(max_scrolls))):
            page.evaluate(
                """
                () => {
                  const scroller = document.querySelector('[data-tid="message-pane-list-runway"]')
                    || document.querySelector('[role="log"]')
                    || document.scrollingElement
                    || document.body;
                  scroller.scrollTop = 0;
                }
                """
            )
            page.wait_for_timeout(450)
        html = page.content()
    messages = parse_teams_channel_html(html, label=label, since_days=since_days)
    if not messages:
        raise TeamsUIScrapeError(
            "No recognizable Teams channel messages were found in the open tab. "
            "Open a channel conversation view and try a larger --max-scrolls value."
        )
    return messages


def scrape_and_store(
    *,
    label: str,
    max_scrolls: int,
    since_days: int,
    store: TeamsContextStore,
    cdp_url: str = "http://127.0.0.1:9222",
) -> dict[str, Any]:
    messages = scrape_open_teams_tab(
        label=label,
        max_scrolls=max_scrolls,
        since_days=since_days,
        cdp_url=cdp_url,
    )
    for message in messages:
        store.upsert_message(message.to_store_message())
    return {"label": label, "stored": len(messages), "source_type": "channel"}


def _looks_like_message_container(node: _Node) -> bool:
    attrs = node.attr_text().lower()
    if node.tag not in {"div", "li", "article"}:
        return False
    return (
        node.attrs.get("role") in {"article", "listitem"}
        or "message" in attrs
        or "chat-message" in attrs
    ) and len(node.text()) >= 3


def _has_message_ancestor(node: _Node) -> bool:
    parent = node.parent
    while parent is not None:
        if _looks_like_message_container(parent):
            return True
        parent = parent.parent
    return False


def _message_from_node(
    node: _Node,
    *,
    label: str,
    channel_title: str | None,
) -> ParsedTeamsUIMessage | None:
    text = node.text()
    if not text:
        return None
    sender = _first_text(node, ("message-author", "author", "sender"))
    timestamp_text = _first_time_text(node)
    timestamp = _parse_timestamp(timestamp_text)
    body_node = _first_node(node, ("messagebody", "message-body", "message content", "message-text"))
    body_text = body_node.text() if body_node else text
    html_body = None
    platform_id = (
        node.attrs.get("data-mid")
        or node.attrs.get("data-message-id")
        or node.attrs.get("id")
    )
    web_url = _first_link(node)
    thread_hint = _first_text(node, ("reply", "thread"))
    message_id = platform_id or _synthetic_ui_message_id(label, timestamp_text, sender, body_text)
    return ParsedTeamsUIMessage(
        source_label=label,
        message_id=str(message_id),
        sender_name=sender,
        timestamp=timestamp,
        text=body_text,
        html=html_body,
        web_url=web_url,
        channel_title=channel_title,
        thread_hint=thread_hint,
        platform_id=platform_id,
    )


def _first_node(node: _Node, needles: tuple[str, ...]) -> _Node | None:
    for child in node.find_all(lambda candidate: _has_attr_needle(candidate, needles)):
        if child is not node:
            return child
    return None


def _first_text(node: _Node, needles: tuple[str, ...]) -> str | None:
    found = _first_node(node, needles)
    if found:
        return found.text() or None
    return None


def _first_time_text(node: _Node) -> str | None:
    for candidate in node.find_all(lambda item: item.tag == "time"):
        for key in ("datetime", "title", "aria-label"):
            if candidate.attrs.get(key):
                return candidate.attrs[key]
        if candidate.text():
            return candidate.text()
    for candidate in node.find_all(lambda item: _has_attr_needle(item, ("timestamp", "time"))):
        value = candidate.attrs.get("datetime") or candidate.attrs.get("title") or candidate.attrs.get("aria-label")
        return value or candidate.text() or None
    return None


def _first_link(node: _Node) -> str | None:
    for candidate in node.find_all(lambda item: item.tag == "a"):
        href = candidate.attrs.get("href")
        if href and ("teams.microsoft.com" in href or "message" in href.lower()):
            return href
    return None


def _has_attr_needle(node: _Node, needles: tuple[str, ...]) -> bool:
    haystack = node.attr_text().lower().replace("_", "-")
    return any(needle in haystack for needle in needles)


def _discover_channel_title(root: _Node) -> str | None:
    for node in root.find_all(lambda item: item.tag in {"h1", "h2"}):
        text = node.text()
        if text:
            return text
    return None


def _parse_timestamp(value: str | None) -> datetime | None:
    if not value:
        return None
    text = _clean_text(value)
    try:
        return parse_graph_datetime(text)
    except Exception:
        pass
    for fmt in ("%m/%d/%Y, %I:%M %p", "%b %d, %Y %I:%M %p", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(text, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


def _synthetic_ui_message_id(label: str, timestamp: str | None, sender: str | None, text: str) -> str:
    material = json.dumps(
        {
            "label": label,
            "timestamp": timestamp,
            "sender": sender,
            "text": text,
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    return "teams_ui:" + hashlib.sha256(material.encode("utf-8")).hexdigest()[:24]


def _clean_text(value: str) -> str:
    text = strip_html(value)
    text = re.sub(r"\s+", " ", text)
    return text.strip()
