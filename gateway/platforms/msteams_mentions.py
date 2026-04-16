from __future__ import annotations

import html
import re
from typing import Any, Iterable

from markdown_it import MarkdownIt

_MENTION_TAG_RE = re.compile(r"<at[^>]*>(.*?)</at>", re.IGNORECASE | re.DOTALL)
_LEADING_MENTION_BLOCK_RE = re.compile(r"^\s*(?:<p[^>]*>\s*)?(?:<at[^>]*>.*?</at>\s*)+", re.IGNORECASE | re.DOTALL)

AI_GENERATED_ENTITY = {
    "type": "https://schema.org/Message",
    "@type": "Message",
    "@id": "",
    "additionalType": ["AIGeneratedContent"],
}

_TEAMS_MARKDOWN = MarkdownIt("commonmark", {"breaks": True, "html": False})


def strip_leading_teams_mentions(text: str) -> str:
    if not text:
        return ""
    return _LEADING_MENTION_BLOCK_RE.sub("", text, count=1)


def strip_teams_mentions(text: str) -> str:
    if not text:
        return ""
    cleaned = _MENTION_TAG_RE.sub(lambda match: f" {match.group(1)} ", text)
    cleaned = re.sub(r'<blockquote[^>]*itemtype=["\']http://schema\.skype\.com/Reply["\'][^>]*>.*?</blockquote>', " ", cleaned, flags=re.IGNORECASE | re.DOTALL)
    cleaned = re.sub(r"<br\s*/?>", "\n", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"</p\s*>", "\n", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"<[^>]+>", " ", cleaned)
    cleaned = html.unescape(cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def to_teams_html(content: str) -> str:
    text = (content or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return text.replace("\n", "<br>")


def build_mention_text_and_entities(content: str, mentions: Iterable[dict[str, Any]] | None) -> tuple[str, list[dict[str, Any]]]:
    text = content or ""
    entities: list[dict[str, Any]] = []
    replacements: list[tuple[str, str]] = []
    for index, mention in enumerate(mentions or []):
        mention_id = str(mention.get("id") or "").strip()
        mention_name = str(mention.get("name") or "").strip()
        if not mention_id or not mention_name:
            continue
        token = f"@[{mention_name}]({mention_id})"
        tag = f"<at>{mention_name}</at>"
        placeholder = f"__HERMES_TEAMS_MENTION_{index}__"
        if token in text:
            text = text.replace(token, placeholder)
            replacements.append((placeholder, tag))
            entities.append(
                {
                    "type": "mention",
                    "text": tag,
                    "mentioned": {"id": mention_id, "name": mention_name},
                }
            )
    if text:
        text = to_teams_html(text)
        for placeholder, tag in replacements:
            text = text.replace(placeholder, tag)
    if text or entities:
        entities.append(dict(AI_GENERATED_ENTITY))
    return text, entities


def build_adaptive_card_attachment(card: dict[str, Any]) -> dict[str, Any]:
    return {
        "contentType": "application/vnd.microsoft.card.adaptive",
        "content": card,
    }


def build_poll_card(question: str, options: list[str], max_selections: int = 1, poll_id: str | None = None) -> dict[str, Any]:
    normalized_options = [str(option).strip() for option in options if str(option).strip()]
    if len(normalized_options) < 2:
        raise ValueError("Teams polls require at least two options")
    limit = max(1, min(int(max_selections or 1), len(normalized_options)))
    effective_poll_id = str(poll_id or "hermes-poll")
    helper_text = "Select one option." if limit == 1 else f"Select up to {limit} options."
    return {
        "type": "AdaptiveCard",
        "version": "1.5",
        "body": [
            {
                "type": "TextBlock",
                "text": question,
                "wrap": True,
                "weight": "Bolder",
                "size": "Medium",
            },
            {
                "type": "Input.ChoiceSet",
                "id": "choices",
                "isMultiSelect": limit > 1,
                "style": "expanded",
                "choices": [
                    {"title": option, "value": str(index)}
                    for index, option in enumerate(normalized_options)
                ],
            },
            {
                "type": "TextBlock",
                "text": helper_text,
                "wrap": True,
                "isSubtle": True,
                "spacing": "Small",
            },
        ],
        "actions": [
            {
                "type": "Action.Submit",
                "title": "Vote",
                "data": {
                    "hermesPollId": effective_poll_id,
                    "pollId": effective_poll_id,
                },
                "msteams": {
                    "type": "messageBack",
                    "text": "hermes poll vote",
                    "displayText": "Vote recorded",
                    "value": {
                        "hermesPollId": effective_poll_id,
                        "pollId": effective_poll_id,
                    },
                },
            }
        ],
    }


def extract_activity_text(activity: dict[str, Any]) -> str:
    text = str(activity.get("text") or "").strip()
    if text:
        return strip_teams_mentions(text)
    value = activity.get("value")
    if isinstance(value, dict):
        for key in ("text", "message", "content"):
            if value.get(key):
                return strip_teams_mentions(str(value[key]))
    return ""
