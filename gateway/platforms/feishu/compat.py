"""Compatibility helpers for the pre-SDK Feishu adapter surface."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence

from gateway.platforms.feishu.types import FeishuMentionRef


FALLBACK_POST_TEXT = "[Rich text message]"
FALLBACK_FORWARD_TEXT = "[Merged forward message]"
FALLBACK_SHARE_CHAT_TEXT = "[Shared chat]"
FALLBACK_INTERACTIVE_TEXT = "[Interactive message]"
FALLBACK_IMAGE_TEXT = "[Image]"
FALLBACK_ATTACHMENT_TEXT = "[Attachment]"

_PREFERRED_LOCALES = ("zh_cn", "en_us")
_MARKDOWN_SPECIAL_CHARS_RE = re.compile(r"([\\`*_{}\[\]()#+\-!|>~])")
_MENTION_PLACEHOLDER_RE = re.compile(r"@_user_\d+")
_WHITESPACE_RE = re.compile(r"\s+")
_MULTISPACE_RE = re.compile(r"[ \t]{2,}")
_SUPPORTED_CARD_TEXT_KEYS = (
    "title",
    "text",
    "content",
    "label",
    "value",
    "name",
    "summary",
    "subtitle",
    "description",
    "placeholder",
    "hint",
)
_SKIP_TEXT_KEYS = {
    "tag",
    "type",
    "msg_type",
    "message_type",
    "chat_id",
    "open_chat_id",
    "share_chat_id",
    "file_key",
    "image_key",
    "user_id",
    "open_id",
    "union_id",
    "url",
    "href",
    "link",
    "token",
    "template",
    "locale",
}


@dataclass(frozen=True)
class FeishuPostMediaRef:
    file_key: str
    file_name: str = ""
    resource_type: str = "file"


@dataclass(frozen=True)
class FeishuPostParseResult:
    text_content: str
    image_keys: list[str] = field(default_factory=list)
    media_refs: list[FeishuPostMediaRef] = field(default_factory=list)


@dataclass(frozen=True)
class FeishuNormalizedMessage:
    raw_type: str
    text_content: str
    preferred_message_type: str = "text"
    image_keys: list[str] = field(default_factory=list)
    media_refs: list[FeishuPostMediaRef] = field(default_factory=list)
    mentions: list[FeishuMentionRef] = field(default_factory=list)
    relation_kind: str = "plain"
    metadata: Dict[str, Any] = field(default_factory=dict)


def _load_feishu_payload(raw_content: str) -> Dict[str, Any]:
    try:
        parsed = json.loads(raw_content) if raw_content else {}
    except json.JSONDecodeError:
        return {"text": raw_content}
    return parsed if isinstance(parsed, dict) else {"content": parsed}


def _escape_markdown_text(text: str) -> str:
    return _MARKDOWN_SPECIAL_CHARS_RE.sub(r"\\\1", text)


def _style_enabled(style: Any, key: str) -> bool:
    if isinstance(style, dict):
        return style.get(key) is True or style.get(key) == 1 or style.get(key) == "true"
    if isinstance(style, (list, tuple, set)):
        return key in style
    return False


def _wrap_inline_code(text: str) -> str:
    max_run = max([0, *[len(run) for run in re.findall(r"`+", text)]])
    fence = "`" * (max_run + 1)
    body = f" {text} " if text.startswith("`") or text.endswith("`") else text
    return f"{fence}{body}{fence}"


def _render_code_block_element(element: Dict[str, Any]) -> str:
    language = str(element.get("language", "") or element.get("lang", "") or "").strip()
    code = str(element.get("text", "") or element.get("content", "") or "").replace("\r\n", "\n")
    trailing_newline = "" if code.endswith("\n") else "\n"
    return f"```{language}\n{code}{trailing_newline}```"


def _render_text_element(element: Dict[str, Any]) -> str:
    text = str(element.get("text", "") or "")
    style = element.get("style")
    if _style_enabled(style, "code"):
        return _wrap_inline_code(text)
    rendered = _escape_markdown_text(text)
    if not rendered:
        return ""
    if _style_enabled(style, "bold"):
        rendered = f"**{rendered}**"
    if _style_enabled(style, "italic"):
        rendered = f"*{rendered}*"
    if _style_enabled(style, "underline"):
        rendered = f"<u>{rendered}</u>"
    if _style_enabled(style, "strikethrough"):
        rendered = f"~~{rendered}~~"
    return rendered


def _extract_mention_ids(mention: Any) -> tuple[str, str]:
    direct_open_id = str(getattr(mention, "open_id", "") or "")
    direct_user_id = str(getattr(mention, "user_id", "") or "")
    if direct_open_id or direct_user_id:
        return direct_open_id, direct_user_id
    mention_id = getattr(mention, "id", None)
    if isinstance(mention_id, dict):
        return str(mention_id.get("open_id", "") or ""), str(mention_id.get("user_id", "") or "")
    if isinstance(mention_id, str):
        id_type = str(getattr(mention, "id_type", "") or "").lower()
        return (mention_id, "") if id_type == "open_id" else ("", mention_id)
    if mention_id is None:
        return "", ""
    return (
        str(getattr(mention_id, "open_id", "") or ""),
        str(getattr(mention_id, "user_id", "") or ""),
    )


def _mention_is_all(mention: Any) -> bool:
    if isinstance(mention, dict):
        if mention.get("key") == "@_all":
            return True
        mention_id = mention.get("id") if isinstance(mention.get("id"), dict) else {}
        return bool(mention_id and mention_id.get("user_id") == "all")
    if getattr(mention, "key", None) == "@_all":
        return True
    mention_id = getattr(mention, "id", None)
    return mention_id is not None and getattr(mention_id, "user_id", None) == "all"


def _mention_key(mention: Any) -> str:
    if isinstance(mention, dict):
        return str(mention.get("key", "") or "")
    return str(getattr(mention, "key", "") or "")


def _mention_name(mention: Any) -> str:
    if isinstance(mention, dict):
        return str(mention.get("name", "") or "")
    return str(getattr(mention, "name", "") or "")


def _bot_matches(bot: Any, *, open_id: str, user_id: str, name: str) -> bool:
    if bot is None:
        return False
    matcher = getattr(bot, "matches", None)
    if callable(matcher):
        return bool(matcher(open_id=open_id, user_id=user_id, name=name))
    bot_open_id = str(getattr(bot, "open_id", "") or "")
    bot_user_id = str(getattr(bot, "user_id", "") or "")
    bot_name = str(getattr(bot, "name", "") or getattr(bot, "display_name", "") or "")
    if open_id and bot_open_id:
        return open_id == bot_open_id
    if user_id and bot_user_id:
        return user_id == bot_user_id
    return bool(bot_name and name == bot_name)


def _build_mentions_map(
    mentions: Optional[Sequence[Any]],
    bot: Any = None,
) -> Dict[str, FeishuMentionRef]:
    out: Dict[str, FeishuMentionRef] = {}
    for mention in mentions or []:
        key = _mention_key(mention)
        if _mention_is_all(mention):
            out[key or "@_all"] = FeishuMentionRef(is_all=True)
            continue
        open_id, user_id = _extract_mention_ids(mention)
        name = _mention_name(mention).strip()
        ref = FeishuMentionRef(
            name=name,
            open_id=open_id,
            is_self=_bot_matches(bot, open_id=open_id, user_id=user_id, name=name),
        )
        if key:
            out[key] = ref
    return out


def _normalize_feishu_text(
    text: str,
    mentions_map: Optional[Dict[str, FeishuMentionRef]] = None,
) -> str:
    def _sub(match: "re.Match[str]") -> str:
        ref = (mentions_map or {}).get(match.group(0))
        if ref is None:
            return " "
        name = ref.name or ref.open_id or "user"
        return f"@{name}"

    cleaned = _MENTION_PLACEHOLDER_RE.sub(_sub, text or "")
    cleaned = cleaned.replace("@_all", "@all")
    cleaned = cleaned.replace("\r\n", "\n").replace("\r", "\n")
    cleaned = "\n".join(_WHITESPACE_RE.sub(" ", line).strip() for line in cleaned.split("\n"))
    cleaned = "\n".join(line for line in cleaned.split("\n") if line)
    cleaned = _MULTISPACE_RE.sub(" ", cleaned)
    return cleaned.strip()


def _resolve_locale_payload(payload: Any) -> Dict[str, Any]:
    direct = _to_post_payload(payload)
    if direct:
        return direct
    if not isinstance(payload, dict):
        return {}
    for key in _PREFERRED_LOCALES:
        candidate = _to_post_payload(payload.get(key))
        if candidate:
            return candidate
    for value in payload.values():
        candidate = _to_post_payload(value)
        if candidate:
            return candidate
    return {}


def _to_post_payload(candidate: Any) -> Dict[str, Any]:
    if not isinstance(candidate, dict):
        return {}
    content = candidate.get("content")
    if not isinstance(content, list):
        return {}
    return {"title": str(candidate.get("title", "") or ""), "content": content}


def _resolve_post_payload(payload: Any) -> Dict[str, Any]:
    direct = _to_post_payload(payload)
    if direct:
        return direct
    if not isinstance(payload, dict):
        return {}
    wrapped = payload.get("post")
    wrapped_direct = _resolve_locale_payload(wrapped)
    if wrapped_direct:
        return wrapped_direct
    return _resolve_locale_payload(payload)


def _render_post_element(
    element: Any,
    image_keys: list[str],
    media_refs: list[FeishuPostMediaRef],
    mentions_map: Optional[Dict[str, FeishuMentionRef]] = None,
) -> str:
    if isinstance(element, str):
        return element
    if not isinstance(element, dict):
        return ""

    tag = str(element.get("tag", "") or "").strip().lower()
    if tag == "text":
        return _render_text_element(element)
    if tag == "a":
        href = str(element.get("href", "") or "").strip()
        label = str(element.get("text", href) or "").strip()
        if not label:
            return ""
        escaped_label = _escape_markdown_text(label)
        return f"[{escaped_label}]({href})" if href else escaped_label
    if tag == "at":
        placeholder = str(element.get("user_id", "") or "").strip()
        if placeholder == "@_all":
            if mentions_map is not None and "@_all" not in mentions_map:
                mentions_map["@_all"] = FeishuMentionRef(is_all=True)
            return "@all"
        ref = (mentions_map or {}).get(placeholder)
        display_name = ref.name or ref.open_id or "user" if ref else str(
            element.get("user_name", "") or "user"
        )
        return f"@{_escape_markdown_text(display_name)}"
    if tag in {"img", "image"}:
        image_key = str(element.get("image_key", "") or "").strip()
        if image_key and image_key not in image_keys:
            image_keys.append(image_key)
        alt = str(element.get("text", "") or element.get("alt", "") or "").strip()
        return f"[Image: {alt}]" if alt else "[Image]"
    if tag in {"media", "file", "audio", "video"}:
        file_key = str(element.get("file_key", "") or "").strip()
        file_name = str(
            element.get("file_name", "")
            or element.get("title", "")
            or element.get("text", "")
            or ""
        ).strip()
        if file_key:
            media_refs.append(
                FeishuPostMediaRef(
                    file_key=file_key,
                    file_name=file_name,
                    resource_type=tag if tag in {"audio", "video"} else "file",
                )
            )
        return f"[Attachment: {file_name}]" if file_name else "[Attachment]"
    if tag in {"emotion", "emoji"}:
        label = str(element.get("text", "") or element.get("emoji_type", "") or "").strip()
        return f":{_escape_markdown_text(label)}:" if label else "[Emoji]"
    if tag == "br":
        return "\n"
    if tag in {"hr", "divider"}:
        return "\n\n---\n\n"
    if tag == "code":
        code = str(element.get("text", "") or element.get("content", "") or "")
        return _wrap_inline_code(code) if code else ""
    if tag in {"code_block", "pre"}:
        return _render_code_block_element(element)

    nested_parts: list[str] = []
    for key in ("text", "title", "content", "children", "elements"):
        extracted = _render_nested_post(element.get(key), image_keys, media_refs, mentions_map)
        if extracted:
            nested_parts.append(extracted)
    return " ".join(part for part in nested_parts if part)


def _render_nested_post(
    value: Any,
    image_keys: list[str],
    media_refs: list[FeishuPostMediaRef],
    mentions_map: Optional[Dict[str, FeishuMentionRef]] = None,
) -> str:
    if isinstance(value, str):
        return _escape_markdown_text(value)
    if isinstance(value, list):
        return " ".join(
            part
            for item in value
            for part in [_render_nested_post(item, image_keys, media_refs, mentions_map)]
            if part
        )
    if isinstance(value, dict):
        direct = _render_post_element(value, image_keys, media_refs, mentions_map)
        if direct:
            return direct
        return " ".join(
            part
            for item in value.values()
            for part in [_render_nested_post(item, image_keys, media_refs, mentions_map)]
            if part
        )
    return ""


def parse_feishu_post_payload(
    payload: Any,
    *,
    mentions_map: Optional[Dict[str, FeishuMentionRef]] = None,
) -> FeishuPostParseResult:
    resolved = _resolve_post_payload(payload)
    if not resolved:
        return FeishuPostParseResult(text_content=FALLBACK_POST_TEXT)

    image_keys: list[str] = []
    media_refs: list[FeishuPostMediaRef] = []
    parts: list[str] = []
    title = _normalize_feishu_text(str(resolved.get("title", "") or "").strip())
    if title:
        parts.append(title)
    for row in resolved.get("content", []) or []:
        if not isinstance(row, list):
            continue
        row_text = _normalize_feishu_text(
            "".join(
                _render_post_element(item, image_keys, media_refs, mentions_map)
                for item in row
            )
        )
        if row_text:
            parts.append(row_text)
    return FeishuPostParseResult(
        text_content="\n".join(parts).strip() or FALLBACK_POST_TEXT,
        image_keys=image_keys,
        media_refs=media_refs,
    )


def _media_ref_from_payload(payload: Dict[str, Any], *, resource_type: str) -> FeishuPostMediaRef:
    return FeishuPostMediaRef(
        file_key=str(
            payload.get("file_key")
            or payload.get("image_key")
            or payload.get("media_key")
            or payload.get("key")
            or ""
        ).strip(),
        file_name=str(
            payload.get("file_name")
            or payload.get("name")
            or payload.get("title")
            or payload.get("text")
            or ""
        ).strip(),
        resource_type=resource_type if resource_type in {"audio", "video"} else "file",
    )


def _attachment_placeholder(file_name: str) -> str:
    normalized_name = _normalize_feishu_text(file_name)
    return f"[Attachment: {normalized_name}]" if normalized_name else FALLBACK_ATTACHMENT_TEXT


def _normalize_merge_forward_message(payload: Dict[str, Any]) -> FeishuNormalizedMessage:
    title = _first_non_empty_text(
        payload.get("title"),
        payload.get("summary"),
        payload.get("preview"),
        _find_first_text(payload, keys=("title", "summary", "preview", "description")),
    )
    entries = _collect_forward_entries(payload)
    lines: list[str] = []
    if title:
        lines.append(title)
    lines.extend(entries[:8])
    return FeishuNormalizedMessage(
        raw_type="merge_forward",
        text_content="\n".join(lines).strip() or FALLBACK_FORWARD_TEXT,
        relation_kind="merge_forward",
        metadata={"entry_count": len(entries), "title": title},
    )


def _normalize_share_chat_message(payload: Dict[str, Any]) -> FeishuNormalizedMessage:
    chat_name = _first_non_empty_text(
        payload.get("chat_name"),
        payload.get("name"),
        payload.get("title"),
        _find_first_text(payload, keys=("chat_name", "name", "title")),
    )
    share_id = _first_non_empty_text(
        payload.get("chat_id"),
        payload.get("open_chat_id"),
        payload.get("share_chat_id"),
    )
    lines = [f"Shared chat: {chat_name}" if chat_name else FALLBACK_SHARE_CHAT_TEXT]
    if share_id:
        lines.append(f"Chat ID: {share_id}")
    return FeishuNormalizedMessage(
        raw_type="share_chat",
        text_content="\n".join(lines),
        relation_kind="share_chat",
        metadata={"chat_id": share_id, "chat_name": chat_name},
    )


def _normalize_interactive_message(message_type: str, payload: Dict[str, Any]) -> FeishuNormalizedMessage:
    card_payload = payload.get("card") if isinstance(payload.get("card"), dict) else payload
    title = _first_non_empty_text(
        _find_header_title(card_payload),
        payload.get("title"),
        _find_first_text(card_payload, keys=("title", "summary", "subtitle")),
    )
    body_lines = _collect_card_lines(card_payload)
    actions = _collect_action_labels(card_payload)
    lines: list[str] = []
    if title:
        lines.append(title)
    for line in body_lines:
        if line != title:
            lines.append(line)
    if actions:
        lines.append(f"Actions: {', '.join(actions)}")
    return FeishuNormalizedMessage(
        raw_type=message_type,
        text_content="\n".join(lines[:12]).strip() or FALLBACK_INTERACTIVE_TEXT,
        relation_kind="interactive",
        metadata={"title": title, "actions": actions},
    )


def _collect_forward_entries(payload: Dict[str, Any]) -> list[str]:
    candidates: list[Any] = []
    for key in ("messages", "items", "message_list", "records", "content"):
        value = payload.get(key)
        if isinstance(value, list):
            candidates.extend(value)
    entries: list[str] = []
    for item in candidates:
        if not isinstance(item, dict):
            text = _normalize_feishu_text(str(item or ""))
            if text:
                entries.append(f"- {text}")
            continue
        sender = _first_non_empty_text(
            item.get("sender_name"),
            item.get("user_name"),
            item.get("sender"),
            item.get("name"),
        )
        nested_type = str(item.get("message_type", "") or item.get("msg_type", "")).strip().lower()
        if nested_type == "post":
            body = parse_feishu_post_payload(item.get("content") or item).text_content
        else:
            body = _first_non_empty_text(
                item.get("text"),
                item.get("summary"),
                item.get("preview"),
                item.get("content"),
                _find_first_text(item, keys=("text", "content", "summary", "preview", "title")),
            )
        body = _normalize_feishu_text(body)
        if sender and body:
            entries.append(f"- {sender}: {body}")
        elif body:
            entries.append(f"- {body}")
    return _unique_lines(entries)


def _collect_card_lines(payload: Any) -> list[str]:
    lines = _collect_text_segments(payload, in_rich_block=False)
    normalized = [_normalize_feishu_text(line) for line in lines]
    return _unique_lines([line for line in normalized if line])


def _collect_action_labels(payload: Any) -> list[str]:
    labels: list[str] = []
    for item in _walk_nodes(payload):
        if not isinstance(item, dict):
            continue
        tag = str(item.get("tag", "") or item.get("type", "")).strip().lower()
        if tag not in {"button", "select_static", "overflow", "date_picker", "picker"}:
            continue
        label = _first_non_empty_text(
            item.get("text"),
            item.get("name"),
            item.get("value"),
            _find_first_text(item, keys=("text", "content", "name", "value")),
        )
        if label:
            labels.append(label)
    return _unique_lines(labels)


def _collect_text_segments(value: Any, *, in_rich_block: bool) -> list[str]:
    if isinstance(value, str):
        return [_normalize_feishu_text(value)] if in_rich_block else []
    if isinstance(value, list):
        segments: list[str] = []
        for item in value:
            segments.extend(_collect_text_segments(item, in_rich_block=in_rich_block))
        return segments
    if not isinstance(value, dict):
        return []

    tag = str(value.get("tag", "") or value.get("type", "")).strip().lower()
    next_in_rich_block = in_rich_block or tag in {
        "plain_text",
        "lark_md",
        "markdown",
        "note",
        "div",
        "column_set",
        "column",
        "action",
        "button",
        "select_static",
        "date_picker",
    }
    segments: list[str] = []
    for key in _SUPPORTED_CARD_TEXT_KEYS:
        item = value.get(key)
        if isinstance(item, str) and next_in_rich_block:
            normalized = _normalize_feishu_text(item)
            if normalized:
                segments.append(normalized)
    for key, item in value.items():
        if key in _SKIP_TEXT_KEYS:
            continue
        segments.extend(_collect_text_segments(item, in_rich_block=next_in_rich_block))
    return segments


def _find_header_title(payload: Any) -> str:
    if not isinstance(payload, dict):
        return ""
    header = payload.get("header")
    if not isinstance(header, dict):
        return ""
    title = header.get("title")
    if isinstance(title, dict):
        return _first_non_empty_text(title.get("content"), title.get("text"), title.get("name"))
    return _normalize_feishu_text(str(title or ""))


def _find_first_text(payload: Any, *, keys: tuple[str, ...]) -> str:
    for node in _walk_nodes(payload):
        if not isinstance(node, dict):
            continue
        for key in keys:
            value = node.get(key)
            if isinstance(value, str):
                normalized = _normalize_feishu_text(value)
                if normalized:
                    return normalized
    return ""


def _walk_nodes(value: Any):
    if isinstance(value, dict):
        yield value
        for item in value.values():
            yield from _walk_nodes(item)
    elif isinstance(value, list):
        for item in value:
            yield from _walk_nodes(item)


def _first_non_empty_text(*values: Any) -> str:
    for value in values:
        if isinstance(value, str):
            normalized = _normalize_feishu_text(value)
            if normalized:
                return normalized
        elif value is not None and not isinstance(value, (dict, list)):
            normalized = _normalize_feishu_text(str(value))
            if normalized:
                return normalized
    return ""


def _unique_lines(lines: list[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for line in lines:
        if not line or line in seen:
            continue
        seen.add(line)
        unique.append(line)
    return unique


def normalize_feishu_message(
    *,
    message_type: str,
    raw_content: str,
    mentions: Optional[Sequence[Any]] = None,
    bot: Any = None,
) -> FeishuNormalizedMessage:
    """Normalize a legacy Feishu raw message payload."""
    normalized_type = str(message_type or "").strip().lower()
    payload = _load_feishu_payload(raw_content)
    mentions_map = _build_mentions_map(mentions, bot)

    if normalized_type == "text":
        text = str(payload.get("text", "") or "")
        if "@_all" in text and "@_all" not in mentions_map:
            mentions_map["@_all"] = FeishuMentionRef(is_all=True)
        return FeishuNormalizedMessage(
            raw_type=normalized_type,
            text_content=_normalize_feishu_text(text, mentions_map),
            mentions=list(mentions_map.values()),
        )
    if normalized_type == "post":
        parsed_post = parse_feishu_post_payload(payload, mentions_map=mentions_map)
        return FeishuNormalizedMessage(
            raw_type=normalized_type,
            text_content=parsed_post.text_content,
            image_keys=list(parsed_post.image_keys),
            media_refs=list(parsed_post.media_refs),
            mentions=list(mentions_map.values()),
            relation_kind="post",
        )

    mention_refs = list(mentions_map.values())
    if normalized_type == "image":
        image_key = str(payload.get("image_key", "") or "").strip()
        alt_text = _normalize_feishu_text(
            str(payload.get("text", "") or payload.get("alt", "") or FALLBACK_IMAGE_TEXT),
            mentions_map,
        )
        return FeishuNormalizedMessage(
            raw_type=normalized_type,
            text_content=alt_text if alt_text != FALLBACK_IMAGE_TEXT else "",
            preferred_message_type="photo",
            image_keys=[image_key] if image_key else [],
            relation_kind="image",
            mentions=mention_refs,
        )
    if normalized_type in {"file", "audio", "media", "video"}:
        media_ref = _media_ref_from_payload(payload, resource_type=normalized_type)
        return FeishuNormalizedMessage(
            raw_type=normalized_type,
            text_content="",
            preferred_message_type="audio" if normalized_type == "audio" else "document",
            media_refs=[media_ref] if media_ref.file_key else [],
            relation_kind=normalized_type,
            metadata={"placeholder_text": _attachment_placeholder(media_ref.file_name)},
            mentions=mention_refs,
        )
    if normalized_type == "merge_forward":
        return _normalize_merge_forward_message(payload)
    if normalized_type == "share_chat":
        return _normalize_share_chat_message(payload)
    if normalized_type in {"interactive", "card"}:
        return _normalize_interactive_message(normalized_type, payload)
    return FeishuNormalizedMessage(raw_type=normalized_type, text_content="")
