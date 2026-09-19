"""Feishu/Lark rich-text (post/card) payload rendering + inbound message normalization.

Extracted byte-verbatim from ``plugins/platforms/feishu/adapter.py`` (2K-law
fracture, issue #79904): renders post/card payloads to markdown or plain text,
collects text out of card JSON, and normalizes an inbound message into
:class:`FeishuNormalizedMessage`. ``adapter`` re-exports every name below, so
``plugins.platforms.feishu.adapter.<name>`` stays the same object.
"""

from __future__ import annotations

import itertools
import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

_MARKDOWN_LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
_MARKDOWN_FENCE_OPEN_RE = re.compile(r"^```([^\n`]*)\s*$")
_MARKDOWN_FENCE_CLOSE_RE = re.compile(r"^```\s*$")
_MULTISPACE_RE = re.compile(r"[ \t]{2,}")
# --- Fallback display strings ---
FALLBACK_POST_TEXT = "[Rich text message]"
FALLBACK_FORWARD_TEXT = "[Merged forward message]"
FALLBACK_SHARE_CHAT_TEXT = "[Shared chat]"
FALLBACK_INTERACTIVE_TEXT = "[Interactive message]"
FALLBACK_IMAGE_TEXT = "[Image]"
FALLBACK_ATTACHMENT_TEXT = "[Attachment]"
# --- Post/card parsing helpers ---
_PREFERRED_LOCALES = ("zh_cn", "en_us")
_MARKDOWN_SPECIAL_CHARS_RE = re.compile(r"([\\`*_{}\[\]()#+\-!|>~])")
_MENTION_PLACEHOLDER_RE = re.compile(r"@_user_\d+")
_MENTION_BOUNDARY_CHARS = frozenset(" \t\n\r.,;:!?、，。；：！？()[]{}<>\"'`")
_TRAILING_TERMINAL_PUNCT = frozenset(" \t\n\r.!?。！？")
_WHITESPACE_RE = re.compile(r"\s+")
_SUPPORTED_CARD_TEXT_KEYS = (
    "title", "text", "content", "label", "value", "name", "summary", "subtitle", "description", "placeholder", "hint",
)
_RICH_BLOCK_TAGS = {
    "plain_text", "lark_md", "markdown", "note", "div", "column_set", "column", "action", "button", "select_static",
    "date_picker",
}
_SKIP_TEXT_KEYS = {
    "tag", "type", "msg_type", "message_type", "chat_id", "open_chat_id", "share_chat_id", "file_key", "image_key",
    "user_id", "open_id", "union_id", "url", "href", "link", "token", "template", "locale",
}


@dataclass(frozen=True)
class FeishuPostMediaRef:
    file_key: str
    file_name: str = ""
    resource_type: str = "file"


@dataclass(frozen=True)
class FeishuMentionRef:
    name: str = ""
    open_id: str = ""
    is_all: bool = False
    is_self: bool = False


@dataclass(frozen=True)
class _FeishuBotIdentity:
    open_id: str = ""
    user_id: str = ""
    name: str = ""

    def matches(self, *, open_id: str, user_id: str, name: str) -> bool:
        # Precedence: open_id > user_id > name. IDs are authoritative when both
        # sides have them; the next tier is only considered when either side
        # lacks the current one.
        if open_id and self.open_id:
            return open_id == self.open_id
        if user_id and self.user_id:
            return user_id == self.user_id
        return bool(self.name) and name == self.name


@dataclass(frozen=True)
class FeishuPostParseResult:
    text_content: str
    image_keys: List[str] = field(default_factory=list)
    media_refs: List[FeishuPostMediaRef] = field(default_factory=list)


@dataclass(frozen=True)
class FeishuNormalizedMessage:
    raw_type: str
    text_content: str
    preferred_message_type: str = "text"
    image_keys: List[str] = field(default_factory=list)
    media_refs: List[FeishuPostMediaRef] = field(default_factory=list)
    mentions: List[FeishuMentionRef] = field(default_factory=list)
    relation_kind: str = "plain"
    metadata: Dict[str, Any] = field(default_factory=dict)

# --- Markdown rendering helpers ---

def _escape_markdown_text(text: str) -> str:
    return _MARKDOWN_SPECIAL_CHARS_RE.sub(r"\\\1", text)


def _to_boolean(value: Any) -> bool:
    return value is True or value == 1 or value == "true"


def _is_style_enabled(style: Dict[str, Any] | None, key: str) -> bool:
    if not style:
        return False
    return _to_boolean(style.get(key))


def _wrap_inline_code(text: str) -> str:
    max_run = max([0, *[len(run) for run in re.findall(r"`+", text)]])
    fence = "`" * (max_run + 1)
    body = f" {text} " if text.startswith("`") or text.endswith("`") else text
    return f"{fence}{body}{fence}"


def _sanitize_fence_language(language: str) -> str:
    return language.strip().replace("\n", " ").replace("\r", " ")


_TEXT_STYLE_WRAPPERS = (("bold", "**", "**"), ("italic", "*", "*"), ("underline", "<u>", "</u>"), ("strikethrough", "~~", "~~"))


def _render_text_element(element: Dict[str, Any]) -> str:
    text = str(element.get("text", "") or "")
    style = element.get("style")
    style_dict = style if isinstance(style, dict) else None
    if _is_style_enabled(style_dict, "code"):
        return _wrap_inline_code(text)
    # Post text elements carry raw text plus separate style flags; the style wrappers below
    # re-create the markdown. Escaping the text here put `\*\*bold\*\*` / `\`code\`` into the
    # model's context and those backslashes came straight back out in replies (#9816).
    rendered = text
    if not rendered:
        return ""
    for key, prefix, suffix in _TEXT_STYLE_WRAPPERS:  # order matters for nesting
        if _is_style_enabled(style_dict, key):
            rendered = f"{prefix}{rendered}{suffix}"
    return rendered


def _render_code_block_element(element: Dict[str, Any]) -> str:
    language = _sanitize_fence_language(str(element.get("language", "") or "") or str(element.get("lang", "") or ""))
    code = (str(element.get("text", "") or "") or str(element.get("content", "") or "")).replace("\r\n", "\n")
    trailing_newline = "" if code.endswith("\n") else "\n"
    return f"```{language}\n{code}{trailing_newline}```"


def _strip_markdown_to_plain_text(text: str) -> str:
    """Plain-text fallback: shared strip_markdown plus Feishu extras (blockquote, ~~, <u>, hr, CRLF)."""
    from gateway.platforms.helpers import strip_markdown
    plain = text.replace("\r\n", "\n")
    plain = _MARKDOWN_LINK_RE.sub(lambda m: f"{m.group(1)} ({m.group(2).strip()})", plain)
    plain = re.sub(r"^>\s?", "", plain, flags=re.MULTILINE)
    plain = re.sub(r"^\s*---+\s*$", "---", plain, flags=re.MULTILINE)
    plain = re.sub(r"~~([^~\n]+)~~", r"\1", plain)
    plain = re.sub(r"<u>([\s\S]*?)</u>", r"\1", plain)
    return strip_markdown(plain)


def _coerce_int(value: Any, default: Optional[int] = None, min_value: int = 0) -> Optional[int]:
    """Coerce value to int with optional default and minimum constraint."""
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed >= min_value else default


def _coerce_required_int(value: Any, default: int, min_value: int = 0) -> int:
    parsed = _coerce_int(value, default=default, min_value=min_value)
    return default if parsed is None else parsed


# --- Post payload builders and parsers ---

def _build_markdown_post_payload(content: str) -> str:
    rows = _build_markdown_post_rows(content)
    return json.dumps({"zh_cn": {"content": rows}}, ensure_ascii=False)


def _build_markdown_post_rows(content: str) -> List[List[Dict[str, str]]]:
    """Build Feishu post rows, giving each fenced code block its own row.

    Feishu's `md` renderer can swallow trailing content when a fence sits inside one
    large element; splitting at real fence lines keeps surrounding prose visible.
    """
    if not content:
        return [[{"tag": "md", "text": ""}]]
    if "```" not in content:
        return [[{"tag": "md", "text": content}]]

    rows: List[List[Dict[str, str]]] = []
    current: List[str] = []
    in_code_block = False

    def _flush_current() -> None:
        nonlocal current
        segment = "\n".join(current)
        if segment.strip():
            rows.append([{"tag": "md", "text": segment}])
        current = []

    for raw_line in content.splitlines():
        fence_re = _MARKDOWN_FENCE_CLOSE_RE if in_code_block else _MARKDOWN_FENCE_OPEN_RE
        is_fence = bool(fence_re.match(raw_line.strip()))
        if is_fence and not in_code_block:  # opening fence: prose before it becomes its own row
            _flush_current()
        current.append(raw_line)
        if is_fence:
            in_code_block = not in_code_block
            if not in_code_block:  # closing fence: the code block becomes its own row
                _flush_current()
    _flush_current()
    return rows or [[{"tag": "md", "text": content}]]


def parse_feishu_post_payload(
    payload: Any, *, mentions_map: Optional[Dict[str, FeishuMentionRef]] = None,
) -> FeishuPostParseResult:
    resolved = _resolve_post_payload(payload)
    if not resolved:
        return FeishuPostParseResult(text_content=FALLBACK_POST_TEXT)
    image_keys: List[str] = []
    media_refs: List[FeishuPostMediaRef] = []
    parts: List[str] = []
    title = _normalize_feishu_text(str(resolved.get("title", "")).strip())
    if title:
        parts.append(title)
    for row in resolved.get("content", []) or []:
        if not isinstance(row, list):
            continue
        row_text = _normalize_feishu_text(
            "".join(_render_post_element(item, image_keys, media_refs, mentions_map) for item in row)
        )
        if row_text:
            parts.append(row_text)
    for entry in resolved.get("files", []) or []:
        if (
            not isinstance(entry, dict)
            or _to_boolean(entry.get("is_folder"))
            or not str(entry.get("file_key", "")).strip()
        ):
            continue
        placeholder = _render_post_element(
            {**entry, "tag": "file"}, image_keys, media_refs, mentions_map,
        )
        if placeholder:
            parts.append(placeholder)
    return FeishuPostParseResult(
        text_content="\n".join(parts).strip() or FALLBACK_POST_TEXT, image_keys=image_keys, media_refs=media_refs,
    )


def _resolve_post_payload(payload: Any) -> Dict[str, Any]:
    direct = _to_post_payload(payload)
    if direct:
        return direct
    if not isinstance(payload, dict):
        return {}
    return _resolve_locale_payload(payload.get("post")) or _resolve_locale_payload(payload)


def _resolve_locale_payload(payload: Any) -> Dict[str, Any]:
    direct = _to_post_payload(payload)
    if direct:
        return direct
    if not isinstance(payload, dict):
        return {}
    # Preferred locales first, then any locale that carries a content list.
    preferred = (payload.get(key) for key in _PREFERRED_LOCALES)
    for candidate in map(_to_post_payload, itertools.chain(preferred, payload.values())):
        if candidate:
            return candidate
    return {}


def _to_post_payload(candidate: Any) -> Dict[str, Any]:
    if not isinstance(candidate, dict):
        return {}
    content = candidate.get("content")
    if not isinstance(content, list):
        return {}
    files = candidate.get("files")
    return {
        "title": str(candidate.get("title", "") or ""),
        "content": content,
        "files": files if isinstance(files, list) else [],
    }


_STATIC_POST_TAGS = {"br": "\n", "hr": "\n\n---\n\n", "divider": "\n\n---\n\n"}


def _render_post_element(
    element: Any, image_keys: List[str], media_refs: List[FeishuPostMediaRef],
    mentions_map: Optional[Dict[str, FeishuMentionRef]] = None,
) -> str:
    if isinstance(element, str):
        return element
    if not isinstance(element, dict):
        return ""

    tag = str(element.get("tag", "")).strip().lower()
    if tag in _STATIC_POST_TAGS:
        return _STATIC_POST_TAGS[tag]
    if tag == "text":
        return _render_text_element(element)
    if tag in {"code_block", "pre"}:
        return _render_code_block_element(element)
    if tag == "a":
        href = str(element.get("href", "")).strip()
        label = str(element.get("text", href) or "").strip()
        if not label:
            return ""
        escaped_label = _escape_markdown_text(label)
        return f"[{escaped_label}]({href})" if href else escaped_label
    if tag == "at":
        # <at>.user_id is a placeholder ("@_user_N" / "@_all"); mentions_map has the real ref.
        placeholder = str(element.get("user_id", "")).strip()
        if placeholder == "@_all":
            # The SDK sometimes omits @_all from top-level mentions; record it so callers see it.
            if mentions_map is not None and "@_all" not in mentions_map:
                mentions_map["@_all"] = FeishuMentionRef(is_all=True)
            return "@all"
        ref = (mentions_map or {}).get(placeholder)
        display_name = (ref.name or ref.open_id or "user") if ref is not None else (
            str(element.get("user_name", "")).strip() or "user"
        )
        return f"@{_escape_markdown_text(display_name)}"
    if tag in {"img", "image"}:
        image_key = str(element.get("image_key", "")).strip()
        if image_key and image_key not in image_keys:
            image_keys.append(image_key)
        alt = str(element.get("text", "")).strip() or str(element.get("alt", "")).strip()
        return f"[Image: {alt}]" if alt else "[Image]"
    if tag in {"media", "file", "audio", "video"}:
        file_key = str(element.get("file_key", "")).strip()
        names = (str(element.get(k, "")).strip() for k in ("file_name", "title", "text"))
        file_name = next((n for n in names if n), "")
        placeholder = f"[Attachment: {file_name}]" if file_name else "[Attachment]"
        if not file_key:
            return placeholder
        if not any(ref.file_key == file_key for ref in media_refs):
            media_refs.append(FeishuPostMediaRef(
                file_key=file_key, file_name=file_name, resource_type=tag if tag in {"audio", "video"} else "file",
            ))
            return placeholder
        return ""
    if tag in {"emotion", "emoji"}:
        label = str(element.get("text", "")).strip() or str(element.get("emoji_type", "")).strip()
        return f":{_escape_markdown_text(label)}:" if label else "[Emoji]"
    if tag == "code":
        code = str(element.get("text", "") or "") or str(element.get("content", "") or "")
        return _wrap_inline_code(code) if code else ""
    nested = (element.get(key) for key in ("text", "title", "content", "children", "elements"))
    return _join_nested_posts(nested, image_keys, media_refs, mentions_map)


def _join_nested_posts(values: Any, image_keys: Any, media_refs: Any, mentions_map: Any) -> str:
    parts = (_render_nested_post(item, image_keys, media_refs, mentions_map) for item in values)
    return " ".join(part for part in parts if part)


def _render_nested_post(
    value: Any, image_keys: List[str], media_refs: List[FeishuPostMediaRef],
    mentions_map: Optional[Dict[str, FeishuMentionRef]] = None,
) -> str:
    if isinstance(value, str):
        return _escape_markdown_text(value)
    if isinstance(value, list):
        return _join_nested_posts(value, image_keys, media_refs, mentions_map)
    if isinstance(value, dict):
        direct = _render_post_element(value, image_keys, media_refs, mentions_map)
        return direct or _join_nested_posts(value.values(), image_keys, media_refs, mentions_map)
    return ""


# --- Message normalization ---

def normalize_feishu_message(
    *, message_type: str, raw_content: str, mentions: Optional[Sequence[Any]] = None,
    bot: _FeishuBotIdentity = _FeishuBotIdentity(),
) -> FeishuNormalizedMessage:
    normalized_type = str(message_type or "").strip().lower()
    payload = _load_feishu_payload(raw_content)
    mentions_map = _build_mentions_map(mentions, bot)

    if normalized_type == "text":
        text = str(payload.get("text", "") or "")
        # Feishu SDK sometimes omits @_all from the mentions payload even when
        # the text literal contains it (confirmed via im.v1.message.get).
        if "@_all" in text and "@_all" not in mentions_map:
            mentions_map["@_all"] = FeishuMentionRef(is_all=True)
        return FeishuNormalizedMessage(
            raw_type=normalized_type, text_content=_normalize_feishu_text(text, mentions_map),
            mentions=list(mentions_map.values()),
        )
    if normalized_type == "post":
        # The walker writes back to mentions_map if it encounters
        # <at user_id="@_all">, so reading .values() after parsing is enough.
        parsed_post = parse_feishu_post_payload(payload, mentions_map=mentions_map)
        return FeishuNormalizedMessage(
            raw_type=normalized_type, text_content=parsed_post.text_content,
            image_keys=list(parsed_post.image_keys), media_refs=list(parsed_post.media_refs),
            mentions=list(mentions_map.values()), relation_kind="post",
        )
    mention_refs = list(mentions_map.values())
    if normalized_type == "image":
        image_key = str(payload.get("image_key", "") or "").strip()
        alt_text = _normalize_feishu_text(
            str(payload.get("text", "") or "")
            or str(payload.get("alt", "") or "")
            or FALLBACK_IMAGE_TEXT,
            mentions_map,
        )
        return FeishuNormalizedMessage(
            raw_type=normalized_type,
            text_content=alt_text if alt_text != FALLBACK_IMAGE_TEXT else "",
            preferred_message_type="photo", image_keys=[image_key] if image_key else [],
            relation_kind="image", mentions=mention_refs,
        )
    if normalized_type in {"file", "audio", "media"}:
        media_ref = _build_media_ref_from_payload(payload, resource_type=normalized_type)
        return FeishuNormalizedMessage(
            raw_type=normalized_type, text_content="",
            preferred_message_type="audio" if normalized_type == "audio" else "document",
            media_refs=[media_ref] if media_ref.file_key else [], relation_kind=normalized_type,
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


def _load_feishu_payload(raw_content: str) -> Dict[str, Any]:
    try:
        parsed = json.loads(raw_content) if raw_content else {}
    except json.JSONDecodeError:
        return {"text": raw_content}
    return parsed if isinstance(parsed, dict) else {"content": parsed}


def _normalize_merge_forward_message(payload: Dict[str, Any]) -> FeishuNormalizedMessage:
    title = _first_text_field(payload, "title", "summary", "preview", deep=("title", "summary", "preview", "description"))
    entries = _collect_forward_entries(payload)
    lines = ([title] if title else []) + entries[:8]
    return FeishuNormalizedMessage(
        raw_type="merge_forward", text_content="\n".join(lines).strip() or FALLBACK_FORWARD_TEXT,
        relation_kind="merge_forward", metadata={"entry_count": len(entries), "title": title},
    )


def _normalize_share_chat_message(payload: Dict[str, Any]) -> FeishuNormalizedMessage:
    chat_name = _first_text_field(payload, "chat_name", "name", "title", deep=("chat_name", "name", "title"))
    share_id = _first_text_field(payload, "chat_id", "open_chat_id", "share_chat_id")
    lines = [f"Shared chat: {chat_name}" if chat_name else FALLBACK_SHARE_CHAT_TEXT]
    if share_id:
        lines.append(f"Chat ID: {share_id}")
    return FeishuNormalizedMessage(
        raw_type="share_chat", text_content="\n".join(lines), relation_kind="share_chat",
        metadata={"chat_id": share_id, "chat_name": chat_name},
    )


def _normalize_interactive_message(message_type: str, payload: Dict[str, Any]) -> FeishuNormalizedMessage:
    card_payload = payload.get("card") if isinstance(payload.get("card"), dict) else payload
    title = _first_non_empty_text(
        _find_header_title(card_payload), payload.get("title"),
        _find_first_text(card_payload, keys=("title", "summary", "subtitle")),
    )
    actions = _collect_action_labels(card_payload)
    lines = ([title] if title else []) + [line for line in _collect_card_lines(card_payload) if line != title]
    if actions:
        lines.append(f"Actions: {', '.join(actions)}")
    return FeishuNormalizedMessage(
        raw_type=message_type,
        text_content="\n".join(lines[:12]).strip() or FALLBACK_INTERACTIVE_TEXT,
        relation_kind="interactive", metadata={"title": title, "actions": actions},
    )


# --- Content extraction utilities (card / forward / text walking) ---

def _collect_forward_entries(payload: Dict[str, Any]) -> List[str]:
    candidates: List[Any] = []
    for key in ("messages", "items", "message_list", "records", "content"):
        value = payload.get(key)
        if isinstance(value, list):
            candidates.extend(value)
    entries: List[str] = []
    for item in candidates:
        if not isinstance(item, dict):
            text = _normalize_feishu_text(str(item or ""))
            if text:
                entries.append(f"- {text}")
            continue
        sender = _first_text_field(item, "sender_name", "user_name", "sender", "name")
        nested_type = str(item.get("message_type", "") or item.get("msg_type", "")).strip().lower()
        if nested_type == "post":
            body = parse_feishu_post_payload(item.get("content") or item).text_content
        else:
            body = _first_text_field(
                item, "text", "summary", "preview", "content", deep=("text", "content", "summary", "preview", "title"),
            )
        body = _normalize_feishu_text(body)
        if sender and body:
            entries.append(f"- {sender}: {body}")
        elif body:
            entries.append(f"- {body}")
    return _unique_lines(entries)


def _collect_card_lines(payload: Any) -> List[str]:
    lines = _collect_text_segments(payload, in_rich_block=False)
    normalized = [_normalize_feishu_text(line) for line in lines]
    return _unique_lines([line for line in normalized if line])


def _collect_action_labels(payload: Any) -> List[str]:
    labels: List[str] = []
    for item in _walk_nodes(payload):
        if not isinstance(item, dict):
            continue
        tag = str(item.get("tag", "") or item.get("type", "")).strip().lower()
        if tag not in {"button", "select_static", "overflow", "date_picker", "picker"}:
            continue
        label = _first_text_field(item, "text", "name", "value", deep=("text", "content", "name", "value"))
        if label:
            labels.append(label)
    return _unique_lines(labels)


def _collect_text_segments(value: Any, *, in_rich_block: bool) -> List[str]:
    if isinstance(value, str):
        return [_normalize_feishu_text(value)] if in_rich_block else []
    if isinstance(value, list):
        return [seg for item in value for seg in _collect_text_segments(item, in_rich_block=in_rich_block)]
    if not isinstance(value, dict):
        return []
    tag = str(value.get("tag", "") or value.get("type", "")).strip().lower()
    next_in_rich_block = in_rich_block or tag in _RICH_BLOCK_TAGS
    segments: List[str] = []
    if next_in_rich_block:
        for key in _SUPPORTED_CARD_TEXT_KEYS:
            item = value.get(key)
            if isinstance(item, str) and _normalize_feishu_text(item):
                segments.append(_normalize_feishu_text(item))
    for key, item in value.items():
        if key not in _SKIP_TEXT_KEYS:
            segments.extend(_collect_text_segments(item, in_rich_block=next_in_rich_block))
    return segments


def _build_media_ref_from_payload(payload: Dict[str, Any], *, resource_type: str) -> FeishuPostMediaRef:
    file_key = str(payload.get("file_key", "") or "").strip()
    file_name = _first_text_field(payload, "file_name", "title", "text")
    effective_type = resource_type if resource_type in {"audio", "video"} else "file"
    return FeishuPostMediaRef(file_key=file_key, file_name=file_name, resource_type=effective_type)


def _attachment_placeholder(file_name: str) -> str:
    normalized_name = _normalize_feishu_text(file_name)
    return f"[Attachment: {normalized_name}]" if normalized_name else FALLBACK_ATTACHMENT_TEXT


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
    """First scalar (non-dict/list, non-None) value that normalizes to non-empty text."""
    for value in values:
        if value is None or isinstance(value, (dict, list)):
            continue
        normalized = _normalize_feishu_text(value if isinstance(value, str) else str(value))
        if normalized:
            return normalized
    return ""


def _first_text_field(payload: Dict[str, Any], *keys: str, deep: tuple[str, ...] = ()) -> str:
    """``_first_non_empty_text`` over ``payload[key]`` for each key, then a deep ``_find_first_text``."""
    values = [payload.get(key) for key in keys]
    if deep:
        values.append(_find_first_text(payload, keys=deep))
    return _first_non_empty_text(*values)


# --- General text utilities ---

def _normalize_feishu_text(text: str, mentions_map: Optional[Dict[str, FeishuMentionRef]] = None) -> str:
    def _sub(match: "re.Match[str]") -> str:
        ref = (mentions_map or {}).get(match.group(0))
        return " " if ref is None else f"@{ref.name or ref.open_id or 'user'}"

    cleaned = _MENTION_PLACEHOLDER_RE.sub(_sub, text or "")
    cleaned = cleaned.replace("@_all", "@all")
    cleaned = cleaned.replace("\r\n", "\n").replace("\r", "\n")
    cleaned = "\n".join(_WHITESPACE_RE.sub(" ", line).strip() for line in cleaned.split("\n"))
    cleaned = "\n".join(line for line in cleaned.split("\n") if line)
    cleaned = _MULTISPACE_RE.sub(" ", cleaned)
    return cleaned.strip()


def _unique_lines(lines: List[str]) -> List[str]:
    seen: set[str] = set()
    unique: List[str] = []
    for line in lines:
        if not line or line in seen:
            continue
        seen.add(line)
        unique.append(line)
    return unique


# --- Mention helpers ---

def _extract_mention_ids(mention: Any) -> tuple[str, str]:
    """(open_id, user_id): message.get gives a string id + id_type; events give a nested UserId object."""
    mention_id = getattr(mention, "id", None)
    if isinstance(mention_id, str):
        id_type = str(getattr(mention, "id_type", "") or "").lower()
        return (mention_id, "") if id_type == "open_id" else ("", mention_id) if id_type == "user_id" else ("", "")
    if mention_id is None:
        return "", ""
    return str(getattr(mention_id, "open_id", "") or ""), str(getattr(mention_id, "user_id", "") or "")


def _build_mentions_map(mentions: Optional[Sequence[Any]], bot: _FeishuBotIdentity) -> Dict[str, FeishuMentionRef]:
    result: Dict[str, FeishuMentionRef] = {}
    for mention in mentions or []:
        key = str(getattr(mention, "key", "") or "")
        if not key:
            continue
        if key == "@_all":
            result[key] = FeishuMentionRef(is_all=True)
            continue
        open_id, user_id = _extract_mention_ids(mention)
        name = str(getattr(mention, "name", "") or "").strip()
        is_self = bot.matches(open_id=open_id, user_id=user_id, name=name)
        result[key] = FeishuMentionRef(name=name, open_id=open_id, is_self=is_self)
    return result


def _build_mention_hint(mentions: Sequence[FeishuMentionRef]) -> str:
    parts: List[str] = []
    seen: set = set()
    for ref in mentions:
        if ref.is_self:
            continue
        signature = (ref.is_all, ref.open_id, ref.name)
        if signature in seen:
            continue
        seen.add(signature)
        if ref.is_all:
            parts.append("@all")
        elif ref.open_id:
            parts.append(f"{ref.name or 'unknown'} (open_id={ref.open_id})")
        else:
            parts.append(ref.name or "unknown")
    return f"[Mentioned: {', '.join(parts)}]" if parts else ""


def _strip_edge_self_mentions(text: str, mentions: Sequence[FeishuMentionRef]) -> str:
    # Leading self-mentions are stripped unconditionally (word-boundary so @Al can't eat @Alice);
    # trailing ones only when followed by whitespace/terminal punct so "don't @Bot again" survives.
    if not text:
        return text
    self_names = [f"@{ref.name or ref.open_id or 'user'}" for ref in mentions if ref.is_self]
    if not self_names:
        return text
    remaining = text.lstrip()
    while True:
        for nm in self_names:
            if not remaining.startswith(nm):
                continue
            after = remaining[len(nm):]
            if after and after[0] not in _MENTION_BOUNDARY_CHARS:
                continue
            remaining = after.lstrip()
            break
        else:
            break
    while True:
        i = len(remaining)
        while i > 0 and remaining[i - 1] in _TRAILING_TERMINAL_PUNCT:
            i -= 1
        body = remaining[:i]
        tail = remaining[i:]
        for nm in self_names:
            if body.endswith(nm):
                remaining = body[: -len(nm)].rstrip() + tail
                break
        else:
            return remaining
