"""Slack platform adapter: slack-bolt Socket Mode (messages, slash commands, threads)."""

import asyncio
import contextvars
import functools
import inspect
import json
import logging
import os
import re
import time
import unicodedata
from dataclasses import dataclass, field
from typing import Awaitable, Callable, ClassVar, Dict, Optional, Any, Tuple, List

import aiohttp

try:
    from slack_bolt.async_app import AsyncApp
    from slack_bolt.adapter.socket_mode.async_handler import AsyncSocketModeHandler
    from slack_sdk.web.async_client import AsyncWebClient

    SLACK_AVAILABLE = True
except ImportError:
    SLACK_AVAILABLE = False
    AsyncApp = Any
    AsyncSocketModeHandler = Any
    AsyncWebClient = Any

import sys
from pathlib import Path as _Path

sys.path.insert(0, str(_Path(__file__).resolve().parents[3]))

from agent.secret_scope import UnscopedSecretError, get_secret
from gateway.config import Platform, PlatformConfig
from gateway.platforms.helpers import MessageDeduplicator
from gateway.platforms.base import (
    gateway_trust_env, BasePlatformAdapter, MessageEvent, MessageType, ProcessingOutcome,
    SendResult, SUPPORTED_DOCUMENT_TYPES, SUPPORTED_VIDEO_TYPES, _TEXT_INJECT_EXTENSIONS,
    is_host_excluded_by_no_proxy, resolve_proxy_url, safe_url_for_log, _ssrf_redirect_guard,
    cache_document_from_bytes_async, cache_video_from_bytes_async)

try:  # sibling module; support both package and flat plugin-dir import
    from .block_kit import render_blocks, sanitize_blocks
except ImportError:  # pragma: no cover - plugin loaded outside package context
    from block_kit import render_blocks, sanitize_blocks  # type: ignore


logger = logging.getLogger(__name__)

# User-Agent prefix (``HermesAgent/<version>``) for platform-partner attribution of API calls.
try:
    from hermes_cli import __version__ as _HERMES_VERSION
except Exception:
    _HERMES_VERSION = "unknown"
_HERMES_SLACK_USER_AGENT_PREFIX = f"HermesAgent/{_HERMES_VERSION}"

_SLACK_ERROR_BODY_LIMIT_BYTES = 8 * 1024
_BOOL_WORDS = frozenset({"1", "0", "true", "false", "yes", "no", "on", "off"})

# Model picker Block Kit action IDs. The picker is a two-step drill-down:
# provider static_select → model static_select, plus Back/Cancel buttons.
_MODEL_PICKER_PROVIDER_ACTION = "hermes_model_provider"
_MODEL_PICKER_MODEL_ACTION = "hermes_model_model"
_MODEL_PICKER_BACK_ACTION = "hermes_model_back"
_MODEL_PICKER_CANCEL_ACTION = "hermes_model_cancel"
# Rendered when a live-looking picker message can no longer resolve (gateway
# restart, aged-out state entry, or a value the stored state no longer
# covers): the message is rewritten to this so the control visibly dies.
_MODEL_PICKER_EXPIRED_NOTICE = "⏳ This model picker expired — please run /model again."
_MODEL_PICKER_ACTION_IDS = (
    _MODEL_PICKER_PROVIDER_ACTION,
    _MODEL_PICKER_MODEL_ACTION,
    _MODEL_PICKER_BACK_ACTION,
    _MODEL_PICKER_CANCEL_ACTION,
)


def _slack_unfurl_kwargs(extra: Optional[Dict[str, Any]]) -> Dict[str, bool]:
    """Explicitly configured link-preview controls (omitted key = Slack default). String bools are
    coerced (config tooling persists YAML bools as strings); junk is dropped, NOT coerced to False,
    so bad config keeps Slack's default rather than suppressing previews."""
    settings = extra or {}
    kwargs: Dict[str, bool] = {}
    for key in ("unfurl_links", "unfurl_media"):
        val = settings.get(key)
        if isinstance(val, bool):
            kwargs[key] = val
        elif isinstance(val, str) and val.strip().lower() in _BOOL_WORDS:
            kwargs[key] = val.strip().lower() in {"1", "true", "yes", "on"}
    return kwargs


async def _read_error_text_limited(
    response: Any, *, limit: int = _SLACK_ERROR_BODY_LIMIT_BYTES) -> str:
    content = getattr(response, "content", None)
    read = getattr(content, "read", None)
    if callable(read):
        chunks: list[bytes] = []
        total = 0
        while total <= limit:
            size = min(4096, limit + 1 - total)
            chunk = await read(size)
            if not chunk:
                break
            data = bytes(chunk)
            chunks.append(data)
            total += len(data)
        if total > limit:
            release = getattr(response, "release", None)
            if callable(release):
                release()
        return b"".join(chunks)[:limit].decode("utf-8", errors="replace")
    text = await response.text()
    return str(text)[:limit]


def _slack_response_payload(response: Any) -> Dict[str, Any]:
    """Return a Slack Web API response as a plain dict (``{}`` for unknown shapes).
    ``SlackResponse`` is mapping-like but not a ``dict``, so an ``isinstance(resp, dict)`` gate is
    always False at runtime and silently degrades results; normalize here instead."""
    if isinstance(response, dict):
        return response
    data = getattr(response, "data", None)
    return data if isinstance(data, dict) else {}


_SLACK_SPECIAL_MENTION_RE = re.compile(r"<!(?:everyone|channel|here)(?:\|[^>\n]*)?>", re.IGNORECASE)

# Thread-root images delivered on a mid-thread cold start; other messages' files
# are text markers only (the root is usually the artifact the mention is about).
_THREAD_ROOT_IMAGE_MAX = 4


def _slack_file_marker(file_obj: Dict[str, Any]) -> str:
    """Render a compact text marker for a Slack file so text-only context shows attachments. Name is
    sanitized (newlines/brackets stripped) so a hostile filename can't fake context structure."""
    name = str(file_obj.get("name") or file_obj.get("title") or file_obj.get("id") or "file")
    name = re.sub(r"[\r\n\[\]]+", " ", name).strip() or "file"
    mimetype = str(file_obj.get("mimetype") or "")
    for kind in ("image", "video", "audio"):
        if mimetype.startswith(kind + "/"):
            return f"[{kind}: {name}]"
    return f"[file: {name} ({mimetype})]" if mimetype else f"[file: {name}]"


# GFM tables: Slack mrkdwn shows pipe tables as literal pipes, so they are wrapped in ```
# fences (monospace) and cells padded to per-column display width (CJK-wide aware).

_TABLE_SEPARATOR_RE = re.compile(r"^\s*\|?\s*:?-+:?\s*(?:\|\s*:?-+:?\s*){1,}\|?\s*$")


def _is_table_row(line: str) -> bool:
    """Return True if *line* could plausibly be a table data row."""
    stripped = line.strip()
    return bool(stripped) and "|" in stripped


def _disp_width(s: str) -> int:
    """Monospace display width: East-Asian Wide / Full-width chars count as 2."""
    return sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in s)


def _pad(cell: str, width: int) -> str:
    """Right-pad *cell* with spaces until its display width equals *width*."""
    return cell + " " * max(width - _disp_width(cell), 0)


def _split_table_row(line: str) -> List[str]:
    """Split a ``| a | b | c |`` row into trimmed cells (outer pipes optional)."""
    s = line.strip()
    s = s[1:] if s.startswith("|") else s
    s = s[:-1] if s.endswith("|") else s
    return [c.strip() for c in s.split("|")]


def _align_table(rows: List[str]) -> List[str]:
    """Re-emit a markdown table padded to per-column display width. rows[1] is the GFM separator
    (regenerated); short rows are padded to a uniform column count first."""
    if len(rows) < 2:
        return rows
    parsed = [_split_table_row(r) for r in rows]
    n_cols = max(len(r) for r in parsed)
    parsed = [r + [""] * (n_cols - len(r)) for r in parsed]
    parsed[1] = ["---"] * n_cols  # placeholder; regenerated below
    widths = [max(_disp_width(r[c]) for r in parsed) for c in range(n_cols)]
    out: List[str] = []
    for idx, row in enumerate(parsed):
        cells = ["-" * widths[c] if idx == 1 else _pad(row[c], widths[c]) for c in range(n_cols)]
        out.append("| " + " | ".join(cells) + " |")
    return out


def _wrap_markdown_tables(text: str) -> str:
    """Wrap GFM pipe tables in ``` fences and align columns; tables already in fences are left alone."""
    if not text or "|" not in text or "-" not in text:
        return text
    lines = text.split("\n")
    out: List[str] = []
    in_fence = False
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
        elif (
            not in_fence and "|" in line and i + 1 < len(lines)
            and _TABLE_SEPARATOR_RE.match(lines[i + 1])):
            block = [line, lines[i + 1]]
            j = i + 2
            while j < len(lines) and _is_table_row(lines[j]):
                block.append(lines[j])
                j += 1
            out.append("```")
            out.extend(_align_table(block))
            out.append("```")
            i = j
            continue
        out.append(line)
        i += 1
    return "\n".join(out)


# Slash invoker's user_id: set in _handle_slash_command, read in send() to pick the right stashed
# response_url under concurrent slashes (ContextVars propagate to the background task).
_slash_user_id: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "_slash_user_id", default=None)


@dataclass
class _ThreadContextCache:
    """Cache entry for fetched thread context."""

    content: str
    fetched_at: float = field(default_factory=time.monotonic)
    message_count: int = 0
    parent_text: str = ""  # root text, for mention wake checks
    # Root author ("" unknown): lets _bot_authored_thread_root spot roots posted outside send().
    # The Slack user_id of the thread parent message author. Used by _bot_authored_thread_root (#63530) to
    # detect threads whose root was posted by the bot via direct chat.postMessage (outside the gateway's
    # send() path). Empty string when the parent could not be fetched or did not have a user_id field.
    parent_user_id: str = ""
    # Raw conversations.replies payloads so a watermark (``after_ts``) re-format needs no API call.
    # Kept so context can be re-formatted with a different watermark (``after_ts``) without an extra API
    # call (#23918).
    messages: List[Dict[str, Any]] = field(default_factory=list)


def slack_deps_present() -> bool:
    """PASSIVE probe: are slack-bolt/slack-sdk importable right now?
    Registry ``check_fn`` (status displays, config loading) — must never install. The active
    installer is ``check_slack_requirements`` (``ensure_deps_fn``).

    The ACTIVE lazy-installer (``check_slack_requirements``) is registered as ``ensure_deps_fn`` and runs
    from ``create_adapter()`` when this returns False (#79812).
    """
    return SLACK_AVAILABLE


@dataclass
class _NativeTaskCardStream:
    """Serialized state for one workspace-scoped Slack progress stream."""

    team_id: str
    channel: str
    thread_ts: str
    stream_ts: str = ""
    stopped: bool = False
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


def check_slack_requirements() -> bool:
    """Lazy-install slack-bolt/slack-sdk if missing and rebind module globals on success."""
    if SLACK_AVAILABLE:
        return True

    def _import():
        from slack_bolt.async_app import AsyncApp
        from slack_bolt.adapter.socket_mode.async_handler import AsyncSocketModeHandler
        from slack_sdk.web.async_client import AsyncWebClient
        import aiohttp
        return {
            "AsyncApp": AsyncApp, "AsyncSocketModeHandler": AsyncSocketModeHandler,
            "AsyncWebClient": AsyncWebClient, "aiohttp": aiohttp, "SLACK_AVAILABLE": True}

    from tools.lazy_deps import ensure_and_bind
    return ensure_and_bind("platform.slack", _import, globals(), prompt=False)


def _collect_slack_block_mentions(blocks: list) -> list:
    """``<@UID>`` mentions authored in non-quoted Block Kit text (flat ``text`` omits block-only
    mentions); ``rich_text_quote`` is ignored so quoted/forwarded text can't summon the bot.

    Slack's flat top-level ``text`` field does NOT contain mentions that were authored only inside Block Kit
    ``blocks`` (e.g. a ``rich_text_section`` with a ``user`` element). This walker recovers those mentions
    so the gates can see Block-Kit-only mentions instead of silently dropping them (#52387).
    """
    mentions: list = []

    def _walk(node, in_quote: bool) -> None:
        if isinstance(node, list):
            for item in node:
                _walk(item, in_quote)
            return
        if not isinstance(node, dict):
            return
        node_type = node.get("type")
        quoted = in_quote or node_type == "rich_text_quote"
        if node_type == "user" and not quoted and node.get("user_id", ""):
            mentions.append(f"<@{node['user_id']}>")
        for key in ("elements", "element"):
            child = node.get(key)
            if child is not None:
                _walk(child, quoted)

    try:
        _walk(blocks, False)
    except Exception:  # pragma: no cover - defensive, never break gating
        return []
    return mentions


def _slack_mention_detection_text(event: dict) -> str:
    """Text for @mention detection: flat ``text`` plus non-quoted Block-Kit-only mentions.

    Combines the flat top-level ``text`` with any ``<@UID>`` mentions recovered from non-quoted Block Kit
    blocks (#52387), so a genuine Block-Kit-only mention reaches the gates while quoted/forwarded mentions
    stay ignored.
    """
    flat = event.get("text", "") or ""
    blocks = event.get("blocks")
    extra = [m for m in _collect_slack_block_mentions(blocks) if m not in flat] if blocks else []
    return (flat.strip() + "\n" + " ".join(extra)).strip() if extra else flat


def _rewrite_known_bang_command(text: str) -> str:
    """Rewrite a known leading ``!cmd`` to the gateway ``/cmd`` form."""
    if not text.startswith("!"):
        return text
    try:
        from hermes_cli.commands import is_gateway_known_command
        first_token = text[1:].split(maxsplit=1)[0]
        cmd_name = first_token.split("@", 1)[0].lower()
        if cmd_name and "/" not in cmd_name and is_gateway_known_command(cmd_name):
            return "/" + text[1:]
    except Exception:  # pragma: no cover - defensive
        pass
    return text


def _slack_permalink_path(channel_id: str | None, message_ts: str | None) -> str:
    """Workspace-independent tail (``archives/<channel>/p<ts>``) of a permalink.
    Only the tail can be rebuilt from a payload, so dedupe compares on it."""
    if not channel_id or not message_ts:
        return ""
    return f"archives/{channel_id}/p{str(message_ts).replace('.', '')}"


def _str_or_empty(value: Any) -> str:
    return str(value) if value else ""


def _int_or_zero(value: str) -> int:
    try:
        return int(value)
    except ValueError:
        return 0


def _first_truthy(mapping: Dict[str, Any], keys: Tuple[str, ...]) -> Any:
    """First truthy ``mapping[key]`` in ``keys`` order, else None."""
    for key in keys:
        value = mapping.get(key)
        if value:
            return value
    return None


def _slack_str_field(el: dict, name: str) -> str:
    """Read a string field of a Block Kit element; non-strings (text objects) would break ``str.join``."""
    value = el.get(name)
    return value if isinstance(value, str) else ""


# Inline rich_text entity → (mrkdwn format, source key, default).
_INLINE_ENTITY_FORMATS = {
    "channel": ("<#{}>", "channel_id", ""), "user": ("<@{}>", "user_id", ""),
    "usergroup": ("<!subteam^{}>", "usergroup_id", ""), "team": ("<!team^{}>", "team_id", ""),
    "emoji": (":{}:", "name", ""), "broadcast": ("<!{}>", "range", "here")}


def _render_slack_inline_element(el: dict) -> str:
    """Render one Block Kit inline element; unknown types fall back to any readable field (Slack adds types unannounced)."""
    el_type = el.get("type", "")
    if el_type == "text":
        return _slack_str_field(el, "text")
    if el_type == "color":
        return _slack_str_field(el, "value")
    entity = _INLINE_ENTITY_FORMATS.get(el_type)
    if entity is not None:
        fmt, key, default = entity
        return fmt.format(el.get(key, default))
    if el_type == "date":
        fallback = _slack_str_field(el, "fallback")
        if fallback:
            return fallback
    # link / message_mention / date-without-fallback / unknown: URL + optional label.
    url = _slack_str_field(el, "url")
    label = _slack_str_field(el, "text") or _slack_str_field(el, "fallback")
    if not url and el_type == "message_mention":
        # ``url`` is optional; channel_id + message_ts are required and form the permalink.
        url = _slack_permalink_path(el.get("channel_id"), el.get("message_ts"))
    if url:
        return f"{label} ({url})" if label and label != url else url
    return label


def _render_inline_elements(elements: list) -> str:
    return "".join(_render_slack_inline_element(el) for el in elements)


def _extract_text_from_slack_blocks(blocks: list) -> str:
    """Render ``rich_text`` blocks to readable lines, preserving quotes, lists and code.
    Quoted/forwarded content lives in nested ``rich_text_quote`` elements that the event's plain
    ``text`` field omits."""
    if not blocks:
        return ""
    parts: list[str] = []

    def _append_line(text: str, quote_depth: int = 0, bullet: str = "") -> None:
        if not text or not text.strip():
            return
        prefix = ((">" * quote_depth) + " ") if quote_depth else ""
        parts.append(f"{prefix}{bullet}{text}".rstrip())

    def _walk_elements(elements: list, quote_depth: int = 0, bullet: str = "") -> None:
        for elem in elements:
            elem_type = elem.get("type", "")
            if elem_type == "rich_text_section":
                _append_line(_render_inline_elements(elem.get("elements", [])), quote_depth, bullet)
            elif elem_type == "rich_text_quote":
                _walk_elements(elem.get("elements", []), quote_depth=quote_depth + 1)
            elif elem_type == "rich_text_list":
                list_style = elem.get("style")
                for idx, item in enumerate(elem.get("elements", [])):
                    item_bullet = "• " if list_style == "bullet" else f"{idx + 1}. "
                    _walk_elements([item], quote_depth=quote_depth, bullet=item_bullet)
            elif elem_type == "rich_text_preformatted":
                code_lines = [
                    _render_inline_elements(
                        child.get("elements", [])
                        if child.get("type", "") == "rich_text_section"
                        else [child])
                    for child in elem.get("elements", [])]
                code_text = "\n".join(line for line in code_lines if line)
                if code_text:
                    lang = elem.get("language", "")
                    _append_line(f"```{lang}\n{code_text}\n```", quote_depth, bullet)
            else:
                _append_line(_render_inline_elements([elem]), quote_depth, bullet)

    for block in blocks:
        if (block or {}).get("type") == "rich_text":
            _walk_elements(block.get("elements", []))
    return "\n".join(parts)


def _extract_text_from_slack_attachments(attachments: list) -> str:
    """Extract readable text from legacy ``attachments`` (alert/CI bots post empty ``text``).
    Prefers structured fields; uses ``fallback`` only when nothing else exists."""
    if not attachments:
        return ""
    lines: list[str] = []
    for att in attachments:
        if not isinstance(att, dict):
            continue
        # Permalink unfurls repeat a message the agent already reads (inbound path skips them too).
        if att.get("is_msg_unfurl"):
            continue
        got: list[str] = [str(att[key]) for key in ("pretext", "title", "text") if att.get(key)]
        for field in att.get("fields", []) or []:
            if isinstance(field, dict):
                got += [str(field[k]) for k in ("title", "value") if field.get(k)]
        block_text = _extract_text_from_slack_blocks(att.get("blocks")) if att.get("blocks") else ""
        if block_text:
            got.append(block_text)
        if not got and att.get("fallback"):
            got.append(str(att["fallback"]))
        lines += got
    return "\n".join(line for line in lines if line).strip()


#: Any ``<scheme:target|label>`` autolink (Slack is not limited to https/mailto).
_SLACK_MRKDWN_LINK_RE = re.compile(r"<([a-zA-Z][a-zA-Z0-9+.\-]*:[^>|]+)(?:\|([^>]+))?>")
#: Optional label Slack adds to a mention in flat text while blocks carry the bare id
#: (``<@U…|name>``, ``<#C…|general>``, ``<!subteam^S…|@marketing>``, ``<!here|@here>``).
_SLACK_ENTITY_LABEL_RE = re.compile(r"<([@#!][^>|]*)\|[^>]*>")
_SLACK_FENCED_CODE_RE = re.compile(r"(?<!`)\n*```[ \t]*\n?(.*?)\n?[ \t]*```\n*(?!`)", re.DOTALL)
_SLACK_INLINE_CODE_RE = re.compile(r"`([^`\n]+)`")
_SLACK_DATE_RE = re.compile(r"<!date\^([^>|]*)(?:\|([^>]*))?>")
#: Message permalink reduced to the tail :func:`_slack_permalink_path` rebuilds (host and thread
#: query differ between flat text and a ``channel_id``/``message_ts``-only payload).
_SLACK_PERMALINK_RE = re.compile(r"https?://[^\s/]+/(archives/[A-Za-z0-9]+/p\d+)(?:\?[^\s)]*)?")
_SLACK_INLINE_STYLE_RE = re.compile(r"([*_~])([^\n]+?)\1")
_SLACK_HTML_ENTITY_RE = re.compile(r"&(amp|lt|gt);")
_SLACK_HTML_ENTITIES = {"amp": "&", "lt": "<", "gt": ">"}


def _unescape_slack_entities(text: str) -> str:
    """Undo Slack's ``&``/``<``/``>`` escaping in flat ``text``.
    ``blocks`` are raw, so text-vs-blocks comparison needs a common form (every "Copy link"
    permalink carries ``?thread_ts=…&cid=…``)."""
    return _SLACK_HTML_ENTITY_RE.sub(lambda match: _SLACK_HTML_ENTITIES[match.group(1)], text or "")


def _normalize_slack_text_for_dedupe(text: str, bot_uid: str = "") -> str:
    """Normalize Slack text for comparison with rendered rich text."""

    def _link(match: re.Match) -> str:
        url, label = match.group(1), match.group(2)
        return f"{label} ({url})" if label and label != url else url

    def _date(match: re.Match) -> str:
        # ``<!date^ts^format^url|fallback>`` → what rich-text renders: fallback, else URL.
        fallback = match.group(2)
        if fallback:
            return fallback
        parts = match.group(1).split("^")
        return parts[2] if len(parts) > 2 else ""

    canonical = text or ""
    # Order matters: unescape before links (same brackets/``&``); permalinks after links (bare
    # URL); labels after dates (dates carry a label); bot mention after labels (``<@U…|hermes>``).
    canonical = _unescape_slack_entities(canonical)
    canonical = _SLACK_MRKDWN_LINK_RE.sub(_link, canonical)
    canonical = _SLACK_DATE_RE.sub(_date, canonical)
    canonical = _SLACK_PERMALINK_RE.sub(r"\1", canonical)
    canonical = _SLACK_ENTITY_LABEL_RE.sub(r"<\1>", canonical)
    if bot_uid:
        canonical = canonical.replace(f"<@{bot_uid}>", "")
    canonical = _SLACK_FENCED_CODE_RE.sub(r"\1", canonical)
    canonical = _SLACK_INLINE_CODE_RE.sub(r"\1", canonical)
    while True:
        unstyled = _SLACK_INLINE_STYLE_RE.sub(r"\2", canonical)
        if unstyled == canonical:
            break
        canonical = unstyled
    return re.sub(r"\s+", " ", canonical).strip()


def _extract_additional_text_from_slack_blocks(
    blocks: list, primary_text: str, bot_uid: str = "") -> str:
    """Render rich-text content not already represented by primary_text."""
    primary = _normalize_slack_text_for_dedupe(primary_text, bot_uid)
    primary_fenced = {
        _normalize_slack_text_for_dedupe(match.group(0), bot_uid)
        for match in _SLACK_FENCED_CODE_RE.finditer(primary_text or "")}
    parts: list[str] = []
    for block in blocks or []:
        if (block or {}).get("type") != "rich_text":
            continue
        for element in block.get("elements", []):
            element_type = element.get("type", "")
            rendered = _extract_text_from_slack_blocks(
                [{"type": "rich_text", "elements": [element]}]).strip()
            if not rendered:
                continue
            normalized = _normalize_slack_text_for_dedupe(rendered, bot_uid)
            if element_type == "rich_text_preformatted":
                is_duplicate = normalized in primary_fenced
            else:
                is_duplicate = normalized == primary or normalized in primary
            if normalized and is_duplicate:
                continue
            parts.append(rendered)
    return "\n".join(parts)


# Block Kit keys kept in the agent-facing payload dump (scalars copied; containers recursed).
_BLOCK_SCALAR_KEYS = frozenset(
    "type block_id action_id style dispatch_action optional multiple emoji".split())
_BLOCK_RECURSIVE_KEYS = frozenset(
    "text title description label placeholder accessory fields elements options "
    "option_groups confirm submit close hint".split())


def _serialize_slack_blocks_for_agent(blocks: list, max_chars: int = 6000) -> str:
    """Compact, redacted JSON view of non-``rich_text`` Block Kit blocks.
    ``rich_text`` is already rendered into the message text; dumping it here would repeat the
    author's words with every ``url`` stripped by the allowlist."""
    inspectable = [block for block in (blocks or []) if (block or {}).get("type") != "rich_text"]
    if not inspectable:
        return ""
    def _sanitize(value):
        if isinstance(value, list):
            return [
                item for item in (_sanitize(v) for v in value) if item not in (None, {}, [], "")]
        if isinstance(value, dict):
            sanitized = {}
            for key, item in value.items():
                if key in _BLOCK_SCALAR_KEYS:
                    sanitized[key] = item
                elif key in _BLOCK_RECURSIVE_KEYS:
                    cleaned = _sanitize(item)
                    if cleaned not in (None, {}, [], ""):
                        sanitized[key] = cleaned
            return sanitized
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        return repr(value)

    try:
        payload = json.dumps(_sanitize(inspectable), ensure_ascii=False, indent=2)
    except Exception:
        payload = repr(inspectable)
    if len(payload) > max_chars:
        payload = payload[: max_chars - 18].rstrip() + "\n... [truncated]"
    return f"[Slack Block Kit payload for this message]\n```json\n{payload}\n```"


def _extract_urls_from_slack_blocks(blocks: list) -> list[str]:
    """Return deduped URLs from a Block Kit tree in discovery order.
    Targeted opt-in for alert links; ``_serialize_slack_blocks_for_agent`` deliberately strips
    ``url`` from the generic payload dump."""
    if not blocks:
        return []
    found: list[str] = []
    seen: set[str] = set()

    def _walk(node: Any) -> None:
        if isinstance(node, dict):
            for key in ("url", "image_url", "external_url"):
                value = node.get(key)
                is_url = isinstance(value, str) and value.startswith(("http://", "https://"))
                if is_url and value not in seen:
                    seen.add(value)
                    found.append(value)
            for value in node.values():
                _walk(value)
        elif isinstance(node, list):
            for item in node:
                _walk(item)

    _walk(blocks)
    return found


def _apply_slack_proxy(client: Any, proxy_url: Optional[str]) -> None:
    """Apply a resolved proxy to a Slack SDK client or clear it explicitly."""
    if hasattr(client, "proxy"):
        client.proxy = proxy_url


def _slack_per_request_proxy_middleware(proxy_url: Optional[str]) -> Callable[..., Awaitable[Any]]:
    """Bolt ``before_authorize`` middleware re-applying *proxy_url* per request: Bolt builds a fresh
    ``AsyncWebClient`` per request and ``slack_sdk`` treats ``proxy=None`` as "unspecified" (reloads
    ``HTTP(S)_PROXY``, bypassing NO_PROXY), so "go direct" only survives if re-set
    post-construction. Symptom otherwise: sends work but every inbound ``auth.test`` fails."""

    async def pin_per_request_proxy(client: Any, next_: Callable[[], Awaitable[Any]]) -> Any:
        _apply_slack_proxy(client, proxy_url)
        return await next_()

    return pin_per_request_proxy


# SocketModeClient's background tasks (getattr-looked-up so an SDK rename degrades to a no-op).
_SOCKET_CLIENT_TASK_ATTRS = ("current_session_monitor", "message_processor", "message_receiver")
# Teardown wait cap: a task wedged in a network call must not hold up shutdown.
_SOCKET_TASK_CANCEL_TIMEOUT_S = 3.0


async def _cancel_socket_tasks(tasks: Any) -> None:
    """Cancel Socket Mode tasks and await them (bounded); unawaited cancel still races the work."""
    live = [
        task
        for task in tasks
        if task is not None
        and callable(getattr(task, "cancel", None))
        and not (callable(getattr(task, "done", None)) and task.done())]
    for task in live:
        task.cancel()
    pending = set(live)
    if not pending:
        return
    done, still_running = await asyncio.wait(pending, timeout=_SOCKET_TASK_CANCEL_TIMEOUT_S)
    for task in done:
        if task.cancelled():
            continue
        if task.exception() is not None:  # pragma: no cover - defensive logging
            logger.debug("[Slack] Socket Mode task failed while stopping", exc_info=True)
    if still_running:  # pragma: no cover - defensive logging
        logger.warning(
            "[Slack] %d Socket Mode task(s) did not stop within %.1fs", len(still_running),
            _SOCKET_TASK_CANCEL_TIMEOUT_S)


_SLACK_PROXY_HOSTS = ("slack.com", "files.slack.com", "wss-primary.slack.com")


def _resolve_slack_proxy_url() -> Optional[str]:
    """Resolve a proxy URL that Slack SDK clients can safely use."""
    proxy_url = resolve_proxy_url()
    if not proxy_url:
        return None
    normalized = proxy_url.lower()
    if not normalized.startswith(("http://", "https://")):
        logger.info(
            "[Slack] Ignoring unsupported proxy scheme for Slack transport: %s",
            safe_url_for_log(proxy_url))
        return None
    if any(is_host_excluded_by_no_proxy(host) for host in _SLACK_PROXY_HOSTS):
        logger.info("[Slack] NO_PROXY bypasses Slack proxy configuration")
        return None
    return proxy_url


def _slack_dedup_ttl_seconds() -> float:
    """Dedup window for Socket Mode replays (override: ``SLACK_DEDUP_TTL_SECONDS``).
    Slack replays un-acked events on reconnect, sometimes minutes later, so the window must span the
    worst-case gap; memory is bounded by ``MessageDeduplicator(max_size=...)``, not the TTL.

    See #4777.
    """
    raw = os.getenv("SLACK_DEDUP_TTL_SECONDS", "")
    if raw:
        try:
            value = float(raw)
            if value > 0:
                return value
        except ValueError:
            logger.warning("[Slack] Invalid SLACK_DEDUP_TTL_SECONDS=%r; using default", raw)
    return 3600.0  # 1 hour — covers Slack reconnect redelivery windows


# Audio mimetype → extension matching the container bytes: Slack voice clips are MP4/AAC, and
# OpenAI STT sniffs the container from the extension, so MP4 bytes cached as ``.ogg`` fail.
_SLACK_AUDIO_MIME_TO_EXT = {
    "audio/ogg": ".ogg", "audio/opus": ".ogg", "audio/mpeg": ".mp3", "audio/mp3": ".mp3",
    "audio/wav": ".wav", "audio/x-wav": ".wav", "audio/webm": ".webm", "audio/mp4": ".m4a",
    "audio/x-m4a": ".m4a", "audio/m4a": ".m4a", "audio/aac": ".m4a", "audio/flac": ".flac",
    "audio/x-flac": ".flac"}

# Extensions Whisper-family STT accepts (in sync with tools/transcription_tools.SUPPORTED_FORMATS).
_SLACK_STT_SUPPORTED_EXTS = frozenset(
    {".mp3", ".mp4", ".mpeg", ".mpga", ".m4a", ".wav", ".webm", ".ogg", ".aac", ".flac"})

# Cached extension → ``audio/*`` mimetype for ``video/mp4``-mislabeled voice clips (the STT gate
# keys on the ``audio/`` prefix). Unmapped → ``audio/mp4``.
_SLACK_EXT_TO_AUDIO_MIME = {
    ".mp4": "audio/mp4", ".m4a": "audio/mp4", ".mp3": "audio/mpeg", ".mpeg": "audio/mpeg",
    ".mpga": "audio/mpeg", ".wav": "audio/wav", ".webm": "audio/webm", ".ogg": "audio/ogg",
    ".aac": "audio/aac", ".flac": "audio/flac"}


def _resolve_slack_audio_ext(file_obj: Dict[str, Any], mimetype: str) -> str:
    """Pick a cache extension matching an inbound audio file's bytes.
    Order: STT-accepted filename ext → mimetype lookup → ``.m4a``. Never ``.ogg``: OpenAI rejects
    MP4/AAC bytes whose extension claims Ogg."""
    name_ext = os.path.splitext((file_obj.get("name") or "").strip())[1].lower()
    if name_ext in _SLACK_STT_SUPPORTED_EXTS:
        return name_ext
    mime_key = (mimetype or "").split(";", 1)[0].strip().lower()
    return _SLACK_AUDIO_MIME_TO_EXT.get(mime_key, ".m4a")


def _is_slack_voice_clip(file_obj: Dict[str, Any]) -> bool:
    """True for audio-only voice clips (``slack_audio`` subtype or ``audio_message*`` name).
    Slack sometimes reports them as ``video/mp4``, which would misroute them to video understanding
    instead of STT."""
    # slack_video clips carry a real video track — deliberately NOT matched.
    return (file_obj.get("subtype") or "").strip().lower() == "slack_audio" or (
        file_obj.get("name") or "").strip().lower().startswith("audio_message")


# content-type substring → upload filename extension (first match wins; default png).
_IMAGE_CT_EXTS = (("jpeg", "jpg"), ("jpg", "jpg"), ("gif", "gif"), ("webp", "webp"))

_TRANSIENT_UPLOAD_MARKERS = (
    "rate_limited", "ratelimited", "429", "connection reset", "service unavailable",
    "temporarily unavailable")


_SLACK_PERMISSION_ERRORS = frozenset(
    {"access_denied", "file_access_denied", "no_permission", "not_allowed_token_type", "restricted_action"}
)
_SLACK_HTTP_STATUS_TEMPLATES = {
    401: "Slack attachment access failed for {file_label} with HTTP 401. The bot token is not "
         "authorized for this file.",
    403: "Slack attachment access failed for {file_label} with HTTP 403. The bot likely lacks "
         "permission or scope to read this file.",
    404: "Slack attachment {file_label} returned HTTP 404 and is no longer reachable."}
# (error codes, user-facing template) for ``_describe_slack_api_error``; first match wins.
_SLACK_API_ERROR_TEMPLATES = (
    ({"not_authed", "invalid_auth", "account_inactive", "token_revoked"},
     "Slack attachment access failed for {file_label} because the bot token is not authorized "
     "({error}). Refresh the token/reinstall the app."),
    ({"file_not_found", "file_deleted"},
     "Slack attachment {file_label} is no longer available ({error})."),
    (_SLACK_PERMISSION_ERRORS,
     "Slack attachment access failed for {file_label} because the bot does not have permission "
     "({error}). Check workspace permissions/scopes and reinstall if needed."))


def _attachment_label(file_obj: Optional[Dict[str, Any]]) -> str:
    """Human label for a Slack file object in user-facing diagnostics."""
    return str((file_obj or {}).get("name") or (file_obj or {}).get("id") or "this attachment")


def _is_transient_transport_error(e: BaseException) -> bool:
    """Timeout or aiohttp connection error that is NOT a permanent TLS failure.
    ``aiohttp`` is looked up via ``globals()`` so tests can stub/remove it."""
    aiohttp_module = globals().get("aiohttp")
    connection_error_type = getattr(aiohttp_module, "ClientConnectionError", None)
    permanent_tls_error_types = tuple(
        error_type
        for error_type in (
            getattr(aiohttp_module, "ClientSSLError", None),
            getattr(aiohttp_module, "ServerFingerprintMismatch", None))
        if isinstance(error_type, type))
    is_permanent_tls_error = bool(permanent_tls_error_types) and isinstance(
        e, permanent_tls_error_types)
    return isinstance(e, TimeoutError) or (
        isinstance(connection_error_type, type)
        and isinstance(e, connection_error_type)
        and not is_permanent_tls_error)


def _extra_or_env_flag_getter(key: str, env_var: str, *, strip: bool = False) -> Callable[..., bool]:
    """Method factory: ``self._extra_or_env_flag(key, env_var, strip=strip)``."""

    def getter(self) -> bool:
        return self._extra_or_env_flag(key, env_var, strip=strip)

    getter.__name__ = f"_slack_{key}"
    return getter


def _extra_or_env_channel_set_getter(
    key: str, env_var: str, *, coerce_scalar: bool = False) -> Callable[..., set]:
    """Method factory: ``self._extra_or_env_channel_set(key, env_var, coerce_scalar=...)``."""

    def getter(self) -> set:
        return self._extra_or_env_channel_set(key, env_var, coerce_scalar=coerce_scalar)

    getter.__name__ = f"_slack_{key}"
    return getter


from .adapter_lifecycle import SlackLifecycleMixin
from .adapter_delivery import SlackDeliveryMixin
from .adapter_format import SlackFormatMixin
from .adapter_context import SlackContextMixin
from .adapter_events import SlackEventsMixin
from .adapter_prompts import SlackPromptsMixin
from .adapter_commands import SlackCommandsMixin


class SlackAdapter(
    SlackLifecycleMixin, SlackDeliveryMixin, SlackFormatMixin, SlackContextMixin, SlackEventsMixin, SlackPromptsMixin, SlackCommandsMixin,
    BasePlatformAdapter,
):
    """Slack bot adapter (Socket Mode).
    Needs SLACK_BOT_TOKEN (xoxb-, API calls) and SLACK_APP_TOKEN (xapp-, Socket Mode). DMs +
    mention-gated channels, threads, attachments, slash commands, status text."""

    MAX_MESSAGE_LENGTH = 39000  # Slack API allows 40,000 chars; leave margin
    supports_code_blocks = True  # Slack mrkdwn renders fenced code blocks
    # Typing indicator is a text status line (assistant.threads.setStatus): fed live phrases.
    supports_status_text = True
    splits_long_messages = True  # send() chunks via truncate_message(MAX_MESSAGE_LENGTH)
    # Slack rejects slash commands inside threads; "!" is rewritten to "/" for known commands.
    typed_command_prefix = "!"
    # ``reply_in_thread: false`` gives both a flat outbound reply and a whole-channel
    # session bucket, so a flat continuable cron continues on a plain reply.
    supports_inchannel_continuable = True

    # Bounded-cache caps (instance assignment in tests overrides per adapter).
    _USER_NAME_CACHE_MAX = _CHANNEL_NAME_CACHE_MAX = _DM_CONVERSATION_CACHE_MAX = 5000
    _PROCESSED_MESSAGE_TS_MAX = _BOT_TS_MAX = _MENTIONED_THREADS_MAX = 5000
    _ASSISTANT_THREADS_MAX = _AGENT_VIEW_CONTEXTS_MAX = _THREAD_REHYDRATION_CHECKED_MAX = 5000
    _REACTING_MESSAGE_IDS_MAX = _TITLED_ASSISTANT_THREADS_MAX = 5000
    _CHANNEL_TEAM_MAX = 10000
    _APPROVAL_RESOLVED_MAX = _CLARIFY_RESOLVED_MAX = _ACTIVE_STATUS_THREADS_MAX = 1000
    # Tighter cap than the approval/clarify dicts: each entry holds the
    # full provider list, and a picker is only live for minutes.
    _MODEL_PICKER_STATE_MAX = 100
    _STATUS_MESSAGE_IDS_MAX = 2000
    _THREAD_CACHE_MAX = 2500
    _THREAD_CACHE_TTL = 60.0
    # Watchdog: poll interval; reconnect after N ping_intervals of silence (Slack pings idle
    # sockets, so silence = wedged transport); grace after (re)connect for the first ping/pong.
    _socket_watchdog_interval_s = 15.0
    _socket_ping_stale_factor = 4
    _socket_first_ping_grace_s = 60.0

    async def _stop_native_task_card_stream(
        self, key: Tuple[str, str, str], stream: _NativeTaskCardStream) -> None:
        async with stream.lock:
            if stream.stopped:
                return
            stream.stopped = True
            try:
                if self._app and stream.stream_ts:
                    await self._get_client(stream.channel, team_id=stream.team_id).api_call(
                        "chat.stopStream", json={"channel": stream.channel, "ts": stream.stream_ts})
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.debug("[Slack] Native task-card stopStream failed: %s", exc)
            finally:
                if self._native_task_card_streams.get(key) is stream:
                    self._native_task_card_streams.pop(key, None)

    def __init__(self, config: PlatformConfig):
        super().__init__(config, Platform.SLACK)
        self._app: Optional[Any] = None
        self._handler: Optional[Any] = None
        self._socket_mode_task: Optional[asyncio.Task] = None
        # Bot identity per workspace (team_id → WebClient / bot_user_id / display name), so the
        # agent never mistakes a human's mention for itself; primary workspace identity separate.
        self._bot_user_id: Optional[str] = None
        self._bot_display_name: Optional[str] = None
        self._team_clients: Dict[str, Any] = {}
        self._team_bot_user_ids: Dict[str, str] = {}
        self._team_bot_names: Dict[str, str] = {}
        # User/channel IDs are workspace-local: name/is_bot caches key by (team_id, id) so
        # multi-workspace processes never reuse another tenant's names (is_bot catches peer-agent
        # posts lacking bot_id/bot_message markers; DM channel IDs are per-user, hence bounded).
        self._user_name_cache: Dict[Tuple[str, str], str] = {}
        self._channel_name_cache: Dict[Tuple[str, str], str] = {}
        self._user_is_bot_cache: Dict[Tuple[str, str], bool] = {}
        # channel_id → owning team_id (bounded; re-learned on the next event, _get_client falls
        # back to primary). Kept only while exactly one workspace claims the id — _channel_teams
        # holds all claimants; an ambiguous id is dropped, not last-writer-wins.
        self._channel_team: Dict[str, str] = {}
        self._channel_teams: Dict[str, set] = {}
        # user target (team_id:user_id) → opened DM conversation ID (D...)
        self._dm_conversation_cache: Dict[str, str] = {}
        # Dedup for Socket Mode reconnect replays; TTL must outlast the worst-case
        # redelivery gap (max_size bounds memory, so a long window is safe).
        # Dedup cache: prevents duplicate bot responses when Socket Mode reconnects redeliver events
        # (#4777).
        self._dedup = MessageDeduplicator(ttl_seconds=_slack_dedup_ttl_seconds())
        # ts of messages already routed to the agent, so later edits don't re-trigger a reply.
        self._processed_message_ts: Dict[str, float] = {}
        # approval / clarify message_ts (or (team_id, ts)) → resolved; blocks double-clicks.
        # Bounded: never-clicked prompts would otherwise leak forever.
        self._approval_resolved: Dict[Any, bool] = {}
        self._clarify_resolved: Dict[Any, bool] = {}
        # Model picker state keyed by workspace message marker (team_id, ts) →
        # picker context (providers, session_key, on_model_selected, stage).
        # The workspace marker prevents cross-tenant session resolution.
        self._model_picker_state: Dict[Any, dict] = {}
        # Bot-sent message ts / @mentioned threads: replies there get answered without a mention.
        self._bot_message_ts: set[str] = set()
        self._mentioned_threads: set[str] = set()
        # (team_id, channel_id, thread_ts) → Assistant thread metadata; lifecycle
        # events may precede message events and carry session-scoping identity.
        self._assistant_threads: Dict[Tuple[str, str, str], Dict[str, str]] = {}
        # Agent-view context per (team, user) — never global, so one person's split-view
        # context can't leak into another's prompt. Bridges lifecycle/message event ordering.
        self._agent_view_contexts: Dict[Tuple[str, str], Dict[str, str]] = {}
        # (channel, thread, status key) → last status bubble ts, so repeated
        # progress callbacks edit ONE message instead of spamming the thread.
        # Status-bubble dedup (issue #30045, extended to Slack): remember the message ts of the last status
        # bubble per (channel, thread, status key) so repeated progress callbacks (compression retries,
        # fallback switches, ...) edit ONE message in place instead of appending a new bubble per event —
        # long retry loops used to spam threads with dozens of out-of-order status messages.
        self._status_message_ids: Dict[Tuple[str, str, str], str] = {}
        self._thread_context_cache: Dict[str, _ThreadContextCache] = {}
        # Threads already rehydration-checked this process (first reply after a restart injects
        # missed messages exactly once); message IDs with reaction lifecycle (bounded: an exception
        # between add and finalize would leak entries).
        # Persistent sessions survive gateway restarts, but messages that arrived while the gateway was DOWN
        # never reached the session. Keys follow the thread session-key scoping. See #63530.
        self._thread_rehydration_checked: set = set()
        self._reacting_message_ids: set = set()
        # Active Assistant statuses by (team_id, channel_id, thread_ts) so cleanup
        # can't clear an overlapping Slack Connect workspace; evicted oldest-thread-first.
        self._active_status_threads: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
        # Native progress streams; each owns a lock so concurrent start/append/stop
        # can't create duplicates or append after finalization.
        self._native_task_card_streams: Dict[Tuple[str, str, str], _NativeTaskCardStream] = {}
        # Guard: set the Slack AI thread title once per DM thread, not per reply.
        self._titled_assistant_threads: set = set()
        # Slash-command contexts so send() can route the first reply ephemerally. Keyed
        # (team_id, channel_id, user_id), two-part when no team id → {"response_url", "ts"}.
        self._slash_command_contexts: Dict[Tuple[str, ...], Dict[str, Any]] = {}
        # Native streaming state per chat_id: {"ts", "draft_id", "sent", "started"}.
        # ``sent`` is raw pre-mrkdwn text; the API is append-only so deltas diff against it.
        self._active_streams: Dict[str, Dict[str, Any]] = {}
        # Set once startStream reports the app lacks streaming (Agents & AI Apps
        # off / missing scope); later responses skip straight to edit-based streaming.
        self._native_stream_unsupported = False
        # Socket Mode self-healing state for silently dropped websockets; the monotonic
        # start time is the grace window for the first ping/pong.
        self._app_token: Optional[str] = None
        self._proxy_url: Optional[str] = None
        self._socket_watchdog_task: Optional[asyncio.Task] = None
        self._socket_reconnect_lock = asyncio.Lock()
        self._socket_handler_started_monotonic: Optional[float] = None

    # Slash-command ephemeral helpers. response_url is valid 30 min; the much shorter TTL avoids
    # routing unrelated messages as ephemeral after a slow/dropped handler. Hard cap because TTL
    # cleanup only runs on lookup, so never-replied contexts would accumulate.
    _SLASH_CTX_TTL = 120.0
    _SLASH_CTX_MAX = 1000

    # Native streaming (chat.startStream/appendStream/stopStream). Unlike Telegram drafts a Slack
    # stream IS the final message: ``send()`` seals it instead of posting a duplicate. Needs the
    # Agents & AI Apps feature; a feature error sets ``_native_stream_unsupported`` → edit-based.
    # Cursor glyphs (streaming.cursor) are stripped before deltas because the API is append-only.
    _STREAM_CURSOR_GLYPHS = ("\u2589", "▍", "▌", "…")
    _NATIVE_STREAM_UNSUPPORTED_MARKERS = (
        "not_allowed", "missing_scope", "feature_not_enabled", "invalid_method", "unknown_method",
        "method_deprecated", "not_authed", "streaming_not_allowed")

    # ----- Markdown → mrkdwn conversion -----

    # Slack caps the cumulative text of all ``markdown`` blocks in a single
    # payload at 12,000 characters.  Leave margin for the feedback block.
    _MARKDOWN_BLOCK_MAX = 11_500

    # ----- Reactions -----

    # ----- User identity resolution -----

    # ----- Internal handlers -----

    # Reaction names → unicode emoji, so skills matching on ``text`` see the same character
    # whether the user typed it or reacted with it.
    _REACTION_EMOJI_MAP: ClassVar[Dict[str, str]] = {
        "thumbsup": "👍", "+1": "👍", "thumbsdown": "👎", "-1": "👎", "white_check_mark": "✅",
        "heavy_check_mark": "✅", "x": "❌", "no_entry": "⛔", "warning": "⚠️", "rotating_light": "🚨",
        "eyes": "👀", "rocket": "🚀", "tada": "🎉", "fire": "🔥", "wave": "👋"}

    # ----- Approval button support (Block Kit) -----

    # Button action_id → choice, and choice → outcome text (``{user}`` = clicker's name).
    _APPROVAL_CHOICES: ClassVar[Dict[str, str]] = {
        "hermes_approve_once": "once", "hermes_approve_session": "session",
        "hermes_approve_always": "always", "hermes_deny": "deny"}
    _APPROVAL_DECISIONS: ClassVar[Dict[str, str]] = {
        "once": "✅ Approved once by {user}", "session": "✅ Approved for session by {user}",
        "always": "✅ Approved permanently by {user}", "deny": "❌ Denied by {user}"}
    _CONFIRM_CHOICES: ClassVar[Dict[str, str]] = {
        "hermes_confirm_once": "once", "hermes_confirm_always": "always",
        "hermes_confirm_cancel": "cancel"}
    _CONFIRM_DECISIONS: ClassVar[Dict[str, str]] = {
        "once": "✅ Approved once by {user}", "always": "🔒 Always approved by {user}",
        "cancel": "❌ Cancelled by {user}"}

    # ----- Thread context fetching -----

    # Slack CDN hosts (``files.slack.com``, Enterprise Grid ``*.slack.com``, legacy
    # ``*.slack-files.com``). Downloads send the bot token as a Bearer header, so a forged URL
    # could exfiltrate it to ANY host; the private-IP SSRF check cannot close that hole.
    _SLACK_CDN_HOST_SUFFIXES = (".slack.com", ".slack-files.com")
    _SLACK_CDN_EXACT_HOSTS = frozenset({"slack.com", "slack-files.com"})

    # ── Channel mention gating ─────────────────────────────────────────────

    # Opt-in flags (``config.extra[key]`` else ``SLACK_*`` env). strict_mention: every thread
    # message needs an explicit @-mention (no auto-triggers); ignore_other_user_mentions: silent
    # when the *leading* token @-mentions someone else; thread_require_mention: thread replies
    # need an @-mention even in free-response channels; disable_dms: incoming DMs are ignored.
    _slack_strict_mention = _extra_or_env_flag_getter("strict_mention", "SLACK_STRICT_MENTION")
    _slack_ignore_other_user_mentions = _extra_or_env_flag_getter(
        "ignore_other_user_mentions", "SLACK_IGNORE_OTHER_USER_MENTIONS")
    _slack_thread_require_mention = _extra_or_env_flag_getter(
        "thread_require_mention", "SLACK_THREAD_REQUIRE_MENTION")
    _slack_disable_dms = _extra_or_env_flag_getter("disable_dms", "SLACK_DISABLE_DMS", strip=True)

    # Channel-ID sets. free_response_channels: no @mention needed; allowed_channels: when set,
    # other channels are ignored even if @mentioned (DMs gated by disable_dms);
    # require_mention_channels: @mention ALWAYS required, overriding ``require_mention: false``
    # and free_response_channels (wake checks still apply); ignored_channels: never touched.
    _slack_free_response_channels = _extra_or_env_channel_set_getter(
        "free_response_channels", "SLACK_FREE_RESPONSE_CHANNELS", coerce_scalar=True)
    _slack_allowed_channels = _extra_or_env_channel_set_getter(
        "allowed_channels", "SLACK_ALLOWED_CHANNELS")
    _slack_require_mention_channels = _extra_or_env_channel_set_getter(
        "require_mention_channels", "SLACK_REQUIRE_MENTION_CHANNELS")
    _slack_ignored_channels = _extra_or_env_channel_set_getter(
        "ignored_channels", "SLACK_IGNORED_CHANNELS", coerce_scalar=True)

# ── Plugin entry point + hooks (register, _standalone_send, interactive_setup,
# _apply_yaml_config, _is_connected, _build_adapter) ──────────────────────────


# Standalone-send cache: user ID -> DM conversation ID, keyed "{token}:{user_id}" (multi-workspace).
# ────────────────────────────────────────────────────────────────────────── Plugin migration glue (#41112 /
# #3823) Everything below this line was added when the Slack adapter moved from
# ``gateway/platforms/slack.py`` into this bundled plugin. It mirrors the Discord migration (PR #24356)
# exactly: a ``register(ctx)`` entry point plus the hook implementations (``_standalone_send``,
# ``interactive_setup``, ``_apply_yaml_config``, ``_is_connected``, ``_build_adapter``) that replace the
# per-platform core touchpoints (the ``Platform.SLACK`` elif in ``gateway/run.py``, the ``slack_cfg``
# YAML→env block in ``gateway/config.py``, the ``_setup_slack`` wizard + ``_PLATFORMS["slack"]`` static dict
# in ``hermes_cli/{setup,gateway}.py``, and the ``_send_slack`` dispatch in ``tools/send_message_tool.py``).
# ──────────────────────────────────────────────────────────────────────────
_slack_dm_cache: Dict[str, str] = {}
_SLACK_DM_CACHE_MAX = 5000


def _trim_slack_dm_cache() -> None:
    """Bound the module-level DM cache, oldest-insertion-first (C16 policy)."""
    while len(_slack_dm_cache) > _SLACK_DM_CACHE_MAX:
        _slack_dm_cache.pop(next(iter(_slack_dm_cache)))


# "Wrong workspace token for this channel" errors: worth retrying with the next token.
_WRONG_WORKSPACE_TOKEN_ERRORS = frozenset(
    {
        "invalid_auth", "not_authed", "token_revoked", "account_inactive", "not_in_channel",
        "channel_not_found"})


def _load_slack_bot_tokens(raw_token: str, *, quiet: bool) -> List[str]:
    """Comma-separated ``raw_token`` plus saved ``slack_tokens.json`` OAuth tokens (deduped, file
    order). ``quiet`` (standalone): no permission warning / per-token INFO; failures swallowed."""
    tokens = [t.strip() for t in raw_token.split(",") if t.strip()]
    try:
        from hermes_constants import get_hermes_home
        tokens_file = get_hermes_home() / "slack_tokens.json"
        present = tokens_file.exists()
    except Exception:
        if quiet:
            return tokens
        raise
    if not present:
        return tokens
    try:
        if not quiet:
            # File holds plaintext bot tokens; warn if group/world-readable.
            from utils import warn_if_credential_file_broadly_readable
            warn_if_credential_file_broadly_readable(tokens_file, label="[Slack]", log=logger)
        saved = json.loads(tokens_file.read_text(encoding="utf-8"))
        for team_id, entry in saved.items():
            tok = entry.get("token", "") if isinstance(entry, dict) else ""
            if tok and tok not in tokens:
                tokens.append(tok)
                if not quiet:
                    team_label = (
                        entry.get("team_name", team_id) if isinstance(entry, dict) else team_id)
                    logger.info("[Slack] Loaded saved token for workspace %s", team_label)
    except Exception as e:
        if not quiet:
            logger.warning("[Slack] Failed to read %s: %s", tokens_file, e)
    return tokens


def _standalone_proxy_kwargs() -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """``(session_kwargs, request_kwargs)`` for aiohttp honoring the configured proxy."""
    from gateway.platforms.base import proxy_kwargs_for_aiohttp
    return proxy_kwargs_for_aiohttp(resolve_proxy_url())


async def _slack_json_post(session, token: str, method: str, payload: dict, req_kw: dict) -> dict:
    """POST ``payload`` to ``https://slack.com/api/<method>`` with a bearer token; JSON body."""
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    async with session.post(
        f"https://slack.com/api/{method}", headers=headers, json=payload, **req_kw) as resp:
        return await resp.json()


async def _resolve_slack_user_dm(token: str, user_id: str) -> Optional[str]:
    """Resolve a user ID (U.../W...) to a DM conversation ID (D...) via ``conversations.open``;
    cached per (token, user). None on failure (e.g. missing ``im:write``)."""
    cache_key = f"{token}:{user_id}"
    if cache_key in _slack_dm_cache:
        return _slack_dm_cache[cache_key]
    try:
        import aiohttp
    except ImportError:
        return None
    try:
        _sess_kw, _req_kw = _standalone_proxy_kwargs()
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=15), **_sess_kw) as session:
            data = await _slack_json_post(
                session, token, "conversations.open", {"users": user_id}, _req_kw)
            if data.get("ok") and data.get("channel", {}).get("id"):
                channel_id = data["channel"]["id"]
                _slack_dm_cache[cache_key] = channel_id
                _trim_slack_dm_cache()
                return channel_id
            logger.warning(
                "[Slack] conversations.open failed for %s: %s", user_id,
                data.get("error", "unknown"))
            return None
    except Exception as e:
        logger.warning("[Slack] conversations.open exception for %s: %s", user_id, e)
        return None


def _standalone_post_kwargs(
    chat_id: str, text: Any, unfurl_kwargs: Dict[str, Any], thread_id: Optional[str]
) -> Dict[str, Any]:
    """``chat.postMessage`` kwargs for the standalone senders (key order is the wire order)."""
    kwargs: Dict[str, Any] = {"channel": chat_id, "text": text, "mrkdwn": True, **unfurl_kwargs}
    if thread_id:
        kwargs["thread_ts"] = thread_id
    return kwargs


async def _standalone_post_text(
    client, chat_id: str, text: Any, unfurl_kwargs: Dict[str, Any], thread_id: Optional[str]
) -> Dict[str, Any]:
    """``chat.postMessage`` via the SDK client; returns the response as a plain dict."""
    kwargs = _standalone_post_kwargs(chat_id, text, unfurl_kwargs, thread_id)
    return _slack_response_payload(await client.chat_postMessage(**kwargs))


async def _standalone_upload_file(
    client, chat_id: str, media_path: str, *, initial_comment: str = "",
    thread_id: Optional[str] = None) -> Dict[str, Any]:
    """Upload one local file via ``files_upload_v2`` (same API as the live adapter)."""
    kwargs: Dict[str, Any] = {
        "channel": chat_id, "file": media_path, "filename": os.path.basename(media_path),
        "initial_comment": initial_comment or ""}
    if thread_id:
        kwargs["thread_ts"] = thread_id
    result = await client.files_upload_v2(**kwargs)
    payload = _slack_response_payload(result)
    if payload.get("ok") is False:
        return {"error": f"Slack API error: {payload.get('error', 'unknown')}"}
    # files_upload_v2 responses vary by sdk version; prefer file timestamp when present.
    message_id = None
    if payload:
        file_obj = payload.get("file") or {}
        shares = file_obj.get("shares") or {}
        for share_bucket in shares.values():
            if isinstance(share_bucket, dict):
                for entries in share_bucket.values():
                    if isinstance(entries, list) and entries:
                        message_id = entries[0].get("ts") or message_id
                        break
            if message_id:
                break
        message_id = message_id or file_obj.get("timestamp") or payload.get("ts")
    return {"success": True, "message_id": message_id, "raw": result}


async def _standalone_send_media(
    token: str, chat_id: str, media_files: list, thread_id: Optional[str], formatted: Optional[str],
    formatted_caption: Optional[str], unfurl_kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Media branch of ``_standalone_send``: ``files_upload_v2`` per file (+ optional text post).
    ``caption`` rides as ``initial_comment`` on the first successful upload unless
    link-preview controls are explicit (the upload API cannot carry them)."""
    warnings: List[str] = []
    # Local import: tests inject a fake slack_sdk; a missing install gets a clean error.
    try:
        from slack_sdk.web.async_client import AsyncWebClient as _AsyncWebClient
    except ImportError:
        return {
            'error': "slack_sdk not installed. Run: pip install 'slack-sdk' (required for Slack MEDIA delivery via send_message)",
        }
    client = _AsyncWebClient(token=token)
    _apply_slack_proxy(client, resolve_proxy_url())
    last_message_id = None
    # The upload API cannot carry unfurl controls; explicit ones need a separate caption post.
    caption_as_upload_comment = bool(formatted_caption) and not unfurl_kwargs
    text_to_send = "" if caption_as_upload_comment else (formatted_caption or formatted or "")
    if text_to_send.strip():
        try:
            post_payload = await _standalone_post_text(
                client, chat_id, text_to_send, unfurl_kwargs, thread_id)
            if not post_payload.get("ok", True):
                return {"error": f"Slack API error: {post_payload.get('error', 'unknown')}"}
            last_message_id = post_payload.get("ts")
        except Exception as e:
            return {"error": f"Slack send failed: {e}"}
    caption_pending = caption_as_upload_comment
    uploaded_any = False
    for media_path, _is_voice in media_files:
        if not os.path.exists(media_path):
            warning = f"Media file not found, skipping: {media_path}"
            logger.warning("[Slack] %s", warning)
            warnings.append(warning)
            if caption_pending:
                # Deliver the caption even though the file is missing.
                try:
                    fb = await _standalone_post_text(
                        client, chat_id, formatted_caption, unfurl_kwargs, thread_id)
                    if fb.get("ok", True):
                        last_message_id = fb.get("ts") or last_message_id
                        caption_pending = False
                except Exception:
                    logger.warning(
                        "[Slack] Caption-fallback send failed for missing media", exc_info=True)
            continue
        try:
            upload_result = await _standalone_upload_file(
                client, chat_id, media_path,
                initial_comment=(formatted_caption or "") if caption_pending else "",
                thread_id=thread_id)
            if upload_result.get("error"):
                warnings.append(f"Failed to send media {media_path}: {upload_result['error']}")
                continue
            uploaded_any = True
            caption_pending = False
            last_message_id = upload_result.get("message_id") or last_message_id
        except Exception as e:
            warning = f"Failed to send media {media_path}: {e}"
            logger.error("[Slack] %s", warning, exc_info=True)
            warnings.append(warning)
    if last_message_id is None and not uploaded_any and not text_to_send.strip():
        result: Dict[str, Any] = {"error": "No deliverable text or media remained after processing"}
    else:
        result = {
            "success": True, "platform": "slack", "chat_id": chat_id, "message_id": last_message_id}
    if warnings:
        result["warnings"] = warnings
    return result


def _standalone_format_mrkdwn(text: str) -> str:
    """``format_message`` without a live adapter; falls back to the raw text."""
    if not text:
        return text
    try:
        return SlackAdapter.__new__(SlackAdapter).format_message(text)
    except Exception:
        logger.debug("Failed to apply Slack mrkdwn formatting in _standalone_send", exc_info=True)
        return text


async def _standalone_send(
    pconfig, chat_id, message, *, thread_id=None, media_files=None, force_document=False,
    caption=None):
    """Out-of-process delivery (``standalone_sender_fn``) for cron/tool processes not co-located
    with the gateway: text via ``chat.postMessage`` (aiohttp), media via ``files_upload_v2``."""
    del force_document  # signature parity with other standalone senders
    media_files = media_files or []
    # Under multiplex os.environ may hold ANOTHER profile's token: read via the secret scope.
    raw_token = getattr(pconfig, "token", None) or get_secret("SLACK_BOT_TOKEN", "")
    # Comma-separated multi-workspace list plus slack_tokens.json; no team map, so try each.
    tokens = _load_slack_bot_tokens(str(raw_token or ""), quiet=True)
    if not tokens:
        return {"error": "Slack send failed: SLACK_BOT_TOKEN not configured"}
    token = tokens[0]
    # Slack rejects bare user IDs (U.../W...) with channel_not_found; open the DM first.
    # User-targeted delivery: chat.postMessage / files_upload_v2 reject bare user IDs (U.../W...) — resolve
    # to a DM conversation ID (D...) first via conversations.open so `deliver=slack:U…` cron jobs reach the
    # user's DM instead of failing with channel_not_found (#17444).
    chat_id = str(chat_id or "")
    if chat_id[:1] in ("U", "W"):
        resolved = None
        for _tok in tokens:
            resolved = await _resolve_slack_user_dm(_tok, chat_id)
            if resolved is not None:
                token = _tok
                break
        if resolved is None:
            return {
                "error": (
                    f"Slack user ID resolution failed for {chat_id} "
                    "(conversations.open — check the bot's im:write scope)")}
        chat_id = resolved
    formatted = _standalone_format_mrkdwn(message) if message else message
    formatted_caption = _standalone_format_mrkdwn(caption) if caption else caption
    unfurl_kwargs = _slack_unfurl_kwargs(getattr(pconfig, "extra", None))
    if media_files:
        return await _standalone_send_media(
            token, chat_id, media_files, thread_id, formatted, formatted_caption, unfurl_kwargs)
    # --- Text-only path (existing aiohttp chat.postMessage) ---
    if not formatted or not formatted.strip():
        logger.debug("[Slack] _standalone_send: skipping empty/whitespace message")
        return {"success": True, "platform": "slack", "skipped": "empty_text"}
    try:
        import aiohttp
    except ImportError:
        return {"error": "aiohttp not installed. Run: pip install aiohttp"}
    try:
        _sess_kw, _req_kw = _standalone_proxy_kwargs()
        last_error = "unknown"
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30), **_sess_kw) as session:
            payload = _standalone_post_kwargs(chat_id, formatted, unfurl_kwargs, thread_id)
            for tok in tokens:
                data = await _slack_json_post(session, tok, "chat.postMessage", payload, _req_kw)
                if data.get("ok"):
                    return {
                        "success": True, "platform": "slack", "chat_id": chat_id,
                        "message_id": data.get("ts")}
                last_error = data.get("error", "unknown")
                if last_error not in _WRONG_WORKSPACE_TOKEN_ERRORS:
                    break
        return {"error": f"Slack API error: {last_error}"}
    except Exception as e:
        return {"error": f"Slack send failed: {e}"}


_SETUP_STEPS = (
    "Steps to create a Slack app:",
    "   1. Go to https://api.slack.com/apps → Create New App",
    "      Pick 'From an app manifest' — we'll generate one for you below.",
    "   2. Enable Socket Mode: Settings → Socket Mode → Enable",
    "      • Create an App-Level Token with 'connections:write' scope",
    "   3. Install to Workspace: Settings → Install App",
    "   4. After installing, invite the bot to channels: /invite @YourBot",)
_SETUP_HOME_CHANNEL_HELP = (
    "📬 Home Channel: where Hermes delivers cron job results,",
    "   cross-platform messages, and notifications.",
    "   To get a channel ID: open the channel in Slack, then right-click",
    "   the channel name → Copy link — the ID starts with C (e.g. C01ABC2DE3F).",
    "   You can also set this later by typing /set-home in a Slack channel.",)


def _write_slack_manifest_and_instruct() -> None:
    """Write the manifest under HERMES_HOME and print paste instructions; non-fatal."""
    from hermes_cli.cli_output import print_info, print_success, print_warning
    try:
        from hermes_cli.slack_cli import _build_full_manifest
        from hermes_constants import get_hermes_home
        manifest = _build_full_manifest(
            bot_name="Hermes", bot_description="Your Hermes agent on Slack")
        target = _Path(get_hermes_home()) / "slack-manifest.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print_success(f"Slack app manifest written to: {target}")
        print_info(
            "   Paste it into https://api.slack.com/apps → your app → Features "
            "→ App Manifest → Edit, then Save.  Slack will prompt to "
            "reinstall if scopes or slash commands changed.")
        print_info(
            "   Re-run `hermes slack manifest --write` anytime to refresh after "
            "Hermes adds new commands.")
    except Exception as e:
        print_warning(f"Could not write Slack manifest: {e}")


def interactive_setup() -> None:
    """Guide the user through Slack bot setup (manifest, tokens, allowlist, home channel).
    CLI helpers are lazy-imported to keep the plugin's import surface small."""
    from hermes_cli.config import get_env_value, remove_env_value, save_env_value
    from hermes_cli.cli_output import (
        prompt, prompt_yes_no, print_header, print_info, print_success, print_warning)

    print_header("Slack")
    if get_env_value("SLACK_BOT_TOKEN"):
        print_info("Slack: already configured")
        if not prompt_yes_no("Reconfigure Slack?", False):
            # Still offer a manifest refresh so new commands get registered.
            if prompt_yes_no(
                "Regenerate the Slack app manifest with the latest command "
                "list? (recommended after `hermes update`)", True):
                _write_slack_manifest_and_instruct()
            return
    for line in _SETUP_STEPS:
        print_info(line)
    print()
    print_info("   Full guide: https://hermes-agent.nousresearch.com/docs/user-guide/messaging/slack/")
    print()
    # Write the manifest up-front for the "Create from manifest" flow.
    _write_slack_manifest_and_instruct()
    print()
    bot_token = prompt("Slack Bot Token (xoxb-...)", password=True)
    if not bot_token:
        return
    save_env_value("SLACK_BOT_TOKEN", bot_token)
    app_token = prompt("Slack App Token (xapp-...)", password=True)
    if app_token:
        save_env_value("SLACK_APP_TOKEN", app_token)
    print_success("Slack tokens saved")
    print()
    print_info("🔒 Security: Restrict who can use your bot")
    print_info(
        "   To find a Member ID: click a user's name → View full profile → ⋮ → Copy member ID")
    print()
    allowed_users = prompt(
        "Allowed user IDs (comma-separated, leave empty to deny everyone except paired users)")
    if allowed_users:
        save_env_value("SLACK_ALLOWED_USERS", allowed_users.replace(" ", ""))
        print_success("Slack allowlist configured")
    else:
        print_warning("⚠️  No Slack allowlist set - unpaired users will be denied by default.")
        print_info(
            "   Set SLACK_ALLOW_ALL_USERS=true or GATEWAY_ALLOW_ALL_USERS=true only if you intentionally want open workspace access."
        )
    print()
    for line in _SETUP_HOME_CHANNEL_HELP:
        print_info(line)
    home_channel = prompt("Home channel ID (leave empty to set later with /set-home)").strip()
    if home_channel:
        save_env_value("SLACK_HOME_CHANNEL", home_channel)
    elif remove_env_value("SLACK_HOME_CHANNEL"):
        print_info("Home channel cleared.")


_YAML_BOOL_KEYS = (
    ("require_mention", "SLACK_REQUIRE_MENTION"), ("strict_mention", "SLACK_STRICT_MENTION"),
    ("ignore_other_user_mentions", "SLACK_IGNORE_OTHER_USER_MENTIONS"),
    ("thread_require_mention", "SLACK_THREAD_REQUIRE_MENTION"), ("allow_bots", "SLACK_ALLOW_BOTS"),
    ("reactions", "SLACK_REACTIONS"), ("disable_dms", "SLACK_DISABLE_DMS"))
# (yaml key, env var, list-ish types joined with ","); str(value) when not a list.
_YAML_LIST_KEYS = (
    ("free_response_channels", "SLACK_FREE_RESPONSE_CHANNELS", list),
    ("require_mention_channels", "SLACK_REQUIRE_MENTION_CHANNELS", list),
    ("reaction_triggers", "SLACK_REACTION_TRIGGERS", (list, tuple, set)),
    ("reaction_trigger_target", "SLACK_REACTION_TRIGGER_TARGET", ()),
    ("allowed_channels", "SLACK_ALLOWED_CHANNELS", list),
    ("ignored_channels", "SLACK_IGNORED_CHANNELS", list))


def _apply_yaml_config(yaml_cfg: dict, slack_cfg: dict) -> dict | None:
    """``apply_yaml_config_fn`` hook: ``slack:`` YAML keys → ``SLACK_*`` env vars (the adapter reads
    ``os.getenv()``; explicit env wins). Returns None: nothing is seeded into ``extra``.

    Implements the ``apply_yaml_config_fn`` contract (#24849). Mirrors the legacy ``slack_cfg`` block that
    used to live in ``gateway/config.py::load_gateway_config()`` before this migration.
    """
    for key, env in _YAML_BOOL_KEYS:
        if key in slack_cfg and not os.getenv(env):
            os.environ[env] = str(slack_cfg[key]).lower()
    for key, env, list_types in _YAML_LIST_KEYS:
        val = slack_cfg.get(key)
        if val is not None and not os.getenv(env):
            if list_types and isinstance(val, list_types):
                val = ",".join(str(v) for v in val)
            os.environ[env] = str(val)
    return None


def _is_connected(config) -> bool:
    """Connected when SLACK_BOT_TOKEN is set. Resolved through ``gateway_mod`` at call
    time (not a bound import) so tests patching ``get_env_value`` take effect."""
    import hermes_cli.gateway as gateway_mod
    return bool((gateway_mod.get_env_value("SLACK_BOT_TOKEN") or "").strip())


def _build_adapter(config):
    """Factory wrapper that constructs SlackAdapter from a PlatformConfig."""
    return SlackAdapter(config)


def register(ctx) -> None:
    """Plugin entry point — called by the Hermes plugin system."""
    ctx.register_platform(
        name="slack",
        label="Slack",
        adapter_factory=_build_adapter,
        check_fn=slack_deps_present,
        ensure_deps_fn=check_slack_requirements,
        is_connected=_is_connected,
        required_env=["SLACK_BOT_TOKEN", "SLACK_APP_TOKEN"],
        install_hint="Run `hermes setup` to install Slack support.",
        setup_fn=interactive_setup,
        # YAML→env bridge: config.yaml slack: keys → SLACK_* env vars read via os.getenv().
        # YAML→env config bridge — owns the translation of config.yaml slack: keys (require_mention,
        # strict_mention, ignore_other_user_mentions, thread_require_mention, allow_bots,
        # free_response_channels, reactions, disable_dms, allowed_channels, ignored_channels) into SLACK_*
        # env vars that the adapter reads via os.getenv(). Replaces the hardcoded block in
        # gateway/config.py. Hook contract: #24849.
        apply_yaml_config_fn=_apply_yaml_config,
        allowed_users_env="SLACK_ALLOWED_USERS",
        allow_all_env="SLACK_ALLOW_ALL_USERS",
        cron_deliver_env_var="SLACK_HOME_CHANNEL",
        # Out-of-process cron delivery; without it deliver=slack cron jobs fail with
        # "No live adapter" when cron runs apart from the gateway.
        standalone_sender_fn=_standalone_send,
        # Slack allows 40,000 chars; leave margin.
        max_message_length=39000,
        emoji="💼",
        allow_update_command=True)
