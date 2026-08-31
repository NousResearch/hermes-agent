"""ACP agent server — exposes Hermes Agent via the Agent Client Protocol."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import contextlib
import contextvars
import logging
import os
import threading
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable, Deque, Optional

import acp
from acp.schema import (
    AgentCapabilities, AgentMessageChunk, AuthenticateResponse, ClientCapabilities, ForkSessionResponse,
    Implementation, InitializeResponse, ListSessionsResponse, LoadSessionResponse, McpServerHttp, McpServerSse,
    McpServerStdio, ModelInfo, NewSessionResponse, PromptCapabilities, PromptResponse, ResumeSessionResponse,
    SessionCapabilities, SessionForkCapabilities, SessionInfo, SessionInfoUpdate, SessionListCapabilities,
    SessionMode, SessionModeState, SessionModelState, SessionResumeCapabilities, SetSessionConfigOptionResponse,
    SetSessionModeResponse, SetSessionModelResponse, TextContentBlock, Usage, UsageUpdate, UserMessageChunk,
)

from acp_adapter.auth import TERMINAL_SETUP_AUTH_METHOD_ID, build_auth_methods, detect_provider
from acp_adapter.commands import HERMES_VERSION, SlashCommandsMixin, _estimate_tokens
from acp_adapter.content import PromptBlock, _content_blocks_to_openai_user_content, _extract_text
from acp_adapter.events import (
    AssistantMessageIdAllocator, _build_plan_update_from_todo_result, make_message_cb, make_step_cb,
    make_thinking_cb, make_tool_progress_cb,
)
from acp_adapter.model_catalog import build_model_state, encode_model_choice
from acp_adapter.permissions import make_approval_callback
from acp_adapter.provenance import session_provenance_meta
from acp_adapter.session import SessionManager, SessionState, _expand_acp_enabled_toolsets
from acp_adapter.tools import build_tool_complete, build_tool_start, coerce_tool_args
from agent.context_compressor import (COMPRESSED_SUMMARY_METADATA_KEY, ContextCompressor)
from agent.interrupt_compat import request_hard_interrupt
from tools.approval_context import reset_hermes_interactive_context, set_hermes_interactive_context

logger = logging.getLogger(__name__)


def _named_custom_provider_catalogs() -> list[tuple[str, str, list[tuple[str, str]]]]:
    """Return ``(slug, label, [(model_id, description), ...])`` for named endpoints.

    Covers both the v12 ``providers:`` mapping and the legacy
    ``custom_providers:`` list.  These endpoints never appear in canonical
    provider enumeration, so without this the ACP model selector hides every
    named endpoint that the TUI ``/model`` picker already renders (#47039
    implemented named-endpoint rows for the TUI surface only).

    Model lists come from the entry's declared models (``default_model`` +
    ``models``), refreshed from the endpoint's live ``/models`` listing when a
    credential is available and ``discover_models`` is not disabled.  Declared
    models are kept even when live discovery fails — some OpenAI-compatible
    endpoints (e.g. Bedrock Mantle Responses) expose no ``/models`` route at
    all yet serve the declared models fine.

    Slugs use the ``custom:<name>`` shape that ``parse_model_input`` and
    ``resolve_runtime_provider`` already resolve, so encoded choice ids
    (``custom:<name>:<model>``) round-trip through ``set_session_model``
    unchanged.
    """
    try:
        from hermes_cli.config import (
            get_compatible_custom_providers,
            is_provider_enabled,
            load_config,
        )
        from hermes_cli.model_switch import (
            _NativePickerModelList,
            _declared_model_ids,
            _entry_models_discovered,
            _fetch_picker_live_models,
            _models_config_is_allowlist,
        )
        from hermes_cli.models import should_use_ollama_native_catalog
        from hermes_cli.providers import custom_provider_slug
    except ImportError:
        return []

    try:
        cfg = load_config()
        entries = get_compatible_custom_providers(cfg)
    except Exception:
        logger.debug("Could not load named custom providers", exc_info=True)
        return []

    # ``get_compatible_custom_providers`` drops the ``enabled`` flag during
    # normalization, so collect explicitly disabled provider keys from the
    # raw config and skip their entries below.
    disabled_keys: set[str] = set()
    raw_providers = cfg.get("providers") if isinstance(cfg, dict) else None
    if isinstance(raw_providers, dict):
        for raw_key, raw_entry in raw_providers.items():
            if isinstance(raw_entry, dict) and not is_provider_enabled(raw_entry):
                disabled_keys.add(str(raw_key).strip().lower())

    catalogs: list[tuple[str, str, list[tuple[str, str]]]] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        provider_key = str(entry.get("provider_key", "") or "").strip()
        if provider_key.lower() in disabled_keys:
            continue
        name = str(entry.get("name", "") or "").strip()
        base_url = str(entry.get("base_url", "") or "").strip()
        if not name or not base_url:
            continue
        slug = custom_provider_slug(name, provider_key)

        api_key = str(entry.get("api_key", "") or "").strip()
        if not api_key:
            key_env = str(
                entry.get("key_env") or entry.get("api_key_env") or ""
            ).strip()
            api_key = os.environ.get(key_env, "").strip() if key_env else ""

        declared: list[str] = []
        default_model = str(entry.get("model", "") or "").strip()
        if default_model:
            declared.append(default_model)
        models_cfg = entry.get("models")
        for mid in _declared_model_ids(models_cfg):
            if mid not in declared:
                declared.append(mid)

        native_headers = entry.get("extra_headers") or None
        native_catalog_provider = (
            provider_key
            if provider_key.lower() in {"ollama", "custom:ollama"}
            else "custom"
        )
        is_native_ollama = should_use_ollama_native_catalog(
            native_catalog_provider, base_url, headers=native_headers
        )
        explicit_catalog = _models_config_is_allowlist(
            models_cfg, _entry_models_discovered(entry)
        )
        if not api_key and not declared and not is_native_ollama:
            # No credential to discover with and nothing declared:
            # not addressable from the selector.
            continue

        model_ids = list(declared)
        discover = entry.get("discover_models", True)
        if isinstance(discover, str):
            discover = discover.lower() not in {"false", "no", "0"}
        native_catalog_provider = native_catalog_provider if is_native_ollama else "custom"
        live = None
        if discover and (api_key or is_native_ollama):
            try:
                live = _fetch_picker_live_models(
                    api_key,
                    base_url,
                    native_catalog_provider,
                    explicit_catalog,
                    headers=native_headers,
                    timeout=1.5,
                    api_mode=entry.get("api_mode"),
                )
            except Exception:
                live = None
            if live is not None:
                if isinstance(live, _NativePickerModelList):
                    model_ids = list(live)
                else:
                    model_ids = declared + [m for m in live if m not in declared]

        if not model_ids:
            if isinstance(live if "live" in locals() else None, _NativePickerModelList):
                catalogs.append((slug, name, []))
            continue
        catalogs.append((slug, name, [(mid, "") for mid in model_ids]))

    return catalogs

try:
    from hermes_cli import __version__ as HERMES_VERSION
except Exception:
    HERMES_VERSION = "0.0.0"

# Thread pool for running AIAgent (synchronous) in parallel.
_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="acp-agent")

# ListSessionsRequest has no client-side limit; clients paginate via `cursor`/`next_cursor`.
_LIST_SESSIONS_PAGE_SIZE = 50


def _flatten_history_text(value: Any) -> str:
    """Persisted content/reasoning (str, or list of ``{"text"}`` / ``{"type": "text", "content"}``
    parts) -> one stripped string; whitespace-only collapses to ``""`` ("nothing to emit")."""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
                elif item.get("type") == "text" and isinstance(item.get("content"), str):
                    parts.append(item["content"])
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(part.strip() for part in parts if part and part.strip()).strip()
    return ""


def _history_reasoning_text(message: dict[str, Any]) -> str:
    """First non-empty of ``reasoning_content`` and ``reasoning`` — both live keys, for
    different transports (not old-vs-new)."""
    for key in ("reasoning_content", "reasoning"):
        text = _flatten_history_text(message.get(key))
        if text:
            return text
    return ""


def _history_summary_meta(message: dict[str, Any], text: str) -> dict[str, Any] | None:
    """``_meta`` for a replayed compaction summary, else None.

    Summaries persist as ordinary messages, standalone (either role) or merged into the first
    preserved tail message. Two keys so clients can't hide real content: ``compactionSummary``
    (whole chunk; safe to collapse) vs ``containsCompactionSummary`` (real content + summary).
    Uses the in-process flag, falling back to content classification for DB-reloaded sessions."""
    kind = ContextCompressor.classify_summary_content(text)
    if kind is None and message.get(COMPRESSED_SUMMARY_METADATA_KEY):
        # Flagged but unclassified (prefix drift): the flag only marks summaries -> standalone.
        kind = "standalone"
    if kind == "standalone":
        return {"hermes": {"compactionSummary": True}}
    if kind == "merged":
        return {"hermes": {"containsCompactionSummary": True}}
    return None


# role -> (chunk class, session_update tag) for history replay.
_HISTORY_CHUNK_TYPES = {
    "user": (UserMessageChunk, "user_message_chunk"), "assistant": (AgentMessageChunk, "agent_message_chunk")
}


def _history_tool_call_name_args(tool_call: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Extract function name/arguments from an OpenAI-style tool_call."""
    function = tool_call.get("function") if isinstance(tool_call.get("function"), dict) else {}
    name = str(function.get("name") or tool_call.get("name") or "unknown_tool")
    raw_args = function.get("arguments") or tool_call.get("arguments") or tool_call.get("args") or {}
    return name, coerce_tool_args(raw_args)


def _history_message_chunk(role: str, message: dict[str, Any]) -> UserMessageChunk | AgentMessageChunk | None:
    text = _flatten_history_text(message.get("content"))
    if not text:
        return None

    parsed = urlparse(raw)
    if parsed.scheme and parsed.scheme != "file":
        return None

    if parsed.scheme == "file":
        if parsed.netloc and parsed.netloc not in {"", "localhost"}:
            return None
        path_text = unquote(parsed.path or "")
    else:
        path_text = unquote(raw)

    # file:///C:/Users/... or C:\Users\...
    if len(path_text) >= 3 and path_text[0] == "/" and path_text[2] == ":" and path_text[1].isalpha():
        drive = path_text[1].lower()
        rest = path_text[3:].lstrip("/\\").replace("\\", "/")
        return Path("/mnt") / drive / rest
    if len(path_text) >= 2 and path_text[1] == ":" and path_text[0].isalpha():
        drive = path_text[0].lower()
        rest = path_text[2:].lstrip("/\\").replace("\\", "/")
        return Path("/mnt") / drive / rest

    return Path(path_text)


def _decode_text_bytes(data: bytes, mime_type: str | None) -> str | None:
    """Decode resource bytes if they are probably text; return None for binary."""
    if b"\x00" in data and not _is_text_resource(mime_type):
        return None
    for encoding in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def _format_resource_text(
    *,
    uri: str,
    body: str,
    name: str | None = None,
    title: str | None = None,
    note: str | None = None,
) -> str:
    display = _resource_display_name(uri, name=name, title=title)
    header = f"[Attached file: {display}]"
    if note:
        header += f" ({note})"
    return f"{header}\nURI: {uri}\n\n{body}"


def _resource_link_to_parts(block: ResourceContentBlock) -> list[dict[str, Any]]:
    """Convert an ACP resource_link block to OpenAI content parts.

    Returns a list of {"type": "text", ...} and/or {"type": "image_url", ...}
    parts. Image resources produce an image_url part with a small text header
    so the model knows which attachment it is. Non-image resources return a
    single text part with the inlined file body (or a binary-omit note).
    """
    uri = str(getattr(block, "uri", "") or "").strip()
    if not uri:
        return []

    name = str(getattr(block, "name", "") or "").strip() or None
    title = str(getattr(block, "title", "") or "").strip() or None
    mime_type = str(getattr(block, "mime_type", "") or "").strip() or None
    path = _path_from_file_uri(uri)

    if path is None:
        return [{
            "type": "text",
            "text": _format_resource_text(
                uri=uri,
                name=name,
                title=title,
                body="[Resource link only; Hermes cannot read non-file ACP resource URIs directly.]",
            ),
        }]

    # Image files: emit a short text header + image_url data URL so vision
    # models can see the attachment instead of a "binary omitted" note.
    image_mime = mime_type if _is_image_resource(mime_type) else _guess_image_mime_from_path(path)
    if image_mime and _is_image_resource(image_mime):
        try:
            size = path.stat().st_size
            if size > _MAX_ACP_RESOURCE_BYTES:
                return [{
                    "type": "text",
                    "text": _format_resource_text(
                        uri=uri,
                        name=name,
                        title=title,
                        body=f"[Image too large to inline: {size} bytes, cap={_MAX_ACP_RESOURCE_BYTES}]",
                    ),
                }]
            with path.open("rb") as fh:
                data = fh.read()
        except OSError as exc:
            logger.warning("ACP image resource read failed: %s", uri, exc_info=True)
            return [{
                "type": "text",
                "text": _format_resource_text(
                    uri=uri,
                    name=name,
                    title=title,
                    body=f"[Could not read attached image: {exc}]",
                ),
            }]
        display = _resource_display_name(uri, name=name, title=title)
        return [
            {"type": "text", "text": f"[Attached image: {display}]\nURI: {uri}"},
            {"type": "image_url", "image_url": {"url": _image_data_url(data, image_mime)}},
        ]

    try:
        size = path.stat().st_size
        read_size = min(size, _MAX_ACP_RESOURCE_BYTES)
        with path.open("rb") as fh:
            data = fh.read(read_size)
        text = _decode_text_bytes(data, mime_type)
        if text is None:
            return [{
                "type": "text",
                "text": _format_resource_text(
                    uri=uri,
                    name=name,
                    title=title,
                    body=f"[Binary file omitted: {size} bytes, mime={mime_type or 'unknown'}]",
                ),
            }]
        note = None
        if size > _MAX_ACP_RESOURCE_BYTES:
            note = f"truncated to {_MAX_ACP_RESOURCE_BYTES} of {size} bytes"
        return [{
            "type": "text",
            "text": _format_resource_text(uri=uri, name=name, title=title, body=text, note=note),
        }]
    except OSError as exc:
        logger.warning("ACP resource read failed: %s", uri, exc_info=True)
        return [{
            "type": "text",
            "text": _format_resource_text(
                uri=uri,
                name=name,
                title=title,
                body=f"[Could not read attached file: {exc}]",
            ),
        }]


def _embedded_resource_to_parts(block: EmbeddedResourceContentBlock) -> list[dict[str, Any]]:
    resource = getattr(block, "resource", None)
    if resource is None:
        return []

    uri = str(getattr(resource, "uri", "") or "").strip()
    mime_type = str(getattr(resource, "mime_type", "") or "").strip() or None

    if isinstance(resource, TextResourceContents):
        return [{"type": "text", "text": _format_resource_text(uri=uri, body=resource.text)}]

    if isinstance(resource, BlobResourceContents):
        blob = resource.blob or ""
        try:
            data = base64.b64decode(blob, validate=True)
        except Exception:
            data = blob.encode("utf-8", errors="replace")

        # Image blobs go through as image_url so vision models can see them.
        if _is_image_resource(mime_type):
            if len(data) > _MAX_ACP_RESOURCE_BYTES:
                return [{
                    "type": "text",
                    "text": _format_resource_text(
                        uri=uri,
                        body=f"[Embedded image too large to inline: {len(data)} bytes, cap={_MAX_ACP_RESOURCE_BYTES}]",
                    ),
                }]
            display = _resource_display_name(uri)
            return [
                {"type": "text", "text": f"[Attached image: {display}]" + (f"\nURI: {uri}" if uri else "")},
                {"type": "image_url", "image_url": {"url": _image_data_url(data, mime_type or "image/png")}},
            ]

        text = _decode_text_bytes(data[:_MAX_ACP_RESOURCE_BYTES], mime_type)
        if text is None:
            body = f"[Binary embedded file omitted: {len(data)} bytes, mime={mime_type or 'unknown'}]"
        else:
            body = text
            if len(data) > _MAX_ACP_RESOURCE_BYTES:
                body += f"\n\n[Truncated to {_MAX_ACP_RESOURCE_BYTES} of {len(data)} bytes]"
        return [{"type": "text", "text": _format_resource_text(uri=uri, body=body)}]

    text = getattr(resource, "text", None)
    if text:
        return [{"type": "text", "text": _format_resource_text(uri=uri, body=str(text))}]
    return []


def _extract_text(
    prompt: list[
        TextContentBlock
        | ImageContentBlock
        | AudioContentBlock
        | ResourceContentBlock
        | EmbeddedResourceContentBlock
    ],
) -> str:
    """Extract plain text from ACP content blocks for display/commands."""
    parts: list[str] = []
    for block in prompt:
        if isinstance(block, TextContentBlock):
            parts.append(block.text)
        elif hasattr(block, "text"):
            parts.append(str(block.text))
    return "\n".join(parts)


def _image_block_to_openai_part(block: ImageContentBlock) -> dict[str, Any] | None:
    """Convert an ACP image content block to OpenAI-style multimodal content."""
    data = str(getattr(block, "data", "") or "").strip()
    uri = str(getattr(block, "uri", "") or "").strip()
    mime_type = str(getattr(block, "mime_type", "") or "image/png").strip() or "image/png"

    if data:
        url = data if data.startswith("data:") else f"data:{mime_type};base64,{data}"
    elif uri:
        url = uri
    else:
        return None

    return {"type": "image_url", "image_url": {"url": url}}


def _content_blocks_to_openai_user_content(
    prompt: list[
        TextContentBlock
        | ImageContentBlock
        | AudioContentBlock
        | ResourceContentBlock
        | EmbeddedResourceContentBlock
    ],
) -> str | list[dict[str, Any]]:
    """Convert ACP prompt blocks into a Hermes/OpenAI-compatible user content payload."""
    parts: list[dict[str, Any]] = []
    text_parts: list[str] = []

    for block in prompt:
        if isinstance(block, TextContentBlock):
            if block.text:
                parts.append({"type": "text", "text": block.text})
                text_parts.append(block.text)
            continue
        if isinstance(block, ImageContentBlock):
            image_part = _image_block_to_openai_part(block)
            if image_part is not None:
                parts.append(image_part)
            continue
        if isinstance(block, ResourceContentBlock):
            resource_parts = _resource_link_to_parts(block)
            for part in resource_parts:
                parts.append(part)
                if part.get("type") == "text":
                    text_parts.append(part["text"])
            continue
        if isinstance(block, EmbeddedResourceContentBlock):
            resource_parts = _embedded_resource_to_parts(block)
            for part in resource_parts:
                parts.append(part)
                if part.get("type") == "text":
                    text_parts.append(part["text"])
            continue

    if not parts:
        return _extract_text(prompt)

    # Keep pure text prompts as strings so slash-command handling and text-only
    # providers keep the exact legacy path. Switch to structured content only
    # when an actual non-text block is present.
    if all(part.get("type") == "text" for part in parts):
        return "\n".join(text_parts)

    return parts


class HermesACPAgent(acp.Agent):
    """ACP Agent implementation wrapping Hermes AIAgent."""

    _SLASH_COMMANDS = {
        "help": "Show available commands",
        "model": "Show or change current model",
        "tools": "List available tools",
        "context": "Show conversation context info",
        "reset": "Clear conversation history",
        "compress": "Compress conversation context",
        "steer": "Inject guidance into the currently running agent turn",
        "queue": "Queue a prompt to run after the current turn finishes",
        "version": "Show Hermes version",
    }

    _ADVERTISED_COMMANDS = (
        {
            "name": "help",
            "description": "List available commands",
        },
        {
            "name": "model",
            "description": "Show current model and provider, or switch models",
            "input_hint": "model name to switch to",
        },
        {
            "name": "tools",
            "description": "List available tools with descriptions",
        },
        {
            "name": "context",
            "description": "Show conversation message counts by role",
        },
        {
            "name": "reset",
            "description": "Clear conversation history",
        },
        {
            "name": "compress",
            "description": "Compress conversation context",
        },
        {
            "name": "steer",
            "description": "Inject guidance into the currently running agent turn",
            "input_hint": "guidance for the active turn",
        },
        {
            "name": "queue",
            "description": "Queue a prompt to run after the current turn finishes",
            "input_hint": "prompt to run next",
        },
        {
            "name": "version",
            "description": "Show Hermes version",
        },
    )


def _history_replay_updates(history: list[dict[str, Any]]):
    """Yield ACP session updates that reconstruct a persisted transcript, in order: user/assistant
    text (with compaction ``_meta``), assistant thoughts, and tool-call start/complete pairs
    (``todo`` results also re-emit the plan)."""
    active_tool_calls: dict[str, tuple[str, dict[str, Any]]] = {}
    for message in history:
        role = str(message.get("role") or "")
        if role == "user":
            if (chunk := _history_message_chunk(role, message)) is not None:
                yield chunk
        elif role == "assistant":
            thought = _history_reasoning_text(message)
            if thought:
                yield acp.update_agent_thought_text(thought)
            if (chunk := _history_message_chunk(role, message)) is not None:
                yield chunk
            tool_calls = message.get("tool_calls")
            for tool_call in tool_calls if isinstance(tool_calls, list) else ():
                if not isinstance(tool_call, dict):
                    continue
                tool_call_id = str(
                    tool_call.get("id") or tool_call.get("call_id") or tool_call.get("tool_call_id") or ""
                ).strip()
                if not tool_call_id:
                    continue
                tool_name, args = _history_tool_call_name_args(tool_call)
                active_tool_calls[tool_call_id] = (tool_name, args)
                yield build_tool_start(tool_call_id, tool_name, args)
        elif role == "tool":
            tool_call_id = str(message.get("tool_call_id") or "").strip()
            tool_name = str(message.get("tool_name") or "").strip()
            function_args: dict[str, Any] | None = None
            if tool_call_id in active_tool_calls:
                tool_name, function_args = active_tool_calls.pop(tool_call_id)
            if not tool_call_id or not tool_name:
                continue
            result = message.get("content")
            result_text = result if isinstance(result, str) else None
            yield build_tool_complete(tool_call_id, tool_name, result=result_text, function_args=function_args)
            if tool_name == "todo":
                plan_update = _build_plan_update_from_todo_result(result_text)
                if plan_update is not None:
                    yield plan_update


def _mcp_server_config(server: McpServerStdio | McpServerHttp | McpServerSse) -> dict:
    if isinstance(server, McpServerStdio):
        return {"command": server.command, "args": list(server.args), "env": {i.name: i.value for i in server.env}}
    return {"url": server.url, "headers": {i.name: i.value for i in server.headers}}


def _restore_env(key: str, value: str | None) -> None:
    if value is None:
        os.environ.pop(key, None)
    else:
        os.environ[key] = value


def _bind_guarded(stack: contextlib.ExitStack, label: str, setup: Callable[[], Callable[[], None]]) -> None:
    """Run ``setup`` (returns its teardown) and register the teardown; failures in either half only
    log — the turn must still run without the binding."""
    try:
        teardown = setup()
    except Exception:
        logger.debug("Could not set ACP %s", label, exc_info=True)
        return

    def _teardown() -> None:
        try:
            teardown()
        except Exception:
            logger.debug("Could not restore ACP %s", label, exc_info=True)

    stack.callback(_teardown)


def _attach_interrupted_prompt(interrupted_prompt: str, guidance: str) -> str:
    return f"{interrupted_prompt}\n\nUser correction/guidance after interrupt: {guidance}"


def _take_interrupted_prompt(state: SessionState) -> tuple[bool, str]:
    """``(idle, interrupted_prompt)``; consumes the cancelled prompt only when the session is idle."""
    with state.runtime_lock:
        if state.is_running:
            return False, ""
        text, state.interrupted_prompt_text = state.interrupted_prompt_text, ""
        return True, text


@dataclass
class _TurnCallbacks:
    """Per-turn ACP streaming callbacks; all None when no client is connected."""

    tool_progress_cb: Any = None
    reasoning_cb: Any = None
    step_cb: Any = None
    stream_delta_cb: Any = None
    approval_cb: Any = None
    edit_approval_requester: Any = None
    streamed: bool = False


class HermesACPAgent(SlashCommandsMixin, acp.Agent):
    """ACP Agent implementation wrapping Hermes AIAgent."""

    _EDIT_APPROVAL_POLICY_CONFIG_ID = "edit_approval_policy"
    _EDIT_APPROVAL_POLICY_DEFAULT = "ask"
    _MODE_DEFAULT = "default"
    # mode id -> (edit approval policy, display name, description)
    _MODES: dict[str, tuple[str, str, str]] = {
        "default": ("ask", "Default", "Ask before edits."),
        "accept_edits": (
            "workspace_session",
            "Accept Edits",
            "Auto-allow workspace and /tmp edits; still asks for sensitive paths.",
        ),
        "dont_ask": (
            "session", "Don't Ask", "Auto-allow file edits for this session except sensitive paths."
        ),
    }
    _MODE_TO_EDIT_APPROVAL_POLICY = {mode: spec[0] for mode, spec in _MODES.items()}
    _EDIT_APPROVAL_POLICY_TO_MODE = {spec[0]: mode for mode, spec in _MODES.items()}

    def __init__(self, session_manager: SessionManager | None = None):
        super().__init__()
        self.session_manager = session_manager or SessionManager()
        self._conn: Optional[acp.Client] = None

    # ---- Connection lifecycle -----------------------------------------------

    def on_connect(self, conn: acp.Client) -> None:
        """Store the client connection for sending session updates."""
        self._conn = conn
        logger.info("ACP client connected")

    async def _send(self, session_id: str, update: Any, *, fail_msg: str, level: int = logging.WARNING) -> bool:
        """``session_update`` that logs instead of raising; False on failure."""
        try:
            await self._conn.session_update(session_id=session_id, update=update)
            return True
        except Exception:
            logger.log(level, fail_msg, session_id, exc_info=True)
            return False

    def _schedule_soon(self, make_coro: Callable[[], Any]) -> None:
        """Run a notification coroutine right after the current response is queued."""
        if not self._conn:
            return
        loop = asyncio.get_running_loop()
        loop.call_soon(asyncio.create_task, make_coro())

    def _session_modes(self, state: SessionState) -> SessionModeState:
        """Edit-approval policy as ACP modes. Zed renders ``config_options`` in the model
        picker's slot; modes (as Claude/Codex use) coexist with the picker."""
        current = str(getattr(state, "mode", "") or self._MODE_DEFAULT)
        if current not in self._MODES:
            current = self._MODE_DEFAULT
        return SessionModeState(
            current_mode_id=current,
            available_modes=[SessionMode(id=m, name=n, description=d) for m, (_p, n, d) in self._MODES.items()],
        )

    def _edit_approval_policy_for_state(self, state: SessionState) -> tuple[str, str | None]:
        mode = str(getattr(state, "mode", "") or self._MODE_DEFAULT)
        policy = self._MODE_TO_EDIT_APPROVAL_POLICY.get(mode, self._EDIT_APPROVAL_POLICY_DEFAULT)
        return policy, state.cwd

    def _build_model_state(self, state: SessionState) -> SessionModelState | None:
        """Authenticated providers + models, from the shared Hermes inventory (same substrate
        as ``hermes model``/TUI/dashboard) so the selector isn't just the current curated list."""
        model = str(state.model or getattr(state.agent, "model", "") or "").strip()
        provider = getattr(state.agent, "provider", None) or detect_provider() or "openrouter"
        try:
            from hermes_cli.inventory import build_models_payload, load_picker_context
            from hermes_cli.models import normalize_provider, provider_label

            normalized_provider = normalize_provider(provider)
            context = load_picker_context().with_overrides(
                current_provider=normalized_provider,
                current_model=model,
                current_base_url=str(getattr(state.agent, "base_url", "") or ""),
            )
            payload = build_models_payload(
                context,
                explicit_only=True,
                include_unconfigured=False,
                picker_hints=False,
                canonical_order=True,
                pricing=False,
                capabilities=False,
                refresh=False,
                probe_custom_providers=False,
                probe_current_custom_provider=False,
                max_models=ACP_MAX_MODELS_PER_PROVIDER,
            )

            available_models: list[ModelInfo] = []
            seen_ids: set[str] = set()
            current_choice_provider = str(provider or "").strip().lower()
            if current_choice_provider == "ollama":
                current_choice_provider = "custom:ollama"
            current_base_url = str(
                getattr(state.agent, "base_url", "") or ""
            ).strip().rstrip("/").lower()

            def semantic_provider(provider_id: str) -> str:
                raw = str(provider_id or "").strip().lower()
                if raw in {"ollama", "custom:ollama"}:
                    return "ollama"
                if raw.startswith("custom:"):
                    return raw
                return normalize_provider(raw)

            seen_semantic_ids: set[str] = set()
            native_empty_rows: set[str] = set()
            current_identity_resolved = current_choice_provider not in {"", "custom"}
            for row in payload.get("providers") or []:
                raw_row_provider = str(row.get("slug") or "").strip().lower()
                row_provider = normalize_provider(raw_row_provider)
                row_base_url = str(row.get("api_url") or "").strip().rstrip("/").lower()
                if row.get("native_catalog_empty"):
                    native_empty_rows.add(raw_row_provider)
                if (
                    not current_identity_resolved
                    and raw_row_provider in {"ollama", "custom:ollama"}
                    and current_base_url
                    and row_base_url == current_base_url
                ):
                    current_choice_provider = "custom:ollama"
                    current_identity_resolved = True
                if not row_provider:
                    continue
                provider_name = str(row.get("name") or "").strip() or provider_label(
                    row_provider
                )
                row_models = row.get("models")
                if not isinstance(row_models, (list, tuple)):
                    continue
                for model_entry in row_models:
                    if isinstance(model_entry, dict):
                        rendered_model = str(
                            model_entry.get("id")
                            or model_entry.get("model")
                            or model_entry.get("name")
                            or ""
                        ).strip()
                    else:
                        rendered_model = str(model_entry or "").strip()
                    if not rendered_model:
                        continue
                    encoded_provider = (
                        "custom:ollama"
                        if raw_row_provider == "ollama"
                        else raw_row_provider
                        if raw_row_provider == "custom:ollama"
                        else raw_row_provider
                        if raw_row_provider.startswith("custom:")
                        else row_provider
                    )
                    choice_id = self._encode_model_choice(
                        encoded_provider, rendered_model
                    )
                    semantic_id = f"{semantic_provider(encoded_provider)}:{rendered_model}"
                    if choice_id in seen_ids or semantic_id in seen_semantic_ids:
                        continue
                    is_current = (
                        semantic_provider(encoded_provider)
                        == semantic_provider(current_choice_provider)
                        and rendered_model == model
                    )
                    description = f"Provider: {provider_name}"
                    if is_current:
                        description += " • current"
                    available_models.append(
                        ModelInfo(
                            model_id=choice_id,
                            name=f"{provider_name} · {rendered_model}",
                            description=description,
                        )
                    )
                    seen_ids.add(choice_id)
                    seen_semantic_ids.add(semantic_id)

            # Named user-defined endpoints (providers: / custom_providers:)
            # are invisible to canonical provider enumeration — append them
            # so editor clients can select them like the TUI /model picker.
            named_empty_authoritative: set[str] = set(native_empty_rows)
            for named_slug, named_label, named_catalog in _named_custom_provider_catalogs():
                if not named_catalog:
                    named_empty_authoritative.add(str(named_slug).strip().lower())
                    continue
                for named_model, named_desc in named_catalog:
                    named_choice = self._encode_model_choice(named_slug, named_model)
                    named_semantic_id = (
                        f"{semantic_provider(named_slug)}:{named_model}"
                    )
                    if (
                        not named_choice
                        or named_choice in seen_ids
                        or named_semantic_id in seen_semantic_ids
                    ):
                        continue
                    named_parts = [f"Provider: {named_label}"]
                    if named_desc:
                        named_parts.append(str(named_desc).strip())
                    if named_slug == normalized_provider and named_model == model:
                        named_parts.append("current")
                    available_models.append(
                        ModelInfo(
                            model_id=named_choice,
                            name=named_model,
                            description=" • ".join(part for part in named_parts if part),
                        )
                    )
                    seen_ids.add(named_choice)
                    seen_semantic_ids.add(named_semantic_id)

            def empty_catalog_applies(provider_id: str) -> bool:
                raw = str(provider_id or "").strip().lower()
                normalized = normalize_provider(raw)
                if normalized == "custom":
                    return any(
                        candidate == raw
                        or f"custom:{candidate}" == raw
                        or (raw == "custom" and candidate == "custom")
                        for candidate in named_empty_authoritative
                    )
                return any(
                    candidate == raw
                    or candidate == f"custom:{normalized}"
                    or candidate == f"custom:{raw}"
                    or normalize_provider(candidate) == normalized
                    for candidate in named_empty_authoritative
                )

            def choice_provider(model_id: str) -> str:
                parts = model_id.split(":")
                if parts[:1] == ["custom"] and len(parts) > 1:
                    from hermes_cli.models import _configured_custom_provider_ids

                    lowered = model_id.lower()
                    for candidate in sorted(
                        (
                            provider_id
                            for provider_id in _configured_custom_provider_ids()
                            if provider_id.startswith("custom:")
                        ),
                        key=len,
                        reverse=True,
                    ):
                        if lowered.startswith(candidate + ":"):
                            return candidate
                    return "custom"
                return parts[0]

            if named_empty_authoritative:
                available_models = [
                    item
                    for item in available_models
                    if not empty_catalog_applies(choice_provider(item.model_id))
                ]
                seen_ids = {item.model_id for item in available_models}

            current_is_empty = empty_catalog_applies(current_choice_provider)
            if current_is_empty:
                available_models = [
                    item
                    for item in available_models
                    if " • current" not in str(item.description or "")
                ]
                seen_ids = {item.model_id for item in available_models}
            current_model_id = (
                "" if current_is_empty else self._encode_model_choice(current_choice_provider, model)
            )
            if (
                current_model_id
                and current_model_id not in seen_ids
                and not current_is_empty
            ):
                provider_name = provider_label(normalized_provider)
                available_models.insert(
                    0,
                    ModelInfo(
                        model_id=current_model_id,
                        name=f"{provider_name} · {model}",
                        description=f"Provider: {provider_name} • current",
                    ),
                )

            if not available_models and current_is_empty:
                return SessionModelState(available_models=[], current_model_id="")
            if available_models:
                return SessionModelState(
                    available_models=available_models,
                    current_model_id=current_model_id
                    if current_model_id or current_is_empty
                    else available_models[0].model_id,
                )
        except Exception:
            logger.debug("Could not build ACP model state", exc_info=True)

        if not model:
            return None
        choice = encode_model_choice(provider, model)
        return SessionModelState(available_models=[ModelInfo(model_id=choice, name=model)], current_model_id=choice)

    def _switch_model(
        self, state: SessionState, raw_model: str, *, keep_endpoint: bool = False
    ) -> tuple[str | None, str, str]:
        """Rebuild the session agent on a new model -> (old provider, new provider, model).

        Resolution goes through ``hermes_cli.model_switch.switch_model`` seeded with the live
        agent route — the same catalog/alias/credential validation as CLI/gateway/TUI ``/model``
        — so ACP never hands the session a model no provider can serve. ``provider:model`` picker
        ids become ``--provider``. ACP never persists. ``keep_endpoint`` carries base_url/api_mode
        over when the provider is unchanged."""
        from hermes_cli.config import get_compatible_custom_providers, load_config
        from hermes_cli.model_switch import switch_model
        from hermes_cli.models import parse_model_input

        current_provider = getattr(state.agent, "provider", None)
        explicit_provider, model_input = parse_model_input(raw_model, "")
        cfg = load_config()
        result = switch_model(
            raw_input=model_input, explicit_provider=explicit_provider,
            current_provider=current_provider or "openrouter", current_model=str(state.model or ""),
            current_base_url=str(getattr(state.agent, "base_url", "") or ""),
            current_api_key=str(getattr(state.agent, "api_key", "") or ""),
            user_providers=cfg.get("providers") if isinstance(cfg.get("providers"), dict) else {},
            custom_providers=get_compatible_custom_providers(cfg))
        if not result.success:
            raise ValueError(result.error_message or f"Cannot switch to {raw_model}")
        target_provider, new_model = result.target_provider, result.new_model
        state.model = new_model
        endpoint: dict[str, Any] = {}
        if keep_endpoint and not (current_provider and target_provider != current_provider):
            endpoint = {
                "base_url": getattr(state.agent, "base_url", None), "api_mode": getattr(state.agent, "api_mode", None)
            }
        state.agent = self.session_manager._make_agent(
            session_id=state.session_id, cwd=state.cwd, model=new_model,
            requested_provider=target_provider, **endpoint,
        )
        self.session_manager.save_session(state.session_id)
        return current_provider, target_provider, new_model

    @staticmethod
    def _build_usage_update(state: SessionState) -> UsageUpdate | None:
        """``usage_update`` for Zed's context indicator: ``size`` = context window, ``used`` =
        estimated request pressure (system prompt + history + tool schemas)."""
        compressor = getattr(state.agent, "context_compressor", None)
        size = int(getattr(compressor, "context_length", 0) or 0)
        if size <= 0:
            return None
        try:
            used = _estimate_tokens(state.history, state.agent)
        except Exception:
            logger.debug("Could not estimate ACP native context usage", exc_info=True)
            used = int(getattr(compressor, "last_prompt_tokens", 0) or 0)
        return UsageUpdate(session_update="usage_update", size=max(size, 0), used=max(used, 0))

    async def _send_usage_update(self, state: SessionState) -> None:
        if self._conn and (update := self._build_usage_update(state)) is not None:
            await self._send(state.session_id, update, fail_msg="Failed to send ACP usage update for session %s")

    def _provenance_meta(
        self, acp_session_id: str, current_hermes_session_id: str, previous_hermes_session_id: Optional[str] = None
    ) -> Optional[dict]:
        """Best-effort ``_meta.hermes.sessionProvenance`` for an ACP session."""
        try:
            return session_provenance_meta(
                self.session_manager._get_db(), acp_session_id, current_hermes_session_id,
                previous_hermes_session_id=previous_hermes_session_id,
            )
        except Exception:
            logger.debug("Could not build ACP session provenance for %s", acp_session_id, exc_info=True)
            return None

    async def _send_session_info_update(
        self, session_id: str, *,
        current_hermes_session_id: Optional[str] = None, previous_hermes_session_id: Optional[str] = None,
    ) -> None:
        """Session metadata update; pass ``previous_hermes_session_id`` when the internal head
        rotated (compression split) so provenance flags the reason."""
        if not self._conn:
            return
        try:
            row = self.session_manager._get_db().get_session(session_id)
        except Exception:
            logger.debug("Could not read ACP session info for %s", session_id, exc_info=True)
            return
        if not row:
            return
        title = row.get("title")
        # `sessions` has no `updated_at`; "now" is right since this fires when the title changed.
        update = SessionInfoUpdate(
            session_update="session_info_update",
            title=title if isinstance(title, str) and title.strip() else None,
            updated_at=datetime.now(timezone.utc).isoformat(),
            field_meta=self._provenance_meta(
                session_id, current_hermes_session_id or session_id, previous_hermes_session_id
            ),
        )
        await self._send(
            session_id, update, fail_msg="Could not send ACP session info update for %s", level=logging.DEBUG
        )

    async def _register_session_mcp_servers(
        self, state: SessionState, mcp_servers: list[McpServerStdio | McpServerHttp | McpServerSse] | None
    ) -> None:
        """Register ACP-provided MCP servers and refresh the agent tool surface."""
        if not mcp_servers:
            return
        try:
            from tools.mcp_tool_discovery import register_mcp_servers

            await asyncio.to_thread(register_mcp_servers, {s.name: _mcp_server_config(s) for s in mcp_servers})
        except Exception:
            logger.warning("Session %s: failed to register ACP MCP servers", state.session_id, exc_info=True)
            return
        try:
            from model_tools import get_tool_definitions
            from agent.memory_manager import inject_memory_provider_tools

            agent = state.agent
            agent.enabled_toolsets = _expand_acp_enabled_toolsets(
                getattr(agent, "enabled_toolsets", None) or ["hermes-acp"],
                mcp_server_names=[s.name for s in mcp_servers],
            )
            agent.tools = get_tool_definitions(
                enabled_toolsets=agent.enabled_toolsets,
                disabled_toolsets=getattr(agent, "disabled_toolsets", None), quiet_mode=True,
            )
            agent.valid_tool_names = {tool["function"]["name"] for tool in agent.tools or []}
            inject_memory_provider_tools(agent)
            if callable(invalidate := getattr(agent, "_invalidate_system_prompt", None)):
                invalidate()
            logger.info(
                "Session %s: refreshed tool surface after ACP MCP registration (%d tools)",
                state.session_id, len(agent.tools or []),
            )
        except Exception:
            logger.warning(
                "Session %s: failed to refresh tool surface after ACP MCP registration", state.session_id, exc_info=True,
            )

    def _schedule_mcp_late_refresh(self, state: SessionState) -> None:
        """Refresh the tool snapshot when background MCP discovery lands after agent build
        (``_make_agent`` only joins ~1.5s). Waits up to 30s off the critical path, then rebuilds
        via ``refresh_agent_mcp_tools`` (same as ``/reload-mcp``).

        Cache safety: only pre-first-turn (nothing cached yet); afterwards the snapshot stays
        frozen and late servers land via the between-turns prologue refresh
        (``agent/turn_context.py``). No-op if discovery finished, join timed out, registry
        unchanged, or session closed."""
        try:
            from hermes_cli.mcp_startup import mcp_discovery_in_flight
        except Exception:
            return
        if not mcp_discovery_in_flight():
            return
        agent, session_id = state.agent, state.session_id

        def _wait_then_refresh() -> None:
            try:
                from hermes_cli.mcp_startup import join_mcp_discovery

                if not join_mcp_discovery(timeout=30.0):
                    return

                # In-memory only: ``get_session()`` would restore from DB and build a new AIAgent.
                with self.session_manager._lock:
                    current = self.session_manager._sessions.get(session_id)
                if current is None or current.agent is not agent:
                    return

                # ``prompt()`` flips ``is_running`` under ``runtime_lock`` before dispatching, so
                # holding it here closes the window where a refresh would swap ``tools=`` mid-turn.
                with current.runtime_lock:
                    if current.is_running:
                        return
                    if any(int(getattr(agent, k, 0) or 0) > 0 for k in ("_user_turn_count", "_api_call_count")):
                        return

                    from tools.mcp_tool_agent import refresh_agent_mcp_tools

                    added = refresh_agent_mcp_tools(agent, quiet_mode=True)
                if added:
                    logger.info(
                        "Session %s: late MCP refresh added %d tools: %s",
                        session_id, len(added), ", ".join(sorted(added)),
                    )
            except Exception:
                logger.debug("Session %s: late MCP refresh failed", session_id, exc_info=True)

        threading.Thread(target=_wait_then_refresh, name=f"acp-mcp-late-refresh-{session_id}", daemon=True).start()

    # ---- ACP lifecycle ------------------------------------------------------

    async def initialize(
        self, protocol_version: int | None = None, client_capabilities: ClientCapabilities | None = None,
        client_info: Implementation | None = None, **kwargs: Any,
    ) -> InitializeResponse:
        auth_methods = build_auth_methods()
        logger.info(
            "Initialize from %s (protocol v%s)", client_info.name if client_info else "unknown",
            protocol_version if isinstance(protocol_version, int) else acp.PROTOCOL_VERSION,
        )

        return InitializeResponse(
            protocol_version=acp.PROTOCOL_VERSION,
            agent_info=Implementation(name="hermes-agent", version=HERMES_VERSION),
            agent_capabilities=AgentCapabilities(
                load_session=True,
                prompt_capabilities=PromptCapabilities(image=True),
                session_capabilities=SessionCapabilities(
                    fork=SessionForkCapabilities(), list=SessionListCapabilities(), resume=SessionResumeCapabilities(),
                ),
            ),
            auth_methods=auth_methods,
        )

    async def authenticate(self, method_id: str, **kwargs: Any) -> AuthenticateResponse | None:
        # Only acknowledge the method_id advertised in initialize().
        if not isinstance(method_id, str):
            return None
        normalized_method = method_id.strip().lower()
        provider = detect_provider()

        if normalized_method == TERMINAL_SETUP_AUTH_METHOD_ID:
            # Terminal auth runs setup out-of-band; succeed only once credentials exist.
            return AuthenticateResponse() if provider else None

        if not provider or normalized_method != provider:
            return None
        return AuthenticateResponse()

    # ---- Session management -------------------------------------------------

    async def _replay_session_history(self, state: SessionState) -> None:
        """Replay history as user/assistant/thought chunks plus reconstructed tool-call
        start/complete events so the editor shows the transcript, not a clean thread."""
        if not self._conn or not state.history:
            return
        for update in _history_replay_updates(state.history):
            if not await self._send(state.session_id, update, fail_msg="Failed to replay ACP history for session %s"):
                return

    async def _session_response_fields(self, state: SessionState, replay_verb: str | None = None) -> dict[str, Any]:
        """``models``/``modes``/``field_meta`` for session responses, after an optional history replay;
        schedules command advertisement + usage refresh.

        Per ACP spec, load/resume must stream history via ``session/update`` BEFORE responding
        (Codex/Claude Code/OpenCode/Zed rely on this; deferring via ``call_soon`` broke them).
        Best-effort: a corrupt message must not turn the load into an error."""
        if replay_verb:
            try:
                # Per ACP spec, `session/load` must stream the prior conversation back to the client via
                # `session/update` notifications BEFORE responding, so the client receives the full
                # transcript within the load request's lifetime. Awaiting the replay here matches Codex /
                # Claude Code / OpenCode / Pi and the Zed client (which registers the session-update routing
                # entry before awaiting the loadSession RPC specifically so in-call history replay updates
                # can find the thread). Deferring this via `loop.call_soon` (as we did briefly in May 2026)
                # broke every spec-compliant ACP client that measures notifications synchronously against
                # the load response — see #12285 follow-up.
                await self._replay_session_history(state)
            except Exception:
                logger.warning(
                    f"ACP history replay raised during session/{replay_verb} for %s — "
                    f"{replay_verb} will still succeed, partial transcript may be missing",
                    state.session_id, exc_info=True,
                )
        self._schedule_available_commands_update(state.session_id)
        self._schedule_soon(lambda: self._send_usage_update(state))
        return {
            "models": self._build_model_state(state),
            "modes": self._session_modes(state),
            "field_meta": self._provenance_meta(state.session_id, getattr(state.agent, "session_id", state.session_id)),
        }

    async def _attach_session_mcp(self, state: SessionState, mcp_servers: list | None, log: str, *log_args) -> None:
        await self._register_session_mcp_servers(state, mcp_servers)
        self._schedule_mcp_late_refresh(state)
        logger.info(log, *log_args)

    async def new_session(self, cwd: str, mcp_servers: list | None = None, **kwargs: Any) -> NewSessionResponse:
        state = self.session_manager.create_session(cwd=cwd)
        await self._attach_session_mcp(state, mcp_servers, "New session %s (cwd=%s)", state.session_id, cwd)
        return NewSessionResponse(session_id=state.session_id, **await self._session_response_fields(state))

    async def load_session(
        self, cwd: str, session_id: str, mcp_servers: list | None = None, **kwargs: Any
    ) -> LoadSessionResponse | None:
        state = self.session_manager.update_cwd(session_id, cwd)
        if state is None:
            logger.warning("load_session: session %s not found", session_id)
            return None
        await self._attach_session_mcp(state, mcp_servers, "Loaded session %s", session_id)
        return LoadSessionResponse(**await self._session_response_fields(state, "load"))

    async def resume_session(
        self, cwd: str, session_id: str, mcp_servers: list | None = None, **kwargs: Any
    ) -> ResumeSessionResponse:
        state = self.session_manager.update_cwd(session_id, cwd)
        if state is None:
            logger.warning("resume_session: session %s not found, creating new", session_id)
            state = self.session_manager.create_session(cwd=cwd)
        await self._attach_session_mcp(state, mcp_servers, "Resumed session %s", state.session_id)
        return ResumeSessionResponse(**await self._session_response_fields(state, "resume"))

    async def cancel(self, session_id: str, **kwargs: Any) -> None:
        state = self.session_manager.get_session(session_id)
        if not (state and state.cancel_event):
            return
        with state.runtime_lock:
            if state.is_running and state.current_prompt_text:
                state.interrupted_prompt_text = state.current_prompt_text
            # Cancel + hard-stop under the lock so no other prompt mistakes this turn for
            # redirectable work.
            state.cancel_event.set()
            try:
                if state.agent:
                    request_hard_interrupt(state.agent)
            except Exception:
                logger.debug("Failed to interrupt ACP session %s", session_id, exc_info=True)
        logger.info("Cancelled session %s", session_id)

    async def fork_session(
        self, cwd: str, session_id: str, mcp_servers: list | None = None, **kwargs: Any
    ) -> ForkSessionResponse:
        state = self.session_manager.fork_session(session_id, cwd=cwd)
        if state is None:
            logger.info("Forked session %s -> %s", session_id, "")
            return ForkSessionResponse(session_id="")
        await self._register_session_mcp_servers(state, mcp_servers)
        logger.info("Forked session %s -> %s", session_id, state.session_id)
        self._schedule_available_commands_update(state.session_id)
        return ForkSessionResponse(
            session_id=state.session_id, models=self._build_model_state(state), modes=self._session_modes(state)
        )

    async def list_sessions(
        self, cursor: str | None = None, cwd: str | None = None, **kwargs: Any
    ) -> ListSessionsResponse:
        """``cursor`` is a ``session_id`` returned as ``next_cursor``; results resume after it
        (unknown cursor -> empty page, never the full list). Pages cap at the fixed size."""
        infos = self.session_manager.list_sessions(cwd=cwd)

        if cursor:
            for idx, s in enumerate(infos):
                if s["session_id"] == cursor:
                    infos = infos[idx + 1:]
                    break
            else:
                infos = []

        has_more = len(infos) > _LIST_SESSIONS_PAGE_SIZE
        sessions = [
            SessionInfo(
                session_id=s["session_id"], cwd=s["cwd"], title=s.get("title"),
                updated_at=None if s.get("updated_at") is None else str(s["updated_at"]),
            )
            for s in infos[:_LIST_SESSIONS_PAGE_SIZE]
        ]
        next_cursor = sessions[-1].session_id if has_more and sessions else None
        return ListSessionsResponse(sessions=sessions, next_cursor=next_cursor)

    # ---- Prompt (core) ------------------------------------------------------

    def _rewrite_prompt_for_interrupt(
        self, state: SessionState, user_text: str, user_content: Any, text_only: bool
    ) -> tuple[str, Any]:
        """Idle ``/steer`` has nothing to inject into (gateway parity): if a prompt was just
        cancelled, replay it with the steer text as explicit correction; otherwise run the steer
        payload as a plain prompt rather than silently queueing it as if ``/queue`` was typed.
        Plain text after a cancel likewise keeps the cancelled request attached ("stop and
        send" clients) so deictic follow-ups have a target."""
        if not (text_only and isinstance(user_content, str)):
            return user_text, user_content

        if user_text.startswith("/steer"):
            split = user_text.split(maxsplit=1)
            steer_text = split[1].strip() if len(split) > 1 else ""
            if not steer_text:
                return user_text, user_content
            idle, interrupted_prompt = _take_interrupted_prompt(state)
            if interrupted_prompt:
                return (_attach_interrupted_prompt(interrupted_prompt, steer_text),) * 2
            return (steer_text, steer_text) if idle else (user_text, user_content)
        if not user_text.startswith("/") and (interrupted_prompt := _take_interrupted_prompt(state)[1]):
            return (_attach_interrupted_prompt(interrupted_prompt, user_text),) * 2
        return user_text, user_content

    def _claim_turn_or_queue(
        self, state: SessionState, session_id: str, user_text: str, user_content: Any, text_only: bool
    ) -> str | None:
        """Mark the session running; if a turn is active, redirect it (text-only, supported
        runtime) or queue it. Returns the client message when absorbed, else None."""
        with state.runtime_lock:
            if not state.is_running:
                state.is_running = True
                state.current_prompt_text = user_text or "[Image attachment]"
                return None
            if text_only and isinstance(user_content, str) and hasattr(state.agent, "redirect") and (
                getattr(state.agent, "_supports_active_turn_redirect", False) is True
            ):
                try:
                    if state.agent.redirect(user_content):
                        return "Redirected the active turn with your correction."
                except Exception:
                    logger.debug("ACP active-turn redirect failed for %s", session_id, exc_info=True)
            state.queued_prompts.append(user_text or "[Image attachment]")
            return f"Queued for the next turn. ({len(state.queued_prompts)} queued)"

    def _run_agent_turn(
        self, *, state: SessionState, session_id: str, user_text: str, user_content: Any, conn: Any,
        loop: asyncio.AbstractEventLoop, approval_cb: Any, edit_approval_requester: Any,
    ) -> dict:
        """Executor-thread body of one turn, run inside ``contextvars.copy_context()`` so
        ContextVar writes are isolated from concurrent sessions.

        Approval routing is thread-local, so it MUST be bound here, not on the loop thread.
        Interactive routing is a ``tools.approval`` contextvar, not ``HERMES_INTERACTIVE`` in
        os.environ, so concurrent workers can't race a global flag onto the non-interactive
        auto-approve path (GHSA-96vc-wcxf-jjff)."""
        agent = state.agent
        with contextlib.ExitStack() as stack:
            # HERMES_SESSION_KEY scopes per-session caches (interactive sudo password) to this
            # session, not the reused thread. ``cwd`` pins what the system prompt reports as the
            # working directory — otherwise it advertises the Hermes workspace while tools are
            # rooted at the client's project and edits land outside it. ``cron_session=""`` masks
            # any leaked process-global HERMES_CRON_SESSION.
            def _session_context() -> Callable[[], None]:
                from gateway.session_context import clear_session_vars, set_session_vars

                tokens = set_session_vars(
                    session_key=session_id, session_id=session_id, cwd=state.cwd, cron_session="",
                )
                return lambda: clear_session_vars(tokens)

            def _approval() -> Callable[[], None]:
                from tools import terminal_tool

                previous = terminal_tool._get_approval_callback()
                terminal_tool.set_approval_callback(approval_cb)
                return lambda: terminal_tool.set_approval_callback(previous)

            def _edit_approval() -> Callable[[], None]:
                from acp_adapter.edit_approval import reset_edit_approval_requester, set_edit_approval_requester

                token = set_edit_approval_requester(edit_approval_requester)
                return lambda: reset_edit_approval_requester(token)

            _bind_guarded(stack, "session context", _session_context)
            if approval_cb:
                _bind_guarded(stack, "approval callback", _approval)
            if edit_approval_requester:
                _bind_guarded(stack, "edit approval requester", _edit_approval)
            stack.callback(reset_hermes_interactive_context, set_hermes_interactive_context(True))
            # Tools tag side-effects with the ACP session (``kanban_create``); save/restore it.
            stack.callback(_restore_env, "HERMES_SESSION_ID", os.environ.get("HERMES_SESSION_ID"))
            os.environ["HERMES_SESSION_ID"] = session_id

            # Auto-titling fires in the turn prologue; push the title now as a session-info update.
            def _notify_title_update(_title: str, _source: str) -> None:
                if conn:
                    loop.call_soon_threadsafe(asyncio.create_task, self._send_session_info_update(session_id))

            agent._on_session_title = _notify_title_update
            try:
                return agent.run_conversation(
                    user_message=user_content, conversation_history=state.history, task_id=session_id,
                    persist_user_message=user_text or "[Image attachment]",
                )
            except Exception as e:
                logger.exception("Agent error in session %s", session_id)
                return {"final_response": f"Error: {e}", "messages": state.history}

    async def prompt(self, prompt: list[PromptBlock], session_id: str, **kwargs: Any) -> PromptResponse:
        """Run Hermes on the user's prompt and stream events back to the editor."""
        state = self.session_manager.get_session(session_id)
        if state is None:
            logger.error("prompt: session %s not found", session_id)
            return PromptResponse(stop_reason="refusal")

        user_text = _extract_text(prompt).strip()
        user_content = _content_blocks_to_openai_user_content(prompt)
        text_only_prompt = all(isinstance(block, TextContentBlock) for block in prompt)
        if not user_text and not (isinstance(user_content, list) and user_content):
            return PromptResponse(stop_reason="end_turn")

        user_text, user_content = self._rewrite_prompt_for_interrupt(state, user_text, user_content, text_only_prompt)

        # Slash commands are text-only; a prompt with media goes to the agent even if it starts with "/".
        if text_only_prompt and isinstance(user_content, str) and user_text.startswith("/"):
            # Off the loop: /model validates through switch_model (network I/O) and /compress
            # calls the LLM; handlers are sync and hold no loop-bound state.
            response_text = await asyncio.to_thread(self._handle_slash_command, user_text, state)
            if response_text is not None:
                if self._conn:
                    await self._conn.session_update(session_id, acp.update_agent_message_text(response_text))
                    await self._send_usage_update(state)
                return PromptResponse(stop_reason="end_turn")

        absorbed = self._claim_turn_or_queue(state, session_id, user_text, user_content, text_only_prompt)
        if absorbed is not None:
            if self._conn:
                await self._conn.session_update(session_id, acp.update_agent_message_text(absorbed))
            return PromptResponse(stop_reason="end_turn")

        logger.info("Prompt on session %s: %s", session_id, user_text[:100])
        conn, loop = self._conn, asyncio.get_running_loop()
        if state.cancel_event:
            state.cancel_event.clear()
        cbs = self._wire_turn_callbacks(state, session_id, conn, loop)

        def _run_agent() -> dict:
            return self._run_agent_turn(
                state=state, session_id=session_id, user_text=user_text, user_content=user_content, conn=conn,
                loop=loop, approval_cb=cbs.approval_cb, edit_approval_requester=cbs.edit_approval_requester,
            )

        try:
            # ACP `session_id` is the stable handle; agent.session_id is the internal head that
            # compression may rotate — snapshot it to detect rotation after the turn.
            pre_turn_hermes_id = getattr(state.agent, "session_id", None)
            # Fresh context copy: concurrent sessions on the shared executor must not share ContextVars.
            ctx = contextvars.copy_context()
            result = await loop.run_in_executor(_executor, ctx.run, _run_agent)
        except Exception:
            logger.exception("Executor error for session %s", session_id)
            with state.runtime_lock:
                state.is_running = False
                state.current_prompt_text = ""
            return PromptResponse(stop_reason="end_turn")

        return await self._finish_turn(state, session_id, conn, result, pre_turn_hermes_id, cbs.streamed)

    def _wire_turn_callbacks(
        self, state: SessionState, session_id: str, conn: Any, loop: asyncio.AbstractEventLoop
    ) -> _TurnCallbacks:
        """Install the ACP streaming callbacks on the session agent for one turn."""
        cbs = _TurnCallbacks()
        if conn:
            tool_call_ids: dict[str, Deque[str]] = defaultdict(deque)
            tool_call_meta: dict[str, dict[str, Any]] = {}
            policy_getter = lambda: self._edit_approval_policy_for_state(state)  # noqa: E731
            cbs.tool_progress_cb = make_tool_progress_cb(
                conn, session_id, loop, tool_call_ids, tool_call_meta, edit_approval_policy_getter=policy_getter
            )
            # Per-session allocator: a new turn must never reuse a previous turn's
            # assistant messageId (ACP clients replace the bubble with that id).
            if state.message_ids is None:
                state.message_ids = AssistantMessageIdAllocator()
            state.message_ids.close()  # new turn -> next chunk opens a fresh id
            cbs.reasoning_cb = make_thinking_cb(conn, session_id, loop, state.message_ids)
            cbs.step_cb = make_step_cb(conn, session_id, loop, tool_call_ids, tool_call_meta)
            message_cb = make_message_cb(conn, session_id, loop, state.message_ids)

            def stream_delta_cb(text: str) -> None:
                cbs.streamed = cbs.streamed or bool(text)
                message_cb(text)

            cbs.stream_delta_cb = stream_delta_cb
            cbs.approval_cb = make_approval_callback(conn.request_permission, loop, session_id)
            try:
                from acp_adapter.edit_approval import make_acp_edit_approval_requester

                cbs.edit_approval_requester = make_acp_edit_approval_requester(
                    conn.request_permission, loop, session_id, auto_approve_getter=policy_getter
                )
            except Exception:
                logger.debug("Could not create ACP edit approval requester", exc_info=True)

        agent = state.agent
        agent.tool_progress_callback = cbs.tool_progress_cb
        # Thought panes get provider reasoning only — no local status updates, no fake accordion.
        agent.thinking_callback = None
        agent.reasoning_callback, agent.step_callback = cbs.reasoning_cb, cbs.step_cb
        agent.stream_delta_callback = cbs.stream_delta_cb
        return cbs

    async def _finish_turn(
        self, state: SessionState, session_id: str, conn: Any, result: dict, pre_turn_hermes_id: Any,
        streamed_message: bool,
    ) -> PromptResponse:
        """Persist, emit provenance/final text, drain queued prompts, report usage."""
        # Key presence, not truthiness: ``messages=[]`` is a legitimate cleared transcript (#10844);
        # only a result without the key leaves the history untouched.
        if "messages" in result and isinstance(result["messages"], list):
            state.history = result["messages"]
            self.session_manager.save_session(session_id)

        # Head rotated (compression split): emit provenance so clients can render the boundary.
        post_turn_hermes_id = getattr(state.agent, "session_id", None)
        if conn and post_turn_hermes_id and pre_turn_hermes_id and post_turn_hermes_id != pre_turn_hermes_id:
            try:
                await self._send_session_info_update(
                    session_id, current_hermes_session_id=post_turn_hermes_id,
                    previous_hermes_session_id=pre_turn_hermes_id,
                )
            except Exception:
                logger.debug("Could not emit ACP provenance update after rotation for %s", session_id, exc_info=True)

        final_response = result.get("final_response", "")
        cancelled = bool(state.cancel_event and state.cancel_event.is_set())
        # The local "waiting for model" interrupt status is metadata, not prose; stop_reason carries it.
        from agent.conversation_loop import INTERRUPT_WAITING_FOR_MODEL_PREFIX

        interrupted = bool(result.get("interrupted")) or cancelled
        suppress = interrupted and final_response.startswith(INTERRUPT_WAITING_FOR_MODEL_PREFIX)
        # Send the final text unless already streamed — or if a plugin hook transformed it after.
        if final_response and conn and not suppress and (not streamed_message or result.get("response_transformed")):
            update = acp.update_agent_message_text(final_response)
            if state.message_ids is not None:
                # A plugin-rewritten reply replaces the streamed bubble (same id); an
                # unstreamed final response opens its own.
                if streamed_message and result.get("response_transformed"):
                    update.message_id = state.message_ids.last() or state.message_ids.current()
                else:
                    update.message_id = state.message_ids.current()
                state.message_ids.close()
            await conn.session_update(session_id, update)

        # Go idle before draining so recursive prompt() calls can acquire the session.
        with state.runtime_lock:
            state.is_running = False
            state.current_prompt_text = ""
        while True:
            with state.runtime_lock:
                if not state.queued_prompts:
                    break
                next_prompt = state.queued_prompts.pop(0)
            if conn:
                await conn.session_update(session_id, acp.update_user_message_text(next_prompt))
            await self.prompt(prompt=[TextContentBlock(type="text", text=next_prompt)], session_id=session_id)

        usage = None
        if any(result.get(k) is not None for k in ("prompt_tokens", "completion_tokens", "total_tokens")):
            usage = Usage(
                input_tokens=result.get("prompt_tokens", 0), output_tokens=result.get("completion_tokens", 0),
                total_tokens=result.get("total_tokens", 0), thought_tokens=result.get("reasoning_tokens"),
                cached_read_tokens=result.get("cache_read_tokens"),
            )
        await self._send_usage_update(state)
        return PromptResponse(stop_reason="cancelled" if cancelled else "end_turn", usage=usage)

    # ---- Session settings (ACP protocol methods) -----------------------------

    async def set_session_model(self, model_id: str, session_id: str, **kwargs: Any) -> SetSessionModelResponse | None:
        """Switch the model for a session (called by ACP protocol)."""
        state = self.session_manager.get_session(session_id)
        if state:
            # switch_model() does synchronous network I/O (models.dev, custom-endpoint probes,
            # ~10 s cold) — off the loop, like the gateway, so other ACP sessions keep flowing.
            _old, requested_provider, resolved_model = await asyncio.to_thread(
                self._switch_model, state, model_id, keep_endpoint=True)
            logger.info(
                "Session %s: model switched to %s via provider %s", session_id, resolved_model, requested_provider
            )
            return SetSessionModelResponse()
        logger.warning("Session %s: model switch requested for missing session", session_id)
        return None

    async def set_session_mode(self, mode_id: str, session_id: str, **kwargs: Any) -> SetSessionModeResponse | None:
        """Persist the editor-requested mode so ACP clients do not fail on mode switches."""
        state = self.session_manager.get_session(session_id)
        if state is None:
            logger.warning("Session %s: mode switch requested for missing session", session_id)
            return None
        normalized_mode = str(mode_id or "").strip()
        if normalized_mode not in self._MODES:
            normalized_mode = self._MODE_DEFAULT
        state.mode = normalized_mode
        self.session_manager.save_session(session_id)
        logger.info("Session %s: mode switched to %s", session_id, normalized_mode)
        return SetSessionModeResponse()

    async def set_config_option(
        self, config_id: str, session_id: str, value: str, **kwargs: Any
    ) -> SetSessionConfigOptionResponse | None:
        """Accept ACP config option updates even when Hermes has no typed ACP config surface yet."""
        state = self.session_manager.get_session(session_id)
        if state is None:
            logger.warning("Session %s: config update requested for missing session", session_id)
            return None

        if str(config_id) == self._EDIT_APPROVAL_POLICY_CONFIG_ID:
            state.mode = self._EDIT_APPROVAL_POLICY_TO_MODE.get(str(value), self._MODE_DEFAULT)
        else:
            options = getattr(state, "config_options", None)
            if not isinstance(options, dict):
                options = {}
            options[str(config_id)] = value
            state.config_options = options
        self.session_manager.save_session(session_id)
        logger.info("Session %s: config option %s updated", session_id, config_id)
        return SetSessionConfigOptionResponse(config_options=[])


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.
from acp.schema import AgentThoughtChunk  # noqa: F401,E402
from acp.schema import AudioContentBlock  # noqa: F401,E402
from acp.schema import AvailableCommand  # noqa: F401,E402
from acp.schema import AvailableCommandsUpdate  # noqa: F401,E402
from acp.schema import BlobResourceContents  # noqa: F401,E402
from acp.schema import EmbeddedResourceContentBlock  # noqa: F401,E402
from acp.schema import ImageContentBlock  # noqa: F401,E402
from pathlib import Path  # noqa: F401,E402
from acp.schema import ResourceContentBlock  # noqa: F401,E402
from acp.schema import TextResourceContents  # noqa: F401,E402
from acp.schema import UnstructuredCommandInput  # noqa: F401,E402
import base64  # noqa: F401,E402
import json  # noqa: F401,E402
from urllib.parse import unquote  # noqa: F401,E402
from urllib.parse import urlparse  # noqa: F401,E402


_PLUGIN_COMPAT_LAZY = {
    'ACP_MAX_MODELS_PER_PROVIDER': ('acp_adapter.model_catalog', 'ACP_MAX_MODELS_PER_PROVIDER'),
}


def __getattr__(name):  # PEP 562 — lazy so no import cycles
    target = _PLUGIN_COMPAT_LAZY.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib
    from hermes_cli.plugin_compat import warn_once
    warn_once(__name__, name, *target)
    return getattr(importlib.import_module(target[0]), target[1])
# ---- END PLUGIN-COMPAT ----
