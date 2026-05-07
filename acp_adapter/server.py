"""ACP agent server — exposes Hermes Agent via the Agent Client Protocol."""

from __future__ import annotations

import asyncio
import base64
import contextvars
import json
import logging
import os
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Deque, Optional
from urllib.parse import unquote, urlparse

import acp
from acp.schema import (
    AgentCapabilities,
    AgentMessageChunk,
    AuthenticateResponse,
    AvailableCommand,
    AvailableCommandsUpdate,
    BlobResourceContents,
    ClientCapabilities,
    EmbeddedResourceContentBlock,
    ForkSessionResponse,
    ImageContentBlock,
    AudioContentBlock,
    Implementation,
    InitializeResponse,
    ListSessionsResponse,
    LoadSessionResponse,
    McpServerHttp,
    McpServerSse,
    McpServerStdio,
    ModelInfo,
    NewSessionResponse,
    PromptCapabilities,
    PromptResponse,
    ResumeSessionResponse,
    SetSessionConfigOptionResponse,
    SetSessionModelResponse,
    SetSessionModeResponse,
    ResourceContentBlock,
    SessionCapabilities,
    SessionForkCapabilities,
    SessionListCapabilities,
    SessionModelState,
    SessionResumeCapabilities,
    SessionInfo,
    TextContentBlock,
    TextResourceContents,
    UnstructuredCommandInput,
    Usage,
    UsageUpdate,
    UserMessageChunk,
)

# AuthMethodAgent was renamed from AuthMethod in agent-client-protocol 0.9.0
try:
    from acp.schema import AuthMethodAgent
except ImportError:
    from acp.schema import AuthMethod as AuthMethodAgent  # type: ignore[attr-defined]

from acp_adapter.auth import detect_provider
from acp_adapter.events import (
    make_message_cb,
    make_step_cb,
    make_thinking_cb,
    make_tool_progress_cb,
)
from acp_adapter.permissions import make_approval_callback
from acp_adapter.session import SessionManager, SessionState, _expand_acp_enabled_toolsets
from acp_adapter.tools import build_tool_complete, build_tool_start

logger = logging.getLogger(__name__)

try:
    from hermes_cli import __version__ as HERMES_VERSION
except Exception:
    HERMES_VERSION = "0.0.0"

# Thread pool for running AIAgent (synchronous) in parallel.
_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="acp-agent")

# Server-side page size for list_sessions. The ACP ListSessionsRequest schema
# does not expose a client-side limit, so this is a fixed cap that clients
# paginate against using `cursor` / `next_cursor`.
_LIST_SESSIONS_PAGE_SIZE = 50
_MAX_ACP_RESOURCE_BYTES = 512 * 1024
_TEXT_RESOURCE_MIME_PREFIXES = ("text/",)
_TEXT_RESOURCE_MIME_TYPES = {
    "application/json",
    "application/javascript",
    "application/typescript",
    "application/xml",
    "application/x-yaml",
    "application/yaml",
    "application/toml",
    "application/sql",
}


def _resource_display_name(uri: str, name: str | None = None, title: str | None = None) -> str:
    """Human-readable attachment name for prompt context."""
    raw_name = (name or "").strip()
    raw_title = (title or "").strip()
    if raw_title and raw_name and raw_title != raw_name:
        return f"{raw_title} ({raw_name})"
    if raw_title:
        return raw_title
    if raw_name:
        return raw_name
    parsed = urlparse(uri)
    candidate = parsed.path if parsed.scheme else uri
    return Path(unquote(candidate)).name or uri or "resource"


def _is_text_resource(mime_type: str | None) -> bool:
    mime = (mime_type or "").split(";", 1)[0].strip().lower()
    if not mime:
        return False
    return mime.startswith(_TEXT_RESOURCE_MIME_PREFIXES) or mime in _TEXT_RESOURCE_MIME_TYPES


def _path_from_file_uri(uri: str) -> Path | None:
    """Convert local file URIs/paths from ACP clients into a readable Path.

    Zed may send POSIX file URIs from Linux/WSL workspaces or Windows-ish paths
    when launched through wsl.exe. Translate the common Windows drive form to
    /mnt/<drive>/... so Hermes running in WSL can read it.
    """
    raw = (uri or "").strip()
    if not raw:
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


def _resource_link_to_text(block: ResourceContentBlock) -> str | None:
    uri = str(getattr(block, "uri", "") or "").strip()
    if not uri:
        return None

    name = str(getattr(block, "name", "") or "").strip() or None
    title = str(getattr(block, "title", "") or "").strip() or None
    mime_type = str(getattr(block, "mime_type", "") or "").strip() or None
    path = _path_from_file_uri(uri)

    if path is None:
        return _format_resource_text(
            uri=uri,
            name=name,
            title=title,
            body="[Resource link only; Hermes cannot read non-file ACP resource URIs directly.]",
        )

    try:
        size = path.stat().st_size
        read_size = min(size, _MAX_ACP_RESOURCE_BYTES)
        with path.open("rb") as fh:
            data = fh.read(read_size)
        text = _decode_text_bytes(data, mime_type)
        if text is None:
            return _format_resource_text(
                uri=uri,
                name=name,
                title=title,
                body=f"[Binary file omitted: {size} bytes, mime={mime_type or 'unknown'}]",
            )
        note = None
        if size > _MAX_ACP_RESOURCE_BYTES:
            note = f"truncated to {_MAX_ACP_RESOURCE_BYTES} of {size} bytes"
        return _format_resource_text(uri=uri, name=name, title=title, body=text, note=note)
    except OSError as exc:
        logger.warning("ACP resource read failed: %s", uri, exc_info=True)
        return _format_resource_text(
            uri=uri,
            name=name,
            title=title,
            body=f"[Could not read attached file: {exc}]",
        )


def _embedded_resource_to_text(block: EmbeddedResourceContentBlock) -> str | None:
    resource = getattr(block, "resource", None)
    if resource is None:
        return None

    uri = str(getattr(resource, "uri", "") or "").strip()
    mime_type = str(getattr(resource, "mime_type", "") or "").strip() or None

    if isinstance(resource, TextResourceContents):
        return _format_resource_text(uri=uri, body=resource.text)

    if isinstance(resource, BlobResourceContents):
        blob = resource.blob or ""
        try:
            data = base64.b64decode(blob, validate=True)
        except Exception:
            data = blob.encode("utf-8", errors="replace")
        text = _decode_text_bytes(data[:_MAX_ACP_RESOURCE_BYTES], mime_type)
        if text is None:
            body = f"[Binary embedded file omitted: {len(data)} bytes, mime={mime_type or 'unknown'}]"
        else:
            body = text
            if len(data) > _MAX_ACP_RESOURCE_BYTES:
                body += f"\n\n[Truncated to {_MAX_ACP_RESOURCE_BYTES} of {len(data)} bytes]"
        return _format_resource_text(uri=uri, body=body)

    text = getattr(resource, "text", None)
    if text:
        return _format_resource_text(uri=uri, body=str(text))
    return None


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
            resource_text = _resource_link_to_text(block)
            if resource_text:
                parts.append({"type": "text", "text": resource_text})
                text_parts.append(resource_text)
            continue
        if isinstance(block, EmbeddedResourceContentBlock):
            resource_text = _embedded_resource_to_text(block)
            if resource_text:
                parts.append({"type": "text", "text": resource_text})
                text_parts.append(resource_text)
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
        "compact": "Compress conversation context",
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
            "name": "compact",
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

    def __init__(self, session_manager: SessionManager | None = None):
        super().__init__()
        self.session_manager = session_manager or SessionManager()
        self._conn: Optional[acp.Client] = None

    # ---- Connection lifecycle -----------------------------------------------

    def on_connect(self, conn: acp.Client) -> None:
        """Store the client connection for sending session updates."""
        self._conn = conn
        logger.info("ACP client connected")

    @staticmethod
    def _encode_model_choice(provider: str | None, model: str | None) -> str:
        """Encode a model selection so ACP clients can keep provider context."""
        raw_model = str(model or "").strip()
        if not raw_model:
            return ""
        raw_provider = str(provider or "").strip().lower()
        if not raw_provider:
            return raw_model
        return f"{raw_provider}:{raw_model}"

    def _build_model_state(self, state: SessionState) -> SessionModelState | None:
        """Return the ACP model selector payload for editors like Zed."""
        model = str(state.model or getattr(state.agent, "model", "") or "").strip()
        provider = getattr(state.agent, "provider", None) or detect_provider() or "openrouter"

        try:
            from hermes_cli.models import curated_models_for_provider, normalize_provider, provider_label

            normalized_provider = normalize_provider(provider)
            provider_name = provider_label(normalized_provider)
            available_models: list[ModelInfo] = []
            seen_ids: set[str] = set()

            for model_id, description in curated_models_for_provider(normalized_provider):
                rendered_model = str(model_id or "").strip()
                if not rendered_model:
                    continue
                choice_id = self._encode_model_choice(normalized_provider, rendered_model)
                if choice_id in seen_ids:
                    continue
                desc_parts = [f"Provider: {provider_name}"]
                if description:
                    desc_parts.append(str(description).strip())
                if rendered_model == model:
                    desc_parts.append("current")
                available_models.append(
                    ModelInfo(
                        model_id=choice_id,
                        name=rendered_model,
                        description=" • ".join(part for part in desc_parts if part),
                    )
                )
                seen_ids.add(choice_id)

            current_model_id = self._encode_model_choice(normalized_provider, model)
            if current_model_id and current_model_id not in seen_ids:
                available_models.insert(
                    0,
                    ModelInfo(
                        model_id=current_model_id,
                        name=model,
                        description=f"Provider: {provider_name} • current",
                    ),
                )

            if available_models:
                return SessionModelState(
                    available_models=available_models,
                    current_model_id=current_model_id or available_models[0].model_id,
                )
        except Exception:
            logger.debug("Could not build ACP model state", exc_info=True)

        if not model:
            return None

        fallback_choice = self._encode_model_choice(provider, model)
        return SessionModelState(
            available_models=[ModelInfo(model_id=fallback_choice, name=model)],
            current_model_id=fallback_choice,
        )

    @staticmethod
    def _resolve_model_selection(raw_model: str, current_provider: str) -> tuple[str, str]:
        """Resolve ``provider:model`` input into the provider and normalized model id."""
        target_provider = current_provider
        new_model = raw_model.strip()

        try:
            from hermes_cli.models import detect_provider_for_model, parse_model_input

            target_provider, new_model = parse_model_input(new_model, current_provider)
            if target_provider == current_provider:
                detected = detect_provider_for_model(new_model, current_provider)
                if detected:
                    target_provider, new_model = detected
        except Exception:
            logger.debug("Provider detection failed, using model as-is", exc_info=True)

        return target_provider, new_model

    @staticmethod
    def _build_usage_update(state: SessionState) -> UsageUpdate | None:
        """Build ACP native context-usage data for clients like Zed.

        Zed's circular context indicator is driven by ACP ``usage_update``
        session updates: ``size`` is the model context window and ``used`` is
        the current request pressure.  Hermes estimates ``used`` from the same
        buckets it sends to providers: system prompt, conversation history, and
        tool schemas.
        """
        agent = state.agent
        compressor = getattr(agent, "context_compressor", None)
        size = int(getattr(compressor, "context_length", 0) or 0)
        if size <= 0:
            return None

        try:
            from agent.model_metadata import estimate_request_tokens_rough

            used = estimate_request_tokens_rough(
                state.history,
                system_prompt=getattr(agent, "_cached_system_prompt", "") or "",
                tools=getattr(agent, "tools", None) or None,
            )
        except Exception:
            logger.debug("Could not estimate ACP native context usage", exc_info=True)
            used = int(getattr(compressor, "last_prompt_tokens", 0) or 0)

        return UsageUpdate(
            session_update="usage_update",
            size=max(size, 0),
            used=max(used, 0),
        )

    async def _send_usage_update(self, state: SessionState) -> None:
        """Send ACP native context usage to the connected client."""
        if not self._conn:
            return
        update = self._build_usage_update(state)
        if update is None:
            return
        try:
            await self._conn.session_update(
                session_id=state.session_id,
                update=update,
            )
        except Exception:
            logger.warning(
                "Failed to send ACP usage update for session %s",
                state.session_id,
                exc_info=True,
            )

    def _schedule_usage_update(self, state: SessionState) -> None:
        """Schedule native context indicator refresh after ACP responses."""
        if not self._conn:
            return
        loop = asyncio.get_running_loop()
        loop.call_soon(asyncio.create_task, self._send_usage_update(state))

    async def _register_session_mcp_servers(
        self,
        state: SessionState,
        mcp_servers: list[McpServerStdio | McpServerHttp | McpServerSse] | None,
    ) -> None:
        """Register ACP-provided MCP servers and refresh the agent tool surface."""
        if not mcp_servers:
            return

        try:
            from tools.mcp_tool import register_mcp_servers

            config_map: dict[str, dict] = {}
            for server in mcp_servers:
                name = server.name
                if isinstance(server, McpServerStdio):
                    config = {
                        "command": server.command,
                        "args": list(server.args),
                        "env": {item.name: item.value for item in server.env},
                    }
                else:
                    config = {
                        "url": server.url,
                        "headers": {item.name: item.value for item in server.headers},
                    }
                config_map[name] = config

            await asyncio.to_thread(register_mcp_servers, config_map)
        except Exception:
            logger.warning(
                "Session %s: failed to register ACP MCP servers",
                state.session_id,
                exc_info=True,
            )
            return

        try:
            from model_tools import get_tool_definitions

            enabled_toolsets = _expand_acp_enabled_toolsets(
                getattr(state.agent, "enabled_toolsets", None) or ["hermes-acp"],
                mcp_server_names=[server.name for server in mcp_servers],
            )
            state.agent.enabled_toolsets = enabled_toolsets
            disabled_toolsets = getattr(state.agent, "disabled_toolsets", None)
            state.agent.tools = get_tool_definitions(
                enabled_toolsets=enabled_toolsets,
                disabled_toolsets=disabled_toolsets,
                quiet_mode=True,
            )
            state.agent.valid_tool_names = {
                tool["function"]["name"] for tool in state.agent.tools or []
            }
            invalidate = getattr(state.agent, "_invalidate_system_prompt", None)
            if callable(invalidate):
                invalidate()
            logger.info(
                "Session %s: refreshed tool surface after ACP MCP registration (%d tools)",
                state.session_id,
                len(state.agent.tools or []),
            )
        except Exception:
            logger.warning(
                "Session %s: failed to refresh tool surface after ACP MCP registration",
                state.session_id,
                exc_info=True,
            )

    # ---- ACP lifecycle ------------------------------------------------------

    async def initialize(
        self,
        protocol_version: int | None = None,
        client_capabilities: ClientCapabilities | None = None,
        client_info: Implementation | None = None,
        **kwargs: Any,
    ) -> InitializeResponse:
        resolved_protocol_version = (
            protocol_version if isinstance(protocol_version, int) else acp.PROTOCOL_VERSION
        )
        provider = detect_provider()
        auth_methods = None
        if provider:
            auth_methods = [
                AuthMethodAgent(
                    id=provider,
                    name=f"{provider} runtime credentials",
                    description=f"Authenticate Hermes using the currently configured {provider} runtime credentials.",
                )
            ]

        client_name = client_info.name if client_info else "unknown"
        logger.info(
            "Initialize from %s (protocol v%s)",
            client_name,
            resolved_protocol_version,
        )

        return InitializeResponse(
            protocol_version=acp.PROTOCOL_VERSION,
            agent_info=Implementation(name="hermes-agent", version=HERMES_VERSION),
            agent_capabilities=AgentCapabilities(
                load_session=True,
                prompt_capabilities=PromptCapabilities(image=True),
                session_capabilities=SessionCapabilities(
                    fork=SessionForkCapabilities(),
                    list=SessionListCapabilities(),
                    resume=SessionResumeCapabilities(),
                ),
            ),
            auth_methods=auth_methods,
        )

    async def authenticate(self, method_id: str, **kwargs: Any) -> AuthenticateResponse | None:
        # Only accept authenticate() calls whose method_id matches the
        # provider we advertised in initialize(). Without this check,
        # authenticate() would acknowledge any method_id as long as the
        # server has provider credentials configured — harmless under
        # Hermes' threat model (ACP is stdio-only, local-trust), but poor
        # API hygiene and confusing if ACP ever grows multi-method auth.
        provider = detect_provider()
        if not provider:
            return None
        if not isinstance(method_id, str) or method_id.strip().lower() != provider:
            return None
        return AuthenticateResponse()

    # ---- Session management -------------------------------------------------

    @staticmethod
    def _history_message_text(message: dict[str, Any]) -> str:
        """Extract displayable text from a persisted OpenAI-style message."""
        content = message.get("content")
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
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

    @staticmethod
    def _history_message_update(
        *,
        role: str,
        text: str,
    ) -> UserMessageChunk | AgentMessageChunk | None:
        """Build an ACP history replay update for a user/assistant message."""
        block = TextContentBlock(type="text", text=text)
        if role == "user":
            return UserMessageChunk(
                session_update="user_message_chunk",
                content=block,
            )
        if role == "assistant":
            return AgentMessageChunk(
                session_update="agent_message_chunk",
                content=block,
            )
        return None

    @staticmethod
    def _history_tool_call_name_args(tool_call: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        """Extract function name/arguments from an OpenAI-style tool_call."""
        function = tool_call.get("function") if isinstance(tool_call.get("function"), dict) else {}
        name = str(function.get("name") or tool_call.get("name") or "unknown_tool")
        raw_args = function.get("arguments") or tool_call.get("arguments") or tool_call.get("args") or {}
        if isinstance(raw_args, str):
            try:
                parsed = json.loads(raw_args)
            except Exception:
                parsed = {"raw": raw_args}
            raw_args = parsed
        if not isinstance(raw_args, dict):
            raw_args = {}
        return name, raw_args

    @staticmethod
    def _history_tool_call_id(tool_call: dict[str, Any]) -> str:
        """Return the stable provider tool call id for ACP history replay."""
        return str(
            tool_call.get("id")
            or tool_call.get("call_id")
            or tool_call.get("tool_call_id")
            or ""
        ).strip()

    async def _replay_session_history(self, state: SessionState) -> None:
        """Send persisted user/assistant history to clients during session/load.

        Zed's ACP history UI calls ``session/load`` after the user picks an item
        from the Agents sidebar. The agent must then replay the full conversation
        as user/assistant chunks plus reconstructed tool-call start/completion
        notifications; merely restoring server-side state makes Hermes remember
        context, but leaves the editor looking like a clean thread.
        """
        if not self._conn or not state.history:
            return

        active_tool_calls: dict[str, tuple[str, dict[str, Any]]] = {}

        async def _send(update: Any) -> bool:
            try:
                await self._conn.session_update(session_id=state.session_id, update=update)
                return True
            except Exception:
                logger.warning(
                    "Failed to replay ACP history for session %s",
                    state.session_id,
                    exc_info=True,
                )
                return False

        for message in state.history:
            role = str(message.get("role") or "")

            if role in {"user", "assistant"}:
                text = self._history_message_text(message)
                if text:
                    update = self._history_message_update(role=role, text=text)
                    if update is not None and not await _send(update):
                        return

            if role == "assistant" and isinstance(message.get("tool_calls"), list):
                for tool_call in message["tool_calls"]:
                    if not isinstance(tool_call, dict):
                        continue
                    tool_call_id = self._history_tool_call_id(tool_call)
                    if not tool_call_id:
                        continue
                    tool_name, args = self._history_tool_call_name_args(tool_call)
                    active_tool_calls[tool_call_id] = (tool_name, args)
                    if not await _send(build_tool_start(tool_call_id, tool_name, args)):
                        return
                continue

            if role == "tool":
                tool_call_id = str(message.get("tool_call_id") or "").strip()
                tool_name = str(message.get("tool_name") or "").strip()
                function_args: dict[str, Any] | None = None
                if tool_call_id in active_tool_calls:
                    tool_name, function_args = active_tool_calls.pop(tool_call_id)
                if not tool_call_id or not tool_name:
                    continue
                result = message.get("content")
                if not await _send(
                    build_tool_complete(
                        tool_call_id,
                        tool_name,
                        result=result if isinstance(result, str) else None,
                        function_args=function_args,
                    )
                ):
                    return

    async def new_session(
        self,
        cwd: str,
        mcp_servers: list | None = None,
        **kwargs: Any,
    ) -> NewSessionResponse:
        state = self.session_manager.create_session(cwd=cwd)
        await self._register_session_mcp_servers(state, mcp_servers)
        logger.info("New session %s (cwd=%s)", state.session_id, cwd)
        self._schedule_available_commands_update(state.session_id)
        self._schedule_usage_update(state)
        return NewSessionResponse(
            session_id=state.session_id,
            models=self._build_model_state(state),
        )

    def _schedule_history_replay(self, state: SessionState) -> None:
        """Replay persisted history after session/load or session/resume returns.

        Zed only attaches streamed transcript/tool updates once the load/resume
        response has completed. Sending replay notifications while the request is
        still in-flight can make the server look correct in logs while the editor
        drops or fails to attach the tool-call history.
        """
        loop = asyncio.get_running_loop()
        replay_coro = self._replay_session_history(state)
        loop.call_soon(asyncio.create_task, replay_coro)

    async def load_session(
        self,
        cwd: str,
        session_id: str,
        mcp_servers: list | None = None,
        **kwargs: Any,
    ) -> LoadSessionResponse | None:
        state = self.session_manager.update_cwd(session_id, cwd)
        if state is None:
            logger.warning("load_session: session %s not found", session_id)
            return None
        await self._register_session_mcp_servers(state, mcp_servers)
        logger.info("Loaded session %s", session_id)
        self._schedule_history_replay(state)
        self._schedule_available_commands_update(session_id)
        self._schedule_usage_update(state)
        return LoadSessionResponse(models=self._build_model_state(state))

    async def resume_session(
        self,
        cwd: str,
        session_id: str,
        mcp_servers: list | None = None,
        **kwargs: Any,
    ) -> ResumeSessionResponse:
        state = self.session_manager.update_cwd(session_id, cwd)
        if state is None:
            logger.warning("resume_session: session %s not found, creating new", session_id)
            state = self.session_manager.create_session(cwd=cwd)
        await self._register_session_mcp_servers(state, mcp_servers)
        logger.info("Resumed session %s", state.session_id)
        self._schedule_history_replay(state)
        self._schedule_available_commands_update(state.session_id)
        self._schedule_usage_update(state)
        return ResumeSessionResponse(models=self._build_model_state(state))

    async def cancel(self, session_id: str, **kwargs: Any) -> None:
        state = self.session_manager.get_session(session_id)
        if state and state.cancel_event:
            with state.runtime_lock:
                if state.is_running and state.current_prompt_text:
                    state.interrupted_prompt_text = state.current_prompt_text
            state.cancel_event.set()
            try:
                if getattr(state, "agent", None) and hasattr(state.agent, "interrupt"):
                    state.agent.interrupt()
            except Exception:
                logger.debug("Failed to interrupt ACP session %s", session_id, exc_info=True)
            logger.info("Cancelled session %s", session_id)

    async def fork_session(
        self,
        cwd: str,
        session_id: str,
        mcp_servers: list | None = None,
        **kwargs: Any,
    ) -> ForkSessionResponse:
        state = self.session_manager.fork_session(session_id, cwd=cwd)
        new_id = state.session_id if state else ""
        if state is not None:
            await self._register_session_mcp_servers(state, mcp_servers)
        logger.info("Forked session %s -> %s", session_id, new_id)
        if new_id:
            self._schedule_available_commands_update(new_id)
        return ForkSessionResponse(session_id=new_id)

    async def list_sessions(
        self,
        cursor: str | None = None,
        cwd: str | None = None,
        **kwargs: Any,
    ) -> ListSessionsResponse:
        """List ACP sessions with optional ``cwd`` filtering and cursor pagination.

        ``cwd`` is passed through to ``SessionManager.list_sessions`` which already
        normalizes and filters by working directory. ``cursor`` is a ``session_id``
        previously returned as ``next_cursor``; results resume after that entry.
        Server-side page size is capped at ``_LIST_SESSIONS_PAGE_SIZE``; when more
        results remain, ``next_cursor`` is set to the last returned ``session_id``.
        """
        infos = self.session_manager.list_sessions(cwd=cwd)

        if cursor:
            for idx, s in enumerate(infos):
                if s["session_id"] == cursor:
                    infos = infos[idx + 1:]
                    break
            else:
                # Unknown cursor -> empty page (do not fall back to full list).
                infos = []

        has_more = len(infos) > _LIST_SESSIONS_PAGE_SIZE
        infos = infos[:_LIST_SESSIONS_PAGE_SIZE]

        sessions = []
        for s in infos:
            updated_at = s.get("updated_at")
            if updated_at is not None and not isinstance(updated_at, str):
                updated_at = str(updated_at)
            sessions.append(
                SessionInfo(
                    session_id=s["session_id"],
                    cwd=s["cwd"],
                    title=s.get("title"),
                    updated_at=updated_at,
                )
            )

        next_cursor = sessions[-1].session_id if has_more and sessions else None
        return ListSessionsResponse(sessions=sessions, next_cursor=next_cursor)

    # ---- Prompt (core) ------------------------------------------------------

    async def prompt(
        self,
        prompt: list[
            TextContentBlock
            | ImageContentBlock
            | AudioContentBlock
            | ResourceContentBlock
            | EmbeddedResourceContentBlock
        ],
        session_id: str,
        **kwargs: Any,
    ) -> PromptResponse:
        """Run Hermes on the user's prompt and stream events back to the editor."""
        state = self.session_manager.get_session(session_id)
        if state is None:
            logger.error("prompt: session %s not found", session_id)
            return PromptResponse(stop_reason="refusal")

        user_text = _extract_text(prompt).strip()
        user_content = _content_blocks_to_openai_user_content(prompt)
        text_only_prompt = all(isinstance(block, TextContentBlock) for block in prompt)
        has_content = bool(user_text) or (
            isinstance(user_content, list) and bool(user_content)
        )
        if not has_content:
            return PromptResponse(stop_reason="end_turn")

        # /steer on an idle session has no in-flight tool call to inject into.
        # Rewrite it so the payload runs as a normal user prompt, matching the
        # gateway's behavior (gateway/run.py ~L4898). Two sub-cases:
        #   1. Zed-interrupt salvage — a prior prompt was cancelled by the
        #      client right before /steer arrived; replay it with the steer
        #      text attached as explicit correction/guidance so the user's
        #      in-flight work isn't lost.
        #   2. Plain idle — no prior work to salvage; just run the steer
        #      payload as a regular prompt. Without this, _cmd_steer would
        #      silently append to state.queued_prompts and respond with
        #      "No active turn — queued for the next turn", which looks like
        #      /queue even though the user never typed /queue.
        if text_only_prompt and isinstance(user_content, str) and user_text.startswith("/steer"):
            steer_text = user_text.split(maxsplit=1)[1].strip() if len(user_text.split(maxsplit=1)) > 1 else ""
            interrupted_prompt = ""
            rewrite_idle = False
            with state.runtime_lock:
                if not state.is_running and steer_text:
                    if state.interrupted_prompt_text:
                        interrupted_prompt = state.interrupted_prompt_text
                        state.interrupted_prompt_text = ""
                    else:
                        rewrite_idle = True
            if interrupted_prompt:
                user_text = (
                    f"{interrupted_prompt}\n\n"
                    f"User correction/guidance after interrupt: {steer_text}"
                )
                user_content = user_text
            elif rewrite_idle:
                user_text = steer_text
                user_content = steer_text

        # Intercept slash commands — handle locally without calling the LLM.
        # Slash commands are text-only; if the client included images/resources,
        # send the whole multimodal prompt to the agent instead of treating it as
        # an ACP command.
        if text_only_prompt and isinstance(user_content, str) and user_text.startswith("/"):
            response_text = self._handle_slash_command(user_text, state)
            if response_text is not None:
                if self._conn:
                    update = acp.update_agent_message_text(response_text)
                    await self._conn.session_update(session_id, update)
                    await self._send_usage_update(state)
                return PromptResponse(stop_reason="end_turn")

        # If Zed sends another regular prompt while the same ACP session is
        # still running, queue it instead of racing two AIAgent loops against
        # the same state.history. /steer and /queue are handled above and can
        # land immediately.
        with state.runtime_lock:
            if state.is_running:
                queued_text = user_text or "[Image attachment]"
                state.queued_prompts.append(queued_text)
                depth = len(state.queued_prompts)
                if self._conn:
                    update = acp.update_agent_message_text(
                        f"Queued for the next turn. ({depth} queued)"
                    )
                    await self._conn.session_update(session_id, update)
                return PromptResponse(stop_reason="end_turn")
            state.is_running = True
            state.current_prompt_text = user_text or "[Image attachment]"

        logger.info("Prompt on session %s: %s", session_id, user_text[:100])

        conn = self._conn
        loop = asyncio.get_running_loop()

        if state.cancel_event:
            state.cancel_event.clear()

        tool_call_ids: dict[str, Deque[str]] = defaultdict(deque)
        tool_call_meta: dict[str, dict[str, Any]] = {}
        previous_approval_cb = None

        streamed_message = False

        if conn:
            tool_progress_cb = make_tool_progress_cb(conn, session_id, loop, tool_call_ids, tool_call_meta)
            reasoning_cb = make_thinking_cb(conn, session_id, loop)
            step_cb = make_step_cb(conn, session_id, loop, tool_call_ids, tool_call_meta)
            message_cb = make_message_cb(conn, session_id, loop)

            def stream_delta_cb(text: str) -> None:
                nonlocal streamed_message
                if text:
                    streamed_message = True
                message_cb(text)

            approval_cb = make_approval_callback(conn.request_permission, loop, session_id)
        else:
            tool_progress_cb = None
            reasoning_cb = None
            step_cb = None
            stream_delta_cb = None
            approval_cb = None

        agent = state.agent
        agent.tool_progress_callback = tool_progress_cb
        # ACP thought panes should not receive Hermes' local kawaii waiting/status
        # updates. Route provider/model reasoning deltas instead; if the provider
        # emits no reasoning, Zed should not get a fake "thinking" accordion.
        agent.thinking_callback = None
        agent.reasoning_callback = reasoning_cb
        agent.step_callback = step_cb
        agent.stream_delta_callback = stream_delta_cb

        # Approval callback is per-thread (thread-local, GHSA-qg5c-hvr5-hjgr).
        # Set it INSIDE _run_agent so the TLS write happens in the executor
        # thread — setting it here would write to the event-loop thread's TLS,
        # not the executor's. Also set HERMES_INTERACTIVE so approval.py
        # takes the CLI-interactive path (which calls the registered
        # callback via prompt_dangerous_approval) instead of the
        # non-interactive auto-approve branch (GHSA-96vc-wcxf-jjff).
        # ACP's conn.request_permission maps cleanly to the interactive
        # callback shape — not the gateway-queue HERMES_EXEC_ASK path,
        # which requires a notify_cb registered in _gateway_notify_cbs.
        previous_approval_cb = None
        previous_interactive = None

        def _run_agent() -> dict:
            nonlocal previous_approval_cb, previous_interactive
            # Bind HERMES_SESSION_KEY for this session so per-session caches
            # (e.g. the interactive sudo password cache in tools.terminal_tool)
            # scope to the ACP session rather than leaking across sessions
            # that land on the same reused executor thread. This call runs
            # inside a contextvars.copy_context() below, so the ContextVar
            # write is isolated from other concurrent ACP sessions.
            try:
                from gateway.session_context import (
                    clear_session_vars,
                    set_session_vars,
                )
                session_tokens = set_session_vars(session_key=session_id)
            except Exception:
                session_tokens = None
                clear_session_vars = None  # type: ignore[assignment]
                logger.debug("Could not set ACP session context", exc_info=True)
            if approval_cb:
                try:
                    from tools import terminal_tool as _terminal_tool
                    previous_approval_cb = _terminal_tool._get_approval_callback()
                    _terminal_tool.set_approval_callback(approval_cb)
                except Exception:
                    logger.debug("Could not set ACP approval callback", exc_info=True)
            # Signal to tools.approval that we have an interactive callback
            # and the non-interactive auto-approve path must not fire.
            previous_interactive = os.environ.get("HERMES_INTERACTIVE")
            os.environ["HERMES_INTERACTIVE"] = "1"
            try:
                result = agent.run_conversation(
                    user_message=user_content,
                    conversation_history=state.history,
                    task_id=session_id,
                    persist_user_message=user_text or "[Image attachment]",
                )
                return result
            except Exception as e:
                logger.exception("Agent error in session %s", session_id)
                return {"final_response": f"Error: {e}", "messages": state.history}
            finally:
                # Restore HERMES_INTERACTIVE.
                if previous_interactive is None:
                    os.environ.pop("HERMES_INTERACTIVE", None)
                else:
                    os.environ["HERMES_INTERACTIVE"] = previous_interactive
                if approval_cb:
                    try:
                        from tools import terminal_tool as _terminal_tool
                        _terminal_tool.set_approval_callback(previous_approval_cb)
                    except Exception:
                        logger.debug("Could not restore approval callback", exc_info=True)
                if session_tokens is not None and clear_session_vars is not None:
                    try:
                        clear_session_vars(session_tokens)
                    except Exception:
                        logger.debug("Could not clear ACP session context", exc_info=True)

        try:
            # Wrap the executor call in a fresh copy of the current context so
            # concurrent ACP sessions on the shared ThreadPoolExecutor don't
            # stomp on each other's ContextVar writes (HERMES_SESSION_KEY in
            # particular — used by the interactive sudo password cache scope).
            ctx = contextvars.copy_context()
            result = await loop.run_in_executor(_executor, ctx.run, _run_agent)
        except Exception:
            logger.exception("Executor error for session %s", session_id)
            with state.runtime_lock:
                state.is_running = False
                state.current_prompt_text = ""
            return PromptResponse(stop_reason="end_turn")

        if result.get("messages"):
            state.history = result["messages"]
            # Persist updated history so sessions survive process restarts.
            self.session_manager.save_session(session_id)

        final_response = result.get("final_response", "")
        if final_response:
            try:
                from agent.title_generator import maybe_auto_title

                maybe_auto_title(
                    self.session_manager._get_db(),
                    session_id,
                    user_text,
                    final_response,
                    state.history,
                )
            except Exception:
                logger.debug("Failed to auto-title ACP session %s", session_id, exc_info=True)
        if final_response and conn and not streamed_message:
            update = acp.update_agent_message_text(final_response)
            await conn.session_update(session_id, update)

        # Mark this turn idle before draining queued work so recursive prompt()
        # calls can acquire the session. Queued turns are intentionally run as
        # normal follow-up user prompts, preserving role alternation and history.
        with state.runtime_lock:
            state.is_running = False
            state.current_prompt_text = ""

        while True:
            with state.runtime_lock:
                if not state.queued_prompts:
                    break
                next_prompt = state.queued_prompts.pop(0)
            if conn:
                await conn.session_update(
                    session_id,
                    acp.update_user_message_text(next_prompt),
                )
            await self.prompt(
                prompt=[TextContentBlock(type="text", text=next_prompt)],
                session_id=session_id,
            )

        usage = None
        if any(result.get(key) is not None for key in ("prompt_tokens", "completion_tokens", "total_tokens")):
            usage = Usage(
                input_tokens=result.get("prompt_tokens", 0),
                output_tokens=result.get("completion_tokens", 0),
                total_tokens=result.get("total_tokens", 0),
                thought_tokens=result.get("reasoning_tokens"),
                cached_read_tokens=result.get("cache_read_tokens"),
            )

        await self._send_usage_update(state)

        stop_reason = "cancelled" if state.cancel_event and state.cancel_event.is_set() else "end_turn"
        return PromptResponse(stop_reason=stop_reason, usage=usage)

    # ---- Slash commands (headless) -------------------------------------------

    @classmethod
    def _available_commands(cls) -> list[AvailableCommand]:
        commands: list[AvailableCommand] = []
        for spec in cls._ADVERTISED_COMMANDS:
            input_hint = spec.get("input_hint")
            commands.append(
                AvailableCommand(
                    name=spec["name"],
                    description=spec["description"],
                    input=UnstructuredCommandInput(hint=input_hint)
                    if input_hint
                    else None,
                )
            )
        return commands

    async def _send_available_commands_update(self, session_id: str) -> None:
        """Advertise supported slash commands to the connected ACP client."""
        if not self._conn:
            return

        try:
            await self._conn.session_update(
                session_id=session_id,
                update=AvailableCommandsUpdate(
                    session_update="available_commands_update",
                    available_commands=self._available_commands(),
                ),
            )
        except Exception:
            logger.warning(
                "Failed to advertise ACP slash commands for session %s",
                session_id,
                exc_info=True,
            )

    def _schedule_available_commands_update(self, session_id: str) -> None:
        """Send the command advertisement after the session response is queued."""
        if not self._conn:
            return
        loop = asyncio.get_running_loop()
        loop.call_soon(
            asyncio.create_task, self._send_available_commands_update(session_id)
        )

    def _handle_slash_command(self, text: str, state: SessionState) -> str | None:
        """Dispatch a slash command and return the response text.

        Returns ``None`` for unrecognized commands so they fall through
        to the LLM (the user may have typed ``/something`` as prose).
        """
        parts = text.split(maxsplit=1)
        cmd = parts[0].lstrip("/").lower()
        args = parts[1].strip() if len(parts) > 1 else ""

        handler = {
            "help": self._cmd_help,
            "model": self._cmd_model,
            "tools": self._cmd_tools,
            "context": self._cmd_context,
            "reset": self._cmd_reset,
            "compact": self._cmd_compact,
            "steer": self._cmd_steer,
            "queue": self._cmd_queue,
            "version": self._cmd_version,
        }.get(cmd)

        if handler is None:
            return None  # not a known command — let the LLM handle it

        try:
            return handler(args, state)
        except Exception as e:
            logger.error("Slash command /%s error: %s", cmd, e, exc_info=True)
            return f"Error executing /{cmd}: {e}"

    def _cmd_help(self, args: str, state: SessionState) -> str:
        lines = ["Available commands:", ""]
        for cmd, desc in self._SLASH_COMMANDS.items():
            lines.append(f"  /{cmd:10s}  {desc}")
        lines.append("")
        lines.append("Unrecognized /commands are sent to the model as normal messages.")
        return "\n".join(lines)

    def _cmd_model(self, args: str, state: SessionState) -> str:
        if not args:
            model = state.model or getattr(state.agent, "model", "unknown")
            provider = getattr(state.agent, "provider", None) or "auto"
            return f"Current model: {model}\nProvider: {provider}"

        current_provider = getattr(state.agent, "provider", None) or "openrouter"
        target_provider, new_model = self._resolve_model_selection(args, current_provider)

        state.model = new_model
        state.agent = self.session_manager._make_agent(
            session_id=state.session_id,
            cwd=state.cwd,
            model=new_model,
            requested_provider=target_provider,
        )
        self.session_manager.save_session(state.session_id)
        provider_label = getattr(state.agent, "provider", None) or target_provider or current_provider
        logger.info("Session %s: model switched to %s", state.session_id, new_model)
        return f"Model switched to: {new_model}\nProvider: {provider_label}"

    def _cmd_tools(self, args: str, state: SessionState) -> str:
        try:
            from model_tools import get_tool_definitions
            toolsets = _expand_acp_enabled_toolsets(
                getattr(state.agent, "enabled_toolsets", None) or ["hermes-acp"]
            )
            tools = get_tool_definitions(enabled_toolsets=toolsets, quiet_mode=True)
            if not tools:
                return "No tools available."
            lines = [f"Available tools ({len(tools)}):"]
            for t in tools:
                name = t.get("function", {}).get("name", "?")
                desc = t.get("function", {}).get("description", "")
                # Truncate long descriptions
                if len(desc) > 80:
                    desc = desc[:77] + "..."
                lines.append(f"  {name}: {desc}")
            return "\n".join(lines)
        except Exception as e:
            return f"Could not list tools: {e}"

    def _cmd_context(self, args: str, state: SessionState) -> str:
        """Show ACP session context pressure and compression guidance."""
        n_messages = len(state.history)

        # Count by role.
        roles: dict[str, int] = {}
        for msg in state.history:
            role = msg.get("role", "unknown")
            roles[role] = roles.get(role, 0) + 1

        agent = state.agent
        model = state.model or getattr(agent, "model", "")
        provider = getattr(agent, "provider", None) or "auto"
        compressor = getattr(agent, "context_compressor", None)
        context_length = int(getattr(compressor, "context_length", 0) or 0)
        threshold_tokens = int(getattr(compressor, "threshold_tokens", 0) or 0)

        try:
            from agent.model_metadata import estimate_request_tokens_rough

            system_prompt = getattr(agent, "_cached_system_prompt", "") or ""
            tools = getattr(agent, "tools", None) or None
            approx_tokens = estimate_request_tokens_rough(
                state.history,
                system_prompt=system_prompt,
                tools=tools,
            )
        except Exception:
            logger.debug("Could not estimate ACP context usage", exc_info=True)
            approx_tokens = 0

        if threshold_tokens <= 0 and context_length > 0:
            threshold_tokens = int(context_length * 0.80)

        lines = [
            f"Conversation: {n_messages} messages"
            if n_messages
            else "Conversation is empty (no messages yet).",
            f"  user: {roles.get('user', 0)}, assistant: {roles.get('assistant', 0)}, "
            f"tool: {roles.get('tool', 0)}, system: {roles.get('system', 0)}",
        ]
        if model:
            lines.append(f"Model: {model}")
        lines.append(f"Provider: {provider}")

        if approx_tokens > 0:
            if context_length > 0:
                usage_pct = (approx_tokens / context_length) * 100
                lines.append(
                    f"Context usage: ~{approx_tokens:,} / {context_length:,} tokens ({usage_pct:.1f}%)"
                )
            else:
                lines.append(f"Context usage: ~{approx_tokens:,} tokens")

        if threshold_tokens > 0:
            if approx_tokens > 0:
                threshold_pct = (threshold_tokens / context_length) * 100 if context_length > 0 else 0
                remaining = max(threshold_tokens - approx_tokens, 0)
                if approx_tokens >= threshold_tokens:
                    lines.append(
                        f"Compression: due now (threshold ~{threshold_tokens:,}"
                        + (f", {threshold_pct:.0f}%" if threshold_pct else "")
                        + "). Run /compact."
                    )
                else:
                    lines.append(
                        f"Compression: ~{remaining:,} tokens until threshold "
                        f"(~{threshold_tokens:,}"
                        + (f", {threshold_pct:.0f}%" if threshold_pct else "")
                        + ")."
                    )
            else:
                lines.append(f"Compression threshold: ~{threshold_tokens:,} tokens")

        if getattr(agent, "compression_enabled", True) is False:
            lines.append("Compression is disabled for this agent.")
        else:
            lines.append("Tip: run /compact to compress manually before the threshold.")

        return "\n".join(lines)

    def _cmd_reset(self, args: str, state: SessionState) -> str:
        state.history.clear()
        self.session_manager.save_session(state.session_id)
        return "Conversation history cleared."

    def _cmd_compact(self, args: str, state: SessionState) -> str:
        if not state.history:
            return "Nothing to compress — conversation is empty."
        try:
            agent = state.agent
            if not getattr(agent, "compression_enabled", True):
                return "Context compression is disabled for this agent."
            if not hasattr(agent, "_compress_context"):
                return "Context compression not available for this agent."

            from agent.model_metadata import estimate_request_tokens_rough

            original_count = len(state.history)
            # Include system prompt + tool schemas so the figure reflects real
            # request pressure, not a transcript-only underestimate (#6217).
            _sys_prompt = getattr(agent, "_cached_system_prompt", "") or ""
            _tools = getattr(agent, "tools", None) or None
            approx_tokens = estimate_request_tokens_rough(
                state.history, system_prompt=_sys_prompt, tools=_tools
            )
            original_session_db = getattr(agent, "_session_db", None)

            try:
                # ACP sessions must keep a stable session id, so avoid the
                # SQLite session-splitting side effect inside _compress_context.
                agent._session_db = None
                compressed, _ = agent._compress_context(
                    state.history,
                    getattr(agent, "_cached_system_prompt", "") or "",
                    approx_tokens=approx_tokens,
                    task_id=state.session_id,
                )
            finally:
                agent._session_db = original_session_db

            state.history = compressed
            self.session_manager.save_session(state.session_id)

            new_count = len(state.history)
            _sys_prompt_after = getattr(agent, "_cached_system_prompt", "") or _sys_prompt
            _tools_after = getattr(agent, "tools", None) or _tools
            new_tokens = estimate_request_tokens_rough(
                state.history,
                system_prompt=_sys_prompt_after,
                tools=_tools_after,
            )
            return (
                f"Context compressed: {original_count} -> {new_count} messages\n"
                f"~{approx_tokens:,} -> ~{new_tokens:,} tokens"
            )
        except Exception as e:
            return f"Compression failed: {e}"

    def _cmd_steer(self, args: str, state: SessionState) -> str:
        steer_text = args.strip()
        if not steer_text:
            return "Usage: /steer <guidance>"

        if state.is_running and hasattr(state.agent, "steer"):
            try:
                if state.agent.steer(steer_text):
                    preview = steer_text[:80] + ("..." if len(steer_text) > 80 else "")
                    return f"⏩ Steer queued for the active turn: {preview}"
            except Exception as exc:
                logger.warning("ACP steer failed for session %s: %s", state.session_id, exc)
                return f"⚠️ Steer failed: {exc}"

        with state.runtime_lock:
            state.queued_prompts.append(steer_text)
            depth = len(state.queued_prompts)
        return f"No active turn — queued for the next turn. ({depth} queued)"

    def _cmd_queue(self, args: str, state: SessionState) -> str:
        queued_text = args.strip()
        if not queued_text:
            return "Usage: /queue <prompt>"
        with state.runtime_lock:
            state.queued_prompts.append(queued_text)
            depth = len(state.queued_prompts)
        return f"Queued for the next turn. ({depth} queued)"

    def _cmd_version(self, args: str, state: SessionState) -> str:
        return f"Hermes Agent v{HERMES_VERSION}"

    # ---- Model switching (ACP protocol method) -------------------------------

    async def set_session_model(
        self, model_id: str, session_id: str, **kwargs: Any
    ) -> SetSessionModelResponse | None:
        """Switch the model for a session (called by ACP protocol)."""
        state = self.session_manager.get_session(session_id)
        if state:
            current_provider = getattr(state.agent, "provider", None)
            requested_provider, resolved_model = self._resolve_model_selection(
                model_id,
                current_provider or "openrouter",
            )
            state.model = resolved_model
            provider_changed = bool(current_provider and requested_provider != current_provider)
            current_base_url = None if provider_changed else getattr(state.agent, "base_url", None)
            current_api_mode = None if provider_changed else getattr(state.agent, "api_mode", None)
            state.agent = self.session_manager._make_agent(
                session_id=session_id,
                cwd=state.cwd,
                model=resolved_model,
                requested_provider=requested_provider,
                base_url=current_base_url,
                api_mode=current_api_mode,
            )
            self.session_manager.save_session(session_id)
            logger.info(
                "Session %s: model switched to %s via provider %s",
                session_id,
                resolved_model,
                requested_provider,
            )
            return SetSessionModelResponse()
        logger.warning("Session %s: model switch requested for missing session", session_id)
        return None

    async def set_session_mode(
        self, mode_id: str, session_id: str, **kwargs: Any
    ) -> SetSessionModeResponse | None:
        """Persist the editor-requested mode so ACP clients do not fail on mode switches."""
        state = self.session_manager.get_session(session_id)
        if state is None:
            logger.warning("Session %s: mode switch requested for missing session", session_id)
            return None
        setattr(state, "mode", mode_id)
        self.session_manager.save_session(session_id)
        logger.info("Session %s: mode switched to %s", session_id, mode_id)
        return SetSessionModeResponse()

    async def set_config_option(
        self, config_id: str, session_id: str, value: str, **kwargs: Any
    ) -> SetSessionConfigOptionResponse | None:
        """Accept ACP config option updates even when Hermes has no typed ACP config surface yet."""
        state = self.session_manager.get_session(session_id)
        if state is None:
            logger.warning("Session %s: config update requested for missing session", session_id)
            return None

        options = getattr(state, "config_options", None)
        if not isinstance(options, dict):
            options = {}
        options[str(config_id)] = value
        setattr(state, "config_options", options)
        self.session_manager.save_session(session_id)
        logger.info("Session %s: config option %s updated", session_id, config_id)
        return SetSessionConfigOptionResponse(config_options=[])
