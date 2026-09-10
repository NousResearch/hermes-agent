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
    _build_plan_update_from_todo_result, make_message_cb, make_step_cb, make_thinking_cb, make_tool_progress_cb,
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

# Runs the synchronous AIAgent off the event loop.
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
    cls, session_update = _HISTORY_CHUNK_TYPES[role]
    return cls(
        session_update=session_update, content=TextContentBlock(type="text", text=text),
        field_meta=_history_summary_meta(message, text),
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
            picker = build_model_state(model, provider, str(getattr(state.agent, "base_url", "") or ""))
            if picker is not None:
                return picker
        except Exception:
            logger.debug("Could not build ACP model state", exc_info=True)

        if not model:
            return None
        choice = encode_model_choice(provider, model)
        return SessionModelState(available_models=[ModelInfo(model_id=choice, name=model)], current_model_id=choice)

    @staticmethod
    def _resolve_model_selection(raw_model: str, current_provider: str) -> tuple[str, str]:
        """Resolve ``provider:model`` input into the provider and normalized model id."""
        target_provider, new_model = current_provider, raw_model.strip()
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

    def _switch_model(
        self, state: SessionState, raw_model: str, *, keep_endpoint: bool = False
    ) -> tuple[str | None, str, str]:
        """Rebuild the session agent on a new model -> (old provider, new provider, model).
        ``keep_endpoint`` carries base_url/api_mode over when the provider is unchanged."""
        current_provider = getattr(state.agent, "provider", None)
        target_provider, new_model = self._resolve_model_selection(raw_model, current_provider or "openrouter")
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
            response_text = self._handle_slash_command(user_text, state)
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
            cbs.reasoning_cb = make_thinking_cb(conn, session_id, loop)
            cbs.step_cb = make_step_cb(conn, session_id, loop, tool_call_ids, tool_call_meta)
            message_cb = make_message_cb(conn, session_id, loop)

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
        if result.get("messages"):
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

        # ``or ""`` rather than a ``.get`` default: the conversation loop can
        # return the key with an explicit ``None`` (e.g. the truncation path's
        # ``partial_response or None``), and a default only applies when the
        # key is absent. Other consumers already tolerate None — see
        # ``turn_finalizer``'s ``len(final_response) if final_response else 0``.
        final_response = result.get("final_response") or ""
        cancelled = bool(state.cancel_event and state.cancel_event.is_set())
        # The local "waiting for model" interrupt status is metadata, not prose; stop_reason carries it.
        from agent.conversation_loop import INTERRUPT_WAITING_FOR_MODEL_PREFIX

        interrupted = bool(result.get("interrupted")) or cancelled
        suppress = interrupted and final_response.startswith(INTERRUPT_WAITING_FOR_MODEL_PREFIX)
        # Send the final text unless already streamed — or if a plugin hook transformed it after.
        if final_response and conn and not suppress and (not streamed_message or result.get("response_transformed")):
            await conn.session_update(session_id, acp.update_agent_message_text(final_response))

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
            _old, requested_provider, resolved_model = self._switch_model(state, model_id, keep_endpoint=True)
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
