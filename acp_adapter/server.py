"""HermesACPAgent — ACP server that wraps Hermes's AIAgent.

Implements the Agent Client Protocol lifecycle: initialize, authenticate,
session management, prompt execution with streaming updates, and cancellation.
Agent work runs in a ThreadPoolExecutor so the asyncio event loop stays
free for JSON-RPC I/O.
"""

from __future__ import annotations

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from acp import (
    PROTOCOL_VERSION,
    Agent,
    AuthenticateResponse,
    InitializeResponse,
    LoadSessionResponse,
    NewSessionResponse,
    PromptResponse,
    RequestError,
)
from acp.interfaces import Client
from acp.schema import (
    AgentCapabilities,
    AudioContentBlock,
    AuthMethod,
    ClientCapabilities,
    EmbeddedResourceContentBlock,
    ForkSessionResponse,
    HttpMcpServer,
    ImageContentBlock,
    Implementation,
    ListSessionsResponse,
    McpServerStdio,
    ResourceContentBlock,
    ResumeSessionResponse,
    SessionCapabilities,
    SessionForkCapabilities,
    SessionInfo,
    SessionListCapabilities,
    SessionResumeCapabilities,
    SetSessionConfigOptionResponse,
    SetSessionModelResponse,
    SetSessionModeResponse,
    SseMcpServer,
    TextContentBlock,
)

from acp_adapter.auth import check_auth
from acp_adapter.session import SessionManager

logger = logging.getLogger(__name__)


class HermesACPAgent(Agent):
    """ACP agent backed by Hermes ``AIAgent``."""

    _conn: Client

    def __init__(self) -> None:
        self._sessions = SessionManager()
        self._executor = ThreadPoolExecutor(max_workers=4)

    def shutdown(self) -> None:
        """Release resources (ThreadPoolExecutor, sessions)."""
        self._sessions.cleanup_all()
        self._executor.shutdown(wait=False)

    def __del__(self) -> None:
        try:
            self._executor.shutdown(wait=False)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # ACP lifecycle
    # ------------------------------------------------------------------

    def on_connect(self, conn: Client) -> None:
        self._conn = conn

    async def initialize(
        self,
        protocol_version: int,
        client_capabilities: ClientCapabilities | None = None,
        client_info: Implementation | None = None,
        **kwargs: Any,
    ) -> InitializeResponse:
        logger.info("initialize: client=%s", client_info)
        if protocol_version != PROTOCOL_VERSION:
            raise RequestError.invalid_params(
                {"message": f"Unsupported protocol version {protocol_version}, expected {PROTOCOL_VERSION}"}
            )
        return InitializeResponse(
            protocol_version=PROTOCOL_VERSION,
            agent_capabilities=AgentCapabilities(
                load_session=True,
                session_capabilities=SessionCapabilities(
                    list=SessionListCapabilities(),
                    fork=SessionForkCapabilities(),
                    resume=SessionResumeCapabilities(),
                ),
            ),
            agent_info=Implementation(
                name="hermes-agent",
                title="Hermes Agent",
                version="0.1.0",
            ),
            auth_methods=[AuthMethod(id="env-check", name="API Key")],
        )

    async def authenticate(self, method_id: str, **kwargs: Any) -> AuthenticateResponse | None:
        logger.info("authenticate: method_id=%s", method_id)
        result = check_auth()
        if result["success"]:
            return AuthenticateResponse()
        logger.warning("Authentication failed: %s", result["error"])
        return None

    async def new_session(
        self,
        cwd: str,
        mcp_servers: list[HttpMcpServer | SseMcpServer | McpServerStdio] | None = None,
        **kwargs: Any,
    ) -> NewSessionResponse:
        loop = asyncio.get_running_loop()
        session_id = self._sessions.generate_id()
        self._sessions.create(session_id, self._conn, loop, cwd)
        logger.info("new_session: id=%s cwd=%s", session_id, cwd)
        return NewSessionResponse(session_id=session_id, modes=None)

    async def load_session(
        self,
        cwd: str,
        session_id: str,
        mcp_servers: list[HttpMcpServer | SseMcpServer | McpServerStdio] | None = None,
        **kwargs: Any,
    ) -> LoadSessionResponse | None:
        loop = asyncio.get_running_loop()
        self._sessions.load(session_id, self._conn, loop, cwd)
        logger.info("load_session: id=%s", session_id)
        return LoadSessionResponse()

    async def list_sessions(
        self, cursor: str | None = None, cwd: str | None = None, **kwargs: Any
    ) -> ListSessionsResponse:
        logger.info("list_sessions: cursor=%s cwd=%s", cursor, cwd)
        sessions = [
            SessionInfo(session_id=sid, cwd=state.cwd or "/")
            for sid, state in self._sessions.list_all()
        ]
        return ListSessionsResponse(sessions=sessions)

    async def fork_session(
        self,
        cwd: str,
        session_id: str,
        mcp_servers: list[HttpMcpServer | SseMcpServer | McpServerStdio] | None = None,
        **kwargs: Any,
    ) -> ForkSessionResponse:
        parent = self._sessions.get(session_id)
        loop = asyncio.get_running_loop()
        new_id = self._sessions.generate_id()
        new_state = self._sessions.create(
            new_id, self._conn, loop, cwd or (parent.cwd if parent else "")
        )
        if parent:
            new_state.history = list(parent.history)
        logger.info("fork_session: %s -> %s (%d messages copied)",
                     session_id, new_id, len(new_state.history))
        return ForkSessionResponse(session_id=new_id)

    async def resume_session(
        self,
        cwd: str,
        session_id: str,
        mcp_servers: list[HttpMcpServer | SseMcpServer | McpServerStdio] | None = None,
        **kwargs: Any,
    ) -> ResumeSessionResponse:
        # Check if session exists in memory
        existing = self._sessions.get(session_id)
        if existing is not None:
            if cwd:
                existing.cwd = cwd
            logger.info("resume_session: id=%s (in memory)", session_id)
            return ResumeSessionResponse()

        # Try to hydrate from DB
        has_db_history = False
        try:
            from hermes_state import SessionDB

            db = SessionDB()
            messages = db.get_messages(session_id)
            db.close()
            has_db_history = bool(messages)
        except Exception:
            pass

        if not has_db_history:
            raise RequestError.invalid_params(
                {"message": f"Session {session_id} not found"}
            )

        # Session exists in DB — create state and hydrate
        loop = asyncio.get_running_loop()
        self._sessions.load(session_id, self._conn, loop, cwd)
        logger.info("resume_session: id=%s (from DB)", session_id)
        return ResumeSessionResponse()

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
        logger.info("prompt: session=%s blocks=%d", session_id, len(prompt))

        state = self._sessions.get(session_id)
        if state is None:
            logger.warning("prompt: auto-creating session %s (client did not call new_session/load_session first)", session_id)
            loop = asyncio.get_running_loop()
            state = self._sessions.create(session_id, self._conn, loop, cwd=".")

        # Reset cancel flag
        state.cancel_event.clear()

        # Extract text from prompt blocks
        user_text = _extract_text(prompt)
        if not user_text:
            return PromptResponse(stop_reason="end_turn")

        loop = asyncio.get_running_loop()
        conn = self._conn

        # ------------------------------------------------------------------
        # Build callbacks that stream ACP notifications from the agent thread
        # ------------------------------------------------------------------

        def tool_progress_cb(tool_name: str, preview: str, args: dict) -> None:
            """Called by AIAgent when a tool call starts."""
            try:
                from acp import start_tool_call, text_block as tb

                tool_id = f"tc-{tool_name}-{id(args)}"
                update = start_tool_call(tool_id, f"{tool_name}: {preview}", kind="tool", status="running")
                asyncio.run_coroutine_threadsafe(
                    conn.session_update(session_id, update), loop
                ).result(timeout=5)
            except Exception:
                logger.debug("tool_progress_cb failed", exc_info=True)

        def step_cb(api_call_count: int, prev_tools: list) -> None:
            """Called between AIAgent API call iterations."""
            if state.cancel_event.is_set():
                state.agent.interrupt()

        # Wire callbacks into the agent
        state.agent.tool_progress_callback = tool_progress_cb
        state.agent.step_callback = step_cb

        # ------------------------------------------------------------------
        # Run the agent in a worker thread
        # ------------------------------------------------------------------

        try:
            result = await loop.run_in_executor(
                self._executor,
                self._run_agent_sync,
                state,
                user_text,
            )
        except Exception as exc:
            logger.exception("Agent execution failed for session %s", session_id)
            await _send_text(conn, session_id, f"Error: {exc}")
            return PromptResponse(stop_reason="error")

        # Stream the final response text back to the editor
        response_text = result.get("final_response", "")
        if response_text:
            await _send_text(conn, session_id, response_text)

        stop_reason = "end_turn"
        if state.cancel_event.is_set():
            stop_reason = "cancelled"

        return PromptResponse(stop_reason=stop_reason)

    async def cancel(self, session_id: str, **kwargs: Any) -> None:
        logger.info("cancel: session=%s", session_id)
        state = self._sessions.get(session_id)
        if state is not None:
            state.cancel_event.set()
            state.agent.interrupt()

    async def set_session_mode(self, mode_id: str, session_id: str, **kwargs: Any) -> SetSessionModeResponse | None:
        state = self._sessions.get(session_id)
        if state is None:
            logger.warning("set_session_mode: unknown session %s", session_id)
        logger.info("set_session_mode: session=%s mode=%s (no-op)", session_id, mode_id)
        return SetSessionModeResponse()

    async def set_session_model(self, model_id: str, session_id: str, **kwargs: Any) -> SetSessionModelResponse | None:
        state = self._sessions.get(session_id)
        if state is None:
            logger.warning("set_session_model: unknown session %s", session_id)
        logger.info("set_session_model: session=%s model=%s (no-op)", session_id, model_id)
        return SetSessionModelResponse()

    async def set_config_option(
        self, config_id: str, session_id: str, value: str, **kwargs: Any
    ) -> SetSessionConfigOptionResponse | None:
        state = self._sessions.get(session_id)
        if state is None:
            logger.warning("set_config_option: unknown session %s", session_id)
        logger.info("set_config_option: session=%s config=%s (no-op)", session_id, config_id)
        return SetSessionConfigOptionResponse(config_options=[])

    async def ext_method(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        raise RequestError.method_not_found(f"_{method}")

    async def ext_notification(self, method: str, params: dict[str, Any]) -> None:
        pass

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _run_agent_sync(state: Any, user_text: str) -> dict:
        """Execute ``AIAgent.run_conversation`` synchronously (runs in ThreadPoolExecutor)."""
        result = state.agent.run_conversation(
            user_message=user_text,
            conversation_history=state.history,
        )
        # Update stored history for multi-turn conversations.
        if isinstance(result, dict) and "messages" in result:
            state.history = result["messages"]
        return result


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _extract_text(
    blocks: list[
        TextContentBlock
        | ImageContentBlock
        | AudioContentBlock
        | ResourceContentBlock
        | EmbeddedResourceContentBlock
    ],
) -> str:
    """Concatenate text from prompt content blocks."""
    parts: list[str] = []
    for block in blocks:
        if isinstance(block, TextContentBlock):
            parts.append(block.text)
        elif hasattr(block, "text"):
            parts.append(str(block.text))
    return "\n".join(parts).strip()


async def _send_text(conn: Client, session_id: str, text: str) -> None:
    """Send a text message update to the editor."""
    from acp import update_agent_message, text_block

    try:
        update = update_agent_message(text_block(text))
        await conn.session_update(session_id, update)
    except Exception:
        logger.debug("_send_text failed", exc_info=True)
