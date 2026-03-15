"""ACP agent server — exposes Hermes Agent via the Agent Client Protocol."""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Deque, Optional

import acp
from acp.schema import (
    AgentCapabilities,
    AuthenticateResponse,
    AuthMethod,
    ClientCapabilities,
    EmbeddedResourceContentBlock,
    ForkSessionResponse,
    ImageContentBlock,
    AudioContentBlock,
    Implementation,
    InitializeResponse,
    ListSessionsResponse,
    LoadSessionResponse,
    NewSessionResponse,
    PromptResponse,
    ResumeSessionResponse,
    ResourceContentBlock,
    SessionCapabilities,
    SessionForkCapabilities,
    SessionListCapabilities,
    SessionInfo,
    TextContentBlock,
    Usage,
)

from acp_adapter.auth import detect_provider, has_provider
from acp_adapter.events import (
    make_message_cb,
    make_step_cb,
    make_thinking_cb,
    make_tool_progress_cb,
)
from acp_adapter.permissions import make_approval_callback
from acp_adapter.session import SessionManager

logger = logging.getLogger(__name__)

try:
    from hermes_cli import __version__ as HERMES_VERSION
except Exception:
    HERMES_VERSION = "0.0.0"

# Thread pool for running AIAgent (synchronous) in parallel.
_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="acp-agent")


def _extract_text(
    prompt: list[
        TextContentBlock
        | ImageContentBlock
        | AudioContentBlock
        | ResourceContentBlock
        | EmbeddedResourceContentBlock
    ],
) -> str:
    """Extract plain text from ACP content blocks."""
    parts: list[str] = []
    for block in prompt:
        if isinstance(block, TextContentBlock):
            parts.append(block.text)
        elif hasattr(block, "text"):
            parts.append(str(block.text))
        # Non-text blocks are ignored for now.
    return "\n".join(parts)


class HermesACPAgent(acp.Agent):
    """ACP Agent implementation wrapping Hermes AIAgent."""

    def __init__(self, session_manager: SessionManager | None = None):
        super().__init__()
        self.session_manager = session_manager or SessionManager()
        self._conn: Optional[acp.Client] = None

    # ---- Connection lifecycle -----------------------------------------------

    def on_connect(self, conn: acp.Client) -> None:
        """Store the client connection for sending session updates."""
        self._conn = conn
        logger.info("ACP client connected")

    # ---- ACP lifecycle ------------------------------------------------------

    async def initialize(
        self,
        protocol_version: int,
        client_capabilities: ClientCapabilities | None = None,
        client_info: Implementation | None = None,
        **kwargs: Any,
    ) -> InitializeResponse:
        provider = detect_provider()
        auth_methods = None
        if provider:
            auth_methods = [
                AuthMethod(
                    id=provider,
                    name=f"{provider} runtime credentials",
                    description=f"Authenticate Hermes using the currently configured {provider} runtime credentials.",
                )
            ]

        client_name = client_info.name if client_info else "unknown"
        logger.info("Initialize from %s (protocol v%s)", client_name, protocol_version)

        return InitializeResponse(
            protocol_version=acp.PROTOCOL_VERSION,
            agent_info=Implementation(name="hermes-agent", version=HERMES_VERSION),
            agent_capabilities=AgentCapabilities(
                session_capabilities=SessionCapabilities(
                    fork=SessionForkCapabilities(),
                    list=SessionListCapabilities(),
                ),
            ),
            auth_methods=auth_methods,
        )

    async def authenticate(self, method_id: str, **kwargs: Any) -> AuthenticateResponse | None:
        if has_provider():
            return AuthenticateResponse()
        return None

    # ---- Session management -------------------------------------------------

    async def new_session(
        self,
        cwd: str,
        mcp_servers: list | None = None,
        **kwargs: Any,
    ) -> NewSessionResponse:
        state = self.session_manager.create_session(cwd=cwd)
        logger.info("New session %s (cwd=%s)", state.session_id, cwd)
        return NewSessionResponse(session_id=state.session_id)

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
        logger.info("Loaded session %s", session_id)
        return LoadSessionResponse()

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
        logger.info("Resumed session %s", state.session_id)
        return ResumeSessionResponse()

    async def cancel(self, session_id: str, **kwargs: Any) -> None:
        state = self.session_manager.get_session(session_id)
        if state and state.cancel_event:
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
        logger.info("Forked session %s -> %s", session_id, new_id)
        return ForkSessionResponse(session_id=new_id)

    async def list_sessions(
        self,
        cursor: str | None = None,
        cwd: str | None = None,
        **kwargs: Any,
    ) -> ListSessionsResponse:
        infos = self.session_manager.list_sessions()
        sessions = [
            SessionInfo(session_id=s["session_id"], cwd=s["cwd"])
            for s in infos
        ]
        return ListSessionsResponse(sessions=sessions)

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

        user_text = _extract_text(prompt)
        if not user_text.strip():
            return PromptResponse(stop_reason="end_turn")

        logger.info("Prompt on session %s: %s", session_id, user_text[:100])

        conn = self._conn
        loop = asyncio.get_running_loop()

        if state.cancel_event:
            state.cancel_event.clear()

        tool_call_ids: dict[str, Deque[str]] = defaultdict(deque)
        previous_approval_cb = None

        if conn:
            tool_progress_cb = make_tool_progress_cb(conn, session_id, loop, tool_call_ids)
            thinking_cb = make_thinking_cb(conn, session_id, loop)
            step_cb = make_step_cb(conn, session_id, loop, tool_call_ids)
            message_cb = make_message_cb(conn, session_id, loop)
            approval_cb = make_approval_callback(conn.request_permission, loop, session_id)
        else:
            tool_progress_cb = None
            thinking_cb = None
            step_cb = None
            message_cb = None
            approval_cb = None

        agent = state.agent
        agent.tool_progress_callback = tool_progress_cb
        agent.thinking_callback = thinking_cb
        agent.step_callback = step_cb
        agent.message_callback = message_cb

        if approval_cb:
            try:
                from tools import terminal_tool as _terminal_tool
                previous_approval_cb = getattr(_terminal_tool, "_approval_callback", None)
                _terminal_tool.set_approval_callback(approval_cb)
            except Exception:
                logger.debug("Could not set ACP approval callback", exc_info=True)

        def _run_agent() -> dict:
            try:
                result = agent.run_conversation(
                    user_message=user_text,
                    conversation_history=state.history,
                    task_id=session_id,
                )
                return result
            except Exception as e:
                logger.exception("Agent error in session %s", session_id)
                return {"final_response": f"Error: {e}", "messages": state.history}
            finally:
                if approval_cb:
                    try:
                        from tools import terminal_tool as _terminal_tool
                        _terminal_tool.set_approval_callback(previous_approval_cb)
                    except Exception:
                        logger.debug("Could not restore approval callback", exc_info=True)

        try:
            result = await loop.run_in_executor(_executor, _run_agent)
        except Exception:
            logger.exception("Executor error for session %s", session_id)
            return PromptResponse(stop_reason="end_turn")

        if result.get("messages"):
            state.history = result["messages"]

        final_response = result.get("final_response", "")
        if final_response and conn:
            update = acp.update_agent_message_text(final_response)
            await conn.session_update(session_id, update)

        usage = None
        usage_data = result.get("usage")
        if usage_data and isinstance(usage_data, dict):
            usage = Usage(
                input_tokens=usage_data.get("prompt_tokens", 0),
                output_tokens=usage_data.get("completion_tokens", 0),
                total_tokens=usage_data.get("total_tokens", 0),
                thought_tokens=usage_data.get("reasoning_tokens"),
                cached_read_tokens=usage_data.get("cached_tokens"),
            )

        stop_reason = "cancelled" if state.cancel_event and state.cancel_event.is_set() else "end_turn"
        return PromptResponse(stop_reason=stop_reason, usage=usage)

    # ---- Model switching ----------------------------------------------------

    async def set_session_model(
        self, model_id: str, session_id: str, **kwargs: Any
    ):
        """Switch the model for a session."""
        state = self.session_manager.get_session(session_id)
        if state:
            state.model = model_id
            state.agent = self.session_manager._make_agent(
                session_id=session_id,
                cwd=state.cwd,
                model=model_id,
            )
            logger.info("Session %s: model switched to %s", session_id, model_id)
        return None
