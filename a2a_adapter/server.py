"""A2A (Agent-to-Agent) server — exposes Hermes Agent via the Google A2A protocol.

Uses the official ``a2a-sdk`` (``pip install 'a2a-sdk[http-server]'``) to build
a standards-compliant FastAPI application that any A2A orchestrator can discover
and call — Vertex AI Agent Engine, LangGraph, Akela, etc.

Exposes (via a2a-sdk):
  GET  /.well-known/agent.json  →  Agent Card
  POST /                         →  JSON-RPC 2.0 (tasks/send, tasks/sendSubscribe)

Configure via environment variables:
  A2A_HOST          — bind host (default: 0.0.0.0)
  A2A_PORT          — bind port (default: 9000)
  A2A_KEY           — optional Bearer token for auth
  AGENT_NAME        — name shown in Agent Card
  AGENT_DESCRIPTION — description shown in Agent Card
  AGENT_SKILLS      — comma-separated skill names
  AGENT_MODEL       — model name shown in Agent Card metadata
"""

from __future__ import annotations

import asyncio
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any

logger = logging.getLogger(__name__)

_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="a2a-agent")

DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 9000


# ---------------------------------------------------------------------------
# SDK imports — fail loudly at startup if a2a-sdk is not installed
# ---------------------------------------------------------------------------
try:
    from a2a.server.agent_execution import AgentExecutor, RequestContext
    from a2a.server.events import EventQueue
    from a2a.server.request_handlers import DefaultRequestHandler
    from a2a.server.tasks import InMemoryTaskStore
    from a2a.server.apps import A2AStarletteApplication
    from a2a.types import (
        AgentCard,
        AgentCapabilities,
        AgentSkill,
        Part,
        Task,
        TaskArtifactUpdateEvent,
        TaskState,
        TaskStatus,
        TaskStatusUpdateEvent,
        TextPart,
        UnsupportedOperationError,
    )
    _SDK_AVAILABLE = True
except ImportError as _sdk_err:
    _SDK_AVAILABLE = False
    _sdk_err_msg = str(_sdk_err)


# ---------------------------------------------------------------------------
# Hermes agent factory (shared with acp_adapter pattern)
# ---------------------------------------------------------------------------

def _make_agent(session_id: str, stream_delta_callback=None) -> Any:
    """Create a Hermes AIAgent using the runtime provider from config."""
    from run_agent import AIAgent
    from hermes_cli.config import load_config
    from hermes_cli.runtime_provider import resolve_runtime_provider

    config = load_config()
    model_cfg = config.get("model")
    default_model = "anthropic/claude-opus-4.6"
    config_provider = None

    if isinstance(model_cfg, dict):
        default_model = str(model_cfg.get("default") or default_model)
        config_provider = model_cfg.get("provider")
    elif isinstance(model_cfg, str) and model_cfg.strip():
        default_model = model_cfg.strip()

    kwargs: dict[str, Any] = {
        "platform": "a2a",
        "enabled_toolsets": ["hermes-acp"],
        "quiet_mode": True,
        "session_id": session_id,
        "model": default_model,
    }
    if stream_delta_callback is not None:
        kwargs["stream_delta_callback"] = stream_delta_callback

    try:
        runtime = resolve_runtime_provider(requested=config_provider)
        kwargs.update(
            {
                "provider": runtime.get("provider"),
                "api_mode": runtime.get("api_mode"),
                "base_url": runtime.get("base_url"),
                "api_key": runtime.get("api_key"),
                "command": runtime.get("command"),
                "args": list(runtime.get("args") or []),
            }
        )
    except Exception:
        logger.debug("A2A falling back to default provider resolution", exc_info=True)

    return AIAgent(**kwargs)


# ---------------------------------------------------------------------------
# Agent Card builder
# ---------------------------------------------------------------------------

def build_agent_card(port: int) -> "AgentCard":
    """Build the A2A Agent Card from env vars."""
    agent_name = os.getenv("AGENT_NAME", "hermes-agent")
    agent_model = os.getenv("AGENT_MODEL", "")
    skills_raw = [s.strip() for s in os.getenv("AGENT_SKILLS", "").split(",") if s.strip()]
    skills = [
        AgentSkill(
            id=s.lower().replace(" ", "_"),
            name=s,
            description=f"{s} capability",
            tags=[s.lower()],
        )
        for s in (skills_raw or [agent_name])
    ]

    public_url = os.getenv("A2A_PUBLIC_URL", f"http://localhost:{port}")

    return AgentCard(
        name=agent_name,
        description=os.getenv("AGENT_DESCRIPTION", f"{agent_name} — Hermes agent"),
        url=public_url,
        version="1.0.0",
        capabilities=AgentCapabilities(streaming=True),
        skills=skills,
        default_input_modes=["text"],
        default_output_modes=["text"],
    )


# ---------------------------------------------------------------------------
# AgentExecutor — the core integration between A2A SDK and Hermes
# ---------------------------------------------------------------------------

class HermesAgentExecutor(AgentExecutor):
    """A2A AgentExecutor that delegates to a Hermes AIAgent."""

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Handle a task: run Hermes and stream results back via EventQueue."""
        task_id = context.task_id
        user_message = _extract_text(context.message)
        session_id = getattr(context, "context_id", None) or task_id

        if not user_message:
            await event_queue.enqueue_event(
                TaskStatusUpdateEvent(
                    id=task_id,
                    status=TaskStatus(state=TaskState.failed),
                    final=True,
                )
            )
            return

        # Emit working status
        await event_queue.enqueue_event(
            TaskStatusUpdateEvent(
                id=task_id,
                status=TaskStatus(state=TaskState.working),
                final=False,
            )
        )

        # Stream tokens back as incremental artifact updates
        accumulated = ""
        loop = asyncio.get_event_loop()

        import queue as _q
        delta_queue: _q.Queue = _q.Queue()

        def _on_delta(delta: str | None) -> None:
            if delta is not None:
                delta_queue.put(delta)

        # Run synchronous Hermes agent in thread executor
        agent_future = loop.run_in_executor(
            _executor,
            lambda: _run_sync(session_id, user_message, stream_delta_callback=_on_delta),
        )

        # Drain delta queue while agent runs
        while not agent_future.done():
            try:
                token = await loop.run_in_executor(
                    None, lambda: delta_queue.get(timeout=0.05)
                )
                if isinstance(token, str):
                    accumulated += token
                    await event_queue.enqueue_event(
                        TaskArtifactUpdateEvent(
                            id=task_id,
                            artifact={"parts": [{"type": "text", "text": accumulated}]},
                            final=False,
                        )
                    )
            except Exception:
                pass  # queue.Empty — keep polling

        result, _ = await agent_future
        final_text = result.get("final_response", "") or accumulated or result.get("error", "")

        if final_text and not accumulated:
            # Non-streaming fallback: emit the full response as a single artifact
            await event_queue.enqueue_event(
                TaskArtifactUpdateEvent(
                    id=task_id,
                    artifact={"parts": [{"type": "text", "text": final_text}]},
                    final=True,
                )
            )

        # Emit completion
        await event_queue.enqueue_event(
            TaskStatusUpdateEvent(
                id=task_id,
                status=TaskStatus(state=TaskState.completed),
                final=True,
            )
        )

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        await event_queue.enqueue_event(
            TaskStatusUpdateEvent(
                id=context.task_id,
                status=TaskStatus(state=TaskState.canceled),
                final=True,
            )
        )


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def build_app(port: int | None = None) -> Any:
    """Build and return the A2A FastAPI/Starlette application.

    Raises RuntimeError if a2a-sdk is not installed.
    """
    if not _SDK_AVAILABLE:
        raise RuntimeError(
            f"a2a-sdk is not installed: {_sdk_err_msg}\n"
            "Install it with: pip install 'a2a-sdk[http-server]'"
        )

    if port is None:
        port = int(os.getenv("A2A_PORT", str(DEFAULT_PORT)))

    agent_card = build_agent_card(port)
    handler = DefaultRequestHandler(
        agent_executor=HermesAgentExecutor(),
        task_store=InMemoryTaskStore(),
    )
    app_builder = A2AStarletteApplication(
        agent_card=agent_card,
        http_handler=handler,
    )
    return app_builder.build()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_text(message: Any) -> str:
    """Extract plain text from an A2A Message (SDK types)."""
    if message is None:
        return ""

    # SDK Pydantic model: message.parts is a list of Part objects
    parts = getattr(message, "parts", None) or []
    texts: list[str] = []
    for part in parts:
        # TextPart has .text; generic Part may have .root.text
        if hasattr(part, "text"):
            texts.append(str(part.text))
        elif hasattr(part, "root") and hasattr(part.root, "text"):
            texts.append(str(part.root.text))
        elif isinstance(part, dict) and part.get("type") == "text":
            texts.append(part.get("text", ""))
    return " ".join(texts).strip()


def _run_sync(
    session_id: str,
    user_message: str,
    stream_delta_callback=None,
) -> tuple[dict, dict]:
    """Synchronous Hermes invocation (runs in thread executor)."""
    agent = _make_agent(session_id=session_id, stream_delta_callback=stream_delta_callback)
    result = agent.run_conversation(user_message=user_message, conversation_history=[])
    usage = {
        "input_tokens": getattr(agent, "session_prompt_tokens", 0) or 0,
        "output_tokens": getattr(agent, "session_completion_tokens", 0) or 0,
        "total_tokens": (
            (getattr(agent, "session_prompt_tokens", 0) or 0)
            + (getattr(agent, "session_completion_tokens", 0) or 0)
        ),
    }
    return result, usage
