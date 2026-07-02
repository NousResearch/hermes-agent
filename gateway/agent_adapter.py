"""AgentAdapter — single execution interface between Gateway and Runtime.

The Gateway executes exclusively through AgentAdapter.  Runtime-specific
logic (AIAgent construction, run_conversation dispatch, lifecycle) lives
only inside HermesAdapter.

Protocol
--------
    AgentAdapter  (Protocol)   — what the Gateway codes against
    HermesAdapter (concrete)   — wraps AIAgent from run_agent
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Protocol — the Gateway's contract with the Runtime
# ---------------------------------------------------------------------------

@runtime_checkable
class AgentAdapter(Protocol):
    """Interface the Gateway uses for agent execution.

    Every attribute/method here is a direct delegate to the underlying
    AIAgent.  The Gateway must never import or reference AIAgent directly;
    all Runtime access goes through this surface.
    """

    # -- post-run introspection (read by Gateway after run()) ----------------
    @property
    def session_id(self) -> Optional[str]: ...

    @property
    def model(self) -> Optional[str]: ...

    @property
    def context_compressor(self) -> Any: ...

    @property
    def session_prompt_tokens(self) -> int: ...

    @property
    def session_completion_tokens(self) -> int: ...

    @property
    def is_interrupted(self) -> bool: ...

    @property
    def _last_compaction_in_place(self) -> bool: ...

    # -- callbacks (set by Gateway before run()) ----------------------------
    tool_progress_callback: Optional[Callable[..., None]]
    tool_start_callback: Optional[Callable[..., None]]
    step_callback: Optional[Callable[..., None]]
    stream_delta_callback: Optional[Callable[..., None]]
    interim_assistant_callback: Optional[Callable[..., None]]
    status_callback: Optional[Callable[..., None]]

    # -- lifecycle -----------------------------------------------------------
    max_iterations: int

    def run(
        self,
        user_message: Any,
        *,
        conversation_history: Optional[List[Dict[str, Any]]] = None,
        task_id: Optional[str] = None,
        persist_user_message: Optional[str] = None,
        persist_user_timestamp: Optional[float] = None,
        moa_config: Optional[dict] = None,
    ) -> Dict[str, Any]:
        """Execute one conversation turn.  Returns the full result dict."""
        ...

    def interrupt(self, reason: str = "") -> None: ...

    def steer(self, text: str) -> bool: ...

    def get_activity_summary(self) -> dict: ...

    def close(self) -> None: ...

    def shutdown_memory_provider(
        self, session_messages: Optional[List[Dict[str, Any]]] = None,
    ) -> None: ...

    @property
    def api_mode(self) -> Optional[str]: ...

    @property
    def provider(self) -> Optional[str]: ...


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_adapter(**kwargs: Any) -> AgentAdapter:
    """Create a HermesAdapter from AIAgent constructor kwargs.

    This is the single entry-point the Gateway calls to obtain an adapter.
    All Runtime-specific imports happen inside HermesAdapter.__init__.
    """
    return HermesAdapter(**kwargs)


# ---------------------------------------------------------------------------
# Concrete adapter — wraps AIAgent
# ---------------------------------------------------------------------------

class HermesAdapter:
    """Thin wrapper around AIAgent that satisfies the AgentAdapter protocol.

    Lifecycle:
        1. Gateway calls ``create_adapter(...)`` → HermesAdapter constructed
        2. Gateway sets callbacks on the adapter
        3. Gateway calls ``adapter.run(user_message, ...)`` → delegates to
           ``AIAgent.run_conversation()``
        4. Gateway reads post-run properties (session_id, token counts, etc.)
        5. Gateway calls ``adapter.close()`` when the session is evicted

    All AIAgent imports are deferred to __init__ so the Gateway module never
    imports ``run_agent`` directly.
    """

    def __init__(self, **kwargs: Any) -> None:
        # Deferred import — keeps the Gateway free of Runtime coupling.
        from run_agent import AIAgent as _AIAgent

        self._agent = _AIAgent(**kwargs)

    # -- property pass-throughs --------------------------------------------

    @property
    def session_id(self) -> Optional[str]:
        return getattr(self._agent, "session_id", None)

    @property
    def model(self) -> Optional[str]:
        return getattr(self._agent, "model", None)

    @property
    def context_compressor(self) -> Any:
        return getattr(self._agent, "context_compressor", None)

    @property
    def session_prompt_tokens(self) -> int:
        return getattr(self._agent, "session_prompt_tokens", 0)

    @property
    def session_completion_tokens(self) -> int:
        return getattr(self._agent, "session_completion_tokens", 0)

    @property
    def is_interrupted(self) -> bool:
        return getattr(self._agent, "is_interrupted", False)

    @property
    def _last_compaction_in_place(self) -> bool:
        return bool(getattr(self._agent, "_last_compaction_in_place", False))

    # -- callback delegates -------------------------------------------------

    @property
    def tool_progress_callback(self):
        return getattr(self._agent, "tool_progress_callback", None)

    @tool_progress_callback.setter
    def tool_progress_callback(self, value):
        self._agent.tool_progress_callback = value

    @property
    def tool_start_callback(self):
        return getattr(self._agent, "tool_start_callback", None)

    @tool_start_callback.setter
    def tool_start_callback(self, value):
        self._agent.tool_start_callback = value

    @property
    def step_callback(self):
        return getattr(self._agent, "step_callback", None)

    @step_callback.setter
    def step_callback(self, value):
        self._agent.step_callback = value

    @property
    def stream_delta_callback(self):
        return getattr(self._agent, "stream_delta_callback", None)

    @stream_delta_callback.setter
    def stream_delta_callback(self, value):
        self._agent.stream_delta_callback = value

    @property
    def interim_assistant_callback(self):
        return getattr(self._agent, "interim_assistant_callback", None)

    @interim_assistant_callback.setter
    def interim_assistant_callback(self, value):
        self._agent.interim_assistant_callback = value

    @property
    def status_callback(self):
        return getattr(self._agent, "status_callback", None)

    @status_callback.setter
    def status_callback(self, value):
        self._agent.status_callback = value

    # -- lifecycle ----------------------------------------------------------

    @property
    def max_iterations(self) -> int:
        return getattr(self._agent, "max_iterations", 90)

    @max_iterations.setter
    def max_iterations(self, value: int) -> None:
        self._agent.max_iterations = value

    def run(
        self,
        user_message: Any,
        *,
        conversation_history: Optional[List[Dict[str, Any]]] = None,
        task_id: Optional[str] = None,
        persist_user_message: Optional[str] = None,
        persist_user_timestamp: Optional[float] = None,
        moa_config: Optional[dict] = None,
    ) -> Dict[str, Any]:
        """Delegate to AIAgent.run_conversation()."""
        kwargs: Dict[str, Any] = {}
        if conversation_history is not None:
            kwargs["conversation_history"] = conversation_history
        if task_id is not None:
            kwargs["task_id"] = task_id
        if persist_user_message is not None:
            kwargs["persist_user_message"] = persist_user_message
        if persist_user_timestamp is not None:
            kwargs["persist_user_timestamp"] = persist_user_timestamp
        if moa_config is not None:
            kwargs["moa_config"] = moa_config
        return self._agent.run_conversation(user_message, **kwargs)

    def interrupt(self, reason: str = "") -> None:
        self._agent.interrupt(reason)

    def steer(self, text: str) -> bool:
        return self._agent.steer(text)

    def get_activity_summary(self) -> dict:
        return self._agent.get_activity_summary()

    def close(self) -> None:
        self._agent.close()

    def shutdown_memory_provider(
        self, session_messages: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        if session_messages is not None:
            self._agent.shutdown_memory_provider(session_messages)
        else:
            self._agent.shutdown_memory_provider()

    @property
    def api_mode(self) -> Optional[str]:
        return getattr(self._agent, "api_mode", None)

    @property
    def provider(self) -> Optional[str]:
        return getattr(self._agent, "provider", None)
