"""Shared fakes for the A2A adapter tests.

All of these let the adapter run without model credentials: ``FakeAgent``
stands in for ``AIAgent`` (injected via ``ContextSessionStore(agent_factory=...)``),
``FakeContext`` / ``RecordingQueue`` let us drive ``HermesAgentExecutor.execute``
directly, and ``make_user_message`` builds a valid A2A request message.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from a2a.types import Message, Part, Role, TextPart


class FakeAgent:
    """Minimal stand-in for ``AIAgent``.

    Exercises the callback bridge (one streamed delta, one tool start, one tool
    result) and returns a deterministic echo response plus updated history.
    """

    def __init__(self) -> None:
        self.stream_delta_callback = None
        self.reasoning_callback = None
        self.tool_progress_callback = None
        self.step_callback = None
        self.thinking_callback = None
        self.interrupted = False
        self.runs: list[str] = []

    def run_conversation(
        self,
        *,
        user_message: str,
        conversation_history: list[dict[str, Any]] | None = None,
        task_id: str | None = None,
        **_kwargs: Any,
    ) -> dict[str, Any]:
        self.runs.append(user_message)
        if self.stream_delta_callback:
            self.stream_delta_callback("thinking... ")
        if self.tool_progress_callback:
            self.tool_progress_callback(
                "tool.started", name="read_file", args={"path": "x.py"}
            )
        if self.step_callback:
            self.step_callback(1, [{"name": "read_file", "result": "file contents"}])
        final = f"echo: {user_message}"
        messages = list(conversation_history or [])
        messages.append({"role": "user", "content": user_message})
        messages.append({"role": "assistant", "content": final})
        return {"final_response": final, "messages": messages}

    def interrupt(self, message: str | None = None) -> None:
        self.interrupted = True


class RecordingQueue:
    """Captures events enqueued by the executor / TaskUpdater."""

    def __init__(self) -> None:
        self.events: list[Any] = []

    async def enqueue_event(self, event: Any) -> None:
        self.events.append(event)


class FakeContext:
    """Stands in for a2a-sdk's ``RequestContext`` for direct executor tests."""

    def __init__(
        self,
        user_text: str,
        message: Message,
        *,
        current_task: Any = None,
        context_id: str | None = None,
    ) -> None:
        self._user_text = user_text
        self.message = message
        self.current_task = current_task
        self.context_id = context_id

    def get_user_input(self, delimiter: str = "\n") -> str:
        return self._user_text


def make_user_message(text: str, context_id: str | None = None) -> Message:
    return Message(
        role=Role.user,
        kind="message",
        message_id="msg-test",
        parts=[Part(root=TextPart(text=text))],
        context_id=context_id,
    )


@pytest.fixture
def fakes() -> SimpleNamespace:
    return SimpleNamespace(
        FakeAgent=FakeAgent,
        RecordingQueue=RecordingQueue,
        FakeContext=FakeContext,
        make_user_message=make_user_message,
    )
