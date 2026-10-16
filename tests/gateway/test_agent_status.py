"""Agent protocol run state (gateway/agent_status.py): the progress bubble reports ``working`` with
the step, the final reply ``done`` / ``failed`` with model, context and spend, and the Matrix adapter
settles the bubble when the run ends."""
import asyncio
import queue as queue_mod
import types
from contextlib import suppress
from unittest.mock import AsyncMock

import pytest

from gateway import agent_status
from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import ProcessingOutcome, SendResult
from gateway.session import SessionSource
from gateway.turn_context import TurnContext

KEY = "org.fluxchat.agent"
ROOM = "!r:example.org"


def test_step_text_drops_the_tool_emoji_and_repeat_count():
    assert agent_status.step_text("💻 terminal: pytest -q") == "terminal: pytest -q"
    assert agent_status.step_text("🔍 search_files (×3)") == "search_files"
    assert agent_status.step_text("plain words") == "plain words"
    assert agent_status.step_text("") == ""
    assert agent_status.step_text(None) == ""


def test_working_and_finished():
    assert agent_status.working("$req", "📖 read_file: a.py", 2) == {
        "v": 1, "state": "working", "run": "$req", "step": "read_file: a.py", "steps": 2}
    result = {"model": "claude-opus-5", "last_prompt_tokens": 42_000, "context_length": 200_000,
              "estimated_cost_usd": 0.1834, "cost_status": "estimated"}
    assert agent_status.finished("$req", result, failed=False) == {
        "v": 1, "state": "done", "run": "$req", "model": "claude-opus-5", "context": 0.21, "cost_usd": 0.1834}
    # The reply's first line rides along as the run's one-line outcome.
    with_reply = agent_status.finished("$req", result, failed=False,
                                       response="**Deployed v0.2.0.**\n\nDetails below, more lines.")
    assert with_reply["result"] == "Deployed v0.2.0."
    assert agent_status.finished("$req", {}, failed=True, response="").get("result") is None
    # Unknown pricing isn't reported as free; a non-dict result still gives the state.
    unknown = agent_status.finished("$req", {"estimated_cost_usd": 0.0, "cost_status": "unknown"}, failed=True)
    assert unknown == {"v": 1, "state": "failed", "run": "$req"}
    assert agent_status.finished(None, None, failed=False) == {"v": 1, "state": "done"}


def test_with_status_copies():
    shared = {"thread_id": "$root"}
    merged = agent_status.with_status(shared, {"state": "working"})
    assert merged == {"thread_id": "$root", "fluxchat_agent": {"state": "working"}}
    assert shared == {"thread_id": "$root"}
    assert agent_status.with_status(shared, None) is shared


class _ProgressAdapter:
    """Records the progress bubble's sends and edits."""
    AGENT_STATUS = True
    name = "fake"

    def __init__(self):
        self.calls = []

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        self.calls.append(("send", content, metadata))
        return SendResult(success=True, message_id="$bubble")

    async def edit_message(self, chat_id, message_id, content, *, finalize=False, metadata=None):
        self.calls.append(("edit", content, metadata))
        return SendResult(success=True, message_id=message_id)

    async def send_typing(self, chat_id, metadata=None):
        pass


def _run_progress(adapter, lines):
    from gateway.run_turn_runner import TurnRunner

    class _Runner:
        def _delivery_adapter_for(self, source):
            return adapter

    q = queue_mod.Queue()
    for line in lines:
        q.put(line)
    source = SessionSource(platform=Platform.MATRIX, chat_id=ROOM, chat_type="group", user_id="@me:example.org")
    ctx = TurnContext(source=source, progress_queue=q, _run_still_current=lambda: True,
                      inbound_message_id="$req", _progress_metadata={"thread_id": "$req"})

    async def go():
        task = asyncio.create_task(TurnRunner(_Runner(), ctx).send_progress_messages())
        await asyncio.sleep(0.6)
        task.cancel()  # turn over: the bubble gets a last edit with what's still queued
        with suppress(asyncio.CancelledError):
            await task

    asyncio.run(go())
    return ctx


def test_progress_bubble_reports_the_step_and_count():
    adapter = _ProgressAdapter()
    ctx = _run_progress(adapter, ["💻 terminal: pytest -q", "📖 read_file: a.py"])
    kind, text, metadata = adapter.calls[0]
    assert kind == "send" and text == "💻 terminal: pytest -q"
    assert metadata["fluxchat_agent"] == {"v": 1, "state": "working", "run": "$req",
                                          "step": "terminal: pytest -q", "steps": 1}
    assert metadata["thread_id"] == "$req", "thread routing still applies"
    kind, text, metadata = adapter.calls[-1]
    assert kind == "edit"
    assert metadata["fluxchat_agent"]["step"] == "read_file: a.py"
    assert metadata["fluxchat_agent"]["steps"] == 2
    assert ctx._progress_metadata == {"thread_id": "$req"}, "the shared metadata is left alone"


def test_adapters_without_agent_status_get_plain_metadata():
    adapter = _ProgressAdapter()
    adapter.AGENT_STATUS = False
    _run_progress(adapter, ["💻 terminal: ls"])
    assert adapter.calls[0][2] == {"thread_id": "$req"}


def _matrix(monkeypatch):
    monkeypatch.setenv("MATRIX_ALLOWED_USERS", "@me:example.org")
    from plugins.platforms.matrix.adapter import MatrixAdapter
    adapter = MatrixAdapter(PlatformConfig(enabled=True, token="tok", extra={"homeserver": "https://matrix.example.org"}))
    adapter._user_id = "@bot:example.org"
    adapter._client = object()
    adapter._reaction_redaction_delay_seconds = 0
    return adapter


def _event(status=None):
    event = types.SimpleNamespace(message_id="$req", source=types.SimpleNamespace(chat_id=ROOM))
    if status is not None:
        event._agent_status = status
    return event


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome, state", [(ProcessingOutcome.SUCCESS, "done"),
                                            (ProcessingOutcome.FAILURE, "failed"),
                                            (ProcessingOutcome.CANCELLED, "failed")])
async def test_matrix_settles_the_progress_bubble(monkeypatch, outcome, state):
    adapter = _matrix(monkeypatch)
    adapter._send_room_message = AsyncMock(return_value="$bubble")
    adapter._send_reaction = AsyncMock(return_value="$r")
    adapter.redact_message = AsyncMock(return_value=True)
    edits = []
    adapter._send_content_event = AsyncMock(
        side_effect=lambda chat, content: edits.append(content) or SendResult(success=True, message_id="$edit"))
    await adapter.send(ROOM, "💻 terminal: ls", metadata={"fluxchat_agent": agent_status.working("$req", "💻 ls", 1)})
    # Progress edits keep it tracked.
    await adapter.edit_message(ROOM, "$bubble", "💻 terminal: ls\n📖 read_file: a",
                               metadata={"fluxchat_agent": agent_status.working("$req", "📖 read_file: a", 2)})
    assert edits[-1]["m.new_content"][KEY]["steps"] == 2

    done = {"v": 1, "state": "done", "run": "$req", "model": "m", "cost_usd": 0.5}
    await adapter.on_processing_complete(_event(done), outcome)
    settle = edits[-1]
    assert settle["m.relates_to"] == {"rel_type": "m.replace", "event_id": "$bubble"}
    assert settle["m.new_content"][KEY] == {**done, "state": state}
    assert "read_file" in settle["m.new_content"]["body"], "the bubble keeps its text"
    # Settled once: a second completion doesn't edit again.
    count = len(edits)
    await adapter.on_processing_complete(_event(done), outcome)
    assert len(edits) == count


@pytest.mark.asyncio
async def test_matrix_runs_without_a_bubble_edit_nothing(monkeypatch):
    adapter = _matrix(monkeypatch)
    adapter._send_reaction = AsyncMock(return_value="$r")
    adapter._send_content_event = AsyncMock(return_value=SendResult(success=True))
    await adapter.on_processing_complete(_event(), ProcessingOutcome.SUCCESS)
    adapter._send_content_event.assert_not_called()
    adapter._send_reaction.assert_awaited_with(ROOM, "$req", "✅")


@pytest.mark.asyncio
async def test_final_reply_carries_how_the_run_ended(monkeypatch):
    adapter = _matrix(monkeypatch)
    captured = {}

    async def ledgered(event, session_key, text, metadata, **kwargs):
        captured["metadata"] = metadata
        return SendResult(success=True, message_id="$answer"), adapter

    adapter.send_final_ledgered = ledgered
    done = {"v": 1, "state": "done", "run": "$req"}
    await adapter._send_final_text(_event(done), "sk", "All green.", {"thread_id": "$req"}, False, 0, lambda r: None)
    assert captured["metadata"] == {"thread_id": "$req", "fluxchat_agent": done}
    # A turn the gateway didn't stamp sends as before.
    await adapter._send_final_text(_event(), "sk", "Hi.", {"thread_id": "$req"}, False, 0, lambda r: None)
    assert captured["metadata"] == {"thread_id": "$req"}


@pytest.mark.asyncio
async def test_approval_names_its_run_and_settling_it_leaves_the_bubble(monkeypatch):
    from unittest.mock import patch

    from gateway.platforms.base import ExecApprovalPrompt
    adapter = _matrix(monkeypatch)
    sent, edits = [], []
    adapter._send_room_message = AsyncMock(side_effect=lambda chat, content: sent.append(content) or f"$e{len(sent)}")
    adapter._send_reaction = AsyncMock(return_value="$r")
    adapter._redact_bot_approval_reactions = AsyncMock()
    adapter._send_content_event = AsyncMock(
        side_effect=lambda chat, content: edits.append(content) or SendResult(success=True))
    await adapter.send(ROOM, "💻 terminal: ls", metadata={"fluxchat_agent": agent_status.working("$req", "💻 ls", 1)})
    prompt = ExecApprovalPrompt(
        chat_id=ROOM, session_key="sess-1", text="Run `rm -rf build`?",
        actions=[("Allow once", "once", "primary"), ("Deny", "deny", "danger")], command="rm -rf build",
        description="", smart_denied=False, metadata={"thread_id": "$req", "fluxchat_agent": {"run": "$req"}})
    await adapter._send_exec_approval_prompt(prompt)
    assert sent[-1][KEY]["state"] == "waiting" and sent[-1][KEY]["run"] == "$req"

    event = types.SimpleNamespace(sender="@me:example.org", event_id="$react", room_id=ROOM,
                                  content={"m.relates_to": {"event_id": "$e2", "key": "✅"}})
    with patch("tools.approval.resolve_gateway_approval", return_value=1):
        await adapter._on_reaction(event)
    assert edits[-1]["m.new_content"][KEY]["state"] == "working"
    assert edits[-1]["m.new_content"][KEY]["run"] == "$req"

    await adapter.on_processing_complete(_event({"v": 1, "state": "done", "run": "$req"}), ProcessingOutcome.SUCCESS)
    assert edits[-1]["m.relates_to"]["event_id"] == "$e1", "the progress bubble is settled, not the prompt"
    assert edits[-1]["m.new_content"][KEY]["state"] == "done"
