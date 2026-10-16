"""Matrix adapter: a gateway restart settles the runs it left in flight.

A gateway killed mid-run never reaches on_processing_complete, so its 👀 stays
on the request and its agent-protocol status bubble stays "working" on the wire
forever — clients show the run as running until their own stall sweep. The
adapter persists in-flight runs to its store dir and, on the next connect,
redacts the 👀s and edits the bubbles to failed.
"""
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from gateway.config import PlatformConfig

KEY = "org.fluxchat.agent"
ROOM = "!r:example.org"


def _adapter(monkeypatch, tmp_path):
    monkeypatch.setenv("MATRIX_ALLOWED_USERS", "@me:example.org")
    from plugins.platforms.matrix.adapter import MatrixAdapter
    adapter = MatrixAdapter(PlatformConfig(enabled=True, token="tok", extra={"homeserver": "https://matrix.example.org"}))
    adapter._user_id = "@bot:example.org"
    adapter._client = object()
    monkeypatch.setattr(type(adapter), "_run_state_path", property(lambda self: tmp_path / "in_flight_runs.json"))
    return adapter


def _event(msg_id="$req"):
    return SimpleNamespace(message_id=msg_id, source=SimpleNamespace(chat_id=ROOM))


def _state(tmp_path):
    return json.loads((tmp_path / "in_flight_runs.json").read_text())


@pytest.mark.asyncio
async def test_in_flight_runs_are_written_and_cleared_around_a_run(monkeypatch, tmp_path):
    from gateway.platforms.base import ProcessingOutcome
    adapter = _adapter(monkeypatch, tmp_path)
    adapter._send_reaction = AsyncMock(return_value="$eyes")
    await adapter.on_processing_start(_event())
    state = _state(tmp_path)
    assert state["runs"] == [f"{ROOM}\x1f$req"]
    assert state["eyes"] == {f"{ROOM}\x1f$req": "$eyes"}
    # A working bubble joins the same run's entry.
    adapter._remember_status_bubble(ROOM, "$prog", "📖 Reading", {"state": "working", "run": "$req"})
    assert _state(tmp_path)["bubbles"] == {f"{ROOM}\x1f$req": ["$prog", "📖 Reading"]}
    # A clean end empties it.
    adapter._redact_reaction = AsyncMock(return_value=True)
    adapter.edit_message = AsyncMock()
    await adapter.on_processing_complete(_event(), ProcessingOutcome.SUCCESS)
    assert _state(tmp_path)["runs"] == []


@pytest.mark.asyncio
async def test_connect_settles_runs_left_over_from_the_last_process(monkeypatch, tmp_path):
    adapter = _adapter(monkeypatch, tmp_path)  # fresh process: the maps start empty
    (tmp_path / "in_flight_runs.json").write_text(json.dumps({
        "runs": [f"{ROOM}\x1f$dead"],
        "eyes": {f"{ROOM}\x1f$dead": "$eyes-old"},
        "bubbles": {f"{ROOM}\x1f$dead": ["$prog-old", "📖 Reading"]},
    }))
    adapter._redact_reaction = AsyncMock(return_value=True)
    adapter.edit_message = AsyncMock()
    await adapter._settle_orphaned_runs()
    adapter._redact_reaction.assert_awaited_once_with(ROOM, "$eyes-old", "gateway restarted mid-run")
    room, ev, text = adapter.edit_message.await_args.args
    status = adapter.edit_message.await_args.kwargs["agent_status"]
    assert (room, ev, text) == (ROOM, "$prog-old", "📖 Reading")
    assert status["state"] == "failed" and status["run"] == "$dead"
    assert KEY not in status  # the edit carries the status via the agent_status kwarg
    # Settled: the file has nothing left, and a second pass is a no-op.
    assert _state(tmp_path)["runs"] == []
    await adapter._settle_orphaned_runs()
    assert adapter.edit_message.await_count == 1


@pytest.mark.asyncio
async def test_a_missing_or_broken_file_is_not_an_error(monkeypatch, tmp_path):
    adapter = _adapter(monkeypatch, tmp_path)
    await adapter._settle_orphaned_runs()  # no file at all
    (tmp_path / "in_flight_runs.json").write_text("{not json")
    await adapter._settle_orphaned_runs()  # garbage is treated as empty


@pytest.mark.asyncio
async def test_only_the_first_working_message_of_a_run_is_the_bubble(monkeypatch, tmp_path):
    """The settle target must not move to a later working send (step log, heartbeat).

    Repro (The Battlefront, 2026-10-12): the queued-follow-up path sent progress working
    messages after the real bubble; the run-end edit chased the last tracked event, so the
    original working bubble stayed state=working on the wire forever.
    """
    from gateway.platforms.base import ProcessingOutcome
    adapter = _adapter(monkeypatch, tmp_path)
    run = "$req"
    adapter._remember_status_bubble(ROOM, "$bubble", "📖 Reading", {"state": "working", "run": run})
    # Later working SENDS (new events) are progress: they must not steal the settle target.
    adapter._remember_status_bubble(ROOM, "$step-log", "🔧 Running tool", {"state": "working", "run": run})
    adapter._remember_status_bubble(ROOM, "$heartbeat", "💓 still working", {"state": "working", "run": run})
    assert adapter._agent_status_bubbles[(ROOM, run)] == ("$bubble", "📖 Reading")
    assert _state(tmp_path)["bubbles"] == {f"{ROOM}\x1f{run}": ["$bubble", "📖 Reading"]}
    # An EDIT of the bubble itself keeps the same target and refreshes the remembered text.
    adapter._remember_status_bubble(ROOM, "$bubble", "📖 Reading + 🔧 Running tool", {"state": "working", "run": run})
    assert adapter._agent_status_bubbles[(ROOM, run)] == ("$bubble", "📖 Reading + 🔧 Running tool")
    assert _state(tmp_path)["bubbles"] == {f"{ROOM}\x1f{run}": ["$bubble", "📖 Reading + 🔧 Running tool"]}
    # The run ends: the FIRST working bubble is what gets settled.
    adapter._redact_reaction = AsyncMock(return_value=True)
    adapter.edit_message = AsyncMock()
    await adapter.on_processing_complete(_event(msg_id=run), ProcessingOutcome.SUCCESS)
    assert adapter.edit_message.await_args.args[1] == "$bubble"
    assert adapter.edit_message.await_args.kwargs["agent_status"]["state"] == "done"


@pytest.mark.asyncio
async def test_a_working_status_without_a_run_is_never_tracked(monkeypatch, tmp_path):
    adapter = _adapter(monkeypatch, tmp_path)
    adapter._remember_status_bubble(ROOM, "$x", "hi", {"state": "working"})
    adapter._remember_status_bubble(ROOM, "$y", "hi", {"state": "done", "run": "$req"})
    adapter._remember_status_bubble(  # a settled approval prompt isn't the bubble either
        ROOM, "$z", "ok", {"state": "working", "run": "$req", "choices": [{"label": "Allow"}]})
    assert adapter._agent_status_bubbles == {}
