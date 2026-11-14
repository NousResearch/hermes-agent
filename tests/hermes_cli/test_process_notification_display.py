"""Background-process completions paint a compact title while the model keeps the raw wall."""
import queue
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from cli import HermesCLI
from tools.process_registry_notifications import (
    PROCESS_COMPLETE_DISPLAY_KIND, format_process_notification, process_completion_display_text)
from tui_gateway import server

def _registry(events):
    return SimpleNamespace(
        drain_notifications=lambda **kw: [(e, format_process_notification(e)) for e in events],
        completion_queue=queue.Queue(), is_completion_consumed=lambda sid: False)

def _event(sid, exit_code, command="cd /tmp && bash long-build.sh"):
    return {"type": "completion", "session_id": sid, "session_key": "display-session", "command": command,
            "exit_code": exit_code, "completion_reason": "exited", "output": "web tsc=0\nSECRET_OUTPUT_LINE"}

@pytest.mark.parametrize("mode,expect_injected", [("off", False), ("concise", True), ("all", True)])
def test_drain_honors_background_process_notifications_off(monkeypatch, mode, expect_injected):
    """``off`` consumes completion events without injecting a REPL input (no spurious post-Ctrl+C
    turn, #123114); every other documented mode keeps injecting, and the durable claim/ack chain
    runs in all modes so an ``off`` session never leaves pending rows to replay on restart."""
    events = [_event("proc_off", 0)]
    cli = HermesCLI.__new__(HermesCLI)
    cli.session_id = "display-session"
    cli._pending_input = queue.Queue()
    monkeypatch.setattr("tools.process_registry.process_registry", _registry(events))
    acked = []
    monkeypatch.setattr("tools.async_delegation.claim_event_delivery", lambda *a: "claimed")
    monkeypatch.setattr("tools.async_delegation.complete_event_delivery", lambda *a: acked.append(a))
    monkeypatch.setattr("cli.CLI_CONFIG", {"display": {"background_process_notifications": mode}})
    cli._drain_process_notifications("cli-post-turn")
    assert acked, "events must still be claimed and acknowledged under every mode"
    assert cli._pending_input.empty() is (not expect_injected)
    if expect_injected:
        assert cli._pending_input.get_nowait() is not None


def test_off_mutes_process_wakes_but_subagent_results_still_land(monkeypatch):
    """``off`` is about background PROCESSES: a finished ``delegate_task(background=true)`` is a
    result the user asked for, never a notification to opt out of."""
    delegation = {"type": "async_delegation", "delegation_id": "d1", "session_key": "display-session",
                  "results": [{"status": "completed", "summary": "done", "goal": "audit"}]}
    events = [_event("proc_a", 0), {**_event("proc_b", 0), "type": "heartbeat", "seq": 1, "output": "line"}, delegation]
    cli = HermesCLI.__new__(HermesCLI)
    cli.session_id = "display-session"
    cli._pending_input = queue.Queue()
    monkeypatch.setattr("tools.process_registry.process_registry", _registry(events))
    monkeypatch.setattr("tools.async_delegation.claim_event_delivery", lambda *a: "claimed")
    monkeypatch.setattr("tools.async_delegation.complete_event_delivery", lambda *a: None)
    monkeypatch.setattr("cli.CLI_CONFIG", {"display": {"background_process_notifications": "off"}})
    cli._drain_process_notifications("cli-post-turn")
    queued = cli._pending_input.get_nowait()
    assert cli._pending_input.empty()
    assert queued.display_kind == "async_delegation_complete"


def test_heartbeat_wake_paints_one_line_and_persists_hidden(monkeypatch):
    """The model gets the output delta; the human gets a one-line receipt, and the persisted row is
    typed ``hidden`` so Desktop/TUI never paint the wake as a user bubble."""
    from tools.process_registry_notifications import HEARTBEAT_DISPLAY_KIND, heartbeat_display_text
    beat = {**_event("proc_hb", None), "type": "heartbeat", "seq": 3, "elapsed": 422, "interval": 60,
            "output": "web tsc=0\nSECRET_OUTPUT_LINE"}
    cli = HermesCLI.__new__(HermesCLI)
    cli.session_id = "display-session"
    cli._pending_input = queue.Queue()
    monkeypatch.setattr("tools.process_registry.process_registry", _registry([beat]))
    monkeypatch.setattr("tools.async_delegation.claim_event_delivery", lambda *a: "claimed")
    monkeypatch.setattr("tools.async_delegation.complete_event_delivery", lambda *a: None)
    cli._drain_process_notifications("cli-idle")
    queued = cli._pending_input.get_nowait()
    assert queued == format_process_notification(beat)
    assert queued.display_kind == HEARTBEAT_DISPLAY_KIND == "hidden"
    assert queued.display_text == heartbeat_display_text(beat)
    assert "SECRET_OUTPUT_LINE" not in queued.display_text and "long-build.sh" in queued.display_text

def test_process_completion_display_keeps_payload_separate_across_surfaces(monkeypatch, capsys, tmp_path):
    events = [_event("proc_1", 0)]
    payload = format_process_notification(events[0])
    cli = HermesCLI.__new__(HermesCLI)
    cli.session_id = "display-session"
    cli._pending_input = queue.Queue()
    registry = _registry(events)
    monkeypatch.setattr("tools.process_registry.process_registry", registry)
    monkeypatch.setattr("tools.async_delegation.claim_event_delivery", lambda *a: "claimed")
    monkeypatch.setattr("tools.async_delegation.complete_event_delivery", lambda *a: None)
    cli._drain_process_notifications("cli-idle")
    for attr, value in {"_pending_resume_sessions": [], "_typed_voice_stop": lambda t: False,
                        "handle_bang_shell": lambda t: False, "_turn_summary_begin": lambda: None,
                        "_tui_after_turn": lambda: None, "_app": SimpleNamespace(invalidate=lambda: None),
                        "chat": Mock()}.items():
        setattr(cli, attr, value)
    cli._tui_process_one_input(cli._pending_input.get_nowait())
    visible = capsys.readouterr().out
    expected = process_completion_display_text(events)
    assert "long-build.sh" in expected and expected in visible
    assert "[IMPORTANT" not in visible and "SECRET_OUTPUT_LINE" not in visible
    queued = cli.chat.call_args.args[0]
    assert queued == payload  # the model still receives the full notification

    cli.conversation_history = []
    cli.agent = SimpleNamespace(run_conversation=Mock(return_value={}))
    cli._chat_stage_user_message(cli.agent, queued)
    staged = cli.conversation_history[-1]
    assert staged["content"] == payload and type(staged["content"]) is str
    assert staged["display_kind"] == PROCESS_COMPLETE_DISPLAY_KIND
    assert staged["display_metadata"]["display_text"] == expected

    from hermes_cli.cli_agent_setup_mixin import _collect_resume_entries
    entries, _, _ = _collect_resume_entries(cli.conversation_history, {}, lambda text: text)
    assert entries == [("event", expected)]

    # TUI gateway: the status line and the persisted turn carry the same compact title.
    emitted, submitted = [], []
    monkeypatch.setattr(server, "_emit", lambda *args: emitted.append(args))
    monkeypatch.setattr(server, "_notif_submit", lambda *args, **kw: submitted.append((args, kw)))
    monkeypatch.setattr(server, "_notif_claim_turn", lambda session: True)
    session = {"session_key": "display-session", "history_lock": threading.RLock()}
    server._notif_handle_ready("ui-session", session, events, set(), registry, format_process_notification, None,
                               owned=True)
    assert emitted[0][2] == {"kind": "process", "text": expected}
    (_rid, _sid, _session, text, _what), kwargs = submitted[0]
    assert text == payload
    assert kwargs["display_kind"] == PROCESS_COMPLETE_DISPLAY_KIND
    assert kwargs["display_metadata"] == {"display_text": expected}


def _drain(monkeypatch, events, mode):
    """Drive the real drain with a stubbed registry/claim chain; return the queued REPL inputs."""
    cli = HermesCLI.__new__(HermesCLI)
    cli.session_id = "display-session"
    cli._pending_input = queue.Queue()
    monkeypatch.setattr("tools.process_registry.process_registry", _registry(events))
    monkeypatch.setattr("tools.async_delegation.claim_event_delivery", lambda *a: "claimed")
    monkeypatch.setattr("tools.async_delegation.complete_event_delivery", lambda *a: None)
    monkeypatch.setattr("cli.CLI_CONFIG", {"display": {"background_process_notifications": mode}})
    cli._drain_process_notifications("cli-post-turn")
    queued = []
    while not cli._pending_input.empty():
        queued.append(cli._pending_input.get_nowait())
    return queued


@pytest.mark.parametrize("mode,exit_code,expect_wake", [
    ("concise", 0, True), ("all", 0, True), ("result", 0, True),
    ("error", 0, False), ("error", 1, True), ("off", 0, False), ("bogus", 0, True),
])
def test_completion_modes_gate_the_turn_start(monkeypatch, mode, exit_code, expect_wake):
    """``result`` always wakes and ``error`` only for a non-clean exit (#153198).

    The gateway watcher has honoured both since the mode was introduced (``notify_mode == "error"
    and session.exit_code not in {0, None}`` in ``gateway/run_notifications.py``); the CLI drain only
    compared against ``off``, so a user who chose ``error`` still started a turn on every success.
    An unrecognised value falls back to ``concise``, the same default the config loader applies.
    """
    assert bool(_drain(monkeypatch, [_event("proc_modes", exit_code)], mode)) is expect_wake


@pytest.mark.parametrize("mode,expect_wake", [
    ("concise", True), ("all", True), ("result", False), ("error", False), ("off", False),
])
def test_only_progress_modes_surface_a_heartbeat(monkeypatch, mode, expect_wake):
    """A heartbeat is running progress, not the final result: the outcome-only modes (``result``,
    ``error``) ask for the outcome and drop it, while ``concise``/``all`` keep the running update."""
    beat = {**_event("proc_beat", None), "type": "heartbeat", "seq": 1, "elapsed": 30, "interval": 60,
            "output": "still building"}
    assert bool(_drain(monkeypatch, [beat], mode)) is expect_wake


def test_error_mode_still_lands_a_subagent_result(monkeypatch):
    """``error`` mutes process noise, never a ``delegate_task(background=true)`` result the user asked for."""
    delegation = {"type": "async_delegation", "delegation_id": "d1", "session_key": "display-session",
                  "results": [{"status": "completed", "summary": "done", "goal": "audit"}]}
    queued = _drain(monkeypatch, [_event("proc_ok", 0), delegation], "error")
    assert [q.display_kind for q in queued] == ["async_delegation_complete"]


@pytest.mark.parametrize("event_type", ["watch_match", "watch_disabled"])
def test_error_mode_keeps_explicit_watch_events(monkeypatch, event_type):
    """A requested watch trigger or diagnostic is not a successful process exit."""
    event = {
        "type": event_type, "session_id": "proc_watched", "session_key": "display-session",
        "command": "python worker.py", "pattern": "READY", "output": "READY",
        "message": "watch patterns disabled",
    }
    assert len(_drain(monkeypatch, [event], "error")) == 1
    assert _drain(monkeypatch, [event], "off") == []
