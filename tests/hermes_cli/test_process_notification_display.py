"""Background-process completions paint a compact title while the model keeps the raw wall."""
import queue
import threading
from types import SimpleNamespace
from unittest.mock import Mock

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
    expected = "Background Process Finished: cd /tmp && bash long-build.sh"
    assert expected in visible
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


def test_process_completion_titles_reflect_outcome_and_batch():
    assert process_completion_display_text([_event("p", 1)]) == (
        "Background Process Failed (exit 1): cd /tmp && bash long-build.sh")
    killed = {**_event("p", -15), "completion_reason": "killed"}
    assert process_completion_display_text([killed]).startswith("Background Process Terminated: ")
    assert process_completion_display_text([_event("a", 0), _event("b", 2)]) == "2 Background Processes Finished"
    long_cmd = "x" * 200
    title = process_completion_display_text([_event("p", 0, command=long_cmd)])
    assert title.endswith("...") and len(title) < 120


def test_cli_clear_rejects_pre_clear_completion_event(monkeypatch, tmp_path):
    from hermes_state import SessionDB

    cli = HermesCLI.__new__(HermesCLI)
    cli.session_id = "display-session"
    cli._pending_input = queue.Queue()
    cli._session_db = SessionDB(tmp_path / "state.db")
    cli._session_db.create_session(cli.session_id, source="cli")
    try:
        cli._session_db.clear_conversation(cli.session_id)
        event = {**_event("proc-old", 0), "started_at": 0.0}
        registry = _registry([event])
        monkeypatch.setattr("tools.process_registry.process_registry", registry)
        monkeypatch.setattr("tools.async_delegation.claim_event_delivery", lambda *a: "claimed")
        monkeypatch.setattr("tools.async_delegation.complete_event_delivery", lambda *a: None)

        cli._drain_process_notifications("cli-idle")
        assert cli._pending_input.empty()
    finally:
        cli._session_db.close()
