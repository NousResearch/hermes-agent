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


