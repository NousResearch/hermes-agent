"""Completion notices stay human-facing while the parent retains full results."""
import copy
import queue
import threading
from types import SimpleNamespace
from unittest.mock import Mock

from cli import HermesCLI
from tools.process_registry_notifications import format_process_notification
from tui_gateway import server


def test_completion_display_keeps_payload_separate_across_surfaces(monkeypatch, capsys, tmp_path):
    for status, truncated, label in [("completed", False, "Completed"), ("failed", False, "Failed"),
                                      ("cancelled", False, "Cancelled"), ("completed", True, "Incomplete"),
                                      ("stalled", False, "Stalled"), ("unknown", False, "Unknown"),
                                      ("rejected", False, "Failed"), ("other_terminal_state", False, "Incomplete")]:
        event = {"type": "async_delegation", "session_key": "display-session", "delegation_id": "deleg-test",
                 "goals": ["Do not display this sibling", "Review [bold]changes[/bold]"],
                 "results": [{"task_index": 1, "status": status, "truncated": truncated, "summary": "Full result evidence"}]}
        original = copy.deepcopy(event)
        payload = format_process_notification(event)
        cli = HermesCLI.__new__(HermesCLI)
        cli.session_id = event["session_key"]
        cli._pending_input = queue.Queue()
        registry = SimpleNamespace(drain_notifications=lambda **kw: [(event, payload)], completion_queue=queue.Queue())
        monkeypatch.setattr("tools.process_registry.process_registry", registry)
        monkeypatch.setattr("tools.async_delegation.claim_event_delivery", lambda *a: "claimed")
        monkeypatch.setattr("tools.async_delegation.complete_event_delivery", lambda *a: None)
        cli._drain_process_notifications("cli-idle")
        cli._pending_resume_sessions = []
        cli._typed_voice_stop = lambda text: False
        cli.handle_bang_shell = lambda text: False
        cli._turn_summary_begin = lambda: None
        cli._tui_after_turn = lambda: None
        cli._app = SimpleNamespace(invalidate=lambda: None)
        cli.chat = Mock()
        cli._tui_process_one_input(cli._pending_input.get_nowait())
        visible = capsys.readouterr().out
        expected = f"Subagent Task {label}: {event['goals'][1]}"
        assert expected in visible
        assert "ASYNC DELEGATION" not in visible and "Full result evidence" not in visible
        queued_message = cli.chat.call_args.args[0]
        assert queued_message == payload
        cli.conversation_history = []
        cli.agent = SimpleNamespace(run_conversation=Mock(return_value={}))
        cli._flush_credit_notices = lambda: None
        cli._chat_stage_user_message(cli.agent, queued_message)
        from cli import _ChatTurn
        cli._chat_run_agent(_ChatTurn(), str(queued_message))
        staged = cli.conversation_history[-1]
        assert staged["content"] == payload
        assert type(staged["content"]) is str
        assert staged["display_kind"] == "async_delegation_complete"
        from agent.turn_context import _stage_turn_user_message
        run_args = cli.agent.run_conversation.call_args.kwargs
        core_message, _ = _stage_turn_user_message(cli.agent, run_args["user_message"],
                                                   run_args["persist_user_message"], None, None, None, None)
        assert core_message is staged
        assert core_message["display_metadata"]["display_text"] == expected
        assert copy.deepcopy(staged) == staged
        from agent.prompt_caching import build_prompt_cache_plan, strip_anthropic_cache_control
        plan = build_prompt_cache_plan([core_message], tools=None)
        assert strip_anthropic_cache_control(plan.messages)[0]["content"] == payload
        from hermes_cli.cli_agent_setup_mixin import _collect_resume_entries
        from hermes_state import SessionDB
        with SessionDB(tmp_path / f"{status}-{truncated}.db") as db:
            db.create_session(cli.session_id, source="cli")
            db.append_messages_batch(cli.session_id, cli.conversation_history)
            restored = db.get_messages(cli.session_id)
        assert restored[0]["content"] == payload
        entries, _, _ = _collect_resume_entries(restored, {}, lambda text: text)
        assert entries == [("event", expected)]
        emitted, dispatched = [], []
        monkeypatch.setattr(server, "_emit", lambda *args: emitted.append(args))
        monkeypatch.setattr(server, "_notif_dispatch_event", lambda *args: dispatched.append(args))
        session = {"session_key": event["session_key"], "history_lock": threading.RLock()}
        server._notif_handle_event("ui-session", session, event, set(), registry, format_process_notification, None)
        assert emitted[0][2]["text"] == expected
        assert dispatched[0][3] == payload
        assert server._async_delegation_display_metadata(event)["display_text"] == expected
        assert event == original

    from tools.process_registry_notifications import async_delegation_display_text
    grouped = {"group": "Review", "goals": ["First", "Second"],
               "results": [{"task_index": 0, "status": "completed"}, {"task_index": 1, "status": "failed"}]}
    assert async_delegation_display_text(grouped) == "Subagent Tasks Finished with Issues: Review (2 tasks)"
    early = {**grouped, "task_failure_notice": True, "results": [grouped["results"][1]]}
    assert async_delegation_display_text(early) == "Subagent Task Failed: Second"
    grouped["results"][1]["status"] = "completed"
    assert async_delegation_display_text(grouped) == "Subagent Tasks Completed: Review (2 tasks)"
    assert async_delegation_display_text({"goal": "Legacy task", "status": "timeout"}) == "Subagent Task Timed Out: Legacy task"
    assert "Failed" in async_delegation_display_text({"results": [], "error": "Worker crashed"})
    cli._print_user_message_preview("[ASYNC DELEGATION BATCH COMPLETE — user-authored]")
    assert "[ASYNC DELEGATION BATCH COMPLETE — user-authored]" in capsys.readouterr().out
