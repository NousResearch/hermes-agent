"""A first Ctrl+C must not turn a pending completion into another CLI turn."""
import queue
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from hermes_cli.cli_process_notifications import CLIProcessNotificationsMixin
from hermes_cli.cli_tui_mixin import CLITuiMixin
from tools.process_registry import ProcessRegistry


class Harness(CLITuiMixin, CLIProcessNotificationsMixin):
    pass


def cli_fixture(monkeypatch):
    cli = Harness()
    cli.session_id = "owner"
    cli._pending_input = queue.Queue()
    cli._session_db = None
    cli._voice_lock = threading.Lock()
    cli._voice_recording = False
    for name in ("_slash_confirm_state", "_model_picker_state", "_command_palette_state",
                 "_sudo_state", "_secret_state", "_approval_state", "_clarify_state", "_connection_state"):
        setattr(cli, name, None)
    cli._close_model_picker = Mock()
    cli._close_command_palette = Mock()
    cli.agent = SimpleNamespace(hard_interrupt=Mock())
    cli._agent_running = True
    cli._last_ctrl_c_time = 0
    cli._should_exit = False
    registry = ProcessRegistry()
    monkeypatch.setattr("tools.process_registry.process_registry", registry)
    import cli as facade
    monkeypatch.setattr(facade, "CLI_CONFIG", {"display": {}})
    return cli, registry, SimpleNamespace(app=Mock())


def completion(sid="proc_owned", owner="owner"):
    return {"type": "completion", "session_id": sid, "session_key": owner,
            "command": "fixture", "exit_code": 0, "output": "done"}


def test_ctrl_c_discards_owned_backlog_and_suppresses_requeued_copy(monkeypatch):
    cli, registry, event = cli_fixture(monkeypatch)
    own, foreign = completion(), completion("proc_foreign", "other")
    registry.completion_queue.put(own)
    registry.completion_queue.put(foreign)
    cli._tui_handle_ctrl_c(event)
    cli.agent.hard_interrupt.assert_called_once()
    cli._drain_process_notifications("cli-post-turn")
    assert cli._pending_input.empty(), "Ctrl+C must not restart the agent from its completion backlog"
    assert registry.is_completion_consumed("proc_owned")
    registry.completion_queue.put(dict(own))
    assert registry.drain_notifications(session_key="owner") == []
    remaining = registry.drain_notifications(session_key="other")
    assert [e for e, _ in remaining] == [foreign]
    cli._tui_handle_ctrl_c(event)
    assert cli._should_exit
    event.app.exit.assert_called_once()


def test_ctrl_c_opt_out_preserves_notification(monkeypatch):
    cli, registry, event = cli_fixture(monkeypatch)
    monkeypatch.setattr("cli.CLI_CONFIG", {"display": {"ctrl_c_purge_notifications": False}})
    registry.completion_queue.put(completion())
    cli._tui_handle_ctrl_c(event)
    cli._drain_process_notifications("cli-post-turn")
    assert not cli._pending_input.empty()
    cli.agent.hard_interrupt.assert_called_once()


@pytest.mark.parametrize("owner", ["other", ""])
def test_ctrl_c_does_not_adopt_unproven_delegation(monkeypatch, owner):
    cli, registry, event = cli_fixture(monkeypatch)
    notice = {"type": "async_delegation", "delegation_id": "foreign", "session_key": owner}
    registry.completion_queue.put(notice)
    cli._tui_handle_ctrl_c(event)
    assert registry.completion_queue.get_nowait() == notice
    cli.agent.hard_interrupt.assert_called_once()


def durable_notice(owner="owner", deleg_id="deleg_fixture", interim=False):
    import time
    from tools import async_delegation as durable
    notice = {"type": "async_delegation", "delegation_id": deleg_id,
              "session_key": owner, "dispatched_at": time.time(),
              "status": "completed", "summary": "preserved result"}
    durable._persist_dispatch(notice)
    if interim:
        notice["task_failure_notice"] = True
    else:
        durable._persist_completion(notice, {"summary": "preserved result"})
    return notice


def test_ctrl_c_drops_durable_completion_without_deleting_result(monkeypatch):
    from tools import async_delegation as durable
    cli, registry, event = cli_fixture(monkeypatch)
    notice = durable_notice()
    foreign = durable_notice("other", "deleg_foreign")
    registry.completion_queue.put(notice)
    registry.completion_queue.put(foreign)
    cli._tui_handle_ctrl_c(event)
    row = durable.get_durable_delegation("deleg_fixture")
    assert row["delivery_state"] == "dropped"
    assert row["result"]["summary"] == "preserved result"
    assert durable.get_durable_delegation("deleg_foreign")["delivery_state"] == "pending"
    restored = queue.Queue()
    durable.restore_undelivered_completions(restored)
    assert restored.get_nowait()["delegation_id"] == "deleg_foreign"
    assert restored.empty()
    registry.completion_queue.put(dict(notice))
    cli._drain_process_notifications("cli-post-turn")
    assert cli._pending_input.empty()


def test_ctrl_c_interim_notice_does_not_drop_future_final(monkeypatch):
    from tools import async_delegation as durable
    cli, registry, event = cli_fixture(monkeypatch)
    notice = durable_notice(interim=True)
    registry.completion_queue.put(notice)
    cli._tui_handle_ctrl_c(event)
    assert registry.completion_queue.empty()
    assert durable.get_durable_delegation("deleg_fixture")["delivery_state"] == "pending"


def test_ctrl_c_preserves_claim_owned_by_another_consumer(monkeypatch):
    from tools import async_delegation as durable
    cli, registry, event = cli_fixture(monkeypatch)
    notice = durable_notice()
    assert durable.claim_completion_delivery("deleg_fixture", "other-consumer")
    registry.completion_queue.put(notice)
    cli._tui_handle_ctrl_c(event)
    assert registry.completion_queue.get_nowait() == notice
    assert durable.get_durable_delegation("deleg_fixture")["delivery_state"] == "pending"


def test_failed_purge_preserves_event_and_still_interrupts(monkeypatch):
    cli, registry, event = cli_fixture(monkeypatch)
    notice = completion()
    registry.completion_queue.put(notice)
    monkeypatch.setattr("tools.async_delegation.claim_event_delivery", Mock(side_effect=OSError("fixture")))
    cli._tui_handle_ctrl_c(event)
    cli.agent.hard_interrupt.assert_called_once()
    assert registry.completion_queue.get_nowait() == notice


def test_empty_purge_still_interrupts(monkeypatch):
    cli, registry, event = cli_fixture(monkeypatch)
    cli._tui_handle_ctrl_c(event)
    cli.agent.hard_interrupt.assert_called_once()
    assert registry.completion_queue.empty()


def test_completed_real_process_keeps_output_after_ctrl_c(monkeypatch, tmp_path):
    import subprocess
    import sys
    cli, registry, event = cli_fixture(monkeypatch)
    proc = subprocess.Popen([sys.executable, "-c", "print('actual-background-output')"],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, encoding="utf-8", cwd=tmp_path)
    session = registry.adopt_local(proc, command="fixture python", cwd=str(tmp_path),
                                   task_id="owner", session_key="owner", notify_on_complete=True)
    assert session._completion_event.wait(10), "real child must complete"
    cli._tui_handle_ctrl_c(event)
    cli._drain_process_notifications("cli-post-turn")
    assert cli._pending_input.empty()
    assert registry.is_completion_consumed(session.id)
    assert registry.get(session.id).exited
    assert "actual-background-output" in registry.read_log(session.id)["output"]


@pytest.mark.parametrize("failure", [False, OSError("drop failed")])
def test_failed_durable_drop_releases_claim_for_normal_delivery(monkeypatch, failure):
    from tools import async_delegation as durable
    cli, registry, event = cli_fixture(monkeypatch)
    notice = durable_notice()
    registry.completion_queue.put(notice)
    drop = Mock(side_effect=failure) if isinstance(failure, Exception) else Mock(return_value=False)
    monkeypatch.setattr(durable, "drop_completion_delivery", drop)
    cli._tui_handle_ctrl_c(event)
    cli.agent.hard_interrupt.assert_called_once()
    cli._drain_process_notifications("cli-post-turn")
    assert not cli._pending_input.empty(), "failed purge must leave a normally deliverable result"
    assert durable.get_durable_delegation("deleg_fixture")["delivery_state"] == "delivered"
