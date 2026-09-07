import argparse
import json
import threading
from datetime import UTC, datetime, timedelta
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import SimpleNamespace

import pytest

from github_pr_feedback.ledger import FeedbackLedger
from github_pr_feedback.policy import FeedbackReceipt


def registered_manager(monkeypatch, home):
    from hermes_cli import plugins

    manager = plugins.PluginManager(scope_key=str(home))
    manager._discovered = True
    ctx = SimpleNamespace(
        get_config=lambda key, default=None: True if key == "enabled" else default,
        register_cli_command=lambda **kwargs: None,
        register_hook=lambda name, callback: manager._hooks.setdefault(name, []).append(callback),
    )
    spec = spec_from_file_location("repair_contract_entry", Path(__file__).parents[1] / "__init__.py")
    entry = module_from_spec(spec)
    spec.loader.exec_module(entry)
    entry.register(ctx)
    monkeypatch.setattr(plugins, "_delivery_manager", lambda: manager)
    return manager


@pytest.mark.parametrize("surface", ["tool", "cli"])
@pytest.mark.parametrize("action", ["pending", "resolving", "completed", "superseded", "unbound"])
def test_real_completion_surfaces_require_the_task_bound_feedback_contract(tmp_path, monkeypatch, surface, action):
    from hermes_cli import kanban as cli, kanban_db as kb, kanban_db_connect as kbc
    from tools import kanban_tools

    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_CONTROL_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    with kbc.connect() as connection:
        tid = kb.create_task(connection, title="Repair an exact PR", assignee="worker")
        kb.claim_task(connection, tid)
    monkeypatch.setenv("HERMES_KANBAN_TASK", tid)
    manager = registered_manager(monkeypatch, home)
    # Completion approval from another plugin cannot mask the repair contract.
    manager._hooks.setdefault("pre_kanban_complete", []).insert(0, lambda **kw: {"action": "approve"})
    ledger = FeedbackLedger(home / "github-pr-feedback" / "ledger.sqlite3")
    now = datetime.now(UTC)
    try:
        if action != "unbound":
            receipt = FeedbackReceipt("acme/repo", 1, "review_comment", "comment-1", "a" * 40)
            claim = ledger.claim(receipt, owner="test", claimed_at=now, stale_before=now - timedelta(minutes=5))
            ledger.finalize(receipt, tid, claim)
            if action == "completed":
                ledger.mark_feedback_actioned(receipt, resolved_head_sha="a" * 40, actioned_at=now)
            elif action == "resolving":
                ledger.begin_feedback_action(receipt, resolved_head_sha="a" * 40, actioned_at=now)
            elif action == "superseded":
                # The fixture represents the durable result of the separately tested retirement transition.
                ledger._connection.execute("UPDATE feedback_receipts SET action_status='superseded' WHERE task_id=?", (tid,))
                ledger._connection.commit()
        if surface == "tool":
            result = json.loads(kanban_tools._handle_complete({"task_id": tid, "summary": "Local tests passed"}))
            allowed = result.get("ok") is True
        else:
            result = cli._cmd_complete(argparse.Namespace(task_id=tid, task_ids=[tid], result=None,
                                                         summary="Local tests passed", metadata=None))
            allowed = result == 0
        expected = action in {"completed", "superseded", "unbound"}
        assert allowed is expected
        with kbc.connect() as connection:
            assert (kb.get_task(connection, tid).status == "done") is expected
    finally:
        ledger.close()


@pytest.mark.parametrize("failure", ["exception", "unavailable_ledger", "timeout", "invalid_decision"])
def test_completion_policy_failure_does_not_mark_done(tmp_path, monkeypatch, failure):
    from hermes_cli import kanban_db as kb, kanban_db_connect as kbc
    from tools import kanban_tools

    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_CONTROL_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    with kbc.connect() as connection:
        tid = kb.create_task(connection, title="Feedback worker", assignee="worker")
        kb.claim_task(connection, tid)
    monkeypatch.setenv("HERMES_KANBAN_TASK", tid)
    manager = registered_manager(monkeypatch, home)
    if failure == "exception":
        def broken(**kwargs):
            raise RuntimeError("cannot verify contract")
        manager._hooks["pre_kanban_complete"] = [broken]
    release = threading.Event()
    if failure == "timeout":
        from hermes_cli import plugins

        monkeypatch.setattr(plugins, "_resolve_hook_callback_timeout", lambda: 0.02)
        def delayed(**kwargs):
            release.wait(2)
        manager._hooks["pre_kanban_complete"] = [delayed]
    elif failure == "invalid_decision":
        manager._hooks["pre_kanban_complete"] = [lambda **kwargs: "approved"]
    try:
        result = json.loads(kanban_tools._handle_complete({"task_id": tid, "summary": "Complete"}))
        assert result.get("ok") is not True
        with kbc.connect() as connection:
            assert kb.get_task(connection, tid).status != "done"
    finally:
        release.set()


def test_disabled_feedback_plugin_ignores_a_legacy_control_ledger(tmp_path, monkeypatch):
    """An optional plugin's stale ledger must not become a core completion gate."""
    import sqlite3

    control = tmp_path / "control"
    ledger_path = control / "github-pr-feedback" / "ledger.sqlite3"
    ledger_path.parent.mkdir(parents=True)
    with sqlite3.connect(ledger_path) as connection:
        connection.execute("CREATE TABLE legacy_receipts (task_id TEXT NOT NULL)")

    monkeypatch.setenv("HERMES_KANBAN_TASK", "ordinary-task")
    monkeypatch.setenv("HERMES_CONTROL_HOME", str(control))
    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", lambda *args, **kwargs: [])

    from hermes_cli.kanban_completion_policy import enforce_completion_policies

    enforce_completion_policies(
        task_id="ordinary-task", board="default", assignee="worker", summary="Complete"
    )
