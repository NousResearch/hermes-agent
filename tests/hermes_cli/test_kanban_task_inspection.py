"""HERMES-ORCH-001D: task admission inspection / diagnostics privacy.

Covers:
- inspect_task_admission fields: enforcement applicability, contract
  version, admission state/reasons, notification-required, subscription
  count, inherited sources, child-creation policy
- no credentials / chat_id / raw Telegram content in inspection payload
- CLI show --json surfaces admission_inspection
- tool kanban_show surfaces admission_inspection
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_cli import kanban as kanban_cli
from hermes_cli import kanban_db as kb


FULL_SHA = "e9b8ae6be137abead6d19ed8a67c523f8c527096"
SECRET_CHAT = "tg-chat-SECRET-99999"
SECRET_USER = "user-SECRET-alice"


def _valid_contract(**overrides):
    base = {
        "version": 1,
        "scope": "ORCH-001D admission inspection only",
        "allowed_files": [
            "hermes_cli/kanban_db.py",
            "tests/hermes_cli/test_kanban_task_inspection.py",
        ],
        "forbidden_files": ["hermes_cli/main.py"],
        "base_commit": FULL_SHA,
        "required_evidence": ["pytest output", "commit SHA"],
        "required_commands": [
            "scripts/run_tests.sh tests/hermes_cli/test_kanban_task_inspection.py -q"
        ],
        "allow_child_creation": False,
        "forbidden_git_actions": [
            "push",
            "merge",
            "amend",
            "reset",
            "clean",
            "restore",
            "stash",
        ],
        "notification_verified": True,
    }
    base.update(overrides)
    return base


@pytest.fixture
def isolated_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("HERMES_KANBAN_ADMISSION_ENFORCE", raising=False)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    return home


def _assert_privacy(payload: dict) -> None:
    blob = json.dumps(payload)
    assert SECRET_CHAT not in blob
    assert SECRET_USER not in blob
    assert "bot_token" not in blob.lower()
    assert "api_key" not in blob.lower()
    # Explicitly no route chat/user fields on the inspection surface.
    assert "chat_id" not in payload
    assert "user_id" not in payload
    assert "thread_id" not in payload


def test_evaluate_child_creation_matrix():
    assert kb.evaluate_child_creation_allowed(None) == (True, None)
    ok, code = kb.evaluate_child_creation_allowed(
        _valid_contract(allow_child_creation=False)
    )
    assert ok is False
    assert code == kb.CHILD_CREATION_DENIED
    ok, code = kb.evaluate_child_creation_allowed(
        _valid_contract(allow_child_creation=True)
    )
    assert ok is True
    assert code is None


def test_inspect_legacy_task_not_enforced(isolated_home):
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="legacy", assignee="w")
        insp = kb.inspect_task_admission(conn, tid)
    assert insp["enforcement_applicable"] is False
    assert insp["contract_present"] is False
    assert insp["contract_version"] is None
    assert insp["admission"]["admitted"] is True
    assert insp["admission"]["enforced"] is False
    assert insp["notification_required"] is False
    assert insp["subscription_count"] == 0
    assert insp["child_creation_allowed"] is True
    assert insp["inherited_sources"] == []
    _assert_privacy(insp)


def test_inspect_enforced_missing_subscription_reasons(isolated_home):
    with kb.connect() as conn:
        tid = kb.create_task(
            conn,
            title="needs sub",
            assignee="w",
            contract=_valid_contract(),
            # create_task will not mark ready without sub when contract present
        )
        insp = kb.inspect_task_admission(conn, tid)
    assert insp["enforcement_applicable"] is True
    assert insp["contract_present"] is True
    assert insp["contract_version"] == 1
    assert insp["notification_required"] is True
    assert insp["subscription_count"] == 0
    assert insp["notification_verified"] is True
    assert insp["allow_child_creation"] is False
    assert insp["child_creation_allowed"] is False
    assert insp["child_creation_denial_code"] == kb.CHILD_CREATION_DENIED
    assert insp["admission"]["admitted"] is False
    assert kb.ADMISSION_REASON_MISSING_NOTIFICATION_SUBSCRIPTION in insp["admission"][
        "codes"
    ]
    _assert_privacy(insp)


def test_inspect_inherited_sources_and_sub_count_no_secrets(isolated_home):
    with kb.connect() as conn:
        parent = kb.create_task(conn, title="parent", assignee="orch")
        kb.add_notify_sub(
            conn,
            task_id=parent,
            platform="telegram",
            chat_id=SECRET_CHAT,
            thread_id="topic-secret",
            user_id=SECRET_USER,
            notifier_profile="ops",
        )
        child = kb.create_task(
            conn,
            title="child",
            assignee="builder",
            parents=(parent,),
            contract=_valid_contract(allow_child_creation=True),
        )
        insp = kb.inspect_task_admission(conn, child)

    assert insp["subscription_count"] >= 1
    assert parent in insp["inherited_sources"]
    assert insp["contract_version"] == 1
    assert insp["admission"]["admitted"] is True
    assert insp["child_creation_allowed"] is True
    _assert_privacy(insp)
    # Full event payload may still exist in DB events, but inspection
    # itself must not re-expose chat/user secrets.
    assert SECRET_CHAT not in json.dumps(insp)
    assert SECRET_USER not in json.dumps(insp)


def test_cli_show_json_includes_admission_inspection(isolated_home, capsys):
    with kb.connect() as conn:
        tid = kb.create_task(
            conn,
            title="show-me",
            assignee="w",
            contract=_valid_contract(),
        )
        kb.add_notify_sub(
            conn,
            task_id=tid,
            platform="telegram",
            chat_id=SECRET_CHAT,
            user_id=SECRET_USER,
            notifier_profile="default",
        )

    args = SimpleNamespace(
        task_id=tid,
        json=True,
        state_type=None,
        state_name=None,
    )
    rc = kanban_cli._cmd_show(args)
    assert rc == 0
    out = capsys.readouterr().out
    payload = json.loads(out)
    insp = payload["admission_inspection"]
    assert insp["task_id"] == tid
    assert insp["contract_version"] == 1
    assert insp["enforcement_applicable"] is True
    assert insp["subscription_count"] == 1
    assert insp["admission"]["admitted"] is True
    assert SECRET_CHAT not in out
    assert SECRET_USER not in out


def test_tool_show_includes_admission_inspection(isolated_home, monkeypatch):
    from tools import kanban_tools as kt

    with kb.connect() as conn:
        tid = kb.create_task(
            conn,
            title="tool-show",
            assignee="w",
            contract=_valid_contract(allow_child_creation=False),
        )
        kb.add_notify_sub(
            conn,
            task_id=tid,
            platform="telegram",
            chat_id=SECRET_CHAT,
            user_id=SECRET_USER,
        )

    monkeypatch.setenv("HERMES_KANBAN_TASK", tid)
    monkeypatch.setenv("HERMES_PROFILE", "test-worker")
    out = json.loads(kt._handle_show({}))
    insp = out["admission_inspection"]
    assert insp["contract_version"] == 1
    assert insp["child_creation_allowed"] is False
    assert insp["subscription_count"] == 1
    assert insp["admission"]["admitted"] is True
    blob = json.dumps(insp)
    assert SECRET_CHAT not in blob
    assert SECRET_USER not in blob
