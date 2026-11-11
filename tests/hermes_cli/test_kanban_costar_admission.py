"""Offline contract tests for CoStar Kanban admission.

These tests intentionally never open a CDP connection.  They exercise the
production tool creation path, DB dependency promotion, and dispatcher path.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd
from hermes_cli.kanban_db_graph import decompose_triage_task


@pytest.fixture
def kanban_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_PROFILE", "operator")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    return home


def _costar_body(extra: str = "") -> str:
    return "CoStar Florida research. " + extra


def test_tool_create_persists_costar_contract_and_omitted_model_uses_terra(kanban_home: Path) -> None:
    from tools import kanban_tools as kt

    result = json.loads(kt._handle_create({
        "title": "Florida CoStar execution",
        "body": _costar_body(),
        "assignee": "operator",
    }))

    assert result["ok"] is True, result
    with kbc.connect() as conn:
        task = kb.get_task(conn, result["task_id"])
        assert (task.model_override, task.provider_override) == ("gpt-5.6-terra", "openai-codex")
        created = next(event for event in kb.list_events(conn, task.id) if event.kind == "created")
        assert created.payload["costar_contract"] == {
            "browser": "HermesCDP-CoStar",
            "cdp_url": "http://127.0.0.1:9224",
            "no_fallback": True,
        }


def test_costar_user_model_exception_and_warden_luna_exception_are_recorded(kanban_home: Path) -> None:
    with kbc.connect() as conn:
        explicit = kb.create_task(
            conn, title="CoStar Florida analysis", body=_costar_body(), assignee="operator",
            model_override="gpt-5.6-sol", provider_override="openai-codex",
        )
        warden = kb.create_task(
            conn, title="Warden saved-search refresh", body=_costar_body(), assignee="operator",
            model_override="gpt-5.6-luna", provider_override="openai-codex",
        )
        assert kb.get_task(conn, explicit).model_override == "gpt-5.6-sol"
        assert kb.get_task(conn, warden).model_override == "gpt-5.6-luna"
        explicit_event = next(event for event in kb.list_events(conn, explicit) if event.kind == "costar_model_exception")
        warden_event = next(event for event in kb.list_events(conn, warden) if event.kind == "costar_model_exception")
        assert explicit_event.payload["kind"] == "user_explicit"
        assert warden_event.payload["kind"] == "warden_saved_search"


def test_do_not_start_card_is_sticky_across_parent_completion_and_recompute(kanban_home: Path) -> None:
    with kbc.connect() as conn:
        parent = kb.create_task(conn, title="parent", assignee="operator")
        child = kb.create_task(
            conn, title="CoStar follow-up", body=_costar_body("Do not start until supervisor approval."),
            assignee="operator", parents=[parent],
        )
        assert kb.get_task(conn, child).status == "blocked"
        kb.claim_task(conn, parent)
        assert kb.complete_task(conn, parent, result="done")
        assert kb.recompute_ready(conn) == 0
        assert kb.get_task(conn, child).status == "blocked"
        held = [event for event in kb.list_events(conn, child) if event.kind == "supervisor_hold"]
        assert held and held[-1].payload["reason"] == "do_not_start"
        ok, error = kb.promote_task(conn, child, actor="tester")
        assert not ok and error == "task has a supervisor hold; only explicit kanban_unblock may release it"


def test_decompose_costar_children_get_contract_and_terra_without_parent_luna(kanban_home: Path) -> None:
    with kbc.connect() as conn:
        root = kb.create_task(
            conn, title="CoStar Florida recovery", body=_costar_body(), assignee="operator",
            model_override="gpt-5.6-luna", provider_override="openai-codex", triage=True,
        )
        children = decompose_triage_task(
            conn, root, root_assignee="operator",
            children=[{"title": "prepare result", "body": "helper child", "assignee": "operator"}],
        )
        assert children is not None
        child = kb.get_task(conn, children[0])
        assert child is not None
        assert (child.model_override, child.provider_override) == ("gpt-5.6-terra", "openai-codex")
        created = next(event for event in kb.list_events(conn, child.id) if event.kind == "created")
        assert created.payload["costar_contract"]["cdp_url"] == "http://127.0.0.1:9224"


def test_tool_create_inherits_costar_contract_from_parent_without_parent_luna(kanban_home: Path) -> None:
    from tools import kanban_tools as kt

    with kbc.connect() as conn:
        parent = kb.create_task(
            conn, title="CoStar Florida recovery", body=_costar_body(), assignee="operator",
            model_override="gpt-5.6-luna", provider_override="openai-codex",
        )

    result = json.loads(kt._handle_create({
        "title": "prepare result",
        "body": "generic helper child",
        "assignee": "operator",
        "parents": [parent],
    }))

    assert result["ok"] is True, result
    with kbc.connect() as conn:
        child = kb.get_task(conn, result["task_id"])
        assert child is not None
        assert (child.model_override, child.provider_override) == ("gpt-5.6-terra", "openai-codex")
        created = next(event for event in kb.list_events(conn, child.id) if event.kind == "created")
        assert created.payload["costar_contract"] == {
            "browser": "HermesCDP-CoStar",
            "cdp_url": "http://127.0.0.1:9224",
            "no_fallback": True,
        }


def test_dispatch_refuses_provider_hold_and_second_costar_owner(kanban_home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(kbd, "_profile_exists_fn", lambda: lambda _name: True)
    spawned: list[str] = []

    def spawn(task, workspace, board=None):
        spawned.append(task.id)

    with kbc.connect() as conn:
        blocked = kb.create_task(conn, title="CoStar denied", body=_costar_body(), assignee="operator")
        kb.claim_task(conn, blocked)
        assert kb.block_task(conn, blocked, reason="Access Denied", expected_run_id=kb.get_task(conn, blocked).current_run_id)
        waiting = kb.create_task(
            conn, title="Warden saved-search hunt", body=_costar_body(), assignee="operator",
            model_override="gpt-5.6-luna", provider_override="openai-codex",
        )
        result = kbd.dispatch_once(conn, spawn_fn=spawn)
        assert spawned == []
        assert waiting not in [task_id for task_id, _, _ in result.spawned]
        denied = [event for event in kb.list_events(conn, waiting) if event.kind == "costar_dispatch_rejected"]
        assert denied[-1].payload["reason"] == "provider_hold"

        assert kb.unblock_task(conn, blocked)
        first = kbd.dispatch_once(conn, spawn_fn=spawn)
        assert first.spawned and first.spawned[0][0] == blocked
        result = kbd.dispatch_once(conn, spawn_fn=spawn)
        assert waiting not in [task_id for task_id, _, _ in result.spawned]
        owner_rejection = [event for event in kb.list_events(conn, waiting) if event.kind == "costar_dispatch_rejected"]
        assert owner_rejection[-1].payload["reason"] == "single_owner"


def test_generic_costar_helper_is_refused_while_provider_hold_is_active(
    kanban_home: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tools import kanban_tools as kt

    monkeypatch.setattr(kbd, "_profile_exists_fn", lambda: lambda _name: True)
    with kbc.connect() as conn:
        parent = kb.create_task(conn, title="CoStar Florida root", body=_costar_body(), assignee="operator")
        kb.claim_task(conn, parent)
        assert kb.complete_task(conn, parent, result="done")
        blocked = kb.create_task(conn, title="CoStar denied", body=_costar_body(), assignee="operator")
        kb.claim_task(conn, blocked)
        blocked_task = kb.get_task(conn, blocked)
        assert blocked_task is not None
        assert kb.block_task(conn, blocked, reason="Access Denied", expected_run_id=blocked_task.current_run_id)

    created = json.loads(kt._handle_create({
        "title": "prepare result", "body": "generic helper child", "assignee": "operator", "parents": [parent],
    }))
    assert created["ok"] is True, created
    spawned: list[str] = []
    with kbc.connect() as conn:
        result = kbd.dispatch_once(conn, spawn_fn=lambda task, workspace, board=None: spawned.append(task.id))
        assert spawned == []
        assert created["task_id"] not in [task_id for task_id, _, _ in result.spawned]
        rejected = [event for event in kb.list_events(conn, created["task_id"]) if event.kind == "costar_dispatch_rejected"]
        assert rejected
        assert rejected[-1].payload["reason"] == "provider_hold"


def test_generic_costar_helper_is_refused_while_another_costar_owner_runs(
    kanban_home: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tools import kanban_tools as kt

    monkeypatch.setattr(kbd, "_profile_exists_fn", lambda: lambda _name: True)
    with kbc.connect() as conn:
        parent = kb.create_task(conn, title="CoStar Florida root", body=_costar_body(), assignee="operator")
        kb.claim_task(conn, parent)
        assert kb.complete_task(conn, parent, result="done")
        owner = kb.create_task(conn, title="CoStar owner", body=_costar_body(), assignee="operator")

    created = json.loads(kt._handle_create({
        "title": "prepare result", "body": "generic helper child", "assignee": "operator", "parents": [parent],
    }))
    assert created["ok"] is True, created
    spawned: list[str] = []
    with kbc.connect() as conn:
        first = kbd.dispatch_once(conn, spawn_fn=lambda task, workspace, board=None: spawned.append(task.id))
        assert first.spawned and first.spawned[0][0] == owner
        result = kbd.dispatch_once(conn, spawn_fn=lambda task, workspace, board=None: spawned.append(task.id))
        assert created["task_id"] not in [task_id for task_id, _, _ in result.spawned]
        rejected = [event for event in kb.list_events(conn, created["task_id"]) if event.kind == "costar_dispatch_rejected"]
        assert rejected
        assert rejected[-1].payload["reason"] == "single_owner"


def test_generic_costar_helper_denial_records_provider_hold(kanban_home: Path) -> None:
    from tools import kanban_tools as kt

    with kbc.connect() as conn:
        parent = kb.create_task(conn, title="CoStar Florida root", body=_costar_body(), assignee="operator")
        kb.claim_task(conn, parent)
        assert kb.complete_task(conn, parent, result="done")

    created = json.loads(kt._handle_create({
        "title": "prepare result", "body": "generic helper child", "assignee": "operator", "parents": [parent],
    }))
    assert created["ok"] is True, created
    with kbc.connect() as conn:
        task = kb.get_task(conn, created["task_id"])
        assert task is not None
        kb.claim_task(conn, task.id)
        claimed = kb.get_task(conn, task.id)
        assert claimed is not None
        assert kb.block_task(conn, task.id, reason="unusual_activity", expected_run_id=claimed.current_run_id)
        holds = [event for event in kb.list_events(conn, task.id) if event.kind == "costar_provider_hold"]
        assert holds
        assert holds[-1].payload == {"task_id": task.id, "reason": "access_denied", "immutable": True}
