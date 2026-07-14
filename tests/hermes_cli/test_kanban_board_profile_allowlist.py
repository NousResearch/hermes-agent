from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_decompose as decomp


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    yield home
    kb._INITIALIZED_PATHS.clear()


def _profiles(names: list[str]):
    return [
        SimpleNamespace(
            name=name,
            is_default=name == "default",
            description=f"description for {name}",
            description_auto=False,
            model="model",
            provider="provider",
            skill_count=0,
        )
        for name in names
    ]


def _aux_response(payload: dict):
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = json.dumps(payload)
    client = MagicMock()
    client.chat.completions.create.return_value = response
    return client


def test_board_allowed_profiles_round_trip_and_normalize(kanban_home):
    kb.create_board("guarded")

    meta = kb.write_board_metadata(
        "guarded",
        allowed_profiles=[" Alex ", "kitt", "ALEX"],
    )

    assert meta["allowed_profiles"] == ["alex", "kitt"]
    assert kb.get_board_allowed_profiles("guarded") == ("alex", "kitt")
    assert kb.is_profile_allowed("guarded", "Alex") is True
    assert kb.is_profile_allowed("guarded", "goggins") is False
    assert kb.get_board_allowed_profiles("default") is None


def test_implicit_policy_lookup_uses_current_board(kanban_home, monkeypatch):
    kb.create_board("guarded", allowed_profiles=["alex"])
    monkeypatch.setenv("HERMES_KANBAN_BOARD", "guarded")

    assert kb.get_board_allowed_profiles() == ("alex",)
    assert kb.is_profile_allowed(None, "alex") is True
    assert kb.is_profile_allowed(None, "goggins") is False


def test_malformed_existing_board_metadata_fails_closed(kanban_home):
    kb.create_board("guarded", allowed_profiles=["alex"])
    kb.board_metadata_path("guarded").write_text("{broken-json", encoding="utf-8")

    assert kb.get_board_allowed_profiles("guarded") == ()
    assert kb.is_profile_allowed("guarded", "alex") is False
    assert kb.is_profile_allowed("guarded", "goggins") is False


def test_empty_allowlist_denies_every_profile_and_none_restores_unrestricted(kanban_home):
    kb.create_board("guarded", allowed_profiles=[])
    assert kb.get_board_allowed_profiles("guarded") == ()
    assert kb.is_profile_allowed("guarded", "alex") is False

    kb.write_board_metadata("guarded", allowed_profiles=None)
    assert kb.get_board_allowed_profiles("guarded") is None
    assert kb.is_profile_allowed("guarded", "goggins") is True


def test_create_and_reassign_reject_disallowed_profiles(kanban_home):
    kb.create_board("guarded", allowed_profiles=["alex", "kitt"])
    with kb.connect(board="guarded") as conn:
        allowed_id = kb.create_task(
            conn,
            title="allowed",
            assignee="alex",
            board="guarded",
        )
        with pytest.raises(ValueError, match="not allowed on board 'guarded'"):
            kb.create_task(
                conn,
                title="disallowed",
                assignee="goggins",
                board="guarded",
            )
        with pytest.raises(ValueError, match="not allowed on board 'guarded'"):
            kb.reassign_task(conn, allowed_id, "chef", board="guarded")
        assert kb.get_task(conn, allowed_id).assignee == "alex"


def test_dispatch_skips_preexisting_disallowed_assignment(kanban_home):
    kb.create_board("guarded")
    with kb.connect(board="guarded") as conn:
        allowed_id = kb.create_task(
            conn, title="allowed", assignee="default", board="guarded"
        )
        disallowed_id = kb.create_task(
            conn, title="stale", assignee="goggins", board="guarded"
        )

    kb.write_board_metadata("guarded", allowed_profiles=["default"])
    spawned: list[str] = []

    def spawn(task, workspace, board=None):
        spawned.append(task.id)
        return 4242

    with kb.connect(board="guarded") as conn:
        result = kb.dispatch_once(conn, spawn_fn=spawn, board="guarded")
        allowed = kb.get_task(conn, allowed_id)
        disallowed = kb.get_task(conn, disallowed_id)

    assert spawned == [allowed_id]
    assert result.skipped_disallowed == [disallowed_id]
    assert allowed.status == "running"
    assert disallowed.status == "ready"


def test_direct_claim_rejects_preexisting_disallowed_assignment(kanban_home):
    kb.create_board("guarded")
    with kb.connect(board="guarded") as conn:
        task_id = kb.create_task(
            conn, title="stale", assignee="goggins", board="guarded"
        )
    kb.write_board_metadata("guarded", allowed_profiles=["alex"])

    with kb.connect(board="guarded") as conn:
        claimed = kb.claim_task(conn, task_id, board="guarded")
        task = kb.get_task(conn, task_id)
        events = kb.list_events(conn, task_id)

    assert claimed is None
    assert task.status == "ready"
    assert any(
        event.kind == "claim_rejected"
        and event.payload
        and event.payload.get("reason") == "profile_not_allowed"
        for event in events
    )


def test_decomposer_filters_roster_and_uses_allowed_fallbacks(kanban_home, monkeypatch):
    kb.create_board("guarded", allowed_profiles=["alex", "kitt"])
    monkeypatch.setenv("HERMES_KANBAN_BOARD", "guarded")
    with kb.connect(board="guarded") as conn:
        task_id = kb.create_task(conn, title="rough", triage=True, board="guarded")

    payload = {
        "fanout": True,
        "rationale": "split",
        "tasks": [
            {"title": "ordinary", "body": "build", "assignee": "goggins", "parents": []},
            {"title": "security", "body": "audit", "assignee": "kitt", "parents": []},
        ],
    }
    client = _aux_response(payload)
    with patch("hermes_cli.profiles.list_profiles", return_value=_profiles(["alex", "kitt", "goggins", "chef"])), \
         patch("hermes_cli.profiles.profile_exists", return_value=True), \
         patch("hermes_cli.profiles.get_active_profile_name", return_value="goggins"), \
         patch("hermes_cli.kanban_decompose._load_config", return_value={"kanban": {}}), \
         patch("agent.auxiliary_client.get_text_auxiliary_client", return_value=(client, "test-model")), \
         patch("agent.auxiliary_client.get_auxiliary_extra_body", return_value={}):
        outcome = decomp.decompose_task(task_id, author="test")

    assert outcome.ok, outcome.reason
    with kb.connect(board="guarded") as conn:
        root = kb.get_task(conn, task_id)
        children = [kb.get_task(conn, child_id) for child_id in outcome.child_ids]

    assert root.assignee == "alex"
    assert [child.assignee for child in children] == ["alex", "kitt"]
    prompt = client.chat.completions.create.call_args.kwargs["messages"][1]["content"]
    assert "alex" in prompt and "kitt" in prompt
    assert "goggins" not in prompt and "chef" not in prompt


def test_decomposer_does_not_call_llm_when_allowlist_is_empty(kanban_home, monkeypatch):
    kb.create_board("frozen", allowed_profiles=[])
    monkeypatch.setenv("HERMES_KANBAN_BOARD", "frozen")
    with kb.connect(board="frozen") as conn:
        task_id = kb.create_task(conn, title="rough", triage=True, board="frozen")

    with patch("agent.auxiliary_client.get_text_auxiliary_client") as get_client:
        outcome = decomp.decompose_task(task_id, author="test")

    assert outcome.ok is False
    assert "allowlist is empty" in outcome.reason
    get_client.assert_not_called()
