"""Replay-oracle case 7: session_search profile routing through both dispatch paths.

The two generic agent-side dispatch sites are independent:
  * sequential executor — agent.tool_executor session_search branch
  * invoke/helper — agent.agent_runtime_helpers.invoke_tool session_search branch

Mocking session_search itself is not enough. These tests seed a current-profile
DB that lacks the source and a target-profile DB that holds it, then dispatch
through the registered surfaces down to the real tool.
"""
from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from hermes_state import SessionDB
from run_agent import AIAgent


TARGET_PROFILE = "coo-astra"
SOURCE_SESSION = "20260912_180441_b28574"
DECOY_SESSION = "s_current_decoy"
SOURCE_TOKEN = "uniquesourceneedle"
DECOY_TOKEN = "uniquedecoyneedle"


def _make_agent(session_db):
    with (
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            session_db=session_db,
            session_id="current-agent-session",
            platform="cli",
        )
    agent.client = MagicMock()
    return agent


def _patch_target_profile(monkeypatch, home, empty_home):
    from hermes_cli import profiles as profiles_mod

    monkeypatch.setattr(profiles_mod, "normalize_profile_name", lambda n: n)
    monkeypatch.setattr(profiles_mod, "validate_profile_name", lambda n: None)
    monkeypatch.setattr(profiles_mod, "profile_exists", lambda n: n == TARGET_PROFILE)
    monkeypatch.setattr(
        profiles_mod,
        "get_profile_dir",
        lambda n: home if n == TARGET_PROFILE else empty_home,
    )
    monkeypatch.setattr(profiles_mod, "list_profiles", lambda: [])


def _seed_current_db(db: SessionDB) -> None:
    db.create_session(DECOY_SESSION, source="cli")
    db.append_message(DECOY_SESSION, role="user", content=f"decoy {DECOY_TOKEN}")
    db.append_message(DECOY_SESSION, role="assistant", content="wrong-profile answer")
    db._conn.commit()


def _seed_target_db(db: SessionDB) -> int:
    db.create_session(SOURCE_SESSION, source="cli")
    db.append_message(SOURCE_SESSION, role="user", content="lead-in")
    anchor = db.append_message(
        SOURCE_SESSION, role="user", content=f"mission {SOURCE_TOKEN}"
    )
    db.append_message(SOURCE_SESSION, role="assistant", content="ack source")
    db.append_message(SOURCE_SESSION, role="user", content="follow-up")
    db._conn.commit()
    return anchor


def _tool_call(args: dict) -> SimpleNamespace:
    return SimpleNamespace(
        id="search-1",
        function=SimpleNamespace(
            name="session_search",
            arguments=json.dumps(args),
        ),
    )


def _sequential_result(agent, args: dict) -> dict:
    messages = []
    agent._execute_tool_calls_sequential(
        SimpleNamespace(tool_calls=[_tool_call(args)]),
        messages,
        "task-id",
    )
    assert messages, "sequential dispatch produced no tool result"
    return json.loads(messages[-1]["content"])


def _invoke_result(agent, args: dict) -> dict:
    return json.loads(agent._invoke_tool("session_search", args, "task-id"))


def test_sequential_scroll_uses_named_profile_not_current_db(tmp_path, monkeypatch):
    current = SessionDB(tmp_path / "current.db")
    target_home = tmp_path / "coo-astra"
    target_home.mkdir()
    target = SessionDB(target_home / "state.db")
    _seed_current_db(current)
    anchor = _seed_target_db(target)
    _patch_target_profile(monkeypatch, target_home, tmp_path / "empty-home")

    result = _sequential_result(
        _make_agent(current),
        {
            "profile": TARGET_PROFILE,
            "session_id": SOURCE_SESSION,
            "around_message_id": anchor,
            "window": 2,
        },
    )

    assert result["success"] is True
    assert result["mode"] == "scroll"
    assert result["session_id"] == SOURCE_SESSION
    assert result["window"] == 2
    assert result["around_message_id"] == anchor
    ids = [m["id"] for m in result["messages"]]
    assert anchor in ids
    assert SOURCE_TOKEN in result["messages"][ids.index(anchor)]["content"]
    assert DECOY_SESSION not in json.dumps(result)


def test_invoke_scroll_uses_named_profile_not_current_db(tmp_path, monkeypatch):
    current = SessionDB(tmp_path / "current.db")
    target_home = tmp_path / "coo-astra"
    target_home.mkdir()
    target = SessionDB(target_home / "state.db")
    _seed_current_db(current)
    anchor = _seed_target_db(target)
    _patch_target_profile(monkeypatch, target_home, tmp_path / "empty-home")

    result = _invoke_result(
        _make_agent(current),
        {
            "profile": TARGET_PROFILE,
            "session_id": SOURCE_SESSION,
            "around_message_id": anchor,
            "window": 2,
        },
    )

    assert result["success"] is True
    assert result["mode"] == "scroll"
    assert result["session_id"] == SOURCE_SESSION
    assert result["window"] == 2
    assert result["around_message_id"] == anchor
    ids = [m["id"] for m in result["messages"]]
    assert anchor in ids
    assert SOURCE_TOKEN in result["messages"][ids.index(anchor)]["content"]
    assert DECOY_SESSION not in json.dumps(result)


def test_sequential_read_and_discovery_use_named_profile(tmp_path, monkeypatch):
    current = SessionDB(tmp_path / "current.db")
    target_home = tmp_path / "coo-astra"
    target_home.mkdir()
    target = SessionDB(target_home / "state.db")
    _seed_current_db(current)
    _seed_target_db(target)
    _patch_target_profile(monkeypatch, target_home, tmp_path / "empty-home")
    agent = _make_agent(current)

    read_result = _sequential_result(
        agent,
        {"profile": TARGET_PROFILE, "session_id": SOURCE_SESSION},
    )
    assert read_result["success"] is True
    assert read_result["mode"] == "read"
    assert read_result["session_id"] == SOURCE_SESSION
    assert any(SOURCE_TOKEN in (m.get("content") or "") for m in read_result["messages"])
    assert DECOY_SESSION not in json.dumps(read_result)

    discovery = _sequential_result(
        agent,
        {"profile": TARGET_PROFILE, "query": SOURCE_TOKEN, "limit": 3, "detail": "full"},
    )
    assert discovery["success"] is True
    assert discovery["mode"] == "discover"
    assert [r["session_id"] for r in discovery["results"]] == [SOURCE_SESSION]
    assert DECOY_SESSION not in json.dumps(discovery)


def test_invoke_read_and_discovery_use_named_profile(tmp_path, monkeypatch):
    current = SessionDB(tmp_path / "current.db")
    target_home = tmp_path / "coo-astra"
    target_home.mkdir()
    target = SessionDB(target_home / "state.db")
    _seed_current_db(current)
    _seed_target_db(target)
    _patch_target_profile(monkeypatch, target_home, tmp_path / "empty-home")
    agent = _make_agent(current)

    read_result = _invoke_result(
        agent,
        {"profile": TARGET_PROFILE, "session_id": SOURCE_SESSION},
    )
    assert read_result["success"] is True
    assert read_result["mode"] == "read"
    assert read_result["session_id"] == SOURCE_SESSION
    assert any(SOURCE_TOKEN in (m.get("content") or "") for m in read_result["messages"])

    discovery = _invoke_result(
        agent,
        {
            "profile": TARGET_PROFILE,
            "query": SOURCE_TOKEN,
            "limit": 1,
            "sort": "newest",
            "detail": "full",
            "role_filter": "user",
        },
    )
    assert discovery["success"] is True
    assert discovery["mode"] == "discover"
    assert [r["session_id"] for r in discovery["results"]] == [SOURCE_SESSION]
    assert DECOY_SESSION not in json.dumps(discovery)


def test_sequential_default_current_profile_still_works(tmp_path):
    current = SessionDB(tmp_path / "current.db")
    _seed_current_db(current)
    result = _sequential_result(
        _make_agent(current),
        {"query": DECOY_TOKEN, "limit": 3},
    )
    assert result["success"] is True
    assert [r["session_id"] for r in result["results"]] == [DECOY_SESSION]


def test_invoke_default_current_profile_still_works(tmp_path):
    current = SessionDB(tmp_path / "current.db")
    _seed_current_db(current)
    result = _invoke_result(
        _make_agent(current),
        {"query": DECOY_TOKEN, "limit": 3},
    )
    assert result["success"] is True
    assert [r["session_id"] for r in result["results"]] == [DECOY_SESSION]


def test_sequential_omitting_profile_does_not_silently_read_target(tmp_path, monkeypatch):
    current = SessionDB(tmp_path / "current.db")
    target_home = tmp_path / "coo-astra"
    target_home.mkdir()
    target = SessionDB(target_home / "state.db")
    _seed_current_db(current)
    anchor = _seed_target_db(target)
    _patch_target_profile(monkeypatch, target_home, tmp_path / "empty-home")

    result = _sequential_result(
        _make_agent(current),
        {
            "session_id": SOURCE_SESSION,
            "around_message_id": anchor,
            "window": 2,
        },
    )
    assert result["success"] is False
    assert "not found" in result.get("error", "").lower()


def test_invoke_omitting_profile_does_not_silently_read_target(tmp_path, monkeypatch):
    current = SessionDB(tmp_path / "current.db")
    target_home = tmp_path / "coo-astra"
    target_home.mkdir()
    target = SessionDB(target_home / "state.db")
    _seed_current_db(current)
    anchor = _seed_target_db(target)
    _patch_target_profile(monkeypatch, target_home, tmp_path / "empty-home")

    result = _invoke_result(
        _make_agent(current),
        {
            "session_id": SOURCE_SESSION,
            "around_message_id": anchor,
            "window": 2,
        },
    )
    assert result["success"] is False
    assert "not found" in result.get("error", "").lower()
