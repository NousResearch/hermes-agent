"""413 evictions belong to the logical session, including reconstructed workers (#148146)."""
from copy import deepcopy
import json
import sqlite3

import pytest

from agent.turn_context import build_api_messages
from hermes_state import SessionDB
from tests.agent.test_413_compression import _new_test_agent

STATE_KEY = "_vision_payload_evictions_v1"
IMAGE = "data:image/png;base64," + "a" * 2000


def _history():
    return [
        {"role": "user", "content": "Inspect"},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "old", "type": "function",
         "function": {"name": "browser_vision", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "old", "content": [
            {"type": "text", "text": "Retained page text"},
            {"type": "image_url", "image_url": {"url": IMAGE}},
        ]},
        {"role": "user", "content": "continue"},
    ]


def _agent(db, sid):
    agent = _new_test_agent()
    agent.provider, agent.model = "custom", "fixture-vision"
    agent.base_url = "https://example.invalid/v1?key=synthetic-secret"
    agent._session_db, agent.session_id = db, sid
    agent._current_turn_timestamp = "2026-11-05T00:00:00"
    return agent


def _build(agent, history):
    return build_api_messages(agent, history, current_turn_user_idx=len(history) - 1,
                              ext_prefetch_cache=None, plugin_user_context=None, moa_config=None,
                              active_system_prompt="Stable prompt")[0]


@pytest.mark.parametrize("case", ["resume", "new-session", "model", "provider", "base_url", "api_mode",
                                  "continuation", "branch", "branch-continuation", "reset", "reset-continuation"])
def test_durable_evictions_follow_the_session_route_and_continuation_boundary(tmp_path, case):
    db = SessionDB(tmp_path / "state.db")
    config = ({"_branched_from": "origin"} if case == "branch-continuation" else
              {"_reset_from": "origin"} if case == "reset-continuation" else None)
    db.create_session("origin", "cli")
    db.create_session("session", "cli", parent_session_id="origin" if config else None, model_config=config)
    history = _history()
    original = deepcopy(history)
    first = _agent(db, "session")
    try:
        assert first._try_strip_image_parts_from_tool_messages(deepcopy(history), remember_model=False, remember_bytes=True)
        sid = "session"
        if case in {"continuation", "branch", "branch-continuation", "reset", "reset-continuation"}:
            child_config = ({"_branched_from": "session"} if case == "branch" else
                            {"_reset_from": "session"} if case == "reset" else (config or {}))
            db.publish_compression_child(parent_session_id="session", child_session_id="child", source="cli",
                                         messages=deepcopy(history), model_config=child_config,
                                         require_compression_lease=False)
            sid = "child"
        if case == "new-session":
            sid = "unrelated"
            db.create_session(sid, "cli")
        restored = first if case == "new-session" else _agent(db, sid)
        restored.session_id = sid
        if case in {"model", "provider", "base_url", "api_mode"}:
            setattr(restored, case, str(getattr(restored, case)) + "/other")
        wire = _build(restored, history)
        should_strip = case in {"resume", "continuation", "branch-continuation", "reset-continuation"}
        assert (IMAGE not in str(wire)) is should_strip
        assert "Retained page text" in str(wire) and history == original
        persisted = json.dumps(db.get_session_model_config_value("session", STATE_KEY))
        assert "synthetic-secret" not in persisted and IMAGE not in persisted
    finally:
        db.close()


def test_worker_receipts_can_restore_the_replay_sidecar_on_a_new_execution(tmp_path):
    from agent.runtime_session_store import RuntimeSessionStore
    from hermes_state_runtime import begin_runtime_epoch, mutate_worker_execution, register_worker_execution

    db = SessionDB(tmp_path / "state.db")
    db.create_session("session", "cli")
    epoch = begin_runtime_epoch(db, instance_id="durable-byte-state")
    stores = []
    try:
        for index in range(2):
            scope = dict(epoch=epoch, execution_id=f"execution-{index}", session_id="session", generation=0)
            register_worker_execution(db, **scope, kind="compute", adoption_secret="fixture-only")
            store = RuntimeSessionStore(lambda method, **params: mutate_worker_execution(db, **params), scope,
                                        tmp_path / f"outbox-{index}")
            stores.append(store)
            agent = _agent(store, "session")
            if index == 0:
                assert agent._try_strip_image_parts_from_tool_messages(deepcopy(_history()), remember_model=False, remember_bytes=True)
                store.finish()
                store.close()
            else:
                assert IMAGE not in str(_build(agent, _history()))
    finally:
        for store in stores:
            store.close()
        db.close()


@pytest.mark.parametrize("failure", ["restore", "persist"])
def test_replay_state_io_failure_is_visible_without_rewriting_history(tmp_path, monkeypatch, failure):
    db = SessionDB(tmp_path / "state.db")
    db.create_session("session", "cli")
    agent = _agent(db, "session")
    history = _history()
    before = deepcopy(history)

    def unavailable(*args, **kwargs):
        raise sqlite3.OperationalError("injected replay-state I/O failure")

    try:
        attribute = {"restore": "get_session_model_config_value", "persist": "patch_session_model_config"}[failure]
        monkeypatch.setattr(db, attribute, unavailable)
        with pytest.raises(sqlite3.OperationalError, match="replay-state"):
            agent._try_strip_image_parts_from_tool_messages(deepcopy(history), remember_model=False, remember_bytes=True)
        assert history == before
    finally:
        db.close()


@pytest.mark.parametrize("corrupt", [[], {"route": []}, {"a" * 64: "bad"}, {"a" * 64: [["old", "not-a-digest"]]}])
def test_corrupt_replay_state_cannot_silently_reenable_evicted_images(tmp_path, corrupt):
    db = SessionDB(tmp_path / "state.db")
    db.create_session("session", "cli")
    try:
        db.patch_session_model_config("session", {STATE_KEY: corrupt})
        with pytest.raises(ValueError, match="vision eviction sidecar"):
            _build(_agent(db, "session"), _history())
    finally:
        db.close()


def test_replay_records_drop_tools_that_no_longer_exist_in_canonical_context(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    db.create_session("session", "cli")
    first = _agent(db, "session")
    try:
        first._try_strip_image_parts_from_tool_messages(deepcopy(_history()), remember_model=False, remember_bytes=True)
        retained = _history()
        retained[1]["tool_calls"][0]["id"] = retained[2]["tool_call_id"] = "fresh"
        first._try_strip_image_parts_from_tool_messages(deepcopy(retained), remember_model=False,
                                                        remember_bytes=True, retained_messages=retained)
        persisted = db.get_session_model_config_value("session", STATE_KEY)
        assert {item[0] for entries in persisted.values() for item in entries} == {"fresh"}
        wire = _build(_agent(db, "session"), _history())
        assert IMAGE in str(wire)
        assert IMAGE not in str(_build(_agent(db, "session"), retained))
    finally:
        db.close()


def test_transient_agent_never_writes_replay_sidecars(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    db.create_session("session", "cli")
    agent = _agent(db, "session")
    agent._persist_disabled = True
    try:
        agent._try_strip_image_parts_from_tool_messages(deepcopy(_history()), remember_model=False, remember_bytes=True)
        assert db.get_session_model_config_value("session", STATE_KEY) is None
        assert IMAGE not in str(_build(agent, _history()))
        assert IMAGE in str(_build(_agent(db, "session"), _history()))
    finally:
        db.close()
