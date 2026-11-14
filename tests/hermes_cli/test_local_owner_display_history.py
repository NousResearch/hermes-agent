"""Regression for #119819: REST history must follow a bound local owner's reset target.

The runtime owns a stable logical id while a canonical reset moves its transcript
into a new physical row. These exercise the real HTTP readers against committed
SQLite state, not a mocked resolver or a Desktop length-based workaround.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import pytest_asyncio
from fastapi import FastAPI
from fastapi.testclient import TestClient

from hermes_state import SessionDB
from hermes_state_local import POLICY_PREFIX, local_receipt
from hermes_state_local_lineage import local_physical_target
from hermes_state_runtime import RuntimeStoreError


async def _mutate(fixture, operation, payload):
    from gateway.session_mutations import mutate_session

    row = fixture.db.get_session(fixture.ref.session_id)
    return await mutate_session(fixture.authority, fixture.owner, fixture.ref, {
        "session_id": fixture.ref.session_id,
        "request_id": operation,
        "expected_revision": row["runtime_revision"],
        "expected_generation": row["runtime_generation"],
        "operation": operation,
        "payload": payload,
    })


@pytest_asyncio.fixture
async def reset_history(tmp_path, monkeypatch):
    # A linked worktree's Git metadata can live under the operator's protected
    # Hermes home. This fixture never runs an updater; keep its marker probe in
    # tmp_path too, without bypassing the real-home I/O guard.
    monkeypatch.setattr("hermes_cli._early_recovery.interrupted_pull_marker", lambda root: tmp_path / "update-pull")
    from gateway import run
    from gateway.config import GatewayConfig
    from gateway.session import SessionStore
    from gateway.session_authority import initialize_session_authority
    from gateway.session_contract import SessionActor as Principal
    from gateway.session_local import create_local_session
    from hermes_cli.web_routers.sessions import manage_router

    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr("hermes_state.DEFAULT_DB_PATH", home / "state.db")
    monkeypatch.setattr(run, "_load_gateway_config", lambda: {"platform_toolsets": {"cli": []}})
    store = SessionStore(home / "sessions", GatewayConfig())
    assert isinstance(store._db, SessionDB)
    db = store._db
    runner = SimpleNamespace(
        session_store=store, _session_db=db, adapters={}, _draining=False,
        _evict_cached_agent=lambda route: None,
    )
    runner._adapter_for_source = lambda source: runner.adapters.get(source.platform)
    try:
        authority = await initialize_session_authority(runner, profile_id="default", instance_id="history-test")
        # No turn executor, LLM, service, or listener is needed for these reads.
        authority._schedule = lambda ref: None
        owner = Principal("history-owner", "default", frozenset({
            "session:create", "session:read", "session:submit", "session:control",
        }), "native")
        ref = create_local_session(authority, owner, {
            "request_id": "history", "source": "gui", "cwd": str(tmp_path),
            "model": "fixture", "toolsets": [],
        })
        fixture = SimpleNamespace(db=db, authority=authority, owner=owner, ref=ref)
        db.append_messages_batch(ref.session_id, [
            {"role": "user", "content": "ENDED_PARENT_ASK"},
            {"role": "assistant", "content": "ENDED_PARENT_ANSWER"},
        ])
        fixture.parent_rows = db.get_messages(ref.session_id)
        reset = await _mutate(fixture, "reset", {})
        fixture.child = reset["target_session_id"]
        db.append_messages_batch(fixture.child, [
            {"role": "user" if index % 2 == 0 else "assistant", "content": f"CURRENT_CHILD_{index}"}
            for index in range(8)
        ])
        fixture.child_rows = db.get_messages(fixture.child)
        fixture.logical = ref.session_id
        parent = db.get_session(fixture.logical)
        assert parent is not None and parent["end_reason"] == "session_reset"
        assert db.resolve_resume_session_id(fixture.logical) == fixture.logical
        with db._read_ctx() as conn:
            assert local_physical_target(conn, fixture.logical) == fixture.child
        # Prove the canonical runtime already serves the child under the logical id.
        snapshot = await authority.attach(owner, ref)
        assert [row["row_id"] for row in snapshot.history] == _ids(fixture.child_rows)
        assert _ids(fixture.parent_rows) == [1, 2]
        assert _ids(fixture.child_rows) == list(range(3, 11))
        app = FastAPI()
        app.include_router(manage_router)
        # Observe HTTP errors rather than letting a store exception escape TestClient.
        with TestClient(app, raise_server_exceptions=False) as client:
            fixture.client = client
            yield fixture
    finally:
        db.close()


def _ids(rows):
    return [row["id"] for row in rows]


@pytest.mark.parametrize(("params", "selection"), [
    pytest.param({}, slice(None), id="default-latest"),
    pytest.param({"limit": 500, "include_compacted": "true"}, slice(None), id="full"),
    pytest.param({"limit": 3, "order": "latest"}, slice(-3, None), id="latest-window"),
    pytest.param({"limit": 3, "offset": 3}, slice(3, 6), id="offset-window"),
    pytest.param({"limit": 3, "order": "oldest"}, slice(0, 3), id="oldest-window"),
])
def test_logical_history_reads_current_reset_target(reset_history, params, selection):
    t = reset_history
    response = t.client.get(f"/api/sessions/{t.logical}/messages", params=params)
    assert response.status_code == 200, response.text
    page = response.json()
    assert page["session_id"] == t.logical
    assert page["profile"] == "default"
    expected = t.child_rows[selection]
    assert _ids(page["messages"]) == _ids(expected), "REST read the ended parent instead of the runtime's target"
    assert [row["content"] for row in page["messages"]] == [row["content"] for row in expected]
    assert page["pagination"]["returned"] == len(expected)


def test_logical_timeline_pages_current_reset_prompts(reset_history):
    t = reset_history
    expected = [row["id"] for row in t.child_rows if row["role"] == "user"]
    first_response = t.client.get(f"/api/sessions/{t.logical}/timeline", params={"limit": 2})
    assert first_response.status_code == 200, first_response.text
    first = first_response.json()
    assert first["session_id"] == t.logical
    assert first["profile"] == "default"
    assert [entry["row_id"] for entry in first["entries"]] == expected[:2]
    assert first["pagination"]["total"] == len(expected)
    assert first["pagination"]["has_more"] is True
    second_response = t.client.get(f"/api/sessions/{t.logical}/timeline", params={
        "limit": 2, "after_row_id": first["pagination"]["next_cursor"],
    })
    assert second_response.status_code == 200, second_response.text
    second = second_response.json()
    assert second["session_id"] == t.logical
    assert [entry["row_id"] for entry in second["entries"]] == expected[2:]
    assert second["pagination"]["has_more"] is False


def test_logical_around_page_accepts_current_child_prompt(reset_history):
    t = reset_history
    response = t.client.get(f"/api/sessions/{t.logical}/messages/around", params={
        "row_id": t.child_rows[2]["id"], "limit": 2,
    })
    assert response.status_code == 200, response.text
    page = response.json()
    assert page["session_id"] == t.logical
    assert page["profile"] == "default"
    assert _ids(page["messages"]) == _ids(t.child_rows[2:4])
    assert page["pagination"]["has_older"] is True
    assert page["pagination"]["has_newer"] is True


@pytest.mark.asyncio
async def test_genuine_rewind_shortens_the_current_transcript(reset_history):
    t = reset_history
    await _mutate(t, "rewind", {"target_message_id": t.child_rows[2]["id"]})
    # Rewind removes the addressed prompt and everything after it, even when
    # include_compacted is requested. Fewer rows is correct here, not a rollback.
    assert _ids(t.db.get_messages(t.child)) == _ids(t.child_rows[:2])
    response = t.client.get(f"/api/sessions/{t.logical}/messages", params={"include_compacted": "true"})
    assert response.status_code == 200, response.text
    assert response.json()["session_id"] == t.logical
    assert _ids(response.json()["messages"]) == _ids(t.child_rows[:2])


@pytest.mark.parametrize("damage", ["broken-lineage", "foreign-owner", "copied-receipt"])
@pytest.mark.parametrize("reader", ["messages", "timeline", "messages/around"])
def test_invalid_receipt_fails_closed_without_parent_fallback(reset_history, damage, reader):
    t = reset_history
    receipt = local_receipt(t.db, t.logical)
    if damage == "broken-lineage":
        receipt["lineage"] = [t.logical]
    elif damage == "foreign-owner":
        receipt["principal_id"] = "another-owner"
    else:
        receipt["session_id"] = "another-logical-session"
    t.db.set_meta(POLICY_PREFIX + t.logical, json.dumps(receipt))
    with t.db._read_ctx() as conn, pytest.raises(RuntimeStoreError, match="storage_unavailable"):
        local_physical_target(conn, t.logical)
    # The old parent's prompt is deliberately valid there: a fallback must not
    # turn an invalid receipt into a successful response containing old history.
    params = {"row_id": t.parent_rows[0]["id"]} if reader == "messages/around" else {}
    response = t.client.get(f"/api/sessions/{t.logical}/{reader}", params=params)
    assert response.status_code in {404, 409, 503}, response.text
    assert "ENDED_PARENT" not in response.text
    assert "CURRENT_CHILD" not in response.text


@pytest.mark.parametrize("reader", ["timeline", "messages/around"])
def test_foreign_profile_reset_target_is_rejected(reset_history, reader):
    t = reset_history
    t.db._write_sql("UPDATE sessions SET profile_name = ? WHERE id = ?", ("other-profile", t.child))
    # Lineage is structurally valid; the HTTP reader must ALSO check the target's
    # owning profile. Check both anchors so neither parent fallback nor a foreign
    # child page can satisfy the around reader by accident.
    anchors = [t.parent_rows[0]["id"], t.child_rows[0]["id"]] if reader == "messages/around" else [None]
    for anchor in anchors:
        params = {"row_id": anchor} if anchor is not None else {}
        response = t.client.get(f"/api/sessions/{t.logical}/{reader}", params=params)
        assert response.status_code == 404, response.text


def test_unregistered_legacy_reset_child_is_not_followed(reset_history):
    t = reset_history
    t.db.create_session("legacy-parent", source="gui")
    parent_id = t.db.append_message("legacy-parent", "user", "LEGACY_PARENT")
    t.db.end_session("legacy-parent", end_reason="session_reset")
    t.db.create_session("legacy-child", source="gui", parent_session_id="legacy-parent",
                        model_config={"_reset_from": "legacy-parent"})
    child_id = t.db.append_message("legacy-child", "user", "LEGACY_CHILD")
    with t.db._read_ctx() as conn:
        assert conn.execute("SELECT 1 FROM state_meta WHERE key = ?", (POLICY_PREFIX + "legacy-parent",)).fetchone() is None
        assert local_physical_target(conn, "legacy-parent") == "legacy-parent"
    response = t.client.get("/api/sessions/legacy-parent/messages")
    assert response.status_code == 200, response.text
    page = response.json()
    assert page["session_id"] == "legacy-parent"
    assert _ids(page["messages"]) == [parent_id]
    timeline = t.client.get("/api/sessions/legacy-parent/timeline")
    assert timeline.status_code == 200, timeline.text
    assert [entry["row_id"] for entry in timeline.json()["entries"]] == [parent_id]
    around = t.client.get("/api/sessions/legacy-parent/messages/around", params={"row_id": parent_id})
    assert around.status_code == 200, around.text
    assert _ids(around.json()["messages"]) == [parent_id]
    assert t.client.get("/api/sessions/legacy-parent/messages/around", params={"row_id": child_id}).status_code == 404
