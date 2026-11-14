"""Public session responses keep client metadata without disclosing storage columns."""

import json
import time

import pytest


PRIVATE_FIELDS = {
    "user_id", "session_key", "chat_id", "chat_type", "thread_id", "display_name",
    "origin_json", "expiry_finalized", "model_config", "system_prompt", "system_prompt_hash",
    "tool_names", "billing_provider", "billing_base_url", "billing_mode", "cost_source",
    "pricing_version", "handoff_error", "compression_failure_cooldown_until",
    "compression_failure_error", "compression_fallback_streak", "future_private_field",
}


@pytest.fixture
def homes(tmp_path, monkeypatch, _isolate_hermes_home):
    from hermes_cli import profiles
    from hermes_cli.web_routers import profiles as profile_routes
    from hermes_constants import get_hermes_home

    default = get_hermes_home()
    profile_root = default / "profiles"
    worker = profile_root / "worker"
    for home in (default, worker):
        home.mkdir(parents=True, exist_ok=True)
        (home / "config.yaml").write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(profiles, "_get_default_hermes_home", lambda: default)
    monkeypatch.setattr(profiles, "_get_profiles_root", lambda: profile_root)
    monkeypatch.setattr(profile_routes, "_SIDEBAR_CACHE_TTL_SECONDS", 0.0)
    return {"default": default, "worker": worker}


@pytest.fixture
def client(homes, monkeypatch):
    from starlette.testclient import TestClient

    import hermes_state
    from hermes_cli.web_server import _SESSION_HEADER_NAME, _SESSION_TOKEN, app

    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", homes["default"] / "state.db")
    client = TestClient(app)
    client.headers[_SESSION_HEADER_NAME] = _SESSION_TOKEN
    return client


def _seed(home, sid, *, source="discord", parent=None, markers=None):
    from hermes_state import SessionDB

    db = SessionDB(db_path=home / "state.db")
    try:
        db.create_session(
            sid, source=source, model="example-model", parent_session_id=parent,
            system_prompt="synthetic private prompt", model_config={"private": "value", **(markers or {})},
            user_id="synthetic-user", session_key="synthetic-route", chat_id="synthetic-chat",
            chat_type="guild", thread_id="synthetic-thread", cwd="/workspace/example",
        )
        db.append_message(sid, role="user", content="public preview")
        db.record_gateway_session_peer(
            sid, source=source, user_id="synthetic-user", session_key="synthetic-route",
            chat_id="synthetic-chat", chat_type="guild", thread_id="synthetic-thread",
            display_name="synthetic private peer", origin_json=json.dumps({"private": "peer"}),
        )

        def write(conn):
            if "future_private_field" not in {r[1] for r in conn.execute("PRAGMA table_info(sessions)")}:
                conn.execute("ALTER TABLE sessions ADD COLUMN future_private_field TEXT")
            conn.execute(
                """UPDATE sessions SET title = ?, git_branch = ?, git_repo_root = ?,
                   billing_provider = ?, billing_base_url = ?, api_call_count = ?,
                   handoff_platform = ?, handoff_state = ?, handoff_error = ?,
                   compression_failure_error = ?, future_private_field = ? WHERE id = ?""",
                (sid, "feature/example", "/workspace/example", "synthetic-provider",
                 "https://backend.invalid/v1", 7, source, "failed", "synthetic raw diagnostic",
                 "synthetic compression diagnostic", "synthetic future secret", sid),
            )

        db._execute_write(write)
    finally:
        db.close()


@pytest.mark.parametrize("endpoint,full", [
    ("sessions", False), ("sessions", True), ("profiles", False),
    ("profiles", True), ("sidebar", False), ("detail", False),
])
def test_public_response_is_allowlisted_across_profile_switches(client, homes, endpoint, full):
    for name, home in homes.items():
        _seed(home, f"{name}-public")
    # The same loaded app serves A -> B -> A; no credentials or rows may stick to a prior scope.
    for name in ("default", "worker", "default"):
        if endpoint == "sidebar":
            response = client.get("/api/profiles/sessions/sidebar", params={"recents_profile": name})
            rows = response.json()["recents"]["sessions"]
        elif endpoint == "detail":
            response = client.get(f"/api/sessions/{name}-public", params={"profile": name})
            rows = [response.json()]
        else:
            route = "/api/sessions" if endpoint == "sessions" else "/api/profiles/sessions"
            response = client.get(route, params={"profile": name, "full": int(full)})
            rows = response.json()["sessions"]
        assert response.status_code == 200
        assert {row["id"] for row in rows} == {f"{name}-public"}
        row = rows[0]
        assert not (PRIVATE_FIELDS & row.keys())
        assert row["profile"] == name
        assert row["is_default_profile"] is (name == "default")
        assert row["model"] == "example-model"
        assert row["git_branch"] == "feature/example"
        assert row["handoff_state"] == "failed"
        assert row["message_count"] == 1
        for flag in ("archived", "pinned", "hidden", "is_active"):
            assert isinstance(row[flag], bool)
        if endpoint == "detail":
            assert row["api_call_count"] == 7


def test_projection_preserves_compression_and_branch_metadata(client, homes):
    from hermes_state import SessionDB

    home = homes["default"]
    _seed(home, "root", source="cli")
    db = SessionDB(db_path=home / "state.db")
    try:
        db.end_session("root", "compression")
    finally:
        db.close()
    _seed(home, "tip", source="cli", parent="root")
    for marker in ("_branched_from", "_reset_from"):
        _seed(home, marker, source="cli", parent="root", markers={marker: "root"})
        response = client.get(f"/api/sessions/{marker}", params={"profile": "default"})
        assert response.status_code == 200
        assert response.json()[marker] == "root"
    for route in ("/api/sessions", "/api/profiles/sessions"):
        response = client.get(route, params={"profile": "default", "order": "recent"})
        assert response.status_code == 200
        tip = next(row for row in response.json()["sessions"] if row["id"] == "tip")
        assert tip["_lineage_root_id"] == "root"
        assert tip["_lineage_ids"] == ["root", "tip"]
        assert tip["continuation_kind"] == "compression"


def test_detail_uses_sql_activity_and_preserves_cron_ownership(client, homes, monkeypatch):
    from hermes_state import SessionDB
    from hermes_cli.web_routers import sessions as session_routes

    home = homes["default"]
    sid = "cron_fixture_20261002_120000"
    _seed(home, sid, source="cron")
    now = time.time()
    db = SessionDB(db_path=home / "state.db")
    try:
        def timestamps(conn):
            conn.execute("UPDATE sessions SET started_at = ?, last_activity_at = ? WHERE id = ?", (now - 3600, now - 30, sid))
            conn.execute("UPDATE messages SET timestamp = ? WHERE session_id = ?", (now - 60, sid))
        db._execute_write(timestamps)
    finally:
        db.close()
    monkeypatch.setattr(session_routes.time, "time", lambda: now)
    response = client.get(f"/api/sessions/{sid}", params={"profile": "default"})
    assert response.status_code == 200
    row = response.json()
    assert row["last_active"] == now - 30
    assert row["is_active"] is True
    # No durable in-flight execution owns this synthetic run.
    assert row["scheduler_owned"] is False
