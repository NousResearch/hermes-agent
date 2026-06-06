"""Focused tests for read-only dashboard crew APIs."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from hermes_cli import web_server


def _client():
    prev_host = getattr(web_server.app.state, "bound_host", None)
    prev_port = getattr(web_server.app.state, "bound_port", None)
    web_server.app.state.bound_host = "127.0.0.1"
    web_server.app.state.bound_port = 9119
    client = TestClient(web_server.app, base_url="http://127.0.0.1:9119")
    try:
        yield client
    finally:
        web_server.app.state.bound_host = prev_host
        web_server.app.state.bound_port = prev_port


@pytest.fixture
def client_loopback():
    yield from _client()


@pytest.fixture
def fake_profiles(tmp_path, monkeypatch):
    default_dir = tmp_path / "default"
    atlas_dir = tmp_path / "profiles" / "atlas"
    new_dir = tmp_path / "profiles" / "new-worker"
    for profile_dir in (default_dir, atlas_dir, new_dir):
        profile_dir.mkdir(parents=True)
    (default_dir / ".env").write_text("SECRET_TOKEN=not-read\n", encoding="utf-8")
    (default_dir / "SOUL.md").write_text("soul", encoding="utf-8")
    (atlas_dir / "SOUL.md").write_text("soul", encoding="utf-8")

    profiles = [
        {
            "name": "default",
            "path": str(default_dir),
            "is_default": True,
            "model": "jarvis-model",
            "provider": "test-provider",
            "gateway_running": True,
            "has_env": True,
            "skill_count": 3,
        },
        {
            "name": "atlas",
            "path": str(atlas_dir),
            "is_default": False,
            "model": None,
            "provider": None,
            "gateway_running": False,
            "has_env": False,
            "skill_count": 7,
        },
        {
            "name": "new-worker",
            "path": str(new_dir),
            "is_default": False,
            "model": None,
            "provider": None,
            "gateway_running": False,
            "has_env": False,
            "skill_count": 0,
        },
    ]
    monkeypatch.setattr(web_server, "_crew_profile_dicts", lambda: profiles)
    monkeypatch.setattr(web_server, "_resolve_profile_dir", lambda name: tmp_path / name)
    return profiles


@pytest.fixture
def fake_profiles_with_sessions(tmp_path, monkeypatch, fake_profiles):
    """Extends fake_profiles with state.db files containing known session rows."""
    import json
    import time
    from hermes_state import SessionDB

    profiles = fake_profiles
    now = time.time()
    one_hour_ago = now - 3600
    one_day_ago = now - 86400

    for p in profiles:
        name = p["name"]
        profile_path = Path(str(p["path"]))
        profile_path.mkdir(parents=True, exist_ok=True)
        state_db_path = profile_path / "state.db"

        db = SessionDB(db_path=state_db_path)
        try:
            # Insert known sessions with src profile data
            db._conn.execute(
                """INSERT INTO sessions
                (source, input_tokens, output_tokens, cache_read_tokens,
                 cache_write_tokens, reasoning_tokens, estimated_cost_usd,
                 started_at, model, billing_provider)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    "kanban" if name != "default" else "manual",
                    100,
                    200,
                    10,
                    5,
                    0,
                    1.50,
                    one_hour_ago,
                    "test-model",
                    "test-provider",
                )
            )
            # Second row: old (outside 7-day window)
            db._conn.execute(
                """INSERT INTO sessions
                (source, input_tokens, output_tokens, cache_read_tokens,
                 cache_write_tokens, reasoning_tokens, estimated_cost_usd,
                 started_at, model, billing_provider)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    "cli",
                    50,
                    30,
                    2,
                    1,
                    0,
                    0.40,
                    one_day_ago - 86400 * 10,  # 11 days ago
                    "old-model",
                    "old-provider",
                )
            )
        finally:
            db.close()

    # Add a profile WITH NO state.db (to test graceful handling)
    no_db_dir = tmp_path / "no-db-profile"
    no_db_dir.mkdir(parents=True, exist_ok=True)
    profiles.append({
        "name": "no-db-profile",
        "path": str(no_db_dir),
        "is_default": False,
        "model": None,
        "provider": None,
        "gateway_running": False,
        "has_env": False,
        "skill_count": 0,
    })

    # Re-apply monkeypatch so the new profile is discovered
    monkeypatch.setattr(web_server, "_crew_profile_dicts", lambda: profiles)
    return profiles


def test_load_crew_metadata_missing_file(tmp_path, monkeypatch):
    metadata_path = tmp_path / "crew" / "organization.yaml"
    monkeypatch.setattr(web_server, "_crew_metadata_path", lambda: metadata_path)

    metadata, warnings = web_server._load_crew_metadata()

    assert metadata == {"version": 1, "profiles": {}}
    assert warnings == []


def test_load_crew_metadata_valid_file_filters_unsafe_keys(tmp_path, monkeypatch):
    metadata_path = tmp_path / "crew" / "organization.yaml"
    metadata_path.parent.mkdir()
    metadata_path.write_text(
        """
version: 1
profiles:
  atlas:
    display_name: Atlas
    role: IT Team Manager
    level: manager
    department: IT Team
    manager: default
    token: should-not-expose
    auth_json: should-not-expose
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setattr(web_server, "_crew_metadata_path", lambda: metadata_path)

    metadata, warnings = web_server._load_crew_metadata()

    assert warnings == []
    assert metadata["profiles"]["atlas"]["display_name"] == "Atlas"
    assert "token" not in metadata["profiles"]["atlas"]
    assert "auth_json" not in metadata["profiles"]["atlas"]


def test_load_crew_metadata_invalid_file_returns_warning(tmp_path, monkeypatch):
    metadata_path = tmp_path / "crew" / "organization.yaml"
    metadata_path.parent.mkdir()
    metadata_path.write_text("profiles: [", encoding="utf-8")
    monkeypatch.setattr(web_server, "_crew_metadata_path", lambda: metadata_path)

    metadata, warnings = web_server._load_crew_metadata()

    assert metadata["profiles"] == {}
    assert warnings


def test_build_crew_nodes_infers_and_includes_unassigned(tmp_path, monkeypatch, fake_profiles):
    metadata_path = tmp_path / "crew" / "organization.yaml"
    monkeypatch.setattr(web_server, "_crew_metadata_path", lambda: metadata_path)

    payload = web_server._crew_organization_payload()
    by_name = {node["profile"]["name"]: node for node in payload["nodes"]}

    assert set(by_name) == {"default", "atlas", "new-worker"}
    assert by_name["default"]["display_name"] == "Jarvis"
    assert by_name["atlas"]["level"] == "manager"
    assert by_name["new-worker"]["metadata_status"] == "missing"
    assert by_name["new-worker"] in payload["unassigned"]
    assert payload["summary"]["total"] == 3
    assert payload["summary"]["unassigned"] == 1


def _walk_keys(value):
    if isinstance(value, dict):
        for key, nested in value.items():
            yield key
            yield from _walk_keys(nested)
    elif isinstance(value, list):
        for nested in value:
            yield from _walk_keys(nested)


def test_profile_snapshot_and_payload_do_not_expose_secret_fields(tmp_path, monkeypatch, fake_profiles):
    metadata_path = tmp_path / "crew" / "organization.yaml"
    monkeypatch.setattr(web_server, "_crew_metadata_path", lambda: metadata_path)

    payload = web_server._crew_organization_payload()
    forbidden_keys = {
        "env",
        "env_values",
        "token",
        "secret",
        "auth",
        "cookie",
        "auth_json",
        "raw_log",
    }

    assert forbidden_keys.isdisjoint(set(_walk_keys(payload)))
    default_snapshot = next(node["profile"] for node in payload["nodes"] if node["profile"]["name"] == "default")
    assert default_snapshot["has_env"] is True
    assert default_snapshot["has_soul"] is True
    assert "SECRET_TOKEN" not in str(payload)
    assert "not-read" not in str(payload)


def test_crew_endpoints_are_auth_gated(client_loopback):
    response = client_loopback.get("/api/crew/organization")
    assert response.status_code == 401


def test_crew_endpoints_return_expected_shapes(client_loopback, tmp_path, monkeypatch, fake_profiles):
    metadata_path = tmp_path / "crew" / "organization.yaml"
    monkeypatch.setattr(web_server, "_crew_metadata_path", lambda: metadata_path)
    headers = {"X-Hermes-Session-Token": web_server._SESSION_TOKEN}

    organization = client_loopback.get("/api/crew/organization", headers=headers)
    control = client_loopback.get("/api/crew/control", headers=headers)
    detail = client_loopback.get("/api/crew/profiles/default", headers=headers)

    assert organization.status_code == 200
    org_body = organization.json()
    assert {"generated_at", "source", "summary", "nodes", "departments", "unassigned"}.issubset(org_body)
    assert org_body["summary"]["total"] == 3

    assert control.status_code == 200
    assert len(control.json()["profiles"]) == 3

    assert detail.status_code == 200
    assert detail.json()["node"]["profile"]["name"] == "default"


def test_crew_usage_returns_200_and_totals(client_loopback, tmp_path, monkeypatch, fake_profiles_with_sessions):
    headers = {"X-Hermes-Session-Token": web_server._SESSION_TOKEN}
    monkeypatch.setattr(web_server, "_crew_task_counts", lambda: {})

    resp = client_loopback.get("/api/crew/usage?days=-1", headers=headers)
    assert resp.status_code == 200
    body = resp.json()
    assert "profiles" in body and "departments" in body and "totals" in body
    assert body["period_days"] == -1
    # Each profile has one recent + one old session (100+200+10+5 = 315 input/output tokens)
    assert body["totals"]["total_tokens"] > 0
    assert body["totals"]["sessions"] >= 6  # 3 profiles × 2 sessions
    for p in body["profiles"]:
        assert "tasks" in p
        assert p["tasks"]["running"] == 0
        assert p["tasks"]["blocked"] == 0
        # Profiles with state.db should have workers; "no-db-profile" has none
        if "no-db" not in p["profile_name"]:
            assert len(p["workers"]) >= 1
            assert p["total"]["model"] is not None


def test_crew_usage_today_window(client_loopback, tmp_path, monkeypatch, fake_profiles_with_sessions):
    """days=1 should include recent sessions only (last 24h)."""
    headers = {"X-Hermes-Session-Token": web_server._SESSION_TOKEN}
    monkeypatch.setattr(web_server, "_crew_task_counts", lambda: {})

    all_time = client_loopback.get("/api/crew/usage?days=-1", headers=headers).json()
    today = client_loopback.get("/api/crew/usage?days=1", headers=headers).json()

    assert today["period_days"] == 1
    assert today["totals"]["total_tokens"] > 0
    # 1-day window should have fewer sessions than all-time (old sessions excluded)
    assert today["totals"]["sessions"] < all_time["totals"]["sessions"]


def test_crew_usage_no_secret_keys(client_loopback, tmp_path, monkeypatch, fake_profiles_with_sessions):
    import json
    headers = {"X-Hermes-Session-Token": web_server._SESSION_TOKEN}
    monkeypatch.setattr(web_server, "_crew_task_counts", lambda: {})

    body = client_loopback.get("/api/crew/usage?days=30", headers=headers).json()
    blob = json.dumps(body).lower()
    for forbidden in ("secret_token", "message_content", "auth_json", "cookie", "password"):
        assert forbidden not in blob


def test_crew_usage_handles_missing_state_db(client_loopback, tmp_path, monkeypatch):
    """A profile with no state.db should yield zeroed totals, not 500."""
    monkeypatch.setattr(web_server, "_crew_task_counts", lambda: {})

    # Create a profile with NO state.db at all
    no_db_dir = tmp_path / "missing-db-profile"
    no_db_dir.mkdir(parents=True, exist_ok=True)
    profiles = [{
        "name": "missing-db-profile",
        "path": str(no_db_dir),
        "is_default": False,
        "model": None,
        "provider": None,
        "gateway_running": False,
        "has_env": False,
        "skill_count": 0,
    }]
    monkeypatch.setattr(web_server, "_crew_profile_dicts", lambda: profiles)

    headers = {"X-Hermes-Session-Token": web_server._SESSION_TOKEN}
    resp = client_loopback.get("/api/crew/usage?days=30", headers=headers)
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["profiles"]) == 1
    assert body["profiles"][0]["total"]["total_tokens"] == 0
    assert body["profiles"][0]["workers"] == []
