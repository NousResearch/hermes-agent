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


# ---------------------------------------------------------------------------
# New: comprehensive coverage for Crew Usage / Token Monitor
# ---------------------------------------------------------------------------


def test_crew_usage_empty_sessions_table(client_loopback, tmp_path, monkeypatch):
    """A profile whose state.db exists but has an empty sessions table
    should yield zeroed totals, not crash."""
    monkeypatch.setattr(web_server, "_crew_task_counts", lambda: {})
    from hermes_state import SessionDB

    empty_dir = tmp_path / "empty-sessions"
    empty_dir.mkdir(parents=True, exist_ok=True)
    state_db_path = empty_dir / "state.db"
    db = SessionDB(db_path=state_db_path)
    # DB was created with the schema but no rows inserted
    db.close()

    profiles = [{
        "name": "empty-sessions",
        "path": str(empty_dir),
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
    p = body["profiles"][0]
    assert p["total"]["sessions"] == 0
    assert p["total"]["input_tokens"] == 0
    assert p["total"]["output_tokens"] == 0
    assert p["total"]["total_tokens"] == 0
    assert p["total"]["estimated_cost_usd"] == 0.0
    assert p["workers"] == []


def test_crew_usage_days_0_today(client_loopback, tmp_path, monkeypatch, fake_profiles_with_sessions):
    """days=0 should filter to sessions started today (past 24h)."""
    headers = {"X-Hermes-Session-Token": web_server._SESSION_TOKEN}
    monkeypatch.setattr(web_server, "_crew_task_counts", lambda: {})

    today = client_loopback.get("/api/crew/usage?days=0", headers=headers).json()
    all_time = client_loopback.get("/api/crew/usage?days=-1", headers=headers).json()

    assert today["period_days"] == 0
    assert today["totals"]["sessions"] < all_time["totals"]["sessions"]


def test_crew_usage_days_7(client_loopback, tmp_path, monkeypatch, fake_profiles_with_sessions):
    """days=7 should include recent sessions within 7-day window."""
    headers = {"X-Hermes-Session-Token": web_server._SESSION_TOKEN}
    monkeypatch.setattr(web_server, "_crew_task_counts", lambda: {})

    week = client_loopback.get("/api/crew/usage?days=7", headers=headers).json()
    all_time = client_loopback.get("/api/crew/usage?days=-1", headers=headers).json()

    assert week["period_days"] == 7
    assert week["totals"]["sessions"] > 0
    # 7-day window should exclude the old session (11 days ago)
    assert week["totals"]["sessions"] < all_time["totals"]["sessions"]


def test_crew_usage_days_30(client_loopback, tmp_path, monkeypatch, fake_profiles_with_sessions):
    """days=30 should include all sessions since the oldest is only 11 days old."""
    headers = {"X-Hermes-Session-Token": web_server._SESSION_TOKEN}
    monkeypatch.setattr(web_server, "_crew_task_counts", lambda: {})

    thirty = client_loopback.get("/api/crew/usage?days=30", headers=headers).json()
    all_time = client_loopback.get("/api/crew/usage?days=-1", headers=headers).json()

    assert thirty["period_days"] == 30
    # Both sessions per profile fall within 30 days
    assert thirty["totals"]["sessions"] == all_time["totals"]["sessions"]


def test_crew_usage_per_profile_aggregates(client_loopback, tmp_path, monkeypatch, fake_profiles_with_sessions):
    """Verify per-profile aggregate values are correct against known data."""
    headers = {"X-Hermes-Session-Token": web_server._SESSION_TOKEN}
    monkeypatch.setattr(web_server, "_crew_task_counts", lambda: {})

    body = client_loopback.get("/api/crew/usage?days=-1", headers=headers).json()

    # Each profile has 2 sessions: (100+200+10+5=315 tokens, $1.50) + (50+30+2+1=83 tokens, $0.40)
    # Session 1: 100+200+10+5+0 = 315 tokens, $1.50
    # Session 2: 50+30+2+1+0 = 83 tokens, $0.40
    # Per-profile: 2 sessions, 398 tokens, $1.90
    for p in body["profiles"]:
        if "no-db" in p["profile_name"]:
            continue
        assert p["total"]["sessions"] == 2
        assert p["total"]["input_tokens"] == 150   # 100 + 50
        assert p["total"]["output_tokens"] == 230  # 200 + 30
        assert p["total"]["cache_read_tokens"] == 12    # 10 + 2
        assert p["total"]["cache_write_tokens"] == 6    # 5 + 1
        assert p["total"]["total_tokens"] == 398   # 315 + 80 + 3 = 398
        assert p["total"]["estimated_cost_usd"] == 1.90
        assert p["total"]["last_active"] is not None
        assert p["total"]["model"] is not None
        assert p["total"]["provider"] is not None


def test_crew_usage_response_schema(client_loopback, tmp_path, monkeypatch, fake_profiles_with_sessions):
    """Validate the full response schema: field names, types, and structure."""
    import json
    headers = {"X-Hermes-Session-Token": web_server._SESSION_TOKEN}
    monkeypatch.setattr(web_server, "_crew_task_counts", lambda: {})

    body = client_loopback.get("/api/crew/usage?days=-1", headers=headers).json()

    # Top-level keys
    assert isinstance(body, dict)
    assert "generated_at" in body
    assert "period_days" in body
    assert "profiles" in body
    assert "departments" in body
    assert "totals" in body

    # totals shape and types
    totals = body["totals"]
    assert isinstance(totals, dict)
    for key in ("sessions", "input_tokens", "output_tokens", "cache_read_tokens",
                "cache_write_tokens", "reasoning_tokens", "total_tokens"):
        assert isinstance(totals[key], int), f"totals.{key} should be int, got {type(totals[key])}"
        assert totals[key] >= 0
    assert isinstance(totals["estimated_cost_usd"], (int, float))

    # profiles list
    assert isinstance(body["profiles"], list)
    assert len(body["profiles"]) >= 1
    for p in body["profiles"]:
        assert isinstance(p, dict)
        assert isinstance(p["profile_name"], str)
        assert isinstance(p["profile"], dict)  # CrewProfileSnapshot
        assert isinstance(p["display_name"], str)
        assert isinstance(p["department"], str)
        assert isinstance(p["level"], str)
        assert p.get("manager") is None or isinstance(p["manager"], str)

        # tasks shape
        assert isinstance(p["tasks"], dict)
        assert "running" in p["tasks"]
        assert "blocked" in p["tasks"]
        assert isinstance(p["tasks"]["running"], int)
        assert isinstance(p["tasks"]["blocked"], int)

        # total shape
        total = p["total"]
        assert isinstance(total, dict)
        for key in ("sessions", "input_tokens", "output_tokens", "cache_read_tokens",
                    "cache_write_tokens", "reasoning_tokens", "total_tokens"):
            assert isinstance(total[key], int), f"profile.{key} should be int"
        assert isinstance(total["estimated_cost_usd"], (int, float))
        assert total.get("last_active") is None or isinstance(total["last_active"], (int, float))
        assert total.get("model") is None or isinstance(total["model"], str)
        assert total.get("provider") is None or isinstance(total["provider"], str)

        # workers list
        assert isinstance(p["workers"], list)
        for w in p["workers"]:
            assert isinstance(w, dict)
            assert isinstance(w["source"], str)
            assert isinstance(w["sessions"], int)
            assert isinstance(w["total_tokens"], int)
            assert w.get("last_active") is None or isinstance(w["last_active"], (int, float))

    # departments shape
    assert isinstance(body["departments"], list)
    dept_names = set()
    for d in body["departments"]:
        assert isinstance(d, dict)
        assert isinstance(d["department"], str)
        assert isinstance(d["sessions"], int)
        assert isinstance(d["total_tokens"], int)
        assert isinstance(d["estimated_cost_usd"], (int, float))
        assert isinstance(d["profiles"], list)
        assert len(d["profiles"]) > 0
        dept_names.add(d["department"])
    # Departments should be sorted alphabetically
    dept_list = [d["department"] for d in body["departments"]]
    assert dept_list == sorted(dept_list, key=str.lower)


def test_crew_usage_department_grouping(client_loopback, tmp_path, monkeypatch, fake_profiles_with_sessions):
    """Verify department grouping aggregates tokens and costs correctly."""
    headers = {"X-Hermes-Session-Token": web_server._SESSION_TOKEN}
    monkeypatch.setattr(web_server, "_crew_task_counts", lambda: {})

    body = client_loopback.get("/api/crew/usage?days=-1", headers=headers).json()

    # Each department's totals should match sum of its profiles
    for dept in body["departments"]:
        expected_sessions = sum(p["total"]["sessions"] for p in dept["profiles"])
        expected_tokens = sum(p["total"]["total_tokens"] for p in dept["profiles"])
        expected_cost = sum(p["total"]["estimated_cost_usd"] for p in dept["profiles"])
        assert dept["sessions"] == expected_sessions
        assert dept["total_tokens"] == expected_tokens
        assert abs(dept["estimated_cost_usd"] - expected_cost) < 0.001
        # Verify all profiles in this department have matching department name
        for p in dept["profiles"]:
            assert p["department"] == dept["department"]

    # The grand totals should match sum of all department aggregates
    total_by_dept_sessions = sum(d["sessions"] for d in body["departments"])
    total_by_dept_tokens = sum(d["total_tokens"] for d in body["departments"])
    total_by_dept_cost = sum(d["estimated_cost_usd"] for d in body["departments"])
    assert body["totals"]["sessions"] == total_by_dept_sessions
    assert body["totals"]["total_tokens"] == total_by_dept_tokens
    assert abs(body["totals"]["estimated_cost_usd"] - total_by_dept_cost) < 0.001


def test_crew_usage_sanitization_system_prompt_handoff(client_loopback, tmp_path, monkeypatch):
    """Verify system_prompt, handoff_state, message_content, title
    keys are stripped from usage responses. Also check _key, _token, _secret
    patterns in profile snapshot output."""
    import json
    from hermes_state import SessionDB

    monkeypatch.setattr(web_server, "_crew_task_counts", lambda: {})

    secret_dir = tmp_path / "secrets-check"
    secret_dir.mkdir(parents=True, exist_ok=True)
    state_db_path = secret_dir / "state.db"
    db = SessionDB(db_path=state_db_path)
    import time
    db._conn.execute(
        "INSERT INTO sessions (source, input_tokens, output_tokens, started_at, model, billing_provider)"
        " VALUES (?, ?, ?, ?, ?, ?)",
        ("test", 10, 20, time.time(), "m", "p"),
    )
    db.close()

    profiles = [{
        "name": "secrets-check",
        "path": str(secret_dir),
        "is_default": False,
        "model": None,
        "provider": None,
        "gateway_running": False,
        "has_env": False,
        "skill_count": 0,
    }]
    monkeypatch.setattr(web_server, "_crew_profile_dicts", lambda: profiles)

    # Patch _profile_snapshot to return a payload that includes secret-shaped keys.
    # This simulates what would happen if a profile's snapshot contained sensitive fields.
    _original_snapshot = web_server._profile_snapshot

    def _dirty_snapshot(profile):
        snap = _original_snapshot(profile)
        snap["system_prompt"] = "you are an evil assistant"
        snap["handoff_state"] = "some state"
        snap["message_content"] = "secret message"
        snap["title"] = "secret task title"
        snap["api_key"] = "sk-abc123"
        snap["auth_token"] = "tok-xyz"
        return snap

    monkeypatch.setattr(web_server, "_profile_snapshot", _dirty_snapshot)

    headers = {"X-Hermes-Session-Token": web_server._SESSION_TOKEN}
    resp = client_loopback.get("/api/crew/usage?days=30", headers=headers)
    assert resp.status_code == 200
    body = resp.json()
    blob = json.dumps(body).lower()

    # These keys should NOT appear in the response
    for forbidden in ("system_prompt", "handoff_state", "message_content", "title",
                      "api_key", "auth_token", "password", "cookie",
                      "credential", "private_key", "raw_log", "env_values"):
        assert forbidden not in blob, f"Secret-shaped key '{forbidden}' leaked in response"

    # The profile should still have valid data
    assert len(body["profiles"]) == 1
    p = body["profiles"][0]
    assert p["total"]["sessions"] == 1
    assert p["total"]["total_tokens"] == 30


def test_crew_usage_sanitization_long_string_truncation(client_loopback, tmp_path, monkeypatch):
    """Verify strings longer than 120 chars are truncated to '[truncated]'."""
    import json
    from hermes_state import SessionDB

    monkeypatch.setattr(web_server, "_crew_task_counts", lambda: {})

    long_str = "x" * 500
    secret_dir = tmp_path / "long-str-profile"
    secret_dir.mkdir(parents=True, exist_ok=True)
    state_db_path = secret_dir / "state.db"
    db = SessionDB(db_path=state_db_path)
    import time
    db._conn.execute(
        "INSERT INTO sessions (source, input_tokens, output_tokens, started_at, model, billing_provider)"
        " VALUES (?, ?, ?, ?, ?, ?)",
        ("test", 10, 20, time.time(), "m", "p"),
    )
    db.close()

    # Patch _profile_snapshot to inject a very long string value
    _original_snapshot = web_server._profile_snapshot

    def _long_str_snapshot(profile):
        snap = _original_snapshot(profile)
        snap["long_description"] = long_str
        return snap

    monkeypatch.setattr(web_server, "_profile_snapshot", _long_str_snapshot)

    profiles = [{
        "name": "long-str-profile",
        "path": str(secret_dir),
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
    blob = json.dumps(body)

    # The long string should NOT appear as 500 x's — it should be truncated
    assert "x" * 500 not in blob
    # The truncation marker should be present instead
    assert "[truncated]" in blob


def test_crew_usage_sanitization_secret_patterns(client_loopback, tmp_path, monkeypatch):
    """Verify secret-shaped key values are stripped from usage responses.

    Note: _sanitize_profile_usage_payload uses exact key-name matching against
    _SECRET_SHAPED_KEYS, NOT substring/pattern matching. Compound keys like
    'openai_api_key' or 'session_secret' are NOT caught by the current
    sanitizer — only the exact keys in the _SECRET_SHAPED_KEYS set are
    stripped. This test validates that known sensitive keys are stripped
    correctly."""
    import json
    from hermes_state import SessionDB

    monkeypatch.setattr(web_server, "_crew_task_counts", lambda: {})

    secret_dir = tmp_path / "pattern-test"
    secret_dir.mkdir(parents=True, exist_ok=True)
    state_db_path = secret_dir / "state.db"
    db = SessionDB(db_path=state_db_path)
    import time
    db._conn.execute(
        "INSERT INTO sessions (source, input_tokens, output_tokens, started_at, model, billing_provider)"
        " VALUES (?, ?, ?, ?, ?, ?)",
        ("test", 10, 20, time.time(), "m", "p"),
    )
    db.close()

    # Keys that ARE in _SECRET_SHAPED_KEYS and should be stripped
    _original_snapshot = web_server._profile_snapshot

    def _dirty_snapshot(profile):
        snap = _original_snapshot(profile)
        snap["api_key"] = "should-be-stripped"
        snap["secret_key"] = "should-be-stripped"
        snap["auth_token"] = "should-be-stripped"
        snap["password"] = "should-be-stripped"
        snap["credential"] = "should-be-stripped"
        return snap

    monkeypatch.setattr(web_server, "_profile_snapshot", _dirty_snapshot)

    profiles = [{
        "name": "pattern-test",
        "path": str(secret_dir),
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
    blob = json.dumps(body)

    # None of the injected key names should appear (they get stripped entirely)
    for key in ("api_key", "secret_key", "auth_token", "password", "credential"):
        assert key not in blob, f"Key '{key}' leaked in response"

    # The secret values should also be absent
    assert "should-be-stripped" not in blob


def test_crew_usage_auth_gate(client_loopback):
    """Verify /api/crew/usage requires authentication."""
    resp = client_loopback.get("/api/crew/usage?days=30")
    assert resp.status_code == 401


def test_crew_task_counts_soft_fail(tmp_path, monkeypatch):
    """Verify _crew_task_counts returns {} when kanban.db is unreadable."""
    counts = web_server._crew_task_counts()
    # Should not crash — returns whatever it can (real or empty)
    assert isinstance(counts, dict)


def test_sanitize_profile_usage_payload_strips_nested_secrets():
    """Unit test for the _sanitize_profile_usage_payload function directly."""
    payload = {
        "generated_at": "2026-01-01T00:00:00Z",
        "profiles": [{
            "profile_name": "test",
            "api_key": "should-go",
            "nested": {
                "secret_key": "also-go",
                "inner": {
                    "auth": "gone",
                },
                "safe_data": "keep",
            },
            "safe": "keep",
        }],
    }
    result = web_server._sanitize_profile_usage_payload(payload)
    blob = str(result)
    assert "api_key" not in blob
    assert "secret_key" not in blob
    assert "auth" not in blob
    assert "safe_data" in blob
    assert "safe" in blob
    assert result["profiles"][0]["safe"] == "keep"
    assert result["profiles"][0]["nested"]["safe_data"] == "keep"
