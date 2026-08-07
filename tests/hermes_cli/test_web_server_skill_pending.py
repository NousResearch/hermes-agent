"""Dashboard API tests for profile-scoped pending skill writes."""

import json

import pytest


SKILL_MD = """---
name: queued-skill
description: a queued test skill
---

# Queued skill
"""


def _stage_skill_write(home, pending_id, *, summary, payload, origin="foreground"):
    pending_dir = home / "pending" / "skills"
    pending_dir.mkdir(parents=True, exist_ok=True)
    record = {
        "id": pending_id,
        "subsystem": "skills",
        "action": payload["action"],
        "summary": summary,
        "origin": origin,
        "created_at": 1_700_000_000.0,
        "payload": payload,
    }
    (pending_dir / f"{pending_id}.json").write_text(
        json.dumps(record), encoding="utf-8"
    )


@pytest.fixture
def isolated_profiles(tmp_path, monkeypatch, _isolate_hermes_home):
    """Default and named profile homes, isolated like a real dashboard."""
    from hermes_constants import get_hermes_home
    from hermes_cli import profiles

    default_home = get_hermes_home()
    profiles_root = default_home / "profiles"
    worker_home = profiles_root / "worker_alpha"
    for home in (default_home, worker_home):
        (home / "skills").mkdir(parents=True, exist_ok=True)
        (home / "config.yaml").write_text("{}\n", encoding="utf-8")

    monkeypatch.setattr(profiles, "_get_default_hermes_home", lambda: default_home)
    monkeypatch.setattr(profiles, "_get_profiles_root", lambda: profiles_root)
    return {"default": default_home, "worker_alpha": worker_home}


@pytest.fixture
def client(monkeypatch, isolated_profiles):
    try:
        from starlette.testclient import TestClient
    except ImportError:
        pytest.skip("fastapi/starlette not installed")

    import hermes_state
    from hermes_constants import get_hermes_home
    from hermes_cli.web_server import app, _SESSION_HEADER_NAME, _SESSION_TOKEN

    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", get_hermes_home() / "state.db")
    dashboard = TestClient(app)
    dashboard.headers[_SESSION_HEADER_NAME] = _SESSION_TOKEN
    return dashboard


def test_pending_list_is_profile_scoped_and_hides_payload(client, isolated_profiles):
    _stage_skill_write(
        isolated_profiles["default"],
        "a1b2c3d4",
        summary="create default-only",
        payload={"action": "create", "name": "default-only", "content": "private"},
    )
    _stage_skill_write(
        isolated_profiles["worker_alpha"],
        "b1c2d3e4",
        summary="create queued-skill",
        origin="background_review",
        payload={"action": "create", "name": "queued-skill", "content": SKILL_MD},
    )

    response = client.get("/api/skills/pending", params={"profile": "worker_alpha"})

    assert response.status_code == 200
    assert response.json() == {
        "pending": [
            {
                "id": "b1c2d3e4",
                "action": "create",
                "summary": "create queued-skill",
                "origin": "background_review",
                "created_at": 1_700_000_000.0,
                "name": "queued-skill",
                "file_path": "",
            }
        ]
    }


def test_pending_diff_uses_selected_profile(client, isolated_profiles):
    _stage_skill_write(
        isolated_profiles["worker_alpha"],
        "b1c2d3e4",
        summary="create queued-skill",
        payload={"action": "create", "name": "queued-skill", "content": SKILL_MD},
    )

    response = client.get(
        "/api/skills/pending/b1c2d3e4/diff", params={"profile": "worker_alpha"}
    )

    assert response.status_code == 200
    assert response.json() == {
        "id": "b1c2d3e4",
        "summary": "create queued-skill",
        "diff": SKILL_MD,
    }


def test_approve_applies_only_selected_profile_pending_write(client, isolated_profiles):
    _stage_skill_write(
        isolated_profiles["worker_alpha"],
        "b1c2d3e4",
        summary="create queued-skill",
        payload={"action": "create", "name": "queued-skill", "content": SKILL_MD},
    )

    response = client.post(
        "/api/skills/pending/b1c2d3e4/approve", params={"profile": "worker_alpha"}
    )

    assert response.status_code == 200
    assert response.json() == {"ok": True, "id": "b1c2d3e4"}
    assert (
        isolated_profiles["worker_alpha"] / "skills" / "queued-skill" / "SKILL.md"
    ).read_text(encoding="utf-8") == SKILL_MD
    assert not (
        isolated_profiles["worker_alpha"] / "pending" / "skills" / "b1c2d3e4.json"
    ).exists()
    assert not (
        isolated_profiles["default"] / "skills" / "queued-skill"
    ).exists()


def test_reject_discards_only_selected_profile_pending_write(client, isolated_profiles):
    _stage_skill_write(
        isolated_profiles["worker_alpha"],
        "b1c2d3e4",
        summary="create queued-skill",
        payload={"action": "create", "name": "queued-skill", "content": SKILL_MD},
    )

    response = client.delete(
        "/api/skills/pending/b1c2d3e4", params={"profile": "worker_alpha"}
    )

    assert response.status_code == 200
    assert response.json() == {"ok": True, "id": "b1c2d3e4"}
    assert not (
        isolated_profiles["worker_alpha"] / "pending" / "skills" / "b1c2d3e4.json"
    ).exists()
    assert not (
        isolated_profiles["worker_alpha"] / "skills" / "queued-skill"
    ).exists()


def test_approve_cannot_cross_profile_boundary(client, isolated_profiles):
    _stage_skill_write(
        isolated_profiles["worker_alpha"],
        "b1c2d3e4",
        summary="create queued-skill",
        payload={"action": "create", "name": "queued-skill", "content": SKILL_MD},
    )

    response = client.post(
        "/api/skills/pending/b1c2d3e4/approve", params={"profile": "default"}
    )

    assert response.status_code == 404
    assert (
        isolated_profiles["worker_alpha"] / "pending" / "skills" / "b1c2d3e4.json"
    ).exists()
    assert not (
        isolated_profiles["worker_alpha"] / "skills" / "queued-skill"
    ).exists()


def test_failed_approval_keeps_the_pending_write(client, isolated_profiles):
    _stage_skill_write(
        isolated_profiles["worker_alpha"],
        "b1c2d3e4",
        summary="patch a missing skill",
        payload={
            "action": "patch",
            "name": "missing-skill",
            "old_string": "old",
            "new_string": "new",
        },
    )

    response = client.post(
        "/api/skills/pending/b1c2d3e4/approve", params={"profile": "worker_alpha"}
    )

    assert response.status_code == 409
    assert (
        isolated_profiles["worker_alpha"] / "pending" / "skills" / "b1c2d3e4.json"
    ).exists()
