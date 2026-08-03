"""Tests for the dashboard skill editor endpoints and cron skill attachment.

The Skills page can now create/edit custom skills (SKILL.md) and the Cron
page can attach skills to jobs — closing the "SSH + nano is the only way"
gap for headless/VPS users. These tests pin:

- GET /api/skills/content returns raw SKILL.md (and profile-scopes).
- POST /api/skills creates a skill through the same validated write path
  as the agent's ``skill_manage`` tool (frontmatter validation enforced).
- PUT /api/skills/content rewrites an existing SKILL.md (404 on unknown).
- POST /api/cron/jobs accepts ``skills`` and persists it on the job;
  PUT /api/cron/jobs/{id} can update the list.
"""
import pytest


SKILL_MD = """---
name: {name}
description: a test skill
---

# {name}

Do the thing.
"""


def _write_skill(skills_dir, name):
    d = skills_dir / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "SKILL.md").write_text(SKILL_MD.format(name=name), encoding="utf-8")


@pytest.fixture
def isolated_profiles(tmp_path, monkeypatch, _isolate_hermes_home):
    """Isolated default home + one named profile, each with its own skills."""
    from hermes_constants import get_hermes_home
    from hermes_cli import profiles

    default_home = get_hermes_home()
    profiles_root = default_home / "profiles"
    worker_home = profiles_root / "worker_alpha"
    for home in (default_home, worker_home):
        (home / "skills").mkdir(parents=True, exist_ok=True)
        (home / "config.yaml").write_text("{}\n", encoding="utf-8")

    _write_skill(default_home / "skills", "dashboard-skill")
    _write_skill(worker_home / "skills", "worker-skill")

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
    c = TestClient(app)
    c.headers[_SESSION_HEADER_NAME] = _SESSION_TOKEN
    return c


class TestSkillContent:
    def test_get_content_returns_raw_skill_md(self, client, isolated_profiles):
        resp = client.get("/api/skills/content", params={"name": "dashboard-skill"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "dashboard-skill"
        assert data["content"].startswith("---")
        assert "Do the thing." in data["content"]

    def test_get_content_scopes_to_profile(self, client, isolated_profiles):
        resp = client.get(
            "/api/skills/content",
            params={"name": "worker-skill", "profile": "worker_alpha"},
        )
        assert resp.status_code == 200
        # ...and the worker skill is invisible without the profile param.
        resp = client.get("/api/skills/content", params={"name": "worker-skill"})
        assert resp.status_code == 404

    def test_get_content_unknown_skill_404(self, client, isolated_profiles):
        resp = client.get("/api/skills/content", params={"name": "nope"})
        assert resp.status_code == 404


class TestSkillCreate:
    def test_create_writes_skill_md(self, client, isolated_profiles):
        resp = client.post(
            "/api/skills",
            json={"name": "my-new-skill", "content": SKILL_MD.format(name="my-new-skill")},
        )
        assert resp.status_code == 200
        assert resp.json()["success"] is True
        skill_md = isolated_profiles["default"] / "skills" / "my-new-skill" / "SKILL.md"
        assert skill_md.exists()
        assert "Do the thing." in skill_md.read_text(encoding="utf-8")

    def test_create_with_category(self, client, isolated_profiles):
        resp = client.post(
            "/api/skills",
            json={
                "name": "cat-skill",
                "category": "devops",
                "content": SKILL_MD.format(name="cat-skill"),
            },
        )
        assert resp.status_code == 200
        assert (
            isolated_profiles["default"] / "skills" / "devops" / "cat-skill" / "SKILL.md"
        ).exists()

    def test_create_scopes_to_profile(self, client, isolated_profiles):
        resp = client.post(
            "/api/skills",
            json={
                "name": "worker-new",
                "content": SKILL_MD.format(name="worker-new"),
                "profile": "worker_alpha",
            },
        )
        assert resp.status_code == 200
        assert (
            isolated_profiles["worker_alpha"] / "skills" / "worker-new" / "SKILL.md"
        ).exists()
        # Dashboard's own skills dir stays clean.
        assert not (
            isolated_profiles["default"] / "skills" / "worker-new"
        ).exists()

    def test_create_rejects_missing_frontmatter(self, client, isolated_profiles):
        resp = client.post(
            "/api/skills",
            json={"name": "bad-skill", "content": "no frontmatter here"},
        )
        assert resp.status_code == 400
        assert "frontmatter" in resp.json()["detail"].lower()
        assert not (isolated_profiles["default"] / "skills" / "bad-skill").exists()

    def test_create_rejects_duplicate_name(self, client, isolated_profiles):
        resp = client.post(
            "/api/skills",
            json={
                "name": "dashboard-skill",
                "content": SKILL_MD.format(name="dashboard-skill"),
            },
        )
        assert resp.status_code == 400
        assert "already exists" in resp.json()["detail"]

    def test_create_rejects_invalid_name(self, client, isolated_profiles):
        resp = client.post(
            "/api/skills",
            json={"name": "../escape", "content": SKILL_MD.format(name="x")},
        )
        assert resp.status_code == 400


class TestSkillUpdate:
    def test_update_rewrites_skill_md(self, client, isolated_profiles):
        new_content = SKILL_MD.format(name="dashboard-skill").replace(
            "Do the thing.", "Do the NEW thing."
        )
        resp = client.put(
            "/api/skills/content",
            json={"name": "dashboard-skill", "content": new_content},
        )
        assert resp.status_code == 200
        skill_md = (
            isolated_profiles["default"] / "skills" / "dashboard-skill" / "SKILL.md"
        )
        assert "Do the NEW thing." in skill_md.read_text(encoding="utf-8")

    def test_update_unknown_skill_404(self, client, isolated_profiles):
        resp = client.put(
            "/api/skills/content",
            json={"name": "nope", "content": SKILL_MD.format(name="nope")},
        )
        assert resp.status_code == 404

    def test_update_invalid_frontmatter_400(self, client, isolated_profiles):
        resp = client.put(
            "/api/skills/content",
            json={"name": "dashboard-skill", "content": "broken"},
        )
        assert resp.status_code == 400


class TestEditorEndpointsAuth:
    @pytest.mark.parametrize(
        "method,path,kwargs",
        [
            ("get", "/api/skills/content?name=dashboard-skill", {}),
            ("post", "/api/skills", {"json": {"name": "x", "content": "y"}}),
            ("put", "/api/skills/content", {"json": {"name": "x", "content": "y"}}),
        ],
    )
    def test_endpoints_401_without_token(
        self, client, isolated_profiles, method, path, kwargs
    ):
        from hermes_cli.web_server import _SESSION_HEADER_NAME

        client.headers.pop(_SESSION_HEADER_NAME, None)
        resp = getattr(client, method)(path, **kwargs)
        assert resp.status_code == 401


class TestCronJobSkills:
    def test_create_job_with_skills(self, client, isolated_profiles):
        resp = client.post(
            "/api/cron/jobs",
            json={
                "prompt": "do work",
                "schedule": "every 1h",
                "name": "skilled-job",
                "skills": ["dashboard-skill"],
            },
        )
        assert resp.status_code == 200
        job = resp.json()
        assert job["skills"] == ["dashboard-skill"]

        # Round-trip: the list endpoint carries the skills field too.
        listed = client.get("/api/cron/jobs", params={"profile": "default"}).json()
        match = [j for j in listed if j["id"] == job["id"]]
        assert match and match[0]["skills"] == ["dashboard-skill"]

    def test_update_job_skills(self, client, isolated_profiles):
        job = client.post(
            "/api/cron/jobs",
            json={"prompt": "do work", "schedule": "every 1h"},
        ).json()
        assert job.get("skills") in (None, [])

        resp = client.put(
            f"/api/cron/jobs/{job['id']}",
            json={"updates": {"skills": ["dashboard-skill", "worker-skill"]}},
            params={"profile": "default"},
        )
        assert resp.status_code == 200
        assert resp.json()["skills"] == ["dashboard-skill", "worker-skill"]

        # Clearing works too.
        resp = client.put(
            f"/api/cron/jobs/{job['id']}",
            json={"updates": {"skills": []}},
            params={"profile": "default"},
        )
        assert resp.status_code == 200
        assert resp.json()["skills"] == []
