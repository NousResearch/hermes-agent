"""Tests for the dashboard skill editor endpoints and cron skill attachment.

The Skills page can now create/edit custom skills (SKILL.md) and the Cron
page can attach skills to jobs — closing the "SSH + nano is the only way"
gap for headless/VPS users. These tests pin:

- GET /api/skills/files lists the allowlisted multi-file skill package.
- GET /api/skills/content returns raw SKILL.md or supporting text files.
- POST /api/skills creates a skill through the same validated write path
  as the agent's ``skill_manage`` tool (frontmatter validation enforced).
- PUT /api/skills/content rewrites SKILL.md or a validated support file.
- DELETE /api/skills/file removes support files but never SKILL.md.
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
    return d


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

    dashboard_skill = _write_skill(default_home / "skills", "dashboard-skill")
    (dashboard_skill / "references" / "nested").mkdir(parents=True)
    (dashboard_skill / "references" / "nested" / "guide.md").write_text(
        "# Nested guide\n", encoding="utf-8"
    )
    (dashboard_skill / "templates").mkdir()
    (dashboard_skill / "templates" / "report.md").write_text(
        "# Report\n", encoding="utf-8"
    )
    (dashboard_skill / "scripts" / "nested").mkdir(parents=True)
    (dashboard_skill / "scripts" / "nested" / "check.py").write_text(
        "print('ok')\n", encoding="utf-8"
    )
    (dashboard_skill / "assets").mkdir()
    (dashboard_skill / "assets" / "logo.png").write_bytes(b"\x89PNG\x00binary")
    (dashboard_skill / "notes.txt").write_text("not package content", encoding="utf-8")

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

    def test_get_nested_support_file(self, client, isolated_profiles):
        resp = client.get(
            "/api/skills/content",
            params={
                "name": "dashboard-skill",
                "file_path": "references/nested/guide.md",
            },
        )
        assert resp.status_code == 200
        assert resp.json()["file_path"] == "references/nested/guide.md"
        assert resp.json()["content"] == "# Nested guide\n"

    def test_get_binary_support_file_415(self, client, isolated_profiles):
        resp = client.get(
            "/api/skills/content",
            params={"name": "dashboard-skill", "file_path": "assets/logo.png"},
        )
        assert resp.status_code == 415
        assert "binary" in resp.json()["detail"].lower()

    @pytest.mark.parametrize(
        "file_path",
        ["../config.yaml", "notes.txt", "references/../../config.yaml"],
    )
    def test_get_rejects_paths_outside_support_dirs(
        self, client, isolated_profiles, file_path
    ):
        resp = client.get(
            "/api/skills/content",
            params={"name": "dashboard-skill", "file_path": file_path},
        )
        assert resp.status_code == 400


class TestSkillFiles:
    def test_lists_complete_grouped_package(self, client, isolated_profiles):
        resp = client.get("/api/skills/files", params={"name": "dashboard-skill"})
        assert resp.status_code == 200
        files = {item["path"]: item for item in resp.json()["files"]}

        assert files["SKILL.md"] == {
            "path": "SKILL.md",
            "kind": "skill",
            "is_binary": False,
        }
        assert files["references/nested/guide.md"]["kind"] == "references"
        assert files["scripts/nested/check.py"]["kind"] == "scripts"
        assert files["assets/logo.png"]["is_binary"] is True
        assert "notes.txt" not in files

    def test_list_scopes_to_profile(self, client, isolated_profiles):
        resp = client.get(
            "/api/skills/files",
            params={"name": "worker-skill", "profile": "worker_alpha"},
        )
        assert resp.status_code == 200
        assert [item["path"] for item in resp.json()["files"]] == ["SKILL.md"]

        assert client.get(
            "/api/skills/files", params={"name": "worker-skill"}
        ).status_code == 404


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

    def test_update_nested_support_file(self, client, isolated_profiles):
        resp = client.put(
            "/api/skills/content",
            json={
                "name": "dashboard-skill",
                "file_path": "references/nested/guide.md",
                "content": "# Updated guide\n",
            },
        )
        assert resp.status_code == 200
        target = (
            isolated_profiles["default"]
            / "skills"
            / "dashboard-skill"
            / "references"
            / "nested"
            / "guide.md"
        )
        assert target.read_text(encoding="utf-8") == "# Updated guide\n"

    def test_update_bundled_skill_is_read_only(self, client, isolated_profiles):
        manifest = isolated_profiles["default"] / "skills" / ".bundled_manifest"
        manifest.write_text("dashboard-skill:abc123\n", encoding="utf-8")

        resp = client.put(
            "/api/skills/content",
            json={
                "name": "dashboard-skill",
                "file_path": "references/nested/guide.md",
                "content": "blocked",
            },
        )
        assert resp.status_code == 403
        assert "read-only" in resp.json()["detail"]

    def test_update_binary_support_file_415(self, client, isolated_profiles):
        resp = client.put(
            "/api/skills/content",
            json={
                "name": "dashboard-skill",
                "file_path": "assets/logo.png",
                "content": "not a png",
            },
        )
        assert resp.status_code == 415
        assert "binary" in resp.json()["detail"].lower()


class TestSkillFileDelete:
    def test_delete_support_file(self, client, isolated_profiles):
        target = (
            isolated_profiles["default"]
            / "skills"
            / "dashboard-skill"
            / "templates"
            / "report.md"
        )
        resp = client.request(
            "DELETE",
            "/api/skills/file",
            json={"name": "dashboard-skill", "file_path": "templates/report.md"},
        )
        assert resp.status_code == 200
        assert not target.exists()

    def test_delete_skill_md_is_refused(self, client, isolated_profiles):
        resp = client.request(
            "DELETE",
            "/api/skills/file",
            json={"name": "dashboard-skill", "file_path": "SKILL.md"},
        )
        assert resp.status_code == 400
        assert "archive" in resp.json()["detail"].lower()
        assert (
            isolated_profiles["default"] / "skills" / "dashboard-skill" / "SKILL.md"
        ).exists()


class TestEditorEndpointsAuth:
    @pytest.mark.parametrize(
        "method,path,kwargs",
        [
            ("get", "/api/skills/files?name=dashboard-skill", {}),
            ("get", "/api/skills/content?name=dashboard-skill", {}),
            ("post", "/api/skills", {"json": {"name": "x", "content": "y"}}),
            ("put", "/api/skills/content", {"json": {"name": "x", "content": "y"}}),
            (
                "delete",
                "/api/skills/file",
                {"json": {"name": "x", "file_path": "references/x.md"}},
            ),
        ],
    )
    def test_endpoints_401_without_token(
        self, client, isolated_profiles, method, path, kwargs
    ):
        from hermes_cli.web_server import _SESSION_HEADER_NAME

        client.headers.pop(_SESSION_HEADER_NAME, None)
        resp = client.request(method.upper(), path, **kwargs)
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
