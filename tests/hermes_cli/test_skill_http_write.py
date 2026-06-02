"""HTTP write surface for skill authoring (hermes-202606-010).

Covers ``POST /api/skills`` + ``POST /api/sessions/{id}/promote-skill``.
Both are thin wrappers around the existing ``skill_manage(action='create')``
tool action; tests assert the wrapper preserves validation + write
semantics and exposes the new write surface without leaking session
content into error responses.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest


@pytest.fixture
def web_client(monkeypatch, _isolate_hermes_home):
    """Authenticated TestClient bound to an isolated HERMES_HOME state DB.

    Mirrors ``TestWebServerEndpoints._setup_test_client`` in
    ``tests/hermes_cli/test_web_server.py``. Additionally rebinds the
    module-level ``SKILLS_DIR`` constants captured at import time so
    the per-test temp HERMES_HOME is the actual write root for
    ``skill_manage`` AND the read root for ``_find_all_skills``.
    """
    try:
        from starlette.testclient import TestClient
    except ImportError:  # pragma: no cover — defensive
        pytest.skip("fastapi/starlette not installed")

    import hermes_state
    from hermes_constants import get_hermes_home
    from hermes_cli.web_server import app, _SESSION_HEADER_NAME, _SESSION_TOKEN

    home = get_hermes_home()
    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", home / "state.db")

    # Module-level constants captured ``SKILLS_DIR`` at import — repoint
    # them so writes and discovery both land under the per-test
    # HERMES_HOME instead of the operator's real ``~/.hermes/skills/``.
    import tools.skill_manager_tool as skill_manager_tool
    isolated_skills = home / "skills"
    isolated_skills.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(skill_manager_tool, "SKILLS_DIR", isolated_skills)
    monkeypatch.setattr(skill_manager_tool, "HERMES_HOME", home)
    import tools.skills_tool as skills_tool
    monkeypatch.setattr(skills_tool, "SKILLS_DIR", isolated_skills)
    monkeypatch.setattr(skills_tool, "HERMES_HOME", home)
    try:
        import agent.skill_utils as skill_utils
        monkeypatch.setattr(
            skill_utils, "get_all_skills_dirs", lambda *a, **k: [isolated_skills]
        )
    except Exception:
        pass

    client = TestClient(app)
    client.headers[_SESSION_HEADER_NAME] = _SESSION_TOKEN
    return client


def _skill_md(name: str = "test-skill", description: str = "Test skill body.") -> str:
    return (
        "---\n"
        f"name: {name}\n"
        f"description: {description}\n"
        "---\n\n"
        "# Test Skill\n\n"
        "Body content.\n"
    )


# ---------------------------------------------------------------------------
# POST /api/skills
# ---------------------------------------------------------------------------


class TestCreateSkillEndpoint:
    def test_create_writes_skill_md(self, web_client):
        from hermes_constants import get_hermes_home

        resp = web_client.post(
            "/api/skills",
            json={"name": "ui-created-skill", "content": _skill_md("ui-created-skill")},
        )
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert data["ok"] is True
        assert data["name"] == "ui-created-skill"
        # SKILL.md actually written under the isolated HERMES_HOME
        skill_md_path = Path(data["skill_md"])
        assert skill_md_path.exists()
        assert str(get_hermes_home()) in str(skill_md_path)
        body = skill_md_path.read_text(encoding="utf-8")
        assert "ui-created-skill" in body

    def test_create_appears_in_get_skills(self, web_client):
        web_client.post(
            "/api/skills",
            json={"name": "listed-skill", "content": _skill_md("listed-skill")},
        )
        resp = web_client.get("/api/skills")
        assert resp.status_code == 200
        names = [s["name"] for s in resp.json()]
        assert "listed-skill" in names

    def test_create_foreground_origin_not_agent_created(self, web_client):
        """``POST /api/skills`` is explicit operator authoring. Per the
        existing ``skill_manage`` rule, only background-review creates
        get the ``created_by: agent`` curator tag. The wrapper must NOT
        flip the origin — the resulting usage record must NOT carry
        ``created_by: agent``."""
        web_client.post(
            "/api/skills",
            json={"name": "foreground-skill", "content": _skill_md("foreground-skill")},
        )
        from tools.skill_usage import get_record  # type: ignore

        rec = get_record("foreground-skill")
        assert rec.get("created_by") != "agent"

    def test_create_rejects_missing_frontmatter(self, web_client):
        resp = web_client.post(
            "/api/skills",
            json={"name": "bad", "content": "no frontmatter here"},
        )
        assert resp.status_code == 400
        assert "frontmatter" in resp.json()["detail"].lower()

    def test_create_rejects_duplicate_name(self, web_client):
        body = {"name": "dup-skill", "content": _skill_md("dup-skill")}
        first = web_client.post("/api/skills", json=body)
        assert first.status_code == 200
        second = web_client.post("/api/skills", json=body)
        assert second.status_code == 400
        assert "already exists" in second.json()["detail"]

    def test_create_requires_auth_token(self, web_client):
        from hermes_cli.web_server import app, _SESSION_HEADER_NAME
        from starlette.testclient import TestClient

        unauth = TestClient(app)
        resp = unauth.post(
            "/api/skills",
            json={"name": "noauth", "content": _skill_md("noauth")},
        )
        assert resp.status_code in {401, 403}


# ---------------------------------------------------------------------------
# POST /api/sessions/{id}/promote-skill
# ---------------------------------------------------------------------------


def _seed_session(messages):
    """Insert a session + messages into the isolated SessionDB."""
    from hermes_state import SessionDB

    db = SessionDB()
    try:
        sid = "promote-test-session"
        db.create_session(session_id=sid, source="cli")
        for role, content in messages:
            db.append_message(session_id=sid, role=role, content=content)
        return sid
    finally:
        db.close()


class TestPromoteSkillEndpoint:
    def test_promote_creates_skill_with_agent_created_tag(self, web_client):
        sid = _seed_session([
            ("user", "every monday morning sort emails by sender"),
            ("assistant", "sorting now"),
            ("user", "mark vendor invoices for tax processing"),
        ])
        resp = web_client.post(
            f"/api/sessions/{sid}/promote-skill",
            json={
                "skill_slug": "monday-inbox-sort",
                "skill_name": "Monday Inbox Sort",
            },
        )
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert data["ok"] is True
        assert data["skill_slug"] == "monday-inbox-sort"
        assert data["skill_name"] == "Monday Inbox Sort"
        skill_md = Path(data["skill_md"])
        assert skill_md.exists()
        body = skill_md.read_text(encoding="utf-8")
        # Frontmatter rendered + slug used as `name:`
        assert "name: monday-inbox-sort" in body
        # Only user-turn clips folded in; no assistant content.
        assert "every monday morning" in body
        assert "sorting now" not in body

        # Background-review origin used → curator-managed agent tag set.
        from tools.skill_usage import get_record  # type: ignore

        rec = get_record("monday-inbox-sort")
        assert rec.get("created_by") == "agent"

    def test_promoted_skill_appears_in_get_skills(self, web_client):
        sid = _seed_session([("user", "hi")])
        web_client.post(
            f"/api/sessions/{sid}/promote-skill",
            json={"skill_slug": "from-promote"},
        )
        resp = web_client.get("/api/skills")
        names = [s["name"] for s in resp.json()]
        assert "from-promote" in names

    def test_promote_unknown_session_returns_404(self, web_client):
        resp = web_client.post(
            "/api/sessions/does-not-exist/promote-skill",
            json={"skill_slug": "x"},
        )
        assert resp.status_code == 404

    def test_promote_rejects_empty_slug(self, web_client):
        sid = _seed_session([("user", "hi")])
        resp = web_client.post(
            f"/api/sessions/{sid}/promote-skill",
            json={"skill_slug": "   "},
        )
        assert resp.status_code == 400
        assert "skill_slug" in resp.json()["detail"]

    def test_promote_does_not_leak_session_content_on_error(self, web_client):
        """Sensitive evidence policy: error responses must not include
        transcript content. Trigger a duplicate-name failure on the
        SECOND call and check the error string is the static validator
        message, not the seeded transcript."""
        sid = _seed_session([("user", "confidential-customer-payload-xyz")])
        first = web_client.post(
            f"/api/sessions/{sid}/promote-skill",
            json={"skill_slug": "dup-promoted"},
        )
        assert first.status_code == 200
        second = web_client.post(
            f"/api/sessions/{sid}/promote-skill",
            json={"skill_slug": "dup-promoted"},
        )
        assert second.status_code == 400
        detail = second.json()["detail"]
        assert "already exists" in detail
        assert "confidential-customer-payload-xyz" not in detail
