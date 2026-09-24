"""The dashboard editor keeps operator authority over locked regions.

The agent-facing guard in ``tools/skill_manager_tool`` refuses any self-patch that touches an
``<!-- operator-locked -->`` region. The dashboard calls ``_create_skill`` / ``_edit_skill``
DIRECTLY (``hermes_cli/web_routers/skills.py``) — "an authenticated dashboard write IS the
user" — so without an explicit operator scope the guard would also refuse the operator, and
the documented lift path ("git or the dashboard editor") would be git only.

Upstream review of NousResearch/hermes-agent#51258 raised exactly this. These tests pin the
fix end to end, through the real HTTP endpoints.
"""
import pytest


LOCKED_MD = """---
name: {name}
description: a locked test skill
---

# {name}

## Calibration

Nothing yet.

<!-- operator-locked -->
- Keep the executor's --wait-seconds at 30; treat 120s as cron-unsafe.
<!-- /operator-locked -->
"""

LIFTED_MD = """---
name: {name}
description: a locked test skill
---

# {name}

## Calibration

Nothing yet.

- Wait-seconds policy retired by the operator on 2026-09-20.
"""


def _write_skill(skills_dir, name, template=LOCKED_MD):
    d = skills_dir / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "SKILL.md").write_text(template.format(name=name), encoding="utf-8")
    return d


@pytest.fixture
def isolated_profiles(tmp_path, monkeypatch, _isolate_hermes_home):
    from hermes_constants import get_hermes_home
    from hermes_cli import profiles

    default_home = get_hermes_home()
    profiles_root = default_home / "profiles"
    (default_home / "skills").mkdir(parents=True, exist_ok=True)
    (default_home / "config.yaml").write_text("{}\n", encoding="utf-8")
    _write_skill(default_home / "skills", "locked-skill")

    monkeypatch.setattr(profiles, "_get_default_hermes_home", lambda: default_home)
    monkeypatch.setattr(profiles, "_get_profiles_root", lambda: profiles_root)
    return {"default": default_home}


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


class TestDashboardOperatorAuthority:
    def test_editor_can_lift_a_lock(self, client, isolated_profiles):
        """The flow the guard exists to preserve: an operator retires a locked rule."""
        resp = client.put("/api/skills/content",
                          json={"name": "locked-skill", "content": LIFTED_MD.format(name="locked-skill")})
        assert resp.status_code == 200, resp.text
        md = (isolated_profiles["default"] / "skills" / "locked-skill" / "SKILL.md").read_text()
        assert "operator-locked" not in md
        assert "retired by the operator" in md

    def test_editor_can_rewrite_inside_a_locked_region(self, client, isolated_profiles):
        changed = LOCKED_MD.format(name="locked-skill").replace(
            "treat 120s as cron-unsafe", "treat 90s as the new ceiling")
        resp = client.put("/api/skills/content", json={"name": "locked-skill", "content": changed})
        assert resp.status_code == 200, resp.text
        md = (isolated_profiles["default"] / "skills" / "locked-skill" / "SKILL.md").read_text()
        assert "treat 90s as the new ceiling" in md

    def test_editor_can_author_a_new_locked_skill(self, client, isolated_profiles):
        resp = client.post("/api/skills",
                           json={"name": "new-locked", "content": LOCKED_MD.format(name="new-locked")})
        assert resp.status_code == 200, resp.text
        md = (isolated_profiles["default"] / "skills" / "new-locked" / "SKILL.md").read_text()
        assert "<!-- operator-locked -->" in md

    def test_agent_path_is_still_refused_after_a_dashboard_write(self, client, isolated_profiles):
        """Authority must not leak: the tool path stays guarded in the same process."""
        assert client.post(
            "/api/skills",
            json={"name": "new-locked", "content": LOCKED_MD.format(name="new-locked")}
        ).status_code == 200

        from tools.skill_manager_tool import _edit_skill
        reversed_md = LOCKED_MD.format(name="new-locked").replace(
            "treat 120s as cron-unsafe", "120s is fine, my runs prove it")
        result = _edit_skill("new-locked", reversed_md)
        assert result["success"] is False
        assert "operator-locked" in result["error"]
        md = (isolated_profiles["default"] / "skills" / "new-locked" / "SKILL.md").read_text()
        assert "treat 120s as cron-unsafe" in md
