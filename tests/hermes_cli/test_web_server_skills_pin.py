"""The Capabilities Skills pane's pin endpoint (``PUT /api/skills/pin``).

The desktop pin button is the only way to protect a learned skill from the
curator's inactivity rule without a terminal, so the contract that matters is
that it writes the SAME ``.usage.json`` flag ``hermes curator pin`` writes (and
refuses what the CLI refuses) rather than a parallel UI-only state.

Regression for the missing pin control: before this route existed the pane
could only read the flag (once ``GET /api/skills`` exposed it) and the user had
to drop to the CLI to set it.
"""
import json

import pytest


LEARNED = "learned-skill"
BUNDLED = "bundled-skill"

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
def home(tmp_path, monkeypatch, _isolate_hermes_home):
    """Isolated home with one learned skill and one built-in.

    The built-in is faked through ``.bundled_manifest`` — the same file the
    installer writes — because bundled-ness is a name-set lookup, not a path.
    """
    from hermes_constants import get_hermes_home

    home = get_hermes_home()
    skills = home / "skills"
    skills.mkdir(parents=True, exist_ok=True)
    (home / "config.yaml").write_text("{}\n", encoding="utf-8")

    _write_skill(skills, LEARNED)
    _write_skill(skills, BUNDLED)
    (skills / ".bundled_manifest").write_text(f"{BUNDLED}:0\n", encoding="utf-8")
    return home


@pytest.fixture
def client(monkeypatch, home):
    try:
        from starlette.testclient import TestClient
    except ImportError:
        pytest.skip("fastapi/starlette not installed")

    import hermes_state
    from hermes_cli.web_server import _SESSION_HEADER_NAME, _SESSION_TOKEN, app

    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", home / "state.db")
    c = TestClient(app)
    c.headers[_SESSION_HEADER_NAME] = _SESSION_TOKEN
    return c


def _record(home, name):
    """The skill's record as it sits on disk (real I/O — the point of the test).

    A missing file is a legitimate answer for the refusal case: nothing was
    written, so there is nothing to read.
    """
    path = home / "skills" / ".usage.json"
    if not path.exists():
        return {}
    usage = json.loads(path.read_text(encoding="utf-8-sig"))
    return usage.get(name, {})


def _pin(client, name, pinned):
    resp = client.put("/api/skills/pin", json={"name": name, "pinned": pinned})
    assert resp.status_code == 200, resp.text
    return resp.json()


class TestSkillPin:
    def test_pin_round_trips_to_the_usage_record(self, client, home):
        """Pin then unpin: the flag lands in `.usage.json`, so `hermes curator
        status` (which reads the same file) agrees with what the pane shows."""
        from tools import skill_usage

        skill_usage.mark_agent_created(LEARNED)

        on = _pin(client, LEARNED, True)
        assert on["ok"] is True and on["pinned"] is True and on["managed"] is True
        assert _record(home, LEARNED)["pinned"] is True

        off = _pin(client, LEARNED, False)
        assert off["ok"] is True and off["pinned"] is False
        assert _record(home, LEARNED)["pinned"] is False

    def test_list_endpoint_serves_the_flag_written_by_the_cli(self, client, home):
        """One source of truth: a pin written the CLI way (``set_pinned``) is what
        ``GET /api/skills`` reports, not a mirrored copy."""
        from tools import skill_usage

        skill_usage.mark_agent_created(LEARNED)
        assert skill_usage.set_pinned(LEARNED, True) is True

        rows = {s["name"]: s for s in client.get("/api/skills").json()}
        assert rows[LEARNED]["pinned"] is True
        assert rows[BUNDLED]["pinned"] is False

    def test_pin_refuses_a_built_in_and_writes_nothing(self, client, home):
        """A bundled skill is managed by its source: refuse with a reason the pane
        can roll back on, and leave no half-written pin behind."""
        result = _pin(client, BUNDLED, True)

        assert result["ok"] is False
        assert result["reason"] == "not_agent_created"
        assert result["message"]
        assert not _record(home, BUNDLED).get("pinned")

    def test_pin_reports_an_unmanaged_skill_as_recorded_but_inert(self, client, home):
        """Curation-eligible but not curator-managed (no `created_by` marker, i.e.
        never adopted): the pin is stored, and the answer says so — mirroring
        `hermes curator pin`'s "recorded, but unmanaged" note instead of promising
        protection the curator never had to lift."""
        result = _pin(client, LEARNED, True)

        assert result["ok"] is True and result["pinned"] is True
        assert result["managed"] is False
        assert "recorded" in result["message"].lower()
        assert _record(home, LEARNED)["pinned"] is True
