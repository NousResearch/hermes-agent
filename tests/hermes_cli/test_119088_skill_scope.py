"""RED probes for issue #119088 (temporary; removed or renamed once verified).

Symptom 1: Desktop skill toggle for a selected profile must write ONLY that
profile's config.yaml, never the global default config.
Symptom 2: a fresh profile must inherit the global skills.disabled list.
"""
import pytest
import yaml
import hermes_cli.web_server_profiles as _web_server_profiles


def _write_skill(skills_dir, name, description="test skill"):
    d = skills_dir / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: {description}\n---\n\n# {name}\n",
        encoding="utf-8",
    )


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


def _load_cfg(home):
    p = home / "config.yaml"
    if not p.exists():
        return {}
    return yaml.safe_load(p.read_text()) or {}


class TestToggleQueryProfile119088:
    def test_toggle_with_query_profile_writes_target_only(self, client, isolated_profiles):
        """Desktop-shaped request: profile in the query string, body carries
        only name/enabled (no body.profile). Must land in worker_alpha."""
        resp = client.put(
            "/api/skills/toggle",
            params={"profile": "worker_alpha"},
            json={"name": "worker-skill", "enabled": False},
        )
        assert resp.status_code == 200
        assert resp.json() == {"ok": True, "name": "worker-skill", "enabled": False}

        worker_cfg = _load_cfg(isolated_profiles["worker_alpha"])
        assert "worker-skill" in worker_cfg.get("skills", {}).get("disabled", [])
        default_cfg = _load_cfg(isolated_profiles["default"])
        assert "worker-skill" not in default_cfg.get("skills", {}).get("disabled", [])


class TestFreshProfileInheritsDisabled119088:
    def test_fresh_profile_inherits_global_disabled(self, isolated_profiles, monkeypatch):
        """Globally disabled skills must stay disabled in a fresh profile."""
        from hermes_cli import profiles
        from hermes_cli.skills_config import get_disabled_skills

        default_home = isolated_profiles["default"]
        (default_home / "config.yaml").write_text(
            "skills:\n  disabled:\n    - airtable\n    - notion\n", encoding="utf-8"
        )
        monkeypatch.setattr(
            profiles, "seed_profile_skills", lambda path, quiet=True: None
        )

        new_path = profiles.create_profile("fresh_writer")
        assert new_path.exists()
        fresh_cfg = _load_cfg(new_path)
        fresh_disabled = get_disabled_skills(fresh_cfg)
        assert {"airtable", "notion"} <= fresh_disabled, fresh_cfg

    def test_fresh_profile_without_global_disabled_has_no_skills_section(
        self, isolated_profiles, monkeypatch
    ):
        """No global disabled list -> fresh profile owns no skills section,
        exactly as before (no empty stub written)."""
        from hermes_cli import profiles

        monkeypatch.setattr(
            profiles, "seed_profile_skills", lambda path, quiet=True: None
        )

        new_path = profiles.create_profile("fresh_plain")
        assert new_path.exists()
        assert "skills" not in _load_cfg(new_path)
