"""Regression tests for dashboard profile-scoped skills/toolsets management.

"Set as active" on the Profiles page only flips the sticky ``active_profile``
file (future CLI/gateway runs) — it never retargets the running dashboard
process. Before the ``profile`` parameter existed, toggling a skill after
"activating" a profile silently wrote into the dashboard's own config.
These tests pin the new behavior: reads and writes land in the REQUESTED
profile's HERMES_HOME, and the dashboard's own profile stays untouched.
"""
import pytest
import yaml


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
    return yaml.safe_load((home / "config.yaml").read_text()) or {}


class TestProfileScopedSkills:
    def test_skills_list_scopes_to_requested_profile(self, client, isolated_profiles):
        resp = client.get("/api/skills", params={"profile": "worker_alpha"})
        assert resp.status_code == 200
        names = {s["name"] for s in resp.json()}
        assert "worker-skill" in names
        assert "dashboard-skill" not in names

    def test_skills_list_without_profile_uses_dashboard_home(
        self, client, isolated_profiles
    ):
        resp = client.get("/api/skills")
        assert resp.status_code == 200
        names = {s["name"] for s in resp.json()}
        assert "dashboard-skill" in names
        assert "worker-skill" not in names

    def test_toggle_writes_into_target_profile_only(self, client, isolated_profiles):
        resp = client.put(
            "/api/skills/toggle",
            json={"name": "worker-skill", "enabled": False, "profile": "worker_alpha"},
        )
        assert resp.status_code == 200
        assert resp.json() == {"ok": True, "name": "worker-skill", "enabled": False}

        worker_cfg = _load_cfg(isolated_profiles["worker_alpha"])
        assert "worker-skill" in worker_cfg.get("skills", {}).get("disabled", [])
        # The dashboard's own config must stay untouched — this was the bug.
        default_cfg = _load_cfg(isolated_profiles["default"])
        assert "worker-skill" not in default_cfg.get("skills", {}).get("disabled", [])

    def test_toggle_reenable_round_trip(self, client, isolated_profiles):
        for enabled in (False, True):
            client.put(
                "/api/skills/toggle",
                json={
                    "name": "worker-skill",
                    "enabled": enabled,
                    "profile": "worker_alpha",
                },
            )
        worker_cfg = _load_cfg(isolated_profiles["worker_alpha"])
        assert "worker-skill" not in worker_cfg.get("skills", {}).get("disabled", [])

    def test_unknown_profile_returns_404(self, client, isolated_profiles):
        resp = client.get("/api/skills", params={"profile": "no_such_profile"})
        assert resp.status_code == 404

    def test_invalid_profile_name_returns_400(self, client, isolated_profiles):
        resp = client.get("/api/skills", params={"profile": "Bad Name!"})
        assert resp.status_code == 400

    def test_scope_restores_module_globals(self, client, isolated_profiles):
        """The SKILLS_DIR swap is per-request; the module global must be
        restored even after a scoped call (cron-style locked swap)."""
        import tools.skills_tool as skills_tool

        before = skills_tool.SKILLS_DIR
        client.get("/api/skills", params={"profile": "worker_alpha"})
        assert skills_tool.SKILLS_DIR == before


class TestProfileScopedToolsets:
    def test_toolset_toggle_scopes_to_profile(self, client, isolated_profiles):
        resp = client.put(
            "/api/tools/toolsets/x_search",
            json={"enabled": True, "profile": "worker_alpha"},
        )
        assert resp.status_code == 200

        worker_cfg = _load_cfg(isolated_profiles["worker_alpha"])
        assert "x_search" in worker_cfg.get("platform_toolsets", {}).get("cli", [])
        default_cfg = _load_cfg(isolated_profiles["default"])
        assert "x_search" not in default_cfg.get("platform_toolsets", {}).get("cli", [])

        listing = client.get(
            "/api/tools/toolsets", params={"profile": "worker_alpha"}
        ).json()
        assert {t["name"]: t for t in listing}["x_search"]["enabled"] is True
        # Unscoped listing reflects the dashboard's own (untouched) config.
        listing = client.get("/api/tools/toolsets").json()
        assert {t["name"]: t for t in listing}["x_search"]["enabled"] is False

    def test_toolset_toggle_unknown_profile_404(self, client, isolated_profiles):
        resp = client.put(
            "/api/tools/toolsets/x_search",
            json={"enabled": True, "profile": "ghost"},
        )
        assert resp.status_code == 404


class TestProfileScopedHubActions:
    def test_hub_install_spawns_with_profile_flag(
        self, client, isolated_profiles, monkeypatch
    ):
        """Hub installs must go through a fresh ``hermes -p <profile>``
        subprocess — the in-process scope can't reach skills_hub's
        import-time SKILLS_DIR binding."""
        import hermes_cli.web_server as web_server

        calls = []

        class _FakeProc:
            pid = 4242

        def _fake_spawn(subcommand, name):
            calls.append((list(subcommand), name))
            return _FakeProc()

        monkeypatch.setattr(web_server, "_spawn_hermes_action", _fake_spawn)
        resp = client.post(
            "/api/skills/hub/install",
            json={"identifier": "official/demo", "profile": "worker_alpha"},
        )
        assert resp.status_code == 200
        assert calls == [
            (["-p", "worker_alpha", "skills", "install", "official/demo"], "skills-install")
        ]

    def test_hub_install_without_profile_keeps_legacy_argv(
        self, client, isolated_profiles, monkeypatch
    ):
        import hermes_cli.web_server as web_server

        calls = []

        class _FakeProc:
            pid = 4242

        monkeypatch.setattr(
            web_server,
            "_spawn_hermes_action",
            lambda subcommand, name: calls.append(list(subcommand)) or _FakeProc(),
        )
        resp = client.post(
            "/api/skills/hub/install", json={"identifier": "official/demo"}
        )
        assert resp.status_code == 200
        assert calls == [["skills", "install", "official/demo"]]

    def test_hub_install_unknown_profile_404(self, client, isolated_profiles):
        resp = client.post(
            "/api/skills/hub/install",
            json={"identifier": "official/demo", "profile": "ghost"},
        )
        assert resp.status_code == 404
