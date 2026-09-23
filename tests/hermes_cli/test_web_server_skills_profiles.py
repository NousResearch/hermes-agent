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
import hermes_cli.web_server_gateway as _web_server_gateway
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
    return yaml.safe_load((home / "config.yaml").read_text()) or {}


class TestProfileScopedSkills:


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



    def test_scope_restores_module_globals(self, client, isolated_profiles):
        """The SKILLS_DIR swap is per-request; the module global must be
        restored even after a scoped call (cron-style locked swap)."""
        import tools.skills_tool as skills_tool

        before = skills_tool.SKILLS_DIR
        client.get("/api/skills", params={"profile": "worker_alpha"})
        assert skills_tool.SKILLS_DIR == before


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

        monkeypatch.setattr(_web_server_gateway, "_spawn_hermes_action", _fake_spawn)
        resp = client.post(
            "/api/skills/hub/install",
            json={"identifier": "official/demo", "profile": "worker_alpha"},
        )
        assert resp.status_code == 200
        assert calls == [
            (
                ["-p", "worker_alpha", "skills", "install", "official/demo", "--yes"],
                _web_server_profiles._hub_action_name("install", "official/demo"),
            )
        ]


    def test_hub_install_unknown_profile_404(self, client, isolated_profiles):
        resp = client.post(
            "/api/skills/hub/install",
            json={"identifier": "official/demo", "profile": "ghost"},
        )
        assert resp.status_code == 404


    def test_hub_install_scoped_to_default_still_carries_the_selector(
        self, client, isolated_profiles, monkeypatch
    ):
        """``default`` is a real named profile, not an alias for "the dashboard's own".
        A pooled ``hermes -p worker_alpha serve`` dashboard answering a hub action with
        ``profile=default`` must still emit ``-p default``: without it the child's env
        carries this process's home verbatim and the install lands on worker_alpha."""
        calls = []

        class _FakeProc:
            pid = 4242

        def _fake_spawn(subcommand, name):
            calls.append(list(subcommand))
            return _FakeProc()

        # Ambient home of a dashboard serving under the named profile.
        monkeypatch.setenv("HERMES_HOME", str(isolated_profiles["worker_alpha"]))
        monkeypatch.setattr(_web_server_gateway, "_spawn_hermes_action", _fake_spawn)
        resp = client.post(
            "/api/skills/hub/install",
            json={"identifier": "official/demo", "profile": "default"},
        )
        assert resp.status_code == 200
        assert calls == [
            ["-p", "default", "skills", "install", "official/demo", "--yes"]
        ]
        # The spawn-path env pin must land the child on the default home, not the
        # ambient one it would inherit from a selector-less argv.
        env = _web_server_gateway._profile_action_environment(calls[0])
        assert env["HERMES_HOME"] == str(isolated_profiles["default"])


    @pytest.mark.parametrize("profile", ["Default", "DEFAULT"])
    def test_hub_install_mixed_case_default_is_rejected_not_ambient(
        self, client, isolated_profiles, monkeypatch, profile
    ):
        """A non-canonical casing is not a valid profile id: the request 400s like any
        mixed-case name instead of silently falling back to the ambient home."""
        calls = []

        class _FakeProc:
            pid = 4242

        def _fake_spawn(subcommand, name):
            calls.append(list(subcommand))
            return _FakeProc()

        monkeypatch.setattr(_web_server_gateway, "_spawn_hermes_action", _fake_spawn)
        resp = client.post(
            "/api/skills/hub/install",
            json={"identifier": "official/demo", "profile": profile},
        )
        assert resp.status_code == 400
        assert calls == []


    @pytest.mark.parametrize("profile", [None, "", "  ", "current", "CURRENT"])
    def test_hub_install_without_a_named_profile_keeps_ambient_argv(
        self, client, isolated_profiles, monkeypatch, profile
    ):
        """Only a real profile name earns ``-p``: unnamed/``current`` requests keep
        the selector-less argv so the child resolves the dashboard's own home."""
        calls = []

        class _FakeProc:
            pid = 4242

        def _fake_spawn(subcommand, name):
            calls.append(list(subcommand))
            return _FakeProc()

        monkeypatch.setattr(_web_server_gateway, "_spawn_hermes_action", _fake_spawn)
        resp = client.post(
            "/api/skills/hub/install",
            json={"identifier": "official/demo", "profile": profile},
        )
        assert resp.status_code == 200
        assert calls == [["skills", "install", "official/demo", "--yes"]]
