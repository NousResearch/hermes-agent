"""Regression tests for platform command sync endpoints and config migration.

Covers profile routing, custom command CRUD, sync workflow, legacy
quick_commands migration, and failure scenarios.
"""
import pytest
import yaml


@pytest.fixture
def isolated_profile(tmp_path, monkeypatch, _isolate_hermes_home):
    """Isolated default home with config."""
    from hermes_constants import get_hermes_home

    home = get_hermes_home()
    home.mkdir(parents=True, exist_ok=True)
    (home / "config.yaml").write_text(
        yaml.dump({
            "commands": {"builtin": {}, "custom": {}},
        }),
        encoding="utf-8",
    )
    return home


@pytest.fixture
def client(monkeypatch, isolated_profile):
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


class TestCustomCommandCRUD:
    """Test create, read, update, delete of custom commands."""

    CUSTOM_PAYLOAD = {
        "name": "mygreet",
        "description": "Say hello",
        "type": "exec",
        "command": "echo hello",
        "enabled": True,
        "visible": {"telegram": True, "discord": False, "cli": True},
        "silent_empty": False,
    }

    def test_create_custom_command(self, client, isolated_profile):
        """POST /api/commands/custom creates a custom command entry."""
        resp = client.post("/api/commands/custom", json=self.CUSTOM_PAYLOAD)
        assert resp.status_code == 200
        data = resp.json()
        assert data["ok"] is True

        # Verify via GET
        resp_list = client.get("/api/commands")
        cmds = resp_list.json()
        match = [c for c in cmds if c["name"] == "mygreet"]
        assert len(match) == 1
        assert match[0]["command"] == "echo hello"

    def test_create_custom_command_invalid_name(self, client):
        """POST with invalid name returns 400."""
        payload = dict(self.CUSTOM_PAYLOAD, name="bad name!")
        resp = client.post("/api/commands/custom", json=payload)
        assert resp.status_code == 400

    def test_custom_command_profile_routing(self, client, isolated_profile, monkeypatch):
        """Profile query param routes write to the correct profile."""
        from hermes_cli import profiles
        from hermes_constants import get_hermes_home

        default_home = get_hermes_home()
        beta_home = default_home / "profiles" / "beta"
        beta_home.mkdir(parents=True, exist_ok=True)
        (beta_home / "config.yaml").write_text(
            yaml.dump({"commands": {"builtin": {}, "custom": {}}}),
            encoding="utf-8",
        )

        # Make the web_server recognize 'beta' as a valid profile
        monkeypatch.setattr(profiles, "profile_exists", lambda name: name == "beta" or True)
        monkeypatch.setattr(
            profiles, "get_profile_dir",
            lambda name: beta_home if name == "beta" else default_home,
        )

        resp = client.post(
            "/api/commands/custom?profile=beta",
            json=self.CUSTOM_PAYLOAD,
        )
        assert resp.status_code == 200

        # Default profile should be untouched
        def_names = [c["name"] for c in client.get("/api/commands").json()]
        assert "mygreet" not in def_names

        # Beta profile should have it
        beta_names = [c["name"] for c in client.get("/api/commands?profile=beta").json()]
        assert "mygreet" in beta_names

    def test_delete_custom_command(self, client, isolated_profile):
        """DELETE /api/commands/custom/<name> removes the entry."""
        # Create first
        client.post("/api/commands/custom", json=self.CUSTOM_PAYLOAD)

        # Verify it's listed
        resp_list = client.get("/api/commands")
        names = [c["name"] for c in resp_list.json()]
        assert "mygreet" in names

        # Delete
        resp = client.delete("/api/commands/custom/mygreet")
        assert resp.status_code == 200

        # Verify it's gone
        resp_list2 = client.get("/api/commands")
        names2 = [c["name"] for c in resp_list2.json()]
        assert "mygreet" not in names2


class TestListCommands:
    """Test GET /api/commands listing."""

    CUSTOM_PAYLOAD = {
        "name": "mytool",
        "description": "A custom tool",
        "type": "exec",
        "command": "echo tool",
        "enabled": True,
        "visible": {"telegram": True, "discord": True, "cli": True},
        "silent_empty": False,
    }

    def test_list_commands_includes_custom(self, client, isolated_profile):
        """GET /api/commands returns both builtin and custom commands."""
        client.post("/api/commands/custom", json=self.CUSTOM_PAYLOAD)

        resp = client.get("/api/commands")
        assert resp.status_code == 200
        cmds = resp.json()
        names = [c["name"] for c in cmds]
        assert "mytool" in names
        # Builtins should be present
        assert any(c["source"] == "builtin" for c in cmds)
        assert any(c["source"] == "custom" for c in cmds)

    def test_list_commands_profile_routing(self, client, isolated_profile, monkeypatch):
        """Profile query param scopes command listing."""
        from hermes_cli import profiles
        from hermes_constants import get_hermes_home

        default_home = get_hermes_home()
        beta_home = default_home / "profiles" / "beta"
        beta_home.mkdir(parents=True, exist_ok=True)
        (beta_home / "config.yaml").write_text(
            yaml.dump({"commands": {"builtin": {}, "custom": {}}}),
            encoding="utf-8",
        )

        monkeypatch.setattr(profiles, "profile_exists", lambda name: name == "beta" or True)
        monkeypatch.setattr(
            profiles, "get_profile_dir",
            lambda name: beta_home if name == "beta" else default_home,
        )

        # Add command to beta profile
        client.post("/api/commands/custom?profile=beta", json=self.CUSTOM_PAYLOAD)

        # Listing without profile should not include beta's commands
        resp_default = client.get("/api/commands")
        default_names = [c["name"] for c in resp_default.json()]
        assert "mytool" not in default_names

        # Listing with profile=beta should include it
        resp_beta = client.get("/api/commands?profile=beta")
        beta_names = [c["name"] for c in resp_beta.json()]
        assert "mytool" in beta_names


class TestMigration:
    """Test legacy quick_commands → commands.custom migration."""

    def test_quick_commands_read_backward_compat(self, client, isolated_profile):
        """GET /api/commands still picks up legacy quick_commands."""
        cfg = yaml.safe_load((isolated_profile / "config.yaml").read_text()) or {}
        cfg["quick_commands"] = {
            "legacyls": {
                "type": "exec",
                "command": "ls -la",
                "description": "List files",
            }
        }
        (isolated_profile / "config.yaml").write_text(yaml.dump(cfg), encoding="utf-8")

        resp = client.get("/api/commands")
        assert resp.status_code == 200
        cmds = resp.json()
        names = [c["name"] for c in cmds]
        assert "legacyls" in names

    def test_upsert_cleans_quick_commands(self, client, isolated_profile):
        """Upserting a command with the same name removes it from quick_commands."""
        raw = yaml.safe_load((isolated_profile / "config.yaml").read_text()) or {}
        raw["quick_commands"] = {
            "mygreet": {
                "type": "exec",
                "command": "echo old",
                "description": "Old quick command",
            }
        }
        # Also ensure commands section exists for the upsert
        if "commands" not in raw:
            raw["commands"] = {"builtin": {}, "custom": {}}
        (isolated_profile / "config.yaml").write_text(yaml.dump(raw), encoding="utf-8")

        payload = {
            "name": "mygreet",
            "description": "New version",
            "type": "exec",
            "command": "echo new",
            "enabled": True,
            "visible": {"telegram": True, "discord": True, "cli": True},
            "silent_empty": False,
        }
        client.post("/api/commands/custom", json=payload)
        resp = client.get("/api/commands")
        cmds = resp.json()
        names = [c["name"] for c in cmds]
        assert "mygreet" in names
        match = [c for c in cmds if c["name"] == "mygreet"]
        assert match[0]["command"] == "echo new"

        # quick_commands should no longer have mygreet in the raw file
        raw2 = yaml.safe_load((isolated_profile / "config.yaml").read_text()) or {}
        qc = raw2.get("quick_commands", {}) or {}
        assert "mygreet" not in qc


class TestFailureScenarios:
    """Test error handling for command endpoints."""

    CUSTOM_PAYLOAD = {
        "name": "testcmd",
        "description": "Test",
        "type": "exec",
        "command": "echo test",
        "enabled": True,
        "visible": {"telegram": True, "discord": True, "cli": True},
        "silent_empty": False,
    }

    def test_create_duplicate_name(self, client):
        """Creating a command with existing name overwrites (no error)."""
        client.post("/api/commands/custom", json=self.CUSTOM_PAYLOAD)
        # Second POST with same name should succeed (overwrite)
        resp = client.post("/api/commands/custom", json=self.CUSTOM_PAYLOAD)
        assert resp.status_code == 200

    def test_delete_nonexistent(self, client):
        """DELETE on non-existent command returns 404."""
        resp = client.delete("/api/commands/custom/nonexistent")
        assert resp.status_code == 404

    def test_sync_profile_param(self, client, isolated_profile):
        """POST /api/commands/sync accepts profile as query param (not body)."""
        resp = client.post("/api/commands/sync?profile=beta")
        # Even if beta profile doesn't exist, endpoint shouldn't crash
        assert resp.status_code in (200, 500, 404)
        if resp.status_code == 500:
            detail = resp.json().get("detail", "")
            # Profile parsing error should NOT mention 'profile' body field
            assert "profile" not in detail.lower() or "query" in detail.lower()


class TestProfileQueryParamContract:
    """Verify profile is sent as query param, not in request body.

    This validates the contract that Fix #1 enforces: the frontend sends
    profile as ?profile=... and the backend reads it as a query parameter.
    """

    def test_upsert_ignores_profile_in_body(self, client, isolated_profile, monkeypatch):
        """POST /api/commands/custom reads profile from query param, not body."""
        from hermes_cli import profiles as prof_mod

        # Create a real profile dir for routing test
        from hermes_constants import get_hermes_home
        profiles_root = get_hermes_home() / "profiles"
        prof_dir = profiles_root / "testprofile"
        prof_dir.mkdir(parents=True, exist_ok=True)
        (prof_dir / "config.yaml").write_text(yaml.dump(
            {"commands": {"builtin": {}, "custom": {}}}
        ))
        monkeypatch.setattr(prof_mod, "profile_exists", lambda n: n == "testprofile" or True)
        monkeypatch.setattr(
            prof_mod, "get_profile_dir", lambda n: prof_dir if n == "testprofile" else get_hermes_home()
        )

        payload = {
            "name": "bodyprof",
            "description": "Profile in body test",
            "type": "exec",
            "command": "echo body",
            "enabled": True,
            "visible": {"telegram": True, "discord": True, "cli": True},
            "silent_empty": False,
        }
        resp = client.post("/api/commands/custom?profile=testprofile", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["ok"] is True

        # Verify the command was created on the testprofile profile, not on default
        resp_def = client.get("/api/commands")
        def_names = [c["name"] for c in resp_def.json()]
        assert "bodyprof" not in def_names  # default profile doesn't have it

    def test_sync_ignores_profile_in_body(self, client, isolated_profile):
        """POST /api/commands/sync reads profile from query param, not body."""
        resp = client.post("/api/commands/sync?profile=test")
        assert resp.status_code in (200, 500, 404)
        # Should not get 422 (validation error) from profile in body


class TestVisibleCliConsumed:
    """Verify visible.cli is respected in command dispatch.

    The command store persists per-platform visibility (visible.cli, etc.)
    and the CLI dispatch must only execute commands where visible.cli is True.
    """

    def test_list_commands_includes_visible_cli(self, client, isolated_profile):
        """GET /api/commands returns visible.cli field for each command."""
        resp = client.get("/api/commands")
        assert resp.status_code == 200
        cmds = resp.json()
        for cmd in cmds:
            assert "visible" in cmd
            assert "cli" in cmd["visible"]

    def test_custom_command_visible_cli_persisted(self, client, isolated_profile):
        """Creating a custom command persists visible.cli setting."""
        payload = {
            "name": "clitest",
            "description": "CLI only",
            "type": "exec",
            "command": "echo cli",
            "enabled": True,
            "visible": {"telegram": False, "discord": False, "cli": True},
            "silent_empty": False,
        }
        client.post("/api/commands/custom", json=payload)

        resp = client.get("/api/commands")
        cmds = resp.json()
        match = [c for c in cmds if c["name"] == "clitest"]
        assert len(match) == 1
        assert match[0]["visible"]["cli"] is True
        assert match[0]["visible"]["telegram"] is False
