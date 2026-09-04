"""Install-on-enable for toolset toggles (dashboard/desktop PUT endpoint).

When a toolset is toggled ON via ``PUT /api/tools/toolsets/{name}`` and its
provider carries a post_setup hook with a registered, UNSATISFIED
install-state predicate (``_POST_SETUP_INSTALLED`` — today: cua-driver),
the endpoint spawns the same background ``hermes tools post-setup <key>``
action the interactive CLI flow runs. Without this, a GUI toggle "saves"
but the tool never appears in the schema because its check_fn can't find
the binary.
"""

import pytest


class TestToggleToolsetInstallOnEnable:
    @pytest.fixture(autouse=True)
    def _setup(self, monkeypatch, _isolate_hermes_home):
        try:
            from starlette.testclient import TestClient
        except ImportError:
            pytest.skip("fastapi/starlette not installed")

        import hermes_state
        from hermes_constants import get_hermes_home
        from hermes_cli.web_server import app, _SESSION_HEADER_NAME, _SESSION_TOKEN

        monkeypatch.setattr(
            hermes_state, "DEFAULT_DB_PATH", get_hermes_home() / "state.db"
        )
        self.client = TestClient(app)
        self.client.headers[_SESSION_HEADER_NAME] = _SESSION_TOKEN

    def _spawn_recorder(self, monkeypatch):
        import hermes_cli.web_server as web_server

        calls = []

        class _FakeProc:
            pid = 4242

        def _fake_spawn(subcommand, name, **kwargs):
            calls.append((tuple(subcommand), name))
            return _FakeProc()

        monkeypatch.setattr(web_server, "_spawn_hermes_action", _fake_spawn)
        return calls

    def test_enable_computer_use_spawns_cua_install_when_binary_missing(
        self, monkeypatch
    ):
        import hermes_cli.tools_config as tools_config

        calls = self._spawn_recorder(monkeypatch)
        # Binary missing → the cua_driver predicate reports unsatisfied.
        monkeypatch.setattr(
            tools_config, "_resolved_cua_driver_cmd", lambda: None
        )
        monkeypatch.setattr(
            tools_config, "_cua_driver_install_ready", lambda: False
        )

        resp = self.client.put(
            "/api/tools/toolsets/computer_use", json={"enabled": True}
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["enabled"] is True
        assert data["post_setup_started"] == "cua_driver"
        assert calls, "expected a background post-setup spawn"
        subcommand, action_name = calls[0]
        assert subcommand[-3:] == ("tools", "post-setup", "cua_driver")
        assert action_name == "tools-post-setup"

    def test_enable_computer_use_skips_install_when_binary_present(
        self, monkeypatch
    ):
        import hermes_cli.tools_config as tools_config

        calls = self._spawn_recorder(monkeypatch)
        monkeypatch.setattr(
            tools_config, "_resolved_cua_driver_cmd", lambda: "/usr/bin/cua-driver"
        )
        monkeypatch.setattr(
            tools_config, "_cua_driver_install_ready", lambda: True
        )

        resp = self.client.put(
            "/api/tools/toolsets/computer_use", json={"enabled": True}
        )
        assert resp.status_code == 200
        assert resp.json()["post_setup_started"] is None
        assert calls == []

    def test_disable_never_spawns_install(self, monkeypatch):
        import hermes_cli.tools_config as tools_config

        calls = self._spawn_recorder(monkeypatch)
        monkeypatch.setattr(
            tools_config, "_resolved_cua_driver_cmd", lambda: None
        )
        monkeypatch.setattr(
            tools_config, "_cua_driver_install_ready", lambda: False
        )

        resp = self.client.put(
            "/api/tools/toolsets/computer_use", json={"enabled": False}
        )
        assert resp.status_code == 200
        assert resp.json()["post_setup_started"] is None
        assert calls == []

    def test_spawn_failure_does_not_fail_the_toggle(self, monkeypatch):
        import hermes_cli.tools_config as tools_config
        import hermes_cli.web_server as web_server

        monkeypatch.setattr(
            tools_config, "_resolved_cua_driver_cmd", lambda: None
        )
        monkeypatch.setattr(
            tools_config, "_cua_driver_install_ready", lambda: False
        )

        def _boom(subcommand, name, **kwargs):
            raise RuntimeError("spawn exploded")

        monkeypatch.setattr(web_server, "_spawn_hermes_action", _boom)

        resp = self.client.put(
            "/api/tools/toolsets/computer_use", json={"enabled": True}
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["ok"] is True
        assert data["enabled"] is True
        assert data["post_setup_started"] is None

    def test_toolsets_without_pending_post_setup_unchanged(self, monkeypatch):
        """A toolset with no registered install predicate keeps today's
        contract — no spawn, post_setup_started stays None."""
        calls = self._spawn_recorder(monkeypatch)

        resp = self.client.put(
            "/api/tools/toolsets/web", json={"enabled": True}
        )
        assert resp.status_code == 200
        assert resp.json()["post_setup_started"] is None
        assert calls == []
