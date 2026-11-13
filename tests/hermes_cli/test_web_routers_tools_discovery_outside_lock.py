"""Plugin discovery must run outside the config mutation span.

Plugin loading runs each plugin's import + register() on a deadline worker
thread, and several plugins read config on that thread. The interprocess
config mutation lock holds the process-local config lock for its whole span,
so a route that discovers plugins *inside* the span makes every config-reading
plugin wait out its load deadline and then get abandoned. The toolset toggle
took ~30s per request that way and dropped the dashboard_auth providers.
"""

import threading

import pytest

import hermes_cli.web_server_gateway as _web_server_gateway


def _lock_is_free_from_another_thread() -> bool:
    from hermes_cli.config_backend import CONFIG_MUTATION_LOCK

    result = []

    def _probe():
        got = CONFIG_MUTATION_LOCK.acquire(timeout=1.0)
        if got:
            CONFIG_MUTATION_LOCK.release()
        result.append(got)

    probe = threading.Thread(target=_probe)
    probe.start()
    probe.join(5)
    return bool(result and result[0])


class TestToolsetRoutesDiscoverOutsideTheMutationLock:
    @pytest.fixture(autouse=True)
    def _setup(self, monkeypatch, _isolate_hermes_home):
        try:
            from starlette.testclient import TestClient
        except ImportError:
            pytest.skip("fastapi/starlette not installed")

        import hermes_state
        from hermes_constants import get_hermes_home
        from hermes_cli.web_server import app, _SESSION_HEADER_NAME, _SESSION_TOKEN

        monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", get_hermes_home() / "state.db")
        monkeypatch.setattr(_web_server_gateway, "_spawn_hermes_action", lambda *a, **k: None)
        self.client = TestClient(app)
        self.client.headers[_SESSION_HEADER_NAME] = _SESSION_TOKEN

    def test_toggle_validates_the_toolset_outside_the_lock(self, monkeypatch):
        from hermes_cli import tools_config

        real = tools_config._get_effective_configurable_toolsets
        seen = []

        def _probing(*args, **kwargs):
            seen.append(_lock_is_free_from_another_thread())
            return real(*args, **kwargs)

        monkeypatch.setattr(tools_config, "_get_effective_configurable_toolsets", _probing)
        resp = self.client.put("/api/tools/toolsets/web", json={"enabled": True})
        assert resp.status_code == 200
        assert seen, "the toggle never validated the toolset name"
        assert all(seen), "plugin discovery ran while the config mutation lock was held"

    def test_unknown_toolset_is_still_refused(self):
        resp = self.client.put("/api/tools/toolsets/no-such-toolset", json={"enabled": True})
        assert resp.status_code == 400

    def test_model_route_discovers_plugins_outside_the_lock(self, monkeypatch):
        seen = self._probe_every_discovery(monkeypatch)
        resp = self.client.put(
            "/api/tools/toolsets/image_gen/model", json={"model": "any-model"})
        # Fresh home: no model-capable backend is active, so the route refuses;
        # what matters is where discovery ran on the way there.
        assert resp.status_code in (200, 400)
        assert seen, "the model route never discovered plugins"
        assert all(seen), "plugin discovery ran while the config mutation lock was held"

    def _probe_every_discovery(self, monkeypatch):
        from hermes_cli.plugins import PluginManager

        real = PluginManager._discover_and_load_inner
        seen = []

        def _probing(manager, *args, **kwargs):
            seen.append(_lock_is_free_from_another_thread())
            return real(manager, *args, **kwargs)

        monkeypatch.setattr(PluginManager, "_discover_and_load_inner", _probing)
        return seen

    @pytest.mark.parametrize("route, body", [
        ("/api/delegation/backend", {"backend": "default"}),
        ("/api/delegation/backends/model_map", {"options": [], "default": None}),
    ])
    def test_delegation_writes_discover_outside_the_lock(self, monkeypatch, route, body):
        seen = self._probe_every_discovery(monkeypatch)
        resp = self.client.put(route, json=body)
        assert resp.status_code in (200, 400), resp.text
        assert seen, f"{route} never discovered plugins"
        assert all(seen), f"{route}: plugin discovery ran while the config mutation lock was held"

    def test_profile_routes_do_not_load_plugins_they_do_not_use(self, monkeypatch):
        seen = self._probe_every_discovery(monkeypatch)
        resp = self.client.put("/api/delegation/backends/profile_routes", json={"profiles": []})
        assert resp.status_code == 200, resp.text
        assert seen == [], "profile-name validation does not need plugin discovery"

    @pytest.mark.parametrize("method,route,body,status", [
        ("put", "/api/tools/terminal/backend", {"backend": "local"}, 200),
        ("post", "/api/mcp/servers", {"name": "lock-probe", "command": "python3", "args": ["-V"]}, 200),
        ("put", "/api/mcp/servers", {"servers": {}}, 200),
        ("delete", "/api/mcp/servers/no-such-server", None, 404),
        ("put", "/api/mcp/servers/no-such-server/enabled", {"enabled": True}, 404),
    ])
    def test_sibling_write_routes_discover_outside_the_lock(self, monkeypatch, method, route, body, status):
        seen = self._probe_every_discovery(monkeypatch)
        resp = self.client.request(method, route, **({"json": body} if body is not None else {}))
        assert resp.status_code == status, resp.text
        assert seen, f"{route} never discovered plugins"
        assert all(seen), f"{route}: plugin discovery ran while the config mutation lock was held"
