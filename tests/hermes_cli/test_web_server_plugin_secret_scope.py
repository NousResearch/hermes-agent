"""A plugin API handler's ``get_secret()`` resolves the requested profile's credentials (#120310),
through the real mount path: the launch profile's copy in-process, a ``?profile=`` profile's own copy
in its plugin host (PLG-3). Moved out of ``test_web_server.py``.
"""
import shutil

import pytest

from tests.hermes_cli import test_web_server as web_server_tests

_EXAMPLE_PLUGIN_FIXTURE = web_server_tests._EXAMPLE_PLUGIN_FIXTURE
_install_example_plugin = web_server_tests._install_example_plugin  # the fixture


class TestPluginAPISecretScopeProductionMount:
    """#120310: a plugin API handler's ``get_secret()`` must resolve the *requested*
    profile's credentials under multi-profile hosting, verified through the REAL mount
    path — discovery → import → ``_mount_plugin_api_routes()`` (which attaches
    ``_plugin_route_secret_scope``) → a live request against ``app`` — not a hand-built
    ``include_router``. The dedicated ``test_plugin_api_secret_scope.py`` suite proves the
    dependency in isolation; this closes the actual reported surface end-to-end and pins
    that the launch profile and a ``?profile=`` request read distinct secrets, that the
    request profile does not leak back into the launch profile, and that an unknown
    profile is rejected in the dependency before the handler runs.
    """

    _PROBE_KEY = "EXAMPLE_PLUGIN_PROBE_KEY"

    @pytest.fixture(autouse=True)
    def _setup(self, monkeypatch, _isolate_hermes_home, _install_example_plugin):
        try:
            from starlette.testclient import TestClient
        except ImportError:
            pytest.skip("fastapi/starlette not installed")

        import hermes_state
        from hermes_constants import get_hermes_home
        from hermes_cli import profiles
        from hermes_cli.web_server import app, _SESSION_HEADER_NAME, _SESSION_TOKEN

        default_home = get_hermes_home()
        monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", default_home / "state.db")

        # Anchor the named-profiles root to the isolated home so ``?profile=workerb``
        # resolves inside the test sandbox (mirrors test_web_server_skills_profiles).
        profiles_root = default_home / "profiles"
        monkeypatch.setattr(profiles, "_get_default_hermes_home", lambda: default_home)
        monkeypatch.setattr(profiles, "_get_profiles_root", lambda: profiles_root)

        # Named profile B: a live profile with its OWN probe key value in ``.env`` (also its identity
        # marker); ``?profile=`` runs B's own installed copy in B's plugin host (PLG-3, #134889).
        (worker_home := profiles_root / "workerb").mkdir(parents=True, exist_ok=True)
        (worker_home / ".env").write_text(f"{self._PROBE_KEY}=sk-workerb\n", encoding="utf-8")
        (worker_home / "config.yaml").write_text("plugins:\n  enabled: [example]\n", encoding="utf-8")
        shutil.copytree(_EXAMPLE_PLUGIN_FIXTURE, worker_home / "plugins" / "example-dashboard")

        # Launch profile A: an env-only credential (systemd ``Environment=`` style), frozen
        # into the launch scope when multi-profile hosting activates.
        monkeypatch.setenv(self._PROBE_KEY, "sk-launch-a")

        import tui_gateway.launch_profile_policy as lpp
        from agent.secret_scope import is_multiplex_active, set_multiplex_active

        monkeypatch.setattr(lpp, "_snapshot", None)  # freeze the launch env fresh
        self._previous_multiplex = is_multiplex_active()
        lpp.activate_multi_profile_hosting()  # freezes os.environ + flips multiplex on

        self.client = TestClient(app)
        self.client.headers[_SESSION_HEADER_NAME] = _SESSION_TOKEN
        try:
            yield
        finally:
            set_multiplex_active(self._previous_multiplex)
            from hermes_cli import plugins as plugins_mod
            for manager in list(plugins_mod._plugin_managers_by_home.values()):
                host = getattr(manager, "_plugin_host_instance", None)
                if host is not None:
                    host.shutdown()

    def _probe(self, profile=None):
        params = {"profile": profile} if profile is not None else None
        return self.client.get("/api/plugins/example/whoami", params=params)

    def test_launch_and_requested_profile_read_distinct_secrets(self):
        # No ``?profile=`` → the launch profile's frozen env-only credential.
        resp = self._probe()
        assert resp.status_code == 200
        assert resp.json() == {"ok": True, "key": "sk-launch-a"}

        # ``?profile=workerb`` → that profile's own credential, through the real mount.
        resp = self._probe("workerb")
        assert resp.status_code == 200
        assert resp.json() == {"ok": True, "key": "sk-workerb"}

        # Back to the launch profile: the request scope reset, so B never leaks into A.
        resp = self._probe()
        assert resp.status_code == 200
        assert resp.json() == {"ok": True, "key": "sk-launch-a"}

    def test_unknown_profile_rejected_before_handler(self):
        # The dependency raises HTTPException(404) before the handler runs, so this is a
        # 404 — NOT the handler's folded ``{"ok": False}`` no-data contract.
        resp = self._probe("ghost")
        assert resp.status_code == 404
