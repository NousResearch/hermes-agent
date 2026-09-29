"""`updates.auto_check: false` gates the passive update-check endpoint (#69947).

The desktop's startup/background/window-focus probes and the unforced
``GET /api/hermes/update/check`` must answer without probing upstream when the
user opted out; the explicit "Check now" (``force``) keeps working.
"""

import pytest

import hermes_cli.config as _cfg_mod


def _client():
    try:
        from starlette.testclient import TestClient
    except ImportError:
        pytest.skip("fastapi/starlette not installed")
    import hermes_state
    from hermes_constants import get_hermes_home
    from hermes_cli.web_server import app, _SESSION_HEADER_NAME, _SESSION_TOKEN

    client = TestClient(app)
    client.headers[_SESSION_HEADER_NAME] = _SESSION_TOKEN
    hermes_state.DEFAULT_DB_PATH = get_hermes_home() / "state.db"
    return client


class TestUpdateAutoCheckGate:
    @pytest.fixture(autouse=True)
    def _setup(self, _isolate_hermes_home):
        self.client = _client()

    def _auto_check_off(self, monkeypatch, probe):
        from hermes_constants import get_hermes_home

        (get_hermes_home() / "config.yaml").write_text("updates:\n  auto_check: false\n", encoding="utf-8")
        monkeypatch.setattr(_cfg_mod, "detect_install_method", lambda *a, **k: "git")
        monkeypatch.setattr("hermes_cli.source_check.check_for_updates", probe)

    def test_auto_check_off_answers_the_passive_check_without_probing(self, monkeypatch):
        self._auto_check_off(monkeypatch, lambda **kw: pytest.fail("passive check must not probe"))

        body = self.client.get("/api/hermes/update/check").json()
        assert body["auto_check_disabled"] is True
        assert body["behind"] is None
        assert body["update_available"] is False
        assert "updates.auto_check" in body["message"]
        # No probe means the 24h source-check cache is never touched either.

    def test_force_bypasses_the_gate(self, monkeypatch):
        self._auto_check_off(monkeypatch, lambda **kw: {"behind": 5, "commits": []})

        body = self.client.get("/api/hermes/update/check?force=true").json()
        assert "auto_check_disabled" not in body
        assert body["behind"] == 5
        assert body["update_available"] is True

    def test_default_config_keeps_passive_checks_running(self, monkeypatch):
        monkeypatch.setattr(_cfg_mod, "detect_install_method", lambda *a, **k: "git")
        monkeypatch.setattr("hermes_cli.source_check.check_for_updates", lambda **kw: {"behind": 0, "commits": []})

        body = self.client.get("/api/hermes/update/check").json()
        assert "auto_check_disabled" not in body
        assert body["behind"] == 0
