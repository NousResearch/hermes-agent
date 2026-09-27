"""GET /api/dashboard/language serves ``display.language`` as the dashboard's
server-side locale default (#125057): browsers with no stored choice (private
windows, cleared site data, new devices) fall back to the configured language
instead of hardcoded English. Regression for #125057."""

import pytest


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


def _write_config(display_language):
    from hermes_cli.config import atomic_config_write
    from hermes_constants import get_config_path

    atomic_config_write(
        get_config_path(),
        {"display": {"language": display_language}},
    )


class TestDashboardLanguageEndpoint:
    @pytest.fixture(autouse=True)
    def _setup(self, _isolate_hermes_home):
        self.client = _client()

    def test_serves_configured_language(self):
        _write_config("de")
        res = self.client.get("/api/dashboard/language")
        assert res.status_code == 200
        assert res.json() == {"language": "de"}

    def test_defaults_to_en_when_unset(self):
        _write_config("en")
        res = self.client.get("/api/dashboard/language")
        assert res.status_code == 200
        assert res.json() == {"language": "en"}
