"""/api/status names an unparseable config.yaml (#93263); the raw YAML error quotes both
duplicated values, so the public probe must carry neither them nor the path."""
import pytest

_SECRET = "sk-dup-0123456789"
_BROKEN = (
    "model:\n"
    "  default: my-model\n"
    f"  api_key: {_SECRET}\n"
    f"  api_key: {_SECRET}-b\n"
)


@pytest.fixture
def client(monkeypatch, _isolate_hermes_home):
    from starlette.testclient import TestClient

    import hermes_constants
    import hermes_state
    from hermes_constants import get_hermes_home
    from hermes_cli.web_server import _SESSION_HEADER_NAME, _SESSION_TOKEN, app

    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", get_hermes_home() / "state.db")
    monkeypatch.setattr(hermes_constants, "_default_hermes_root_memo", None)
    c = TestClient(app)
    c.headers[_SESSION_HEADER_NAME] = _SESSION_TOKEN
    return c


def _config_path():
    from hermes_constants import get_hermes_home

    return get_hermes_home() / "config.yaml"


def test_status_reports_where_config_fails_to_parse_without_leaking_it(client):
    _config_path().write_text(_BROKEN, encoding="utf-8")

    status = client.get("/api/status").json()

    error = status.get("config_parse_error") or ""
    assert "line 4" in error
    assert _SECRET not in error
    assert str(_config_path()) not in error


def test_status_clears_config_parse_error_once_the_file_parses(client):
    path = _config_path()
    path.write_text(_BROKEN, encoding="utf-8")
    assert client.get("/api/status").json().get("config_parse_error")

    path.write_text("model:\n  default: my-model\n", encoding="utf-8")

    assert client.get("/api/status").json().get("config_parse_error") is None
