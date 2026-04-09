import os
import queue
from unittest.mock import patch

from cli import HermesCLI, _normalize_browser_connect_endpoint


class _FakeResponse:
    def __init__(self, status_code=200, json_data=None):
        self.status_code = status_code
        self._json_data = json_data or {}

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return self._json_data


def _bare_cli():
    cli = HermesCLI.__new__(HermesCLI)
    cli._pending_input = queue.Queue()
    return cli


def test_normalize_browser_connect_endpoint_adds_http_for_schemeless_localhost():
    assert _normalize_browser_connect_endpoint("localhost:9377") == "http://localhost:9377"


@patch("tools.browser_tool.cleanup_all_browsers")
def test_browser_connect_autodetects_camofox_on_9377(mock_cleanup, monkeypatch, capsys):
    cli = _bare_cli()
    monkeypatch.delenv("BROWSER_CDP_URL", raising=False)
    monkeypatch.delenv("CAMOFOX_URL", raising=False)
    monkeypatch.delenv("BROWSER_CONNECT_MODE", raising=False)
    monkeypatch.delenv("BROWSER_PREV_CAMOFOX_URL", raising=False)

    def fake_get(url, timeout=10):
        if url == "http://localhost:9377/json/version":
            return _FakeResponse(status_code=404)
        if url == "http://localhost:9377/health":
            return _FakeResponse(status_code=200, json_data={"ok": True})
        raise AssertionError(f"unexpected URL {url}")

    with patch("requests.get", side_effect=fake_get):
        cli._handle_browser_command("/browser connect localhost:9377")

    out = capsys.readouterr().out
    assert os.environ.get("CAMOFOX_URL") == "http://localhost:9377"
    assert os.environ.get("BROWSER_CONNECT_MODE") == "camofox"
    assert not os.environ.get("BROWSER_CDP_URL")
    assert "connected to Camofox" in out
    assert "Endpoint: http://localhost:9377" in out


@patch("tools.browser_tool.cleanup_all_browsers")
def test_browser_status_reports_camofox_when_connected(mock_cleanup, monkeypatch, capsys):
    cli = _bare_cli()
    monkeypatch.setenv("CAMOFOX_URL", "http://localhost:9377")
    monkeypatch.setenv("BROWSER_CONNECT_MODE", "camofox")
    monkeypatch.delenv("BROWSER_CDP_URL", raising=False)

    cli._handle_browser_command("/browser status")

    out = capsys.readouterr().out
    assert "Browser: connected to Camofox" in out
    assert "Endpoint: http://localhost:9377" in out


@patch("tools.browser_tool.cleanup_all_browsers")
def test_browser_disconnect_restores_previous_camofox_url(mock_cleanup, monkeypatch, capsys):
    cli = _bare_cli()
    monkeypatch.setenv("CAMOFOX_URL", "http://localhost:9377")
    monkeypatch.setenv("BROWSER_CONNECT_MODE", "camofox")
    monkeypatch.setenv("BROWSER_PREV_CAMOFOX_URL", "http://localhost:9999")
    monkeypatch.delenv("BROWSER_CDP_URL", raising=False)

    cli._handle_browser_command("/browser disconnect")

    out = capsys.readouterr().out
    assert os.environ.get("CAMOFOX_URL") == "http://localhost:9999"
    assert not os.environ.get("BROWSER_CONNECT_MODE")
    assert not os.environ.get("BROWSER_PREV_CAMOFOX_URL")
    assert "Browser disconnected from Camofox" in out
