"""HERMES_GATEWAY_ONLY_PLATFORMS restricts which adapters a worker initializes."""

from gateway.run import _only_platforms_filter


def test_unset_returns_none(monkeypatch):
    monkeypatch.delenv("HERMES_GATEWAY_ONLY_PLATFORMS", raising=False)
    assert _only_platforms_filter() is None


def test_blank_returns_none(monkeypatch):
    monkeypatch.setenv("HERMES_GATEWAY_ONLY_PLATFORMS", "  ")
    assert _only_platforms_filter() is None


def test_api_server_only_admits_api_server(monkeypatch):
    monkeypatch.setenv("HERMES_GATEWAY_ONLY_PLATFORMS", "api_server")
    only = _only_platforms_filter()
    assert "api_server" in only
    assert "telegram" not in only


def test_comma_separated(monkeypatch):
    monkeypatch.setenv("HERMES_GATEWAY_ONLY_PLATFORMS", "api_server, telegram")
    assert _only_platforms_filter() == {"api_server", "telegram"}
