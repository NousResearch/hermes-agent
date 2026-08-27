from __future__ import annotations

from types import SimpleNamespace

from hermes_cli import kanban


def _set_liveness(monkeypatch, *, pid=None, probe_error=False):
    monkeypatch.setattr(
        "gateway.status.resolve_gateway_liveness",
        lambda **_kwargs: SimpleNamespace(pid=pid, probe_error=probe_error),
    )


def test_dispatcher_readiness_reports_live_gateway(monkeypatch):
    _set_liveness(monkeypatch, pid=4321)
    monkeypatch.setattr(
        "hermes_cli.config.load_config_readonly",
        lambda: {"kanban": {"dispatch_in_gateway": True}},
    )

    result = kanban._dispatcher_readiness()

    assert result == {
        "status": "ready",
        "ready": True,
        "gateway_pid": 4321,
        "message": "gateway pid=4321, dispatch enabled",
    }


def test_dispatcher_readiness_fails_closed_when_gateway_is_offline(monkeypatch):
    _set_liveness(monkeypatch, pid=None)
    monkeypatch.setattr(
        "hermes_cli.config.load_config_readonly",
        lambda: {"kanban": {"dispatch_in_gateway": True}},
    )

    result = kanban._dispatcher_readiness()

    assert result["status"] == "offline"
    assert result["ready"] is False
    assert "No gateway is running" in result["message"]


def test_dispatcher_readiness_distinguishes_disabled_dispatch(monkeypatch):
    _set_liveness(monkeypatch, pid=4321)
    monkeypatch.setattr(
        "hermes_cli.config.load_config_readonly",
        lambda: {"kanban": {"dispatch_in_gateway": False}},
    )

    result = kanban._dispatcher_readiness()

    assert result["status"] == "disabled"
    assert result["ready"] is False
    assert result["gateway_pid"] == 4321


def test_strict_readiness_reports_unknown_but_cli_warning_stays_fail_open(monkeypatch):
    _set_liveness(monkeypatch, probe_error=True)

    result = kanban._dispatcher_readiness()

    assert result["status"] == "unknown"
    assert result["ready"] is False
    assert kanban._check_dispatcher_presence() == (True, "")
