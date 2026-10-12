"""The ``gateway_messaging_configured`` verdict on /api/status.

A stopped gateway projects an empty platform map, so the pill cannot tell "bots are down"
from "messaging was never set up" without this verdict (the false-positive "Messaging
stopped" alarm on every Desktop boot). The verdict must be:
  - False only when the config is readable AND no platform is configured AND no shared
    multiplexer serves profiles this request cannot see;
  - None (unknown) whenever the config is unreadable or a multiplexer roster exists —
    callers must fall back to the pre-verdict behavior rather than trust a partial answer.
"""

import asyncio

import hermes_cli.web_routers.status as status_mod
from gateway.status import GatewayLiveness


def _resolve(monkeypatch, *, runtime, configured, running=False):
    """Drive _resolve_gateway_status with every external probe stubbed."""
    monkeypatch.setattr(status_mod, "read_runtime_status", lambda **kw: runtime)
    monkeypatch.setattr(
        status_mod, "resolve_gateway_liveness",
        lambda **kw: GatewayLiveness(running=running, pid=1 if running else None, source="test"))
    if callable(configured) or isinstance(configured, Exception):
        def _boom():
            raise configured if isinstance(configured, Exception) else ValueError("probe")
        monkeypatch.setattr(status_mod, "_load_configured_gateway_platforms", _boom)
    else:
        monkeypatch.setattr(status_mod, "_load_configured_gateway_platforms", lambda: configured)
    return asyncio.run(status_mod._resolve_gateway_status(None, None))


def test_no_platforms_configured_gives_false(monkeypatch):
    out = _resolve(monkeypatch, runtime={"gateway_state": "stopped", "platforms": {}},
                   configured=set())
    assert out["gateway_running"] is False
    assert out["gateway_state"] == "stopped"
    assert out["gateway_messaging_configured"] is False


def test_configured_platform_gives_true(monkeypatch):
    out = _resolve(monkeypatch, runtime={"gateway_state": "stopped", "platforms": {}},
                   configured={"discord"})
    assert out["gateway_messaging_configured"] is True


def test_unreadable_config_gives_none(monkeypatch):
    out = _resolve(monkeypatch, runtime={"gateway_state": "stopped", "platforms": {}},
                   configured=RuntimeError("config.yaml unreadable"))
    assert out["gateway_messaging_configured"] is None


def test_shared_multiplexer_roster_gives_none(monkeypatch):
    # The polled profile has no platforms, but the retained record shows the multiplexer
    # served others — their config is invisible to this request, so "nothing configured"
    # would be a partial answer and must not suppress the pill.
    out = _resolve(monkeypatch,
                   runtime={"gateway_state": "stopped", "platforms": {},
                            "served_profiles": ["default", "alpha"]},
                   configured=set())
    assert out["gateway_messaging_configured"] is None


def test_single_served_profile_is_not_multiplex(monkeypatch):
    # A standalone gateway records served_profiles: ["default"] — one entry is not a
    # multiplexer roster and must not veto the verdict.
    out = _resolve(monkeypatch,
                   runtime={"gateway_state": "stopped", "platforms": {},
                            "served_profiles": ["default"]},
                   configured=set())
    assert out["gateway_messaging_configured"] is False


def test_stopped_record_retains_served_profiles(tmp_path, monkeypatch):
    """Pin the assumption behind the multiplex guard: the graceful-stop stamp merges
    field-wise into the live snapshot, so served_profiles survives on disk after death."""
    from gateway import status as gw_status

    target = tmp_path / "gateway_state.json"
    monkeypatch.setattr(gw_status, "_get_runtime_status_path", lambda: target)
    monkeypatch.setattr(gw_status, "_runtime_status_state", None)
    monkeypatch.setattr(gw_status, "_runtime_status_state_path", None)
    gw_status.write_runtime_status(gateway_state="running",
                                    served_profiles=["default", "alpha"], wait_timeout=2)
    gw_status.write_runtime_status(gateway_state="stopped", wait_timeout=2)
    rec = gw_status.read_runtime_status(path=target)
    assert rec["gateway_state"] == "stopped"
    assert rec["served_profiles"] == ["default", "alpha"]


def test_running_gateway_still_reports_the_verdict(monkeypatch):
    out = _resolve(monkeypatch, runtime={"gateway_state": "running", "platforms": {}},
                   configured={"telegram"}, running=True)
    assert out["gateway_running"] is True
    assert out["gateway_messaging_configured"] is True
