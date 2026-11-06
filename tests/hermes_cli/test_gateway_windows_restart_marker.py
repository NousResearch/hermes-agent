"""The Windows CLI restart stamps the planned-restart marker — after the old gateway is gone.

``hermes gateway restart`` on Windows is ``stop()`` + ``start()`` (no SIGUSR1), so nothing else would
ever write ``.restart_pending.json`` and the next boot would skip the "Gateway online" home-channel
notice. The marker has to land *after* the old process is gone — written before ``stop()`` it is
visible to a still-draining gateway, which can replay the notice and unlink it — and *before*
``start()``, so a restart that never comes back still leaves it for the next boot.
"""

from __future__ import annotations

import pytest

from gateway import run as gateway_run

MARKER = ".restart_pending.json"


@pytest.fixture
def hermes_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    return tmp_path


@pytest.mark.platforms("windows")
def test_restart_stamps_the_marker_after_the_old_gateway_is_gone(hermes_home, monkeypatch):
    from hermes_cli import gateway as gw_facade
    from hermes_cli import gateway_windows as gw

    marker = hermes_home / MARKER
    seen: dict[str, bool] = {}
    monkeypatch.setattr(gw, "_wait_for_gateway_absent", lambda timeout_s=30.0: True)
    monkeypatch.setattr(gw_facade, "_wait_for_api_server_port_free", lambda: None)
    monkeypatch.setattr(gw, "stop", lambda: seen.setdefault("marker_at_stop", marker.exists()))
    monkeypatch.setattr(gw, "start", lambda: seen.setdefault("marker_at_start", marker.exists()))
    monkeypatch.setattr(gw, "_wait_for_gateway_ready", lambda timeout_s=15.0: True)

    gw.restart()

    assert seen["marker_at_stop"] is False, "the still-draining gateway must not see the marker"
    assert seen["marker_at_start"] is True, "the next boot pass must find it"

    # A restart that never comes back must keep the marker: the notice is still owed.
    monkeypatch.setattr(gw, "stop", lambda: None)
    marker.unlink()

    def not_ready(timeout_s=15.0):
        raise RuntimeError("Gateway restart did not produce a running gateway process.")

    monkeypatch.setattr(gw, "_wait_for_gateway_ready", not_ready)
    with pytest.raises(RuntimeError):
        gw.restart()

    assert marker.exists()
