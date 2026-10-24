"""A Windows CLI restart is a planned restart: the home channels must hear about it.

The gateway writes the planned-restart marker itself when a restart goes through
its in-band path. A Windows CLI restart is stop()+start(), so the gateway never
sees a restart request, the marker was never written, and the "Gateway online"
notice was lost - every home channel got the "shutting down" notice and nothing
on the way back (#142515).
"""
import json
from pathlib import Path

import pytest

from hermes_cli import gateway_windows

pytestmark = pytest.mark.platforms("windows")


def _marker_path(monkeypatch, tmp_path) -> Path:
    """Point the marker at a temp file - the path the boot pass reads."""
    marker = tmp_path / ".restart_pending.json"
    monkeypatch.setattr("gateway.run._planned_restart_notification_path", lambda: marker)
    return marker


@pytest.fixture
def restarted(monkeypatch):
    """Run a Windows CLI restart with the gateway's own side effects stubbed."""
    calls = []

    def fake_stop():
        calls.append("stop")

    monkeypatch.setattr(gateway_windows, "stop", fake_stop)
    monkeypatch.setattr(gateway_windows, "_wait_for_gateway_absent", lambda timeout_s=30.0: True)
    monkeypatch.setattr("hermes_cli.gateway._wait_for_api_server_port_free", lambda: None)
    monkeypatch.setattr(gateway_windows, "start", lambda: calls.append("start"))
    monkeypatch.setattr(gateway_windows, "_wait_for_gateway_ready", lambda timeout_s=15.0: True)
    return calls


def test_a_windows_cli_restart_records_the_planned_restart(monkeypatch, tmp_path, restarted):
    marker = _marker_path(monkeypatch, tmp_path)

    gateway_windows.restart()

    assert restarted == ["stop", "start"]
    assert marker.exists(), "the planned-restart marker was not written"
    data = json.loads(marker.read_text(encoding="utf-8"))
    assert "requested_at" in data, data


def test_the_marker_is_written_before_the_gateway_is_stopped(monkeypatch, tmp_path, restarted):
    """The boot that follows the restart reads the marker, so it has to exist
    by the time the old gateway is gone."""
    marker = _marker_path(monkeypatch, tmp_path)
    seen = {}

    def fake_stop():
        seen["marker_existed"] = marker.exists()

    monkeypatch.setattr(gateway_windows, "stop", fake_stop)

    gateway_windows.restart()

    assert seen["marker_existed"] is True


def test_a_failed_restart_still_leaves_the_marker_for_the_next_boot(
    monkeypatch, tmp_path, restarted
):
    """A restart that fails to come back must not swallow the notice: the marker
    is what tells the next boot that a return is owed."""
    marker = _marker_path(monkeypatch, tmp_path)

    def not_ready(timeout_s=15.0):
        raise RuntimeError("Gateway restart did not produce a running gateway process.")

    monkeypatch.setattr(gateway_windows, "_wait_for_gateway_ready", not_ready)

    with pytest.raises(RuntimeError):
        gateway_windows.restart()
    assert marker.exists()
