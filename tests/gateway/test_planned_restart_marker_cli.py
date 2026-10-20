"""A CLI ``restart`` must stamp the planned-restart marker itself.

The gateway writes ``.restart_pending.json`` only on an in-band restart (SIGUSR1 / pause-for-update).
``hermes gateway restart`` on Windows is ``stop()`` + ``start()`` — Windows has no SIGUSR1 — so
without a CLI writer the next boot reads a plain stop and never sends the "Gateway online" notice to
the configured home channels.
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


def test_writer_stamps_the_marker_the_boot_pass_reads(hermes_home):
    """Writer and reader must agree on path and payload; a rename on either side fails here."""
    from gateway.restart_notice import write_planned_restart_marker

    assert gateway_run._planned_restart_notification_pending() is False

    write_planned_restart_marker()

    assert (hermes_home / MARKER).exists()
    assert gateway_run._planned_restart_notification_pending() is True


def test_windows_restart_stamps_the_marker_before_stopping(hermes_home, monkeypatch):
    """Ordering is the whole fix: ``stop()`` must already see the marker on disk."""
    from hermes_cli import gateway as gw_facade
    from hermes_cli import gateway_windows as gw

    seen: dict[str, bool] = {}
    monkeypatch.setattr(gw, "_assert_windows", lambda: None)
    monkeypatch.setattr(gw, "_wait_for_gateway_absent", lambda timeout_s=30.0: True)
    monkeypatch.setattr(gw, "_wait_for_gateway_ready", lambda timeout_s=15.0: True)
    monkeypatch.setattr(gw_facade, "_wait_for_api_server_port_free", lambda: None)
    monkeypatch.setattr(
        gw, "stop", lambda: seen.setdefault("marker_at_stop", (hermes_home / MARKER).exists())
    )
    monkeypatch.setattr(gw, "start", lambda: None)

    gw.restart()

    assert seen["marker_at_stop"] is True
