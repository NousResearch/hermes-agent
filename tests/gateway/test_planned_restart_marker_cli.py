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


def test_failed_restart_still_leaves_the_marker(hermes_home, monkeypatch):
    """A restart that never comes back must not swallow the notice.

    The marker is what tells the *next* boot that a return is owed, so it has to survive a
    ``start()`` that raises — otherwise a crash-looping gateway goes silent exactly when the user
    most needs to know it is down.
    """
    from hermes_cli import gateway as gw_facade
    from hermes_cli import gateway_windows as gw

    def not_ready(timeout_s=15.0):
        raise RuntimeError("Gateway restart did not produce a running gateway process.")

    monkeypatch.setattr(gw, "_assert_windows", lambda: None)
    monkeypatch.setattr(gw, "stop", lambda: None)
    monkeypatch.setattr(gw, "_wait_for_gateway_absent", lambda timeout_s=30.0: True)
    monkeypatch.setattr(gw_facade, "_wait_for_api_server_port_free", lambda: None)
    monkeypatch.setattr(gw, "start", lambda: None)
    monkeypatch.setattr(gw, "_wait_for_gateway_ready", not_ready)

    with pytest.raises(RuntimeError):
        gw.restart()

    assert (hermes_home / MARKER).exists()


def test_an_unwritable_marker_does_not_block_the_restart(hermes_home, monkeypatch):
    """Losing the notice is acceptable; losing the restart is not."""
    from gateway import restart_notice
    from hermes_cli import gateway as gw_facade
    from hermes_cli import gateway_windows as gw

    stopped: list[str] = []

    def boom(*_a, **_kw):
        raise RuntimeError("disk full")

    monkeypatch.setattr(restart_notice, "atomic_json_write", boom)
    monkeypatch.setattr(gw, "_assert_windows", lambda: None)
    monkeypatch.setattr(gw, "stop", lambda: stopped.append("stop"))
    monkeypatch.setattr(gw, "_wait_for_gateway_absent", lambda timeout_s=30.0: True)
    monkeypatch.setattr(gw_facade, "_wait_for_api_server_port_free", lambda: None)
    monkeypatch.setattr(gw, "start", lambda: stopped.append("start"))
    monkeypatch.setattr(gw, "_wait_for_gateway_ready", lambda timeout_s=15.0: True)

    gw.restart()

    assert stopped == ["stop", "start"]
