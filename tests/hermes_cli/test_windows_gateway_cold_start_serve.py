"""#76129: do not cold-start gateway run when desktop serve is live."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import hermes_cli.update_cmd as update_cmd


def test_cold_start_skips_when_desktop_serve_is_running(capsys):
    fake_main = MagicMock()
    fake_main._is_windows.return_value = True
    with (
        patch.object(update_cmd, "_m", return_value=fake_main),
        patch("hermes_cli.gateway.find_gateway_pids", return_value=[]),
        patch("gateway.status.desktop_serve_is_running", return_value=True),
        patch("hermes_cli.gateway_windows._spawn_detached") as spawn,
    ):
        update_cmd._cold_start_windows_gateway_after_update()

    spawn.assert_not_called()
    assert "Starting Windows gateway after update" not in capsys.readouterr().out


def test_cold_start_spawns_when_nothing_serves(capsys):
    fake_main = MagicMock()
    fake_main._is_windows.return_value = True
    with (
        patch.object(update_cmd, "_m", return_value=fake_main),
        patch("hermes_cli.gateway.find_gateway_pids", return_value=[]),
        patch("gateway.status.desktop_serve_is_running", return_value=False),
        patch("hermes_cli.gateway_windows._spawn_detached", return_value=4242) as spawn,
    ):
        update_cmd._cold_start_windows_gateway_after_update()

    spawn.assert_called_once()
    assert "Starting Windows gateway after update (PID 4242)" in capsys.readouterr().out


def test_pause_skips_cold_start_plan_when_serve_live():
    fake_main = MagicMock()
    fake_main._is_windows.return_value = True
    with (
        patch.object(update_cmd, "_m", return_value=fake_main),
        patch("hermes_cli.gateway.find_gateway_pids", return_value=[]),
        patch("gateway.status.desktop_serve_is_running", return_value=True),
        patch("hermes_cli.gateway_windows.is_installed", return_value=True),
    ):
        token = update_cmd._pause_windows_gateways_for_update()

    assert token is None
