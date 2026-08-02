"""#76129 / #76745: install-scoped desktop serve suppresses gateway cold-start."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import hermes_cli.update_cmd as update_cmd
from gateway.status import looks_like_desktop_serve_command_line


def test_cold_start_skips_when_this_install_serve_is_running(capsys):
    fake_main = MagicMock()
    fake_main._is_windows.return_value = True
    with (
        patch.object(update_cmd, "_m", return_value=fake_main),
        patch("hermes_cli.gateway.find_gateway_pids", return_value=[]),
        patch.object(update_cmd, "_this_install_desktop_serve_is_running", return_value=True),
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
        patch.object(update_cmd, "_this_install_desktop_serve_is_running", return_value=False),
        patch("hermes_cli.gateway_windows._spawn_detached", return_value=4242) as spawn,
    ):
        update_cmd._cold_start_windows_gateway_after_update()

    spawn.assert_called_once()
    assert "Starting Windows gateway after update (PID 4242)" in capsys.readouterr().out


def test_pause_skips_cold_start_plan_when_this_install_serve_live():
    fake_main = MagicMock()
    fake_main._is_windows.return_value = True
    with (
        patch.object(update_cmd, "_m", return_value=fake_main),
        patch("hermes_cli.gateway.find_gateway_pids", return_value=[]),
        patch.object(update_cmd, "_this_install_desktop_serve_is_running", return_value=True),
        patch("hermes_cli.gateway_windows.is_installed", return_value=True),
    ):
        token = update_cmd._pause_windows_gateways_for_update()

    assert token is None


def test_this_install_serve_ignores_foreign_holder_rows():
    """Cross-install regression: only install-scoped holders feed the check."""
    foreign = (
        99,
        "python.exe",
        r"C:\other\hermes\venv\Scripts\python.exe -m hermes_cli.main serve --host 127.0.0.1",
    )
    local = (
        42,
        "python.exe",
        r"C:\this\hermes\venv\Scripts\python.exe -m hermes_cli.main serve --host 127.0.0.1",
    )

    # Foreign-only holder list → do not suppress (scanner already install-scoped;
    # empty list means "no this-install serve").
    with patch.object(update_cmd, "_detect_venv_python_processes", return_value=[]):
        assert update_cmd._this_install_desktop_serve_is_running() is False

    # This-install holder that is serve → suppress.
    with patch.object(update_cmd, "_detect_venv_python_processes", return_value=[local]):
        assert update_cmd._this_install_desktop_serve_is_running() is True

    # This-install holder that is gateway run, not serve → do not suppress via this helper.
    gateway_only = (
        43,
        "python.exe",
        r"C:\this\hermes\venv\Scripts\python.exe -m hermes_cli.main gateway run",
    )
    with patch.object(update_cmd, "_detect_venv_python_processes", return_value=[gateway_only]):
        assert update_cmd._this_install_desktop_serve_is_running() is False

    # Sanity: the foreign cmdline still matches the pure serve matcher (so the
    # install scope, not the matcher, is what filters it out).
    assert looks_like_desktop_serve_command_line(foreign[2]) is True
