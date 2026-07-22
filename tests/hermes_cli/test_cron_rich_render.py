from unittest.mock import patch, MagicMock

import pytest
import rich
from _pytest.assertion import highlight

from hermes_cli import cron


@pytest.fixture
def forced_console():
    """
    Reconfigures the global rich console to force standard ANSI color output
    so tests can assert against raw color escape sequences.
    """
    original_console = rich.get_console()

    rich.reconfigure(force_terminal=True, color_system="standard", highlight=False)

    yield rich.get_console()

    rich.reconfigure(
        force_terminal=original_console.is_terminal,
        color_system=original_console.color_system,
        highlight=True
    )


@patch("hermes_cli.colors.should_use_color")
@patch("cron.jobs.list_jobs")
@patch("hermes_cli.gateway.find_gateway_pids")
def test_cron_list_rendering_and_markup_escape(
        mock_find_pids,
        mock_list_jobs,
        mock_should_use_color,
        forced_console
):
    mock_should_use_color.return_value = True

    mock_find_pids.return_value = [1234]

    mock_list_jobs.return_value = [
        {
            "id": "job_1",
            "name": "[red]malicious_name[/red]",
            "schedule": {"value": "*/5 * * * *"},
            "schedule_display": "every 5 minutes",
            "enabled": True,
            "state": "scheduled",
            "script": "[blue]echo injected[/blue]",
            "workdir": "[dim]/var/log[/dim]",
            "last_status": "ok",
            "last_run_at": "2026-07-12T10:00:00Z"
        },
        {
            "id": "job_2",
            "name": "Failing Job",
            "enabled": False,
            "state": "paused",
            "last_status": "failed",
            "last_error": "exit code 1",
            "last_delivery_error": "[bold]Timeout[/bold]"
        }
    ]

    with forced_console.capture() as capture:
        cron.cron_list(show_all=True)

    output = capture.get()

    # ==========================================
    # 1. ASSERT MARKUP ESCAPE (Literal Rendering)
    # ==========================================
    assert "[red]malicious_name[/red]" in output
    assert "[blue]echo injected[/blue]" in output
    assert "[dim]/var/log[/dim]" in output
    assert "[bold]Timeout[/bold]" in output

    assert "\x1b[31mmalicious_name\x1b[0m" not in output
    assert "\x1b[34mecho injected\x1b[0m" not in output

    # ==========================================
    # 2. ASSERT ANSI COLOR PRESERVATION
    # ==========================================
    assert "\x1b[32m[active]\x1b[0m" in output
    assert "\x1b[32mok\x1b[0m" in output
    assert "\x1b[33m[paused]\x1b[0m" in output
    assert "\x1b[31mfailed: exit code 1\x1b[0m" in output
    assert "\x1b[33m⚠ Delivery failed:\x1b[0m" in output

    # ==========================================
    # 3. ASSERT STANDARD FIELD RENDERING
    # ==========================================
    assert "every 5 minutes" in output
    assert "2026-07-12T10:00:00Z" in output
    assert "    Name:      " in output