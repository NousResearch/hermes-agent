import io
import pytest
from unittest.mock import patch
from argparse import Namespace
from rich.console import Console

from hermes_cli.cron import cron_list, cron_status, cron_create, cron_edit, cron_command


@pytest.fixture
def capture_console():
    """Fixture to intercept global rich.print calls and return clean strings instantly.

    This uses an in-memory StringIO buffer to bypass asynchronous/context-manager
    race conditions entirely.
    """
    string_io = io.StringIO()
    test_console = Console(file=string_io, force_terminal=False, color_system=None, width=80)

    def mock_rich_print(*args, **kwargs):
        test_console.print(*args, **kwargs)

    with patch("rich.print", side_effect=mock_rich_print):
        yield lambda: string_io.getvalue()


@pytest.fixture
def mock_cron_api():
    """Fixture to mock downstream json_loads(cronjob_tool) API vectors."""
    with patch("hermes_cli.cron._cron_api") as mock_api:
        yield mock_api


# ==============================================================================
#  Test Cases: List & Status Commands
# ==============================================================================

def test_cron_list_empty(capture_console):
    """Verify system output strings when zero scheduled jobs exist."""
    with patch("cron.jobs.list_jobs", return_value=[]):
        cron_list(show_all=False)

        output = capture_console()
        assert "No scheduled jobs" in output
        assert "Create one with" in output


def test_cron_list_with_jobs(capture_console):
    """Verify that structured records compile safely into Rich layout structures."""
    mock_jobs = [
        {
            "id": "job_99f1",
            "name": "Database Backup",
            "schedule_display": "Every Sunday",
            "state": "scheduled",
            "enabled": True,
            "next_run_at": "2026-07-12 00:00:00",
            "deliver": ["local"],
            "skills": ["system-admin"]
        }
    ]

    with patch("cron.jobs.list_jobs", return_value=mock_jobs), \
            patch("hermes_cli.cron._warn_if_gateway_not_running"):
        cron_list(show_all=False)
        output = capture_console()

        assert "Scheduled Jobs" in output
        assert "job_99f1" in output
        assert "Database Backup" in output


def test_cron_status_external_provider(capture_console):
    """Verify status text updates gracefully based on target cron engines."""
    with patch("hermes_cli.cron._active_cron_provider_name", return_value="chronos"), \
            patch("cron.jobs.list_jobs", return_value=[]):
        cron_status()
        output = capture_console()

        assert "Cron provider: chronos" in output


# ==============================================================================
#  Test Cases: Create Command
# ==============================================================================

def test_cron_create_success(capture_console, mock_cron_api):
    """Ensure successful job initialization renders flat dictionary targets."""
    mock_cron_api.return_value = {
        "success": True,
        "job_id": "cron_abc123",
        "name": "Telemetry Ping",
        "schedule": "*/15 * * * *",
        "next_run_at": "2026-07-07 15:15:00",
        "skills": ["network-ping"],
        "job": {
            "script": "ping.sh",
            "no_agent": True,
            "workdir": "/var/log"
        }
    }

    args = Namespace(
        schedule="*/15 * * * *",
        prompt="Ping endpoint",
        name="Telemetry Ping",
        skill="network-ping",
        skills=None
    )

    with patch("hermes_cli.cron._warn_if_gateway_not_running"):
        exit_code = cron_create(args)

        output = capture_console()
        assert exit_code == 0
        assert "Created job: cron_abc123" in output


# ==============================================================================
#  Test Cases: Edit Command
# ==============================================================================

def test_cron_edit_success(capture_console, mock_cron_api):
    """Verify editing a job routes its database payload values accurately."""
    mock_cron_api.return_value = {
        "success": True,
        "job": {
            "job_id": "cron_abc123",
            "name": "Updated Task Name",
            "schedule": "0 0 * * *",
            "skills": ["updated-skill"]
        }
    }

    args = Namespace(
        job_id="cron_abc123",
        name="Updated Task Name",
        schedule="0 0 * * *",
        skill="updated-skill",
        skills=None
    )

    with patch("cron.jobs.resolve_job_ref", return_value={"id": "cron_abc123"}), \
            patch("hermes_cli.cron._warn_if_gateway_not_running"):
        exit_code = cron_edit(args)

        output = capture_console()
        assert exit_code == 0
        assert "Updated job: cron_abc123" in output
        assert "Name: Updated Task Name" in output


def test_cron_edit_not_found(capture_console):
    """Verify that editing an unrecognized job returns an exit failure error."""
    args = Namespace(
        job_id="cron_missing",
        name="Ghost Edit",
        schedule=None,
        skill=None,
        skills=None
    )

    with patch("cron.jobs.resolve_job_ref", return_value=None):
        exit_code = cron_edit(args)

        output = capture_console()
        assert exit_code == 1
        assert "Job not found: cron_missing" in output


# ==============================================================================
#  Test Cases: Command Dispatch Routing
# ==============================================================================

def test_cron_command_routing():
    """Verify core CLI parser arguments map out execution tracks cleanly."""
    args = Namespace(cron_command="list", all=False)

    with patch("hermes_cli.cron.cron_list") as mock_list:
        exit_code = cron_command(args)
        assert exit_code == 0
        mock_list.assert_called_once()