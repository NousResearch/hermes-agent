"""Tests for HERMES_HOME-scoped gateway PID lookup.

Verifies that find_gateway_pids() uses the PID file scoped to the current
HERMES_HOME, preventing multi-profile gateway collisions.
"""

import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def fake_hermes_homes(tmp_path):
    """Create two fake HERMES_HOME directories with different PID files."""
    home_a = tmp_path / "profile-a"
    home_b = tmp_path / "profile-b"
    home_a.mkdir()
    home_b.mkdir()
    return home_a, home_b


class TestFindGatewayPidsScoping:
    """find_gateway_pids should only return PIDs for the current HERMES_HOME."""

    def test_returns_pid_from_scoped_file(self, fake_hermes_homes):
        """When a PID file exists, find_gateway_pids should read from it."""
        home_a, _ = fake_hermes_homes

        with patch.dict(os.environ, {"HERMES_HOME": str(home_a)}):
            # Write a PID file for profile A
            pid_data = {"pid": os.getpid(), "kind": "hermes-gateway",
                        "argv": ["hermes", "gateway", "run"]}
            (home_a / "gateway.pid").write_text(json.dumps(pid_data))

            from hermes_cli.gateway import find_gateway_pids
            pids = find_gateway_pids()
            assert os.getpid() in pids

    def test_does_not_see_other_profile_pid(self, fake_hermes_homes):
        """Profile B's gateway PID should not appear when HERMES_HOME points to A."""
        home_a, home_b = fake_hermes_homes

        # Write PID file only in profile B
        pid_data = {"pid": os.getpid(), "kind": "hermes-gateway",
                    "argv": ["hermes", "gateway", "run"]}
        (home_b / "gateway.pid").write_text(json.dumps(pid_data))

        with patch.dict(os.environ, {"HERMES_HOME": str(home_a)}):
            from hermes_cli.gateway import find_gateway_pids
            # get_running_pid is imported locally from gateway.status,
            # so we patch at the source
            with patch("gateway.status.get_running_pid", return_value=None):
                # With no PID file in home_a, and mocking out the global scan,
                # we should get no PIDs
                with patch("subprocess.run") as mock_run:
                    mock_run.return_value = MagicMock(stdout="", returncode=0)
                    pids = find_gateway_pids()
                    assert pids == []

    def test_empty_when_no_pid_file_and_no_processes(self, fake_hermes_homes):
        """When no PID file exists and no gateway processes are found, returns empty."""
        home_a, _ = fake_hermes_homes

        with patch.dict(os.environ, {"HERMES_HOME": str(home_a)}):
            from hermes_cli.gateway import find_gateway_pids
            with patch("gateway.status.get_running_pid", return_value=None):
                with patch("subprocess.run") as mock_run:
                    mock_run.return_value = MagicMock(stdout="", returncode=0)
                    pids = find_gateway_pids()
                    assert pids == []
