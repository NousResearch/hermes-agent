"""The startup supervisor probe must not load the slow CIM ScheduledTasks module."""

import uuid
from unittest.mock import patch

import pytest

from hermes_cli.gateway import _windows_scheduled_task_state


@pytest.mark.windows_only
def test_task_query_without_powershell_module_autoload():
    # A random absent task exercises real in-process COM/RPC without starting PowerShell,
    # creating, stopping or changing any scheduled task on the developer's host.
    task_name = "Hermes_test_'" + uuid.uuid4().hex
    with patch("hermes_cli.gateway.subprocess.run", side_effect=AssertionError("PowerShell fallback used")):
        assert _windows_scheduled_task_state(task_name) == "MISSING"
