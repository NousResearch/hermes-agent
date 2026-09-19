"""The startup supervisor probe must not load the slow CIM ScheduledTasks module."""

import subprocess
import uuid
from unittest.mock import patch

import pytest

from hermes_cli.gateway import _windows_scheduled_task_state


@pytest.mark.platforms("windows")
def test_task_query_without_powershell_module_autoload():
    real_run = subprocess.run

    def run_without_modules(command, **kwargs):
        command = list(command)
        command[-1] = ("$ErrorActionPreference='Stop'; $PSModuleAutoLoadingPreference='None'; "
                       + command[-1])
        result = real_run(command, **kwargs)
        assert result.returncode == 0, result.stderr
        return result

    # A random absent task exercises real COM/RPC and task-name quoting without
    # creating, stopping or changing any scheduled task on the developer's host.
    task_name = "Hermes_test_'" + uuid.uuid4().hex
    with patch("hermes_cli.gateway.subprocess.run", side_effect=run_without_modules):
        assert _windows_scheduled_task_state(task_name) == "MISSING"
