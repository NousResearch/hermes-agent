"""Workspace audit warnings must reflect their runtime and build dependency scope."""

import json
import subprocess
from unittest.mock import patch

import pytest

from hermes_cli import doctor_tools


@pytest.mark.parametrize("workspace, runtime_package", [("web", "react-router"), ("ui-tui", "undici")])
def test_workspace_runtime_findings_are_not_dismissed_as_build_tooling(workspace, runtime_package, tmp_path, capsys):
    audit_args = ["npm", "audit", "--json", "--workspace", workspace]
    audit_output = json.dumps({
        "vulnerabilities": {
            runtime_package: {"severity": "high", "isDirect": True},
            "vitest": {"severity": "moderate", "isDirect": True},
        },
        "metadata": {"vulnerabilities": {"high": 1, "moderate": 1}},
    })
    result = subprocess.CompletedProcess(audit_args, 1, stdout=audit_output)
    issues = []

    with patch.object(doctor_tools.subprocess, "run", return_value=result) as run:
        doctor_tools._audit_one("npm", tmp_path, f"{workspace} workspace", audit_args[3:], issues)

    assert run.call_args.args[0] == audit_args
    output = capsys.readouterr().out
    assert "runtime and build dependencies" in output
    assert "not runtime" not in output
    assert "1 high, 1 moderate" in output
    assert "upstream lockfile bump" in output
    assert issues == [f"{workspace} workspace has 2 npm vulnerabilities"]
