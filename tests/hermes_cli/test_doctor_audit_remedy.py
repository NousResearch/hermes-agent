"""Doctor npm-audit remedy guidance — the fix must persist across `hermes update`.

Regression for #116774: `npm audit fix` on a managed install is reverted by the
next `hermes update`, because `_run_npm_install_deterministic` runs `npm ci`
against the COMMITTED root `package-lock.json` and restores the pinned
(vulnerable) versions. The doctor's root-row remedy used to prescribe exactly
that doomed local command (`run: cd <root> && npm audit fix --workspaces=false`),
sending users into a fix→reintroduce loop. The durable remedy is an upstream
lockfile bump; the doctor must never prescribe a local mutating fix command, and
the shipped root lockfile must audit clean so the finding never appears at all.
"""

import json
import subprocess
from unittest.mock import patch

import pytest

from hermes_cli import doctor_tools


def _audit_json(critical=0, high=0, moderate=0):
    return json.dumps({
        "metadata": {"vulnerabilities": {
            "critical": critical, "high": high, "moderate": moderate,
            "low": 0, "info": 0, "total": critical + high + moderate,
        }},
    })


def _run_audit_one(capsys, audit_extra, audit_stdout):
    issues: list[str] = []
    completed = subprocess.CompletedProcess([], 0, stdout=audit_stdout, stderr="")
    with patch.object(doctor_tools.subprocess, "run", return_value=completed):
        doctor_tools._audit_one("npm", "C:/fake/root", "Browser tools (agent-browser)",
                                audit_extra, issues)
    return capsys.readouterr().out, issues


def test_root_remedy_never_prescribes_local_audit_fix(capsys):
    """The root row must not tell users to run a mutating `npm audit fix`: the
    next `hermes update` reinstalls from the committed lockfile and reverts it
    (#116774)."""
    out, issues = _run_audit_one(capsys, ["--workspaces=false"], _audit_json(high=1, moderate=1))
    assert "run: cd" not in out
    assert "npm audit fix" not in out
    assert "lockfile bump" in out
    assert issues == ["Browser tools (agent-browser) has 2 npm vulnerabilities"]


def test_workspace_remedy_still_names_lockfile_bump(capsys):
    out, _ = _run_audit_one(capsys, ["--workspace", "web"], _audit_json(critical=1))
    assert "run: cd" not in out
    assert "npm audit fix" not in out
    assert "lockfile bump" in out


def test_clean_tree_reports_no_known_vulnerabilities(capsys):
    out, issues = _run_audit_one(capsys, ["--workspaces=false"], _audit_json())
    assert "no known vulnerabilities" in out
    assert issues == []

