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




def test_clean_tree_reports_no_known_vulnerabilities(capsys):
    out, issues = _run_audit_one(capsys, ["--workspaces=false"], _audit_json())
    assert "no known vulnerabilities" in out
    assert issues == []


def test_registry_network_failure_warns_instead_of_false_clean(capsys):
    """#101760: a DNS/network failure reaching registry.npmjs.org still exits 0
    with a JSON body carrying a top-level "error" and no metadata.vulnerabilities.
    Treating that missing shape as zero vulnerabilities used to print a false
    "no known vulnerabilities" instead of surfacing the audit failure."""
    network_failure_stdout = json.dumps({
        "message": ("request to https://registry.npmjs.org/-/npm/v1/security/audits/quick "
                    "failed, reason: getaddrinfo ENOTFOUND registry.npmjs.org"),
        "error": {"summary": "", "detail": ""},
    })
    out, issues = _run_audit_one(capsys, ["--workspaces=false"], network_failure_stdout)
    assert "no known vulnerabilities" not in out
    assert "npm audit unavailable: registry/network error" in out
    assert issues == []


def test_audit_subprocess_exception_warns_instead_of_silently_passing(capsys):
    """A subprocess timeout or JSON parse failure must be visible, not swallowed."""
    issues: list[str] = []
    with patch.object(doctor_tools.subprocess, "run", side_effect=subprocess.TimeoutExpired("npm", 30)):
        doctor_tools._audit_one("npm", "C:/fake/root", "Browser tools (agent-browser)",
                                ["--workspaces=false"], issues)
    out = capsys.readouterr().out
    assert "npm audit unavailable: registry/network error" in out
    assert issues == []
