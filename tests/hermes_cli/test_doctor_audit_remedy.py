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
from hermes_cli.doctor_report import Finding


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
        doctor_tools._audit_one("npm", "C:/fake/root", "Root npm package",
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
    assert issues == [
        "Root npm package has 2 npm vulnerabilities; wait for an upstream lockfile fix, then update "
        "your installation (Docker: pull a rebuilt image; recreating the same image restores the same dependencies)"
    ]


def test_npm_audit_only_finding_does_not_offer_doctor_fix(capsys):
    """npm lockfile findings need an upstream image or lockfile update (#136806)."""
    from hermes_cli import doctor

    doctor._print_summary(False, Finding(manual_issues=["web workspace has 1 npm vulnerability"]))

    out = capsys.readouterr().out
    assert "web workspace has 1 npm vulnerability" in out
    assert "hermes doctor --fix" not in out




def test_clean_tree_reports_no_known_vulnerabilities(capsys):
    out, issues = _run_audit_one(capsys, ["--workspaces=false"], _audit_json())
    assert "no known vulnerabilities" in out
    assert issues == []


def test_root_audit_row_is_not_mislabeled_as_agent_browser(monkeypatch, tmp_path, capsys):
    """The root package.json row (js-yaml, semver, eslint tooling) names itself, not a browser
    driver it never audits (#122223)."""
    import hermes_cli.doctor as doctor_mod
    from gateway.platforms import whatsapp_common

    project = tmp_path / "project"
    (project / "node_modules").mkdir(parents=True)
    monkeypatch.setattr(doctor_mod, "PROJECT_ROOT", project)
    monkeypatch.setattr(doctor_tools, "_pm_tool_path", lambda name: "/usr/bin/npm" if name == "npm" else None)
    monkeypatch.setattr(whatsapp_common, "resolve_whatsapp_bridge_dir", lambda: tmp_path / "no-whatsapp-bridge")

    completed = subprocess.CompletedProcess([], 0, stdout=_audit_json(), stderr="")
    monkeypatch.setattr(doctor_tools.subprocess, "run", lambda *a, **kw: completed)

    doctor_tools._check_npm_audit(False)

    out = capsys.readouterr().out
    assert "agent-browser" not in out.lower()
    assert "Root npm package deps" in out
