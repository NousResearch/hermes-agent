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


def _audit_json_with_vulns(vulns: dict, critical=0, high=0, moderate=0):
    return json.dumps({
        "metadata": {"vulnerabilities": {
            "critical": critical, "high": high, "moderate": moderate,
            "low": 0, "info": 0, "total": critical + high + moderate,
        }},
        "vulnerabilities": vulns,
    })


def test_warn_names_affected_packages(capsys):
    """The counts-only row leaves users guessing which package/advisory the
    finding refers to (#126882): the per-package map must be surfaced."""
    audit = _audit_json_with_vulns(
        {
            "js-yaml": {
                "severity": "high",
                "via": [{"title": "js-yaml: maxTotalMergeKeys does not limit CPU use for empty merge sources",
                         "url": "https://github.com/advisories/GHSA-xxxx", "severity": "high"}],
            },
            "gulp-esbuild": {"severity": "moderate", "via": ["esbuild"]},
        },
        high=1, moderate=1,
    )
    out, _ = _run_audit_one(capsys, ["--workspaces=false"], audit)
    assert "affected: js-yaml (high — js-yaml: maxTotalMergeKeys does not limit CPU use" in out
    assert "affected: gulp-esbuild (moderate)" in out
    assert "+2 more" not in out


def test_warn_matches_the_advisory_to_the_package_severity(capsys):
    """npm's package severity is the max over advisories while ``via`` is sorted
    by advisory source id, so the first titled entry can be a lower-severity one
    (AI review on a real lodash payload: a critical package printed its high
    advisory). The printed title must carry the package's severity, falling back
    to the first titled entry when none matches (metavuln case)."""
    audit = _audit_json_with_vulns(
        {
            "lodash": {
                "severity": "critical",
                "via": [
                    {"title": "Command Injection in lodash", "severity": "high"},
                    {"title": "Prototype Pollution in lodash", "severity": "critical"},
                ],
            },
            "micromatch": {
                "severity": "high",  # raised by a metavuln; no high titled via exists
                "via": [
                    {"title": "ReDoS in micromatch", "severity": "moderate"},
                    "braces",
                ],
            },
        },
        critical=1, high=1,
    )
    out, _ = _run_audit_one(capsys, ["--workspaces=false"], audit)
    assert "affected: lodash (critical — Prototype Pollution in lodash)" in out
    assert "Command Injection" not in out
    assert "affected: micromatch (high — ReDoS in micromatch)" in out


def test_warn_caps_the_affected_package_list(capsys):
    """A heavily affected tree must not flood the doctor output: five named
    packages plus a +N summary line."""
    vulns = {f"pkg-{i}": {"severity": "moderate", "via": []} for i in range(7)}
    out, _ = _run_audit_one(capsys, ["--workspaces=false"],
                            _audit_json_with_vulns(vulns, high=1, moderate=6))
    assert sum(1 for line in out.splitlines() if "affected: pkg-" in line) == 5
    assert "+2 more package(s)" in out


def test_warn_without_per_package_map_keeps_counts_row(capsys):
    """Older npm or a parse quirk may omit the per-package map: the counts row
    and remedy must still print, with no empty affected section."""
    out, issues = _run_audit_one(capsys, ["--workspaces=false"], _audit_json(high=1))
    assert "lockfile bump" in out
    assert "affected:" not in out
    assert issues == ["Browser tools (agent-browser) has 1 npm vulnerability"]
