"""Actionable audit findings from real package metadata and plugin pins (#134652)."""

import argparse
import json
import sys
from pathlib import Path

import pytest

from hermes_cli import security_audit as audit


@pytest.mark.parametrize("json_output", [False, True])
def test_command_identifies_audited_sources_and_preserves_advisory_details(tmp_path, monkeypatch, capsys, json_output):
    packages = tmp_path / "audit environment" / "site-packages"
    metadata = packages / "audit_probe-1.0.dist-info"
    metadata.mkdir(parents=True)
    (metadata / "METADATA").write_text("Metadata-Version: 2.1\nName: audit-probe\nVersion: 1.0\n", encoding="utf-8")
    monkeypatch.syspath_prepend(str(packages))
    home = tmp_path / "profile"
    pins = home / "plugins" / "example" / "requirements.txt"
    pins.parent.mkdir(parents=True)
    pins.write_text("audit-probe==1.0\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    summary = "A long advisory describes the vulnerable operation. " * 4 + "The final mitigation must stay visible."
    fixed = ["1.1", "2.1", "3.1", "4.1"]
    advisory = "GHSA-example"

    def osv_response(url, payload=None):
        if payload is not None:
            return {"results": [
                {"vulns": [{"id": advisory}]} if q["package"]["name"] == "audit-probe" else {}
                for q in payload["queries"]
            ]}
        return {
            "summary": summary,
            "database_specific": {"severity": "HIGH"},
            "affected": [{"ranges": [{"events": [{"fixed": version} for version in fixed]}]}],
        }

    monkeypatch.setattr(audit, "_http_json", osv_response)
    args = argparse.Namespace(json=json_output, fail_on="high", skip_venv=False, skip_plugins=False, skip_mcp=True)
    assert audit.cmd_security_audit(args) == 1
    output = capsys.readouterr().out
    url = f"https://osv.dev/vulnerability/{advisory}"
    if json_output:
        payload = json.loads(output)
        assert payload["python_environment"] == {"executable": sys.executable, "prefix": sys.prefix}
        findings = {finding["source"]: finding for finding in payload["findings"]}
        assert Path(findings["venv"]["source_path"]) == packages
        assert Path(findings["plugin:example"]["source_path"]) == pins
        for finding in findings.values():
            assert finding["advisory_url"] == url
            assert finding["summary"] == summary
            assert finding["fixed_versions"] == fixed
    else:
        assert sys.executable in output
        assert sys.prefix in output
        assert str(packages) in output
        assert str(pins) in output
        assert url in output
        assert summary in " ".join(output.split())
        assert ", ".join(fixed) in output


def test_skipped_environment_is_not_reported(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    args = argparse.Namespace(json=True, fail_on="critical", skip_venv=True, skip_plugins=True, skip_mcp=True)
    assert audit.cmd_security_audit(args) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["findings"] == []
    assert payload.get("python_environment") is None
