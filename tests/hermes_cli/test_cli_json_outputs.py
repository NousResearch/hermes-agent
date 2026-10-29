"""``hermes doctor --json`` and ``hermes profile list --json``: one JSON document on stdout, nothing else."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_cli.doctor_json import run_doctor_json
from hermes_cli.doctor_report import Finding, _fail_and_issue, _section, annotate_last, check_info, check_ok, check_warn
from hermes_cli.subcommands.doctor import build_doctor_parser


def _checks_with_noise() -> Finding:
    """Stands in for DOCTOR_CHECKS: real report primitives plus a stray print, as the real checks do."""
    f = Finding()
    print("  Running 3 connectivity checks in parallel…")
    _section("Configuration Files")
    check_ok("config.yaml exists")
    check_warn("Missing directory", "(will be created on first use)")
    _section("Security Advisories")
    _fail_and_issue("Compromised package", "(pkg==1.0)", "Uninstall pkg==1.0", f.manual_issues)
    annotate_last(remediation="Rotate credentials.")
    check_info("see the advisory page")
    return f


def test_doctor_json_stdout_is_one_document_with_every_row(capsys):
    code = run_doctor_json(_checks_with_noise)
    out = capsys.readouterr().out
    doc = json.loads(out)  # raises if any human text leaked onto stdout
    assert code == 1 and next(iter(doc)) == "schema_version" and doc["schema_version"] == 1
    assert doc["healthy"] is False
    assert doc["summary"] == {"ok": 1, "warn": 1, "fail": 1, "info": 1}
    assert doc["checks"][1] == {"section": "Configuration Files", "status": "warn", "message": "Missing directory",
                                "detail": "will be created on first use"}
    assert doc["checks"][2] == {"section": "Security Advisories", "status": "fail", "message": "Compromised package",
                                "detail": "pkg==1.0", "fix": "Uninstall pkg==1.0", "remediation": "Rotate credentials."}
    assert doc["manual_issues"] == ["Uninstall pkg==1.0"] and doc["issues"] == []


def test_doctor_rows_are_not_collected_outside_json(capsys):
    """The human report is unchanged and annotate_last is inert without --json."""
    _checks_with_noise()
    out = capsys.readouterr().out
    assert "Compromised package" in out and "{" not in out


def test_doctor_json_and_ack_are_exclusive():
    parser = argparse.ArgumentParser()
    build_doctor_parser(parser.add_subparsers(dest="command"), cmd_doctor=lambda _a: None)
    assert parser.parse_args(["doctor", "--json", "--fix"]).json is True
    with pytest.raises(SystemExit):
        parser.parse_args(["doctor", "--json", "--ack", "ADV-1"])


def test_profile_list_json_envelope(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    default = tmp_path / ".hermes"
    (default / "profiles" / "work").mkdir(parents=True)
    (default / "profiles" / "work" / "config.yaml").write_text("model:\n  default: other/model\n", encoding="utf-8")
    (default / "config.yaml").write_text("model:\n  provider: openrouter\n  default: example/model\n", encoding="utf-8")
    (default / ".env").write_text("OPENROUTER_API_KEY=sk-secret-value\n", encoding="utf-8")
    (default / "SOUL.md").write_text("SOUL-CONTENT", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(default))
    from hermes_cli.profile_cmd import cmd_profile

    cmd_profile(SimpleNamespace(profile_action="list", json=True))
    out = capsys.readouterr().out
    doc = json.loads(out)
    assert next(iter(doc)) == "schema_version" and doc["active"] == "default"
    by_name = {p["name"]: p for p in doc["profiles"]}
    assert set(by_name) == {"default", "work"}
    assert by_name["default"]["is_active"] is True and by_name["work"]["is_active"] is False
    assert by_name["default"]["has_env"] is True and by_name["default"]["provider"] == "openrouter"
    assert "sk-secret-value" not in out and "SOUL-CONTENT" not in out


def test_doctor_json_connectivity_status_comes_from_the_probe_not_its_symbol(monkeypatch, capsys):
    """Changing a display symbol must not change which connectivity checks reach the JSON report."""
    from hermes_cli import doctor, doctor_connectivity

    monkeypatch.setattr(doctor_connectivity, "_GLYPH", {s: (f"[{s}]", "") for s in ("ok", "warn", "fail")})
    monkeypatch.setattr(doctor, "build_probes", list)
    monkeypatch.setattr(doctor, "run_probes", lambda _p: [doctor_connectivity._row("Stand-in API", "warn", "(slow)")])
    run_doctor_json(lambda: doctor._check_api_connectivity(False))
    rows = json.loads(capsys.readouterr().out)["checks"]
    assert [(r["status"], r["message"], r["detail"]) for r in rows] == [("warn", "Stand-in API", "slow")]
