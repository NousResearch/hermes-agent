import json
from pathlib import Path

import pytest

from hermes_cli.agent_spec import cmd_agent_spec
from hermes_cli.subcommands.agent_spec import build_agent_spec_parser

FIX = Path("tests/fixtures/agent_specs")


def test_parser_registers_agent_spec_command():
    import argparse

    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command")
    build_agent_spec_parser(sub, cmd_agent_spec=lambda args: None)
    args = parser.parse_args(["agent-spec", "validate", "x.agent.md", "--json"])
    assert args.command == "agent-spec"
    assert args.agent_spec_command == "validate"
    assert args.json is True


def run_cmd(args):
    with pytest.raises(SystemExit) as exc:
        cmd_agent_spec(args)
    return exc.value.code


def ns(**kwargs):
    import argparse

    return argparse.Namespace(**kwargs)


def test_validate_text_and_json_exit_codes(capsys, tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    (tmp_path / "home").mkdir()
    code = run_cmd(ns(agent_spec_command="validate", path_or_id=str(FIX / "valid/minimal.agent.md"), json=True, strict=False, profile=None))
    assert code == 0
    data = json.loads(capsys.readouterr().out)
    assert data["status"] == "pass"
    code = run_cmd(ns(agent_spec_command="validate", path_or_id="minimal-agent", json=True, strict=False, profile=None))
    assert code == 0
    data = json.loads(capsys.readouterr().out)
    assert data["status"] == "pass"
    code = run_cmd(ns(agent_spec_command="validate", path_or_id=str(FIX / "invalid/bad-reasoning.agent.md"), json=False, strict=False, profile=None))
    assert code == 1
    assert "bad-reasoning" in capsys.readouterr().out
    code = run_cmd(ns(agent_spec_command="validate", path_or_id=str(FIX / "invalid/unknown-mcp.agent.md"), json=True, strict=False, profile=None))
    assert code == 1
    data = json.loads(capsys.readouterr().out)
    assert data["errors"][0]["code"] == "unknown_server_id"


def test_preview_and_list_json_are_read_only(capsys, tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    (home / "config.yaml").write_text("model:\n  provider: test\n  model: test-model\n", encoding="utf-8")
    (home / "SOUL.md").write_text("soul", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    before = sorted(str(p.relative_to(home)) for p in home.rglob("*"))
    code = run_cmd(ns(agent_spec_command="preview", profile="default", spec=str(FIX / "valid/full-preview.agent.md"), json=True, strict=False))
    assert code == 0
    data = json.loads(capsys.readouterr().out)
    assert data["profile_id"] == "default"
    assert data["sandbox"]["enforcement_status"] == "declared_only"
    code = run_cmd(ns(agent_spec_command="preview", profile="default", spec=None, json=True, strict=False))
    assert code == 0
    data = json.loads(capsys.readouterr().out)
    assert data["spec_status"] == "missing"
    code = run_cmd(ns(agent_spec_command="list", profiles=True, json=True))
    assert code == 0
    listed = json.loads(capsys.readouterr().out)
    assert listed["profiles"][0]["name"] == "default"
    code = run_cmd(ns(agent_spec_command="list", profiles=True, json=False))
    assert code == 0
    text = capsys.readouterr().out
    assert "Agent spec profile coverage:" in text
    assert "Profile: default" in text
    assert "Spec status: missing" in text
    assert "Profile: None" not in text
    assert "Agent spec status: None" not in text
    after = sorted(str(p.relative_to(home)) for p in home.rglob("*"))
    assert before == after


def test_preview_reports_malformed_structured_fields_without_internal_error(capsys, tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))

    code = run_cmd(ns(agent_spec_command="preview", profile="default", spec=str(FIX / "invalid/malformed-structured-fields.agent.md"), json=True, strict=False))

    assert code == 1
    data = json.loads(capsys.readouterr().out)
    assert data["status"] == "fail"
    assert data["read_only_guarantee"] is True
    assert data["enforcement_enabled"] is False
    error_codes = {error["code"] for error in data["errors"]}
    assert "agent_spec_error" not in error_codes
    assert "invalid_mcp" in error_codes
    assert "invalid_toolsets" in error_codes
    assert "invalid_sandbox" in error_codes
    assert "invalid_skills" in error_codes
    assert "invalid_artifacts" in error_codes
    assert "invalid_gates" in error_codes


def test_strict_mode_returns_nonzero_for_warnings(capsys, tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    code = run_cmd(ns(agent_spec_command="preview", profile="default", spec=None, json=True, strict=True))
    assert code == 1
    assert json.loads(capsys.readouterr().out)["status"] == "fail"
