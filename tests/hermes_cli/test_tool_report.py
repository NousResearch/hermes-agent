"""Effective capability reporting uses the actual resolver and profile config."""
import json

import pytest
import yaml

from hermes_cli.tool_report import report_tools


@pytest.mark.parametrize("disabled, expected", [([], 0), (["file"], 1)])
def test_report_resolves_profile_suppressions(tmp_path, monkeypatch, capsys, disabled, expected):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump({"agent": {"disabled_toolsets": disabled}}))
    original = path.read_bytes()

    assert report_tools("file, terminal", ["read_file", "write_file", "terminal"]) == expected
    report = json.loads(capsys.readouterr().out)
    assert report["schema"] == "hermes.tool-report.v1"
    assert report["ok"] == (expected == 0)
    assert report["toolsets"] == ["file", "terminal"]
    assert ("read_file" in report["tools"]) == (not disabled)
    assert ("read_file" in report["missing_required"]) == bool(disabled)
    assert "terminal" in report["tools"]
    assert path.read_bytes() == original
    assert str(tmp_path) not in json.dumps(report)


def test_unknown_required_tool_fails_closed(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    assert report_tools("file", ["example_missing_tool"]) == 1
    assert json.loads(capsys.readouterr().out)["missing_required"] == ["example_missing_tool"]


@pytest.mark.parametrize("disabled", [[], ["vision"]])
def test_report_checks_real_vision_availability(tmp_path, monkeypatch, capsys, disabled):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text(yaml.safe_dump({
        "agent": {"disabled_toolsets": disabled},
        "auxiliary": {"vision": {"provider": "custom", "model": "example-model",
                                 "base_url": "http://127.0.0.1:1/v1", "api_key": "example-test-key"}},
    }))
    assert report_tools("file,terminal,vision", ["vision_analyze"]) == int(bool(disabled))
    report = json.loads(capsys.readouterr().out)
    assert ("vision_analyze" in report["tools"]) == (not disabled)
    assert "example-test-key" not in json.dumps(report)


def test_report_parser():
    import argparse
    from hermes_cli.subcommands.tools import build_tools_parser

    parser = argparse.ArgumentParser()
    build_tools_parser(parser.add_subparsers(), cmd_tools=lambda args: None)
    args = parser.parse_args(["tools", "report", "--toolsets", "file,vision",
                              "--require-tool", "read_file", "--require-tool", "vision_analyze"])
    assert args.tools_action == "report"
    assert args.required_tools == ["read_file", "vision_analyze"]
