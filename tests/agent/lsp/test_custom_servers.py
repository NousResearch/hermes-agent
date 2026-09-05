"""Custom servers reach the real config loader, CLI and stdio diagnostic loop."""
import argparse
import json
import sys
from pathlib import Path

import pytest
import yaml

from agent.lsp import get_service, shutdown_service
from agent.lsp.cli import register_subparser, run_lsp_command
from agent.lsp.manager import LSPService
from agent.lsp.servers import find_server_for_file
from tools.file_operations_lint import LintMixin


@pytest.mark.parametrize("extension,language,expected_language", [
    (".QMD", "quarto", "quarto"),
    (".py", None, "python"),
    (".custom", None, "plaintext"),
    ("Customfile", "customfile", "customfile"),
])
def test_custom_server_config_to_diagnostics(tmp_path, monkeypatch, capsys, extension, language, expected_language):
    home = tmp_path / "profile"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".git").mkdir()
    (tmp_path / "project.marker").touch()  # must not root a server outside its workspace
    project = repo / "nested project"
    project.mkdir()
    (project / "project.marker").touch()
    filename = "document" + extension if extension.startswith(".") else extension
    source = project / filename
    source.write_text("before\n", encoding="utf-8")
    monkeypatch.chdir(repo)
    messages = tmp_path / "messages.jsonl"
    command = [sys.executable, str(Path(__file__).with_name("_mock_lsp_server.py")), "arg with spaces"]
    config = {"lsp": {"install_strategy": "manual", "idle_timeout": 0, "wait_timeout": 3, "servers": {"custom-test": {
        "command": command,
        "extensions": [extension],
        "root_markers": ["project.marker"],
        "env": {"MOCK_LSP_SCRIPT": "errors", "MOCK_LSP_MESSAGES_LOG": str(messages)},
        "initialization_options": {"example": {"enabled": True}},
    }}}}
    if language is not None:
        config["lsp"]["servers"]["custom-test"]["language_id"] = language
    config["lsp"]["servers"]["aaa-shadowed"] = {
        "command": ["missing-shadowed-lsp"], "extensions": [extension],
    }
    (home / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    original = find_server_for_file(str(source))
    parser = argparse.ArgumentParser()
    register_subparser(parser.add_subparsers())
    shutdown_service()
    try:
        svc = get_service()
        assert svc is not None and svc.enabled_for(str(source))
        lint_hook = LintMixin()
        monkeypatch.setattr(lint_hook, "_lsp_service", lambda: svc)
        assert lint_hook._lsp_will_handle(str(source))
        assert not svc.enabled_for(str(tmp_path / filename))
        svc.snapshot_baseline(str(source))
        source.write_text("after\n", encoding="utf-8")
        assert svc.get_diagnostics_sync(str(source)) == []
        assert svc.get_diagnostics_sync(str(source), delta=False)
        assert svc._loop.run(svc._current_diags_async(str(source)), timeout=3)
        assert {c["server_id"] for c in svc.get_status()["clients"]} == {"custom-test"}
        fallback = repo / filename
        fallback.write_text("fallback\n", encoding="utf-8")
        assert svc.get_diagnostics_sync(str(fallback), delta=False)
        assert {Path(c["workspace_root"]) for c in svc.get_status()["clients"]} == {repo, project}

        for args in (["lsp", "list", "--installed-only"], ["lsp", "status", "--json"]):
            assert run_lsp_command(parser.parse_args(args)) == 0
            assert "custom-test" in capsys.readouterr().out
        assert run_lsp_command(parser.parse_args(["lsp", "which", "custom-test"])) == 0
        assert Path(capsys.readouterr().out.strip()) == Path(sys.executable)
    finally:
        shutdown_service()

    traffic = [json.loads(line) for log in tmp_path.glob("messages.jsonl.*")
               for line in log.read_text(encoding="utf-8").splitlines()]
    initialize = next(row for row in traffic if row["message"].get("method") == "initialize"
                      and Path(row["cwd"]) == project)
    assert initialize["argv"] == command[2:]
    assert Path(initialize["cwd"]) == project
    params = initialize["message"]["params"]
    assert params["rootUri"] == project.as_uri()
    assert params["initializationOptions"] == config["lsp"]["servers"]["custom-test"]["initialization_options"]
    opened = next(row["message"] for row in traffic if row["message"].get("method") == "textDocument/didOpen")
    assert opened["params"]["textDocument"]["languageId"] == expected_language
    assert find_server_for_file(str(source)) is original


@pytest.mark.parametrize("invalid", [
    None,
    {"command": "server --stdio", "extensions": [".bad"]},
    {"command": [], "extensions": [".bad"]},
    {"command": [42], "extensions": [".bad"]},
    {"command": ["server"], "extensions": ".bad"},
    {"command": ["server"], "extensions": []},
    {"command": ["server"], "extensions": [42]},
    {"command": ["server"], "extensions": [".bad"], "root_markers": "project.marker"},
    {"command": ["server"], "extensions": [".bad"], "language_id": []},
])
def test_invalid_custom_server_does_not_break_other_servers(tmp_path, monkeypatch, caplog, invalid):
    home = tmp_path / "profile"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    (tmp_path / ".git").mkdir()
    monkeypatch.chdir(tmp_path)
    config = {"lsp": {"idle_timeout": 0, "servers": {
        "invalid-custom": invalid,
        "valid-custom": {"command": [sys.executable], "extensions": [".custom"]},
        "disabled-custom": {"command": [sys.executable], "extensions": [".disabled"], "disabled": True},
    }}}
    (home / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    svc = LSPService.create_from_config()
    assert svc is not None
    try:
        assert svc.enabled_for(str(tmp_path / "valid.custom"))
        assert svc.enabled_for(str(tmp_path / "builtin.py"))
        assert not svc.enabled_for(str(tmp_path / "bad.bad"))
        assert not svc.enabled_for(str(tmp_path / "off.disabled"))
        assert "lsp.servers.invalid-custom" in caplog.text
        assert find_server_for_file(str(tmp_path / "valid.custom")) is None
    finally:
        svc.shutdown()
