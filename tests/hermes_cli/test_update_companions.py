"""Installed-only companion refresh, including actual local npm subprocesses."""
import json
import subprocess
import sys

import pytest

from hermes_cli import update_companions as companions


@pytest.mark.parametrize("installed,exit_code", [(False, 0), (True, 0), (True, 7)])
def test_real_npm_installed_only(tmp_path, monkeypatch, capsys, installed, exit_code):
    npm = tmp_path / "npm"
    log = tmp_path / "calls.jsonl"
    listing = {"dependencies": {"@openai/codex": {}} if installed else {}}
    npm.write_text(
        f"#!{sys.executable}\nimport json,sys\n"
        f"with open({str(log)!r}, 'a') as f: f.write(json.dumps(sys.argv[1:])+'\\n')\n"
        f"if sys.argv[1]=='list': print(json.dumps({listing!r})); sys.exit(0)\n"
        f"sys.exit({exit_code})\n"
    )
    npm.chmod(0o700)
    monkeypatch.setattr("hermes_cli.main_install_repair._resolve_node_runtime_npm", lambda: str(npm))
    monkeypatch.setattr("hermes_constants.with_hermes_node_path", dict)
    companions.update_installed_codex()
    calls = [json.loads(line) for line in log.read_text().splitlines()]
    assert calls[0] == ["list", "-g", "@openai/codex", "--depth=0", "--json"]
    assert calls[1:] == ([["install", "-g", "@openai/codex@latest"]] if installed else [])
    if installed and exit_code:
        assert "non-fatal" in capsys.readouterr().out


@pytest.mark.parametrize("npm", [None, "/mnt/c/Program Files/node/npm.cmd"])
def test_absent_or_foreign_npm_never_runs(monkeypatch, npm):
    monkeypatch.setattr("hermes_constants.find_node_executable", lambda name: npm)
    monkeypatch.setattr("hermes_cli.main_install_repair._is_windows", lambda: False)
    monkeypatch.setattr(companions.subprocess, "run", lambda *a, **k: pytest.fail("foreign npm ran"))
    companions.update_installed_codex()


@pytest.mark.parametrize("failure", [subprocess.TimeoutExpired("npm", 30), OSError("missing"), ValueError("bad JSON")])
def test_companion_failure_nonfatal(monkeypatch, capsys, failure):
    monkeypatch.setattr("hermes_cli.main_install_repair._resolve_node_runtime_npm", lambda: "/managed/npm")
    monkeypatch.setattr("hermes_constants.with_hermes_node_path", dict)
    def fail(*a, **k):
        raise failure
    monkeypatch.setattr(companions.subprocess, "run", fail)
    companions.update_installed_codex()
    assert "non-fatal" in capsys.readouterr().out
