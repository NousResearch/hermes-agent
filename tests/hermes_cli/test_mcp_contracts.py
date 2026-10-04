"""Real CLI + stdio discovery, persistence, and offline report integration."""

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
SERVER = Path(__file__).with_name("fixtures") / "mcp_contract_server.py"


def tool(name="search", **schema):
    return {"name": name, "inputSchema": {"type": "object", **schema}}


def run_cli(home, *arguments):
    return subprocess.run(
        [sys.executable, "-m", "hermes_cli.main", "mcp", *map(str, arguments)],
        cwd=ROOT, env={**os.environ, "HERMES_HOME": str(home)},
        capture_output=True, text=True, encoding="utf-8", timeout=45,
    )


def setup_server(tmp_path, monkeypatch, data):
    home = tmp_path / "profile"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    manifest_path = tmp_path / "server-tools.json"
    manifest_path.write_text(json.dumps(data), encoding="utf-8")
    audit = tmp_path / "rpc.jsonl"
    cfg = {"mcp_servers": {
        "demo": {"command": sys.executable, "args": [str(SERVER), str(manifest_path), str(audit)],
                 "env": {"MCP_TEST_TOKEN": "SYNTHETIC_CONFIG_TOKEN"},
                 "sampling": {"enabled": True}, "elicitation": {"enabled": True},
                 "tools": {"include": []}, "connect_timeout": 3 if data.get("mode") == "error" else 10},
        "unselected": {"command": "does-not-exist-contract-fixture"},
    }}
    config_path = home / "config.yaml"
    config_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return home, manifest_path, audit, config_path


def test_live_snapshot_and_offline_diff(tmp_path, monkeypatch):
    old_tool = tool(properties={"limit": {"type": "number"}}, additionalProperties=False)
    home, data_path, audit, config_path = setup_server(tmp_path, monkeypatch, {"tools": [old_tool, tool("count")]})
    config_before = config_path.read_bytes()
    before, after = tmp_path / "before.json", tmp_path / "after.json"
    first = run_cli(home, "snapshot", "demo", "--output", before)
    diagnostics = "\n".join(p.read_text(encoding="utf-8") for p in (audit, home / "logs" / "agent.log") if p.exists())
    assert first.returncode == 0, first.stdout + first.stderr + diagnostics
    saved = json.loads(before.read_text(encoding="utf-8"))
    assert {t["name"] for t in saved["tools"]} == {"search", "count"}  # includes the second page and filtered tools
    assert "SYNTHETIC_CONFIG_TOKEN" not in before.read_text(encoding="utf-8")
    assert "env" not in saved and "command" not in saved
    assert config_path.read_bytes() == config_before
    assert not (home / "cache" / "mcp_schema_cache.json").exists()
    events = [json.loads(line) for line in audit.read_text(encoding="utf-8").splitlines()]
    init = next(e for e in events if e.get("method") == "initialize")
    assert "sampling" not in init["params"]["capabilities"]
    assert "elicitation" not in init["params"]["capabilities"]
    assert {e.get("method") for e in events if "method" in e} <= {"initialize", "notifications/initialized", "tools/list"}

    data_path.write_text(json.dumps({"tools": [tool(properties={"limit": {"type": "integer"}},
                                                       required=["limit"], additionalProperties=False)]}), encoding="utf-8")
    second = run_cli(home, "snapshot", "demo", "--output", after)
    assert second.returncode == 0, second.stdout + second.stderr
    # An unrelated profile has no server configuration. Diff must still work without
    # contacting either server, reading credentials, or invoking a tool.
    offline_home = tmp_path / "offline"
    offline_home.mkdir()
    audit_before = audit.read_bytes()
    diff = run_cli(offline_home, "diff", before, after, "--json")
    assert diff.returncode == 1, diff.stdout + diff.stderr
    report = json.loads(diff.stdout)
    assert report["status"] == "breaking"
    assert {c["message"] for c in report["changes"]} >= {"Tool removed.", "New required property: 'limit'."}
    assert any(c["path"] == "/inputSchema/properties/limit/type" for c in report["changes"])
    assert audit.read_bytes() == audit_before
    reverse = run_cli(offline_home, "diff", after, before, "--json")
    assert reverse.returncode == 0 and json.loads(reverse.stdout)["status"] == "compatible"
    unchanged = run_cli(offline_home, "diff", before, before)
    assert unchanged.returncode == 0 and "unchanged" in unchanged.stdout


@pytest.mark.parametrize("mode", ["error", "cycle", "duplicate"])
def test_incomplete_capture_preserves_existing_snapshot(tmp_path, monkeypatch, mode):
    data = {"mode": mode, "tools": [tool(), tool("search" if mode == "duplicate" else "count")]}
    home, _, audit, _ = setup_server(tmp_path, monkeypatch, data)
    target = tmp_path / "baseline.json"
    original = '{"keep": "previous baseline"}'
    target.write_text(original, encoding="utf-8")
    result = run_cli(home, "snapshot", "demo", "--output", target)
    assert result.returncode == 2, result.stdout + result.stderr
    assert "MCP snapshot failed:" in result.stderr
    # The existing connection manager retries discovery errors until its deadline.
    # Pin the actual second-page request as well as the timeout so a startup failure
    # cannot masquerade as successful coverage of partial discovery.
    expected = {"error": "TimeoutError", "cycle": "pagination limit", "duplicate": "Duplicate tool name"}[mode]
    assert expected in result.stderr, result.stdout + result.stderr
    events = [json.loads(line) for line in audit.read_text(encoding="utf-8").splitlines()]
    assert any(e.get("method") == "tools/list" and e["params"].get("cursor") is not None for e in events)
    assert "SYNTHETIC_FIXTURE_TOKEN" not in result.stdout + result.stderr
    assert target.read_text(encoding="utf-8") == original


@pytest.mark.parametrize("payload", [
    '{', '{"version":1,"version":1}', '{"version":NaN}', '[]',
    '[' * 66 + 'null' + ']' * 66, ' ' * (8 * 1024 * 1024) + '{}',
], ids=["malformed", "duplicate-key", "non-json-number", "wrong-envelope", "too-deep", "too-large"])
def test_invalid_snapshot_is_an_error_not_a_compatibility_verdict(tmp_path, monkeypatch, capsys, payload):
    from hermes_cli.mcp_config import mcp_command
    from hermes_cli.subcommands.mcp import build_mcp_parser
    import argparse

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    path = tmp_path / "invalid.json"
    path.write_text(payload, encoding="utf-8")
    parser = argparse.ArgumentParser()
    build_mcp_parser(parser.add_subparsers(dest="command"), cmd_mcp=mcp_command)
    args = parser.parse_args(["mcp", "diff", str(path), str(path), "--json"])
    with pytest.raises(SystemExit) as exc:
        args.func(args)
    assert exc.value.code == 2
    assert json.loads(capsys.readouterr().out)["status"] == "error"
