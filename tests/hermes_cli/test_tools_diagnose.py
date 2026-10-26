"""Tool-surface inspection command regressions (#47404)."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_cli.subcommands.tools import build_tools_parser


def test_diagnose_parser_accepts_platform_and_json():
    parser = argparse.ArgumentParser()
    build_tools_parser(parser.add_subparsers(), cmd_tools=lambda args: None)
    args = parser.parse_args(["tools", "diagnose", "--platform", "telegram", "--json"])
    assert (args.tools_action, args.platform, args.json) == ("diagnose", "telegram", True)


@pytest.fixture
def inspection_home(tmp_path, monkeypatch):
    home = tmp_path / "hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.chdir(tmp_path)
    return home


def _write_config(home, toolsets, search="off"):
    from hermes_cli.config import atomic_config_write

    atomic_config_write(home / "config.yaml", {
        "platform_toolsets": {"cli": toolsets},
        "tools": {"tool_search": {"enabled": search}},
        "memory": {"provider": "diagnose-fixture"},
    })


def _schema(name):
    return {"name": name, "description": "Inspection fixture", "parameters": {"type": "object", "properties": {}}}


@pytest.fixture
def memory_provider(monkeypatch):
    from agent import agent_init
    from agent.memory_provider import MemoryProvider

    class ProbeMemory(MemoryProvider):
        name = "diagnose-fixture"

        def is_available(self):
            return True

        def initialize(self, session_id, **kwargs):
            self.initialized = True

        def get_tool_schemas(self):
            return [_schema("memory_probe")]

    provider = ProbeMemory()
    monkeypatch.setattr(agent_init, "resolve_memory_provider", lambda *args, **kwargs: provider)
    return provider


@pytest.mark.parametrize("search", ["on", "off"])
def test_report_matches_real_agent_with_search_and_injected_providers(inspection_home, monkeypatch, memory_provider, search):
    from agent.context_compressor import ContextCompressor
    from hermes_cli import prompt_size
    from hermes_cli.tools_diagnose import compute_tool_diagnostics
    from tools.registry import registry

    registry.register(name="mcp_diagnose_probe", toolset="mcp-diagnose-probe",
                      schema=_schema("mcp_diagnose_probe"), handler=lambda args, **kw: "{}")
    # Stub only external schema boundaries; real AIAgent initialization, tool-search
    # replacement and provider injection decide the final report surface.
    monkeypatch.setattr(ContextCompressor, "get_tool_schemas", lambda self: [_schema("context_probe")])
    _write_config(inspection_home, ["memory", "context_engine", "mcp-diagnose-probe"], search)
    build = prompt_size._build_inspection_agent
    expected = []
    closed = []

    def observe(platform):
        agent = build(platform)
        expected.extend(tool["function"]["name"] for tool in agent.tools)
        close = agent.close

        def tracked_close():
            closed.append(True)
            close()

        monkeypatch.setattr(agent, "close", tracked_close)
        return agent

    monkeypatch.setattr(prompt_size, "_build_inspection_agent", observe)
    report = compute_tool_diagnostics()
    assert report["tools_visible"] == sorted(set(expected))
    assert {"memory_probe", "context_probe"} <= set(report["tools_visible"])
    assert report["tool_search"]["bridge_visible"] is (search == "on")
    assert ("mcp_diagnose_probe" in report["tools_visible"]) is (search == "off")
    assert report["provider_tools"]["memory"]["visible_tools"] == ["memory_probe"]
    assert report["provider_tools"]["context_engine"]["visible_tools"] == ["context_probe"]
    assert memory_provider.initialized
    assert closed == [True]


def test_empty_platform_keeps_provider_tools_hidden(inspection_home, memory_provider):
    from hermes_cli.tools_diagnose import compute_tool_diagnostics

    _write_config(inspection_home, [])
    report = compute_tool_diagnostics()
    assert report["tools_visible"] == []
    assert report["provider_tools"]["memory"]["visible_tools"] == []
    assert report["config_sources"]["platform_toolsets"] == "platform_toolsets.cli"


def test_agent_closed_when_reporting_fails(inspection_home, monkeypatch):
    from hermes_cli import prompt_size, tools_diagnose

    closed = []
    agent = SimpleNamespace(tools=[], close=lambda: closed.append(True))
    monkeypatch.setattr(prompt_size, "_build_inspection_agent", lambda platform: agent)
    with pytest.raises(AttributeError):
        tools_diagnose.compute_tool_diagnostics()
    assert closed == [True]


def test_json_error_does_not_echo_provider_secrets(monkeypatch, capsys):
    from hermes_cli import tools_diagnose

    def fail(platform):
        print("provider-secret-canary")
        raise RuntimeError("provider-secret-canary")

    monkeypatch.setattr(tools_diagnose, "compute_tool_diagnostics", fail)
    assert tools_diagnose.tools_diagnose_command(SimpleNamespace(platform="cli", json=True)) == 1
    captured = capsys.readouterr()
    assert "provider-secret-canary" not in captured.out + captured.err
    assert json.loads(captured.out) == {"error": "inspection_failed", "error_type": "RuntimeError"}


@pytest.mark.parametrize("platform,code", [("cli", 0), ("not-a-platform", 2)])
def test_real_cli_emits_json_without_running_a_turn(inspection_home, platform, code):
    _write_config(inspection_home, [])
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run([sys.executable, "-m", "hermes_cli.main", "tools", "diagnose",
                             "--platform", platform, "--json"], cwd=root, env=dict(os.environ),
                            capture_output=True, text=True, timeout=60, check=False)
    assert result.returncode == code, result.stderr
    report = json.loads(result.stdout)
    if code:
        assert report == {"error": "unknown_platform"}
    else:
        assert report["scope"] == "fresh_inspection_agent"
        assert report["tools_visible"] == []
