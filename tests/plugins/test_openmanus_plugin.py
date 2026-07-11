from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

PLUGIN_DIR = Path(__file__).resolve().parents[2] / "plugins" / "openmanus"


def load_plugin():
    package_name = "openmanus_test_plugin"
    for name in list(sys.modules):
        if name == package_name or name.startswith(f"{package_name}."):
            del sys.modules[name]
    spec = importlib.util.spec_from_file_location(
        package_name,
        PLUGIN_DIR / "__init__.py",
        submodule_search_locations=[str(PLUGIN_DIR)],
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[package_name] = module
    spec.loader.exec_module(module)
    return module


class _Context:
    def __init__(self):
        self.tools = []
        self.commands = []
        self.cli_commands = {}

    def register_tool(self, **kwargs):
        self.tools.append(kwargs)

    def register_command(self, *args, **kwargs):
        self.commands.append((args, kwargs))

    def register_cli_command(self, name, **kwargs):
        self.cli_commands[name] = kwargs


def test_register_exposes_openmanus_to_agents_and_moa():
    module = load_plugin()
    ctx = _Context()
    module.register(ctx)
    assert {tool["name"] for tool in ctx.tools} == {
        "openmanus_capabilities",
        "openmanus_run",
        "openmanus_wide_research",
    }
    assert {tool["toolset"] for tool in ctx.tools} == {"openmanus"}
    assert "openmanus" in ctx.cli_commands


def test_workspace_confinement_rejects_escape(tmp_path, monkeypatch):
    module = load_plugin()
    root = tmp_path / "allowed"
    root.mkdir()
    monkeypatch.setattr(module.core, "_load_entry", lambda: {"workspace_root": str(root)})
    try:
        module.core.resolve_workspace("..")
    except ValueError as exc:
        assert "escapes" in str(exc)
    else:
        raise AssertionError("workspace escape was accepted")


def test_dry_run_writes_plan_without_spawning(tmp_path, monkeypatch):
    module = load_plugin()
    root = tmp_path / "allowed"
    root.mkdir()
    home = tmp_path / ".hermes"
    monkeypatch.setattr(module.core, "_load_entry", lambda: {"workspace_root": str(root)})
    monkeypatch.setattr(module.core, "get_hermes_home", lambda: home)
    monkeypatch.setattr(module.core, "_source_revision", lambda: "test-revision")
    monkeypatch.setattr(module.core.subprocess, "run", lambda *a, **k: (_ for _ in ()).throw(AssertionError("spawned")))
    payload = module.core.run_task({"task": "inspect only", "dry_run": True})
    assert payload["ok"] is True
    assert payload["status"] == "planned"
    assert Path(payload["receipt_path"]).is_file()
    saved = json.loads(Path(payload["receipt_path"]).read_text(encoding="utf-8"))
    assert saved["source_revision"] == "test-revision"


def test_live_run_requires_explicit_ack(tmp_path, monkeypatch):
    module = load_plugin()
    root = tmp_path / "allowed"
    root.mkdir()
    monkeypatch.setattr(module.core, "_load_entry", lambda: {"workspace_root": str(root)})
    payload = module.core.run_task({"task": "write", "dry_run": False})
    assert payload["ok"] is False
    assert "acknowledge_side_effects" in payload["error"]


def test_wide_research_uses_configured_parallel_cap(tmp_path, monkeypatch):
    module = load_plugin()
    root = tmp_path / "allowed"
    root.mkdir()
    monkeypatch.setattr(module.core, "_load_entry", lambda: {"workspace_root": str(root), "max_parallel": 2})
    calls = []

    def fake_run(args):
        calls.append(args["task"])
        return {"ok": True, "status": "planned", "stdout": ""}

    monkeypatch.setattr(module.core, "run_task", fake_run)
    monkeypatch.setattr(module.core, "get_hermes_home", lambda: tmp_path / ".hermes")
    payload = module.core.wide_research({"items": ["one", "two", "three"], "dry_run": True, "max_parallel": 99})
    assert payload["ok"] is True
    assert payload["parallelism"] == 2
    assert calls == ["one", "two", "three"]
