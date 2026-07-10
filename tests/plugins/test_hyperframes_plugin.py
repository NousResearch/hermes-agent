"""Tests for the hyperframes Hermes plugin bridge."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

PLUGIN_DIR = Path(__file__).resolve().parents[2] / "plugins" / "hyperframes"


def load_plugin():
    package_name = "hyperframes_test_plugin"
    for name in list(sys.modules):
        if name == package_name or name.startswith(f"{package_name}."):
            del sys.modules[name]
    spec = importlib.util.spec_from_file_location(
        package_name,
        PLUGIN_DIR / "__init__.py",
        submodule_search_locations=[str(PLUGIN_DIR)],
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[package_name] = module
    spec.loader.exec_module(module)
    return module


class _FakeContext:
    def __init__(self) -> None:
        self.tools: list[dict] = []
        self.commands: list[tuple] = []
        self.cli_commands: dict[str, dict] = {}
        self.hooks: list[tuple[str, object]] = []

    def register_tool(self, **kwargs):
        self.tools.append(kwargs)

    def register_command(self, *args, **kwargs):
        self.commands.append((args, kwargs))

    def register_cli_command(self, name, **kwargs):
        self.cli_commands[name] = kwargs

    def register_hook(self, hook_name, callback):
        self.hooks.append((hook_name, callback))


def test_register_exposes_tools_and_cli(tmp_path, monkeypatch):
    module = load_plugin()
    monkeypatch.setattr(module.core, "plugin_dir", lambda: PLUGIN_DIR)
    monkeypatch.setattr(module.core, "repo_root", lambda: PLUGIN_DIR.parents[1])
    monkeypatch.setattr(module.core, "get_hermes_home", lambda: tmp_path / ".hermes")
    monkeypatch.setattr(module.core, "display_hermes_home", lambda: str(tmp_path / ".hermes"))

    ctx = _FakeContext()
    module.register(ctx)

    names = {tool["name"] for tool in ctx.tools}
    assert names == {
        "hyperframes_status",
        "hyperframes_setup",
        "hyperframes_install",
        "hyperframes_init",
        "hyperframes_validate",
        "hyperframes_render",
        "hyperframes_preview",
        "hyperframes_capture",
        "hyperframes_audio",
    }
    assert ctx.tools[0]["toolset"] == "hyperframes"
    assert "hyperframes" in ctx.cli_commands
    assert ctx.commands


def test_materialize_skill_symlink(tmp_path):
    module = load_plugin()
    src = tmp_path / "bundled"
    src.mkdir()
    (src / "SKILL.md").write_text("---\nname: hyperframes\n---\n", encoding="utf-8")
    dst = tmp_path / "skills" / "hyperframes"

    result = module.core._materialize_skill_source(src, dst)
    assert result["ok"] is True
    assert (dst / "SKILL.md").is_file()


def test_sync_requires_bundled_skill(tmp_path, monkeypatch):
    module = load_plugin()
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(module.core, "bundled_skill_dir", lambda: home / "missing")
    monkeypatch.setattr(module.core, "user_skill_path", lambda: home / "skills" / "hyperframes")

    result = module.core.sync_skill_link()
    assert result["ok"] is False
    assert "install" in result.get("hint", "").lower() or "missing" in result["error"].lower()


def test_status_reports_paths(tmp_path, monkeypatch):
    module = load_plugin()
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(module.core, "get_hermes_home", lambda: home)
    monkeypatch.setattr(module.core, "display_hermes_home", lambda: str(home))
    monkeypatch.setattr(module.core, "_node_major_version", lambda: 22)
    monkeypatch.setattr(module.core, "_hyperframes_version", lambda: "0.4.2")
    monkeypatch.setattr(module.core, "_which", lambda name: name in {"ffmpeg", "npm"})

    payload = json.loads(module.core.handle_status())
    assert payload["ok"] is True
    assert payload["environment"]["ready"] is True
    assert "projects_root" in payload


def test_validate_project_missing_dir(tmp_path, monkeypatch):
    module = load_plugin()
    monkeypatch.setattr(
        module.core,
        "_resolve_project_dir",
        lambda project_dir=None, project_name=None: tmp_path / "missing",
    )
    result = module.core.validate_project(project_dir=str(tmp_path / "missing"))
    assert result["ok"] is False
    assert "not found" in result["error"].lower()
