from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "optional-skills"
    / "autonomous-ai-agents"
    / "autonomy-stack"
    / "scripts"
    / "autonomy_stack.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location("autonomy_stack_skill", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_install_plugins_copies_selected_plugins_and_utils(tmp_path: Path, monkeypatch):
    mod = load_module()
    hermes_home = tmp_path / ".hermes"
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    repo_dir = tmp_path / "repo"
    (repo_dir / "evey-autonomy").mkdir(parents=True)
    (repo_dir / "evey-status").mkdir(parents=True)
    (repo_dir / "evey-autonomy" / "plugin.yaml").write_text("name: evey-autonomy\n", encoding="utf-8")
    (repo_dir / "evey-status" / "plugin.yaml").write_text("name: evey-status\n", encoding="utf-8")
    (repo_dir / "evey_utils.py").write_text("# helper\n", encoding="utf-8")

    monkeypatch.setattr(mod, "_ensure_repo", lambda update=False: repo_dir)

    result = mod.install_plugins(["evey-autonomy", "evey-status"], update_repo=False)

    assert result["success"] is True
    assert (hermes_home / "plugins" / "evey-autonomy" / "plugin.yaml").exists()
    assert (hermes_home / "plugins" / "evey-status" / "plugin.yaml").exists()
    assert (hermes_home / "plugins" / "evey_utils.py").exists()


def test_detect_plugin_status_reports_missing_and_builtin_loop(tmp_path: Path, monkeypatch):
    mod = load_module()
    hermes_home = tmp_path / ".hermes"
    plugins_dir = hermes_home / "plugins"
    plugins_dir.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    (plugins_dir / "evey-status").mkdir()
    (plugins_dir / "evey_utils.py").write_text("# helper\n", encoding="utf-8")

    result = mod.detect_plugin_status(["evey-status", "evey-goals"])

    assert result["success"] is True
    assert result["installed"] == ["evey-status"]
    assert result["missing"] == ["evey-goals"]
    assert result["evey_utils"] is True
    assert result["hermes_skill_loop"]["built_in"] is True
