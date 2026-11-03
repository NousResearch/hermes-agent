"""Tests for the fsautocomplete (F# / Ionide) server registration.

fsautocomplete is a ``dotnet tool``. These tests cover registry wiring,
project-root resolution (``*.sln`` / ``*.fsproj`` are globs, so they
can't go through ``nearest_root``), spawn flags, and discovery of the
binary in the .NET global-tools directory when it is not on PATH.
"""
from __future__ import annotations

from pathlib import Path

import agent.lsp.install as install_mod
import agent.lsp.servers as srv
from agent.lsp.install import INSTALL_RECIPES, detect_status
from agent.lsp.servers import (
    ServerContext,
    find_server_for_file,
    language_id_for,
)


def test_fsharp_extensions_route_to_fsautocomplete():
    for name in ("App.fs", "App.fsi", "Script.fsx"):
        s = find_server_for_file(name)
        assert s is not None, name
        assert s.server_id == "fsautocomplete"


def test_fsharp_language_id():
    assert language_id_for("src/App.fs") == "fsharp"
    assert language_id_for("src/App.fsi") == "fsharp"
    assert language_id_for("scratch.fsx") == "fsharp"


def test_recipe_is_dotnet_tool():
    recipe = INSTALL_RECIPES["fsautocomplete"]
    assert recipe["strategy"] == "dotnet"
    assert recipe["bin"] == "fsautocomplete"
    assert recipe["pkg"] == "fsautocomplete"


def _make_dotnet_tool(home: Path, name: str = "fsautocomplete") -> Path:
    tools = home / "tools"
    tools.mkdir(parents=True)
    binary = tools / name
    binary.write_text("#!/bin/sh\n")
    binary.chmod(0o755)
    return binary


def test_existing_binary_finds_dotnet_global_tool(tmp_path, monkeypatch):
    """``dotnet tool install -g`` lands in $DOTNET_CLI_HOME/tools, often off PATH."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes_home"))
    monkeypatch.setenv("DOTNET_CLI_HOME", str(tmp_path / "dotnet_home"))
    binary = _make_dotnet_tool(tmp_path / "dotnet_home")
    monkeypatch.setattr(install_mod.shutil, "which", lambda name, **k: None)

    assert install_mod._existing_binary("fsautocomplete") == str(binary)


def test_existing_binary_dotnet_cli_home_hides_default_home(tmp_path, monkeypatch):
    """When DOTNET_CLI_HOME is set, ~/.dotnet/tools is not consulted."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes_home"))
    monkeypatch.setenv("DOTNET_CLI_HOME", str(tmp_path / "cli_home"))
    (tmp_path / "cli_home" / "tools").mkdir(parents=True)
    monkeypatch.setattr(install_mod.shutil, "which", lambda name, **k: None)

    assert install_mod._existing_binary("fsautocomplete") is None


def test_detect_status_installed_from_dotnet_tools(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes_home"))
    monkeypatch.setenv("DOTNET_CLI_HOME", str(tmp_path / "dotnet_home"))
    _make_dotnet_tool(tmp_path / "dotnet_home")
    monkeypatch.setattr(install_mod.shutil, "which", lambda name, **k: None)

    assert detect_status("fsautocomplete") == "installed"


def test_spawn_builds_command_from_path(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes_home"))
    monkeypatch.setattr(srv, "_which", lambda *names: "/usr/bin/fsautocomplete" if "fsautocomplete" in names else None)

    ctx = ServerContext(workspace_root=str(tmp_path), install_strategy="manual")
    spec = srv._spawn_fsautocomplete(str(tmp_path), ctx)
    assert spec is not None
    assert spec.command[0] == "/usr/bin/fsautocomplete"
    assert "--state-directory" in spec.command
    state_dir = spec.command[spec.command.index("--state-directory") + 1]
    assert str(tmp_path / "hermes_home") in state_dir
    assert spec.initialization_options.get("AutomaticWorkspaceInit") is True
    assert spec.cwd == str(tmp_path)
    assert spec.workspace_root == str(tmp_path)


def test_spawn_finds_dotnet_global_tool_when_not_on_path(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes_home"))
    monkeypatch.setenv("DOTNET_CLI_HOME", str(tmp_path / "dotnet_home"))
    binary = _make_dotnet_tool(tmp_path / "dotnet_home")
    monkeypatch.setattr(srv, "_which", lambda *names: None)
    monkeypatch.setattr(install_mod.shutil, "which", lambda name, **k: None)
    monkeypatch.setattr(install_mod, "_install_results", {})

    ctx = ServerContext(workspace_root=str(tmp_path), install_strategy="manual")
    spec = srv._spawn_fsautocomplete(str(tmp_path), ctx)
    assert spec is not None
    assert spec.command[0] == str(binary)


def test_spawn_returns_none_when_missing(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes_home"))
    monkeypatch.setenv("DOTNET_CLI_HOME", str(tmp_path / "empty_dotnet"))
    (tmp_path / "empty_dotnet" / "tools").mkdir(parents=True)
    monkeypatch.setattr(srv, "_which", lambda *names: None)
    monkeypatch.setattr(install_mod.shutil, "which", lambda name, **k: None)
    monkeypatch.setattr(install_mod, "_install_results", {})

    ctx = ServerContext(workspace_root=str(tmp_path), install_strategy="manual")
    assert srv._spawn_fsautocomplete(str(tmp_path), ctx) is None


def test_spawn_init_overrides_can_disable_automatic_workspace_init(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes_home"))
    monkeypatch.setattr(srv, "_which", lambda *names: "/usr/bin/fsautocomplete")

    ctx = ServerContext(
        workspace_root=str(tmp_path),
        install_strategy="manual",
        init_overrides={"fsautocomplete": {"AutomaticWorkspaceInit": False, "foo": 1}},
    )
    spec = srv._spawn_fsautocomplete(str(tmp_path), ctx)
    assert spec is not None
    assert spec.initialization_options["AutomaticWorkspaceInit"] is False
    assert spec.initialization_options["foo"] == 1


def test_spawn_sets_dotnet_root_from_dotnet_binary(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes_home"))
    sdk = tmp_path / "sdk"
    sdk.mkdir()
    dotnet = sdk / "dotnet"
    dotnet.write_text("")
    dotnet.chmod(0o755)

    def which(*names):
        if "fsautocomplete" in names:
            return "/usr/bin/fsautocomplete"
        if "dotnet" in names:
            return str(dotnet)
        return None

    monkeypatch.setattr(srv, "_which", which)
    ctx = ServerContext(workspace_root=str(tmp_path), install_strategy="manual")
    spec = srv._spawn_fsautocomplete(str(tmp_path), ctx)
    assert spec is not None
    assert spec.env.get("DOTNET_ROOT") == str(sdk)


def test_spawn_state_dir_is_stable_under_normcase(monkeypatch, tmp_path):
    """Windows case-folded paths must share one FSAC state directory."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes_home"))
    monkeypatch.setattr(srv, "_which", lambda *names: "/usr/bin/fsautocomplete")
    monkeypatch.setattr(srv.os.path, "normcase", lambda p: p.lower())
    ctx = ServerContext(workspace_root=str(tmp_path), install_strategy="manual")
    spec_a = srv._spawn_fsautocomplete(str(tmp_path / "Repo"), ctx)
    spec_b = srv._spawn_fsautocomplete(str(tmp_path / "repo"), ctx)
    assert spec_a is not None and spec_b is not None
    state_a = spec_a.command[spec_a.command.index("--state-directory") + 1]
    state_b = spec_b.command[spec_b.command.index("--state-directory") + 1]
    assert state_a == state_b


def test_spawn_does_not_clobber_user_dotnet_root(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes_home"))
    monkeypatch.setattr(srv, "_which", lambda *names: "/usr/bin/fsautocomplete")
    ctx = ServerContext(
        workspace_root=str(tmp_path),
        install_strategy="manual",
        env_overrides={"fsautocomplete": {"DOTNET_ROOT": "/custom/dotnet"}},
    )
    spec = srv._spawn_fsautocomplete(str(tmp_path), ctx)
    assert spec is not None
    assert spec.env["DOTNET_ROOT"] == "/custom/dotnet"


def test_root_prefers_nearest_sln(tmp_path: Path):
    repo = tmp_path / "repo"
    src = repo / "src" / "App"
    src.mkdir(parents=True)
    (src / "App.fsproj").write_text("")
    (repo / "App.sln").write_text("")
    found = srv._root_fsharp(str(src / "App.fs"), str(repo))
    assert found == str(repo)


def test_root_uses_fsproj_when_no_sln(tmp_path: Path):
    repo = tmp_path / "repo"
    src = repo / "src" / "App"
    src.mkdir(parents=True)
    (src / "App.fsproj").write_text("")
    found = srv._root_fsharp(str(src / "App.fs"), str(repo))
    assert found == str(src)


def test_root_accepts_slnx(tmp_path: Path):
    repo = tmp_path / "repo"
    src = repo / "src"
    src.mkdir(parents=True)
    (repo / "App.slnx").write_text("")
    found = srv._root_fsharp(str(src / "App.fs"), str(repo))
    assert found == str(repo)


def test_root_falls_back_to_workspace(tmp_path: Path):
    repo = tmp_path / "repo"
    src = repo / "src"
    src.mkdir(parents=True)
    found = srv._root_fsharp(str(src / "Scratch.fsx"), str(repo))
    assert found == str(repo)


def test_root_does_not_walk_above_workspace(tmp_path: Path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "Other.sln").write_text("")
    repo = tmp_path / "repo"
    src = repo / "src"
    src.mkdir(parents=True)
    found = srv._root_fsharp(str(src / "App.fs"), str(repo))
    assert found == str(repo)


def test_root_empty_workspace_does_not_walk(tmp_path: Path):
    """An empty workspace must not walk toward a stray .sln above the file."""
    (tmp_path / "Other.sln").write_text("")
    nested = tmp_path / "a" / "b" / "c"
    nested.mkdir(parents=True)
    (nested / "App.fs").write_text("")
    assert srv._root_fsharp(str(nested / "App.fs"), "") is None
    assert srv._root_fsharp(str(nested / "App.fs"), None) is None


def test_root_prefers_strong_marker_over_inner_directory_build_props(tmp_path: Path):
    repo = tmp_path / "repo"
    nested = repo / "src"
    nested.mkdir(parents=True)
    (nested / "Directory.Build.props").write_text("")
    (repo / "paket.dependencies").write_text("")
    found = srv._root_fsharp(str(nested / "App.fs"), str(repo))
    assert found == str(repo)


def test_root_uses_directory_build_props_as_fallback(tmp_path: Path):
    repo = tmp_path / "repo"
    nested = repo / "src"
    nested.mkdir(parents=True)
    (nested / "Directory.Build.props").write_text("")
    found = srv._root_fsharp(str(nested / "App.fs"), str(repo))
    assert found == str(nested)


def test_root_uses_global_json(tmp_path: Path):
    repo = tmp_path / "repo"
    nested = repo / "src"
    nested.mkdir(parents=True)
    (nested / "global.json").write_text("{}")
    found = srv._root_fsharp(str(nested / "App.fs"), str(repo))
    assert found == str(nested)


def test_install_dotnet_command(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    captured = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        staging = install_mod.hermes_lsp_bin_dir()
        binary = staging / "fsautocomplete"
        binary.write_text("#!/bin/sh\n")
        binary.chmod(0o755)
        from unittest.mock import MagicMock
        return MagicMock(returncode=0, stderr="")

    monkeypatch.setattr(install_mod.subprocess, "run", fake_run)
    monkeypatch.setattr(install_mod.shutil, "which", lambda name, **k: "/usr/bin/dotnet" if name == "dotnet" else None)

    resolved = install_mod._install_dotnet("fsautocomplete", "fsautocomplete")
    assert resolved is not None
    assert resolved.endswith("fsautocomplete")
    assert captured["cmd"][:4] == ["/usr/bin/dotnet", "tool", "install", "--tool-path"]
    assert captured["cmd"][-1] == "fsautocomplete"


def test_install_dotnet_missing_sdk(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(install_mod.shutil, "which", lambda name, **k: None)
    assert install_mod._install_dotnet("fsautocomplete", "fsautocomplete") is None
