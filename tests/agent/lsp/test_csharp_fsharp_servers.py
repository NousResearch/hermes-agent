"""Tests for the C# (csharp-ls) and F# (FsAutoComplete) LSP server entries.

Covers:

1. The two install recipes exist with the ``dotnet`` strategy.
2. ``_install_dotnet`` runs the right ``dotnet tool install -g`` command
   and links the installed binary into the staging dir.
3. The server registry routes .cs/.csx → csharp-ls and
   .fs/.fsi/.fsx → fsautocomplete, and their spawn specs use stdio.
4. ``_root_dotnet`` walks upward and stops at the directory containing
   a ``.sln`` / ``.csproj`` / ``.fsproj`` (per-project names are
   not known in advance, so the resolver uses fnmatch, not
   nearest_root's exact-name matching).
"""
from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock

from agent.lsp.install import INSTALL_RECIPES
from agent.lsp.servers import ServerContext, find_server_for_file


# ---------------------------------------------------------------------------
# Recipes
# ---------------------------------------------------------------------------


def test_csharp_ls_recipe_uses_dotnet_strategy():
    recipe = INSTALL_RECIPES["csharp-ls"]
    assert recipe["strategy"] == "dotnet", recipe
    assert recipe["bin"] == "csharp-ls"


def test_fsautocomplete_recipe_uses_dotnet_strategy():
    recipe = INSTALL_RECIPES["fsautocomplete"]
    assert recipe["strategy"] == "dotnet", recipe
    assert recipe["bin"] == "fsautocomplete"


# ---------------------------------------------------------------------------
# _install_dotnet behavior
# ---------------------------------------------------------------------------


def test_install_dotnet_runs_correct_subprocess(tmp_path, monkeypatch):
    """When the binary doesn't exist yet, _install_dotnet invokes
    ``dotnet tool install -g <pkg>`` and links the resulting binary
    into the staging dir."""
    from agent.lsp import install as install_mod

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    # Pretend a global-tools dir exists with the binary the recipe
    # expects.  Symlinking a fake file is fine for the test — the
    # function doesn't run the binary.
    fake_tools_root = tmp_path / ".dotnet" / "tools"
    fake_tools_root.mkdir(parents=True)
    fake_bin = fake_tools_root / "csharp-ls"
    fake_bin.write_text("#!/bin/sh\n")
    fake_bin.chmod(0o755)

    # PATH probe for dotnet
    monkeypatch.setattr(
        install_mod.shutil, "which",
        lambda c: "/usr/bin/dotnet" if c == "dotnet" else None,
    )

    captured = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        return MagicMock(returncode=0, stderr="")

    monkeypatch.setattr(install_mod.subprocess, "run", fake_run)

    # Also stub `_existing_binary` so the function actually proceeds
    # to install (it would short-circuit otherwise).
    monkeypatch.setattr(install_mod, "_existing_binary", lambda n: None)

    # Point DOTNET_CLI_HOME-less resolution at our fake root.
    # `_install_dotnet` reads $HOME when DOTNET_CLI_HOME is unset,
    # so we have to monkeypatch Path.home() too.
    monkeypatch.setattr(install_mod.Path, "home", classmethod(lambda cls: tmp_path))

    result = install_mod._install_dotnet("csharp-ls", "csharp-ls")

    # The subprocess call should have been `dotnet tool install -g csharp-ls`
    # (we stub `shutil.which` to return the full path, so the first
    # element is whatever PATH resolved to).
    assert os.path.basename(captured["cmd"][0]) == "dotnet"
    assert captured["cmd"][1:4] == ["tool", "install", "-g"]
    assert "csharp-ls" in captured["cmd"]
    # The function should have returned a path (the symlink in staging dir)
    assert result is not None
    assert os.path.basename(result) == "csharp-ls"
    # The symlink should exist in the staging dir
    assert (tmp_path / "lsp" / "bin" / "csharp-ls").exists()


def test_install_dotnet_bails_when_dotnet_missing(tmp_path, monkeypatch):
    """If `dotnet` isn't on PATH, the install is a clean no-op."""
    from agent.lsp import install as install_mod

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(install_mod.shutil, "which", lambda c: None)

    ran = {"subprocess": False}

    def fake_run(*a, **kw):
        ran["subprocess"] = True
        return MagicMock(returncode=0, stderr="")

    monkeypatch.setattr(install_mod.subprocess, "run", fake_run)
    monkeypatch.setattr(install_mod, "_existing_binary", lambda n: None)

    assert install_mod._install_dotnet("csharp-ls", "csharp-ls") is None
    assert ran["subprocess"] is False


def test_install_dotnet_handles_already_installed(tmp_path, monkeypatch):
    """`dotnet tool install -g` returns non-zero if the tool is already
    installed.  We must still link whatever is on disk."""
    from agent.lsp import install as install_mod

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    fake_tools_root = tmp_path / ".dotnet" / "tools"
    fake_tools_root.mkdir(parents=True)
    fake_bin = fake_tools_root / "fsautocomplete"
    fake_bin.write_text("#!/bin/sh\n")
    fake_bin.chmod(0o755)

    monkeypatch.setattr(
        install_mod.shutil, "which",
        lambda c: "/usr/bin/dotnet" if c == "dotnet" else None,
    )
    monkeypatch.setattr(
        install_mod.Path, "home", classmethod(lambda cls: tmp_path),
    )
    monkeypatch.setattr(install_mod, "_existing_binary", lambda n: None)

    def fake_run(cmd, **kwargs):
        # Simulate "already installed" failure
        return MagicMock(
            returncode=1,
            stderr="Tool 'fsautocomplete' is already installed.",
        )

    monkeypatch.setattr(install_mod.subprocess, "run", fake_run)

    result = install_mod._install_dotnet("fsautocomplete", "fsautocomplete")
    assert result is not None
    assert os.path.basename(result) == "fsautocomplete"
    assert (tmp_path / "lsp" / "bin" / "fsautocomplete").exists()


# ---------------------------------------------------------------------------
# Registry routing
# ---------------------------------------------------------------------------


def test_registry_routes_csharp_files_to_csharp_ls():
    srv = find_server_for_file("/tmp/whatever/Foo.cs")
    assert srv is not None
    assert srv.server_id == "csharp-ls"


def test_registry_routes_fsharp_files_to_fsautocomplete():
    for ext in (".fs", ".fsi", ".fsx"):
        srv = find_server_for_file(f"/tmp/whatever/Foo{ext}")
        assert srv is not None, f"no server for {ext}"
        assert srv.server_id == "fsautocomplete", f"wrong server for {ext}"


def test_csharp_ls_spawn_uses_stdio(tmp_path, monkeypatch):
    """_spawn_csharp_ls must spawn the binary in stdio LSP mode (no args)."""
    # Plant a fake binary on PATH
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake = bin_dir / "csharp-ls"
    fake.write_text("#!/bin/sh\n")
    fake.chmod(0o755)
    monkeypatch.setenv("PATH", str(bin_dir) + os.pathsep + os.environ.get("PATH", ""))

    from agent.lsp.servers import _spawn_csharp_ls

    ctx = ServerContext(workspace_root=str(tmp_path), install_strategy="manual")
    spec = _spawn_csharp_ls(str(tmp_path), ctx)
    assert spec is not None
    # csharp-ls / fsautocomplete default to stdio LSP, no --stdio needed
    assert spec.command[0].endswith("csharp-ls")
    assert spec.command[1:] == []
    # First-push diagnostics should be enabled so we get something
    # useful on the very first edit.
    assert spec.seed_diagnostics_on_first_push is True


def test_fsautocomplete_spawn_uses_legacy_engine_by_default(tmp_path, monkeypatch):
    """fsautocomplete default (legacy) engine is what we spawn by default.
    Both engines are equivalent for capabilities (neither advertises
    publishDiagnostics — they use LSP 3.17 pull diagnostics instead)."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake = bin_dir / "fsautocomplete"
    fake.write_text("#!/bin/sh\n")
    fake.chmod(0o755)
    monkeypatch.setenv("PATH", str(bin_dir) + os.pathsep + os.environ.get("PATH", ""))

    from agent.lsp.servers import _spawn_fsautocomplete

    ctx = ServerContext(workspace_root=str(tmp_path), install_strategy="manual")
    spec = _spawn_fsautocomplete(str(tmp_path), ctx)
    assert spec is not None
    assert spec.command[0].endswith("fsautocomplete")
    # No flag = legacy engine.  Either works for the LSP; we just
    # stay on the stable default.
    assert spec.command == [spec.command[0]]


# ---------------------------------------------------------------------------
# _root_dotnet — per-project .sln / .csproj / .fsproj discovery
# ---------------------------------------------------------------------------


def test_root_dotnet_finds_solution_in_project_root(tmp_path: Path):
    repo = tmp_path / "repo"
    nested = repo / "src" / "MyApp"
    nested.mkdir(parents=True)
    (repo / "MyApp.sln").write_text("")
    (nested / "Program.cs").write_text("")

    from agent.lsp.servers import _root_dotnet
    assert _root_dotnet(str(nested / "Program.cs"), str(repo)) == str(repo)


def test_root_dotnet_finds_csproj_in_subdir(tmp_path: Path):
    """No .sln — should still find the .csproj (per-project names differ)."""
    repo = tmp_path / "repo"
    sub = repo / "src" / "MyApp"
    sub.mkdir(parents=True)
    (sub / "MyApp.csproj").write_text("")
    (sub / "Program.cs").write_text("")

    from agent.lsp.servers import _root_dotnet
    assert _root_dotnet(str(sub / "Program.cs"), str(repo)) == str(sub)


def test_root_dotnet_finds_fsproj(tmp_path: Path):
    repo = tmp_path / "repo"
    sub = repo / "src" / "MyLib"
    sub.mkdir(parents=True)
    (sub / "MyLib.fsproj").write_text("")
    (sub / "Library.fs").write_text("")

    from agent.lsp.servers import _root_dotnet
    assert _root_dotnet(str(sub / "Library.fs"), str(repo)) == str(sub)


def test_root_dotnet_finds_global_json(tmp_path: Path):
    """global.json is enough to anchor a .NET workspace."""
    repo = tmp_path / "repo"
    sub = repo / "src"
    sub.mkdir(parents=True)
    (repo / "global.json").write_text('{"sdk": {"version": "10.0.108"}}')
    (sub / "Thing.cs").write_text("")

    from agent.lsp.servers import _root_dotnet
    assert _root_dotnet(str(sub / "Thing.cs"), str(repo)) == str(repo)


def test_root_dotnet_falls_back_to_workspace_when_no_markers(tmp_path: Path):
    """No .sln / .csproj / .fsproj / global.json — use the workspace root."""
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "Loose.cs").write_text("")

    from agent.lsp.servers import _root_dotnet
    assert _root_dotnet(str(repo / "Loose.cs"), str(repo)) == str(repo)


def test_root_dotnet_finds_directory_build_props(tmp_path: Path):
    repo = tmp_path / "repo"
    sub = repo / "src" / "App"
    sub.mkdir(parents=True)
    (repo / "Directory.Build.props").write_text("<Project></Project>")
    (sub / "App.cs").write_text("")

    from agent.lsp.servers import _root_dotnet
    assert _root_dotnet(str(sub / "App.cs"), str(repo)) == str(repo)
