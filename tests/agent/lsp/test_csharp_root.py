"""Root resolution for the csharp-ls server.

``_root_csharp`` can't use ``nearest_root`` because .sln/.csproj files
have project-specific names (exact-name matching can't find them), so
it walks up with suffix matching.  These tests pin the contract:

  - .sln directory wins over a nearer .csproj directory (solution-level
    load gives csharp-ls cross-project resolution).
  - .csproj / global.json is the fallback root when no .sln exists.
  - macOS resource-fork litter (``._foo.sln``) is ignored.
  - Unity projects (``ProjectSettings/ProjectVersion.txt``) gate the
    server OFF — their .csproj files are editor-generated and stale or
    absent in headless checkouts.
  - No marker at all falls back to the workspace root.
"""
from __future__ import annotations


def _resolve(file_path, workspace):
    from agent.lsp.servers import _root_csharp
    return _root_csharp(str(file_path), str(workspace))


def test_sln_dir_wins_over_nearer_csproj(tmp_path):
    (tmp_path / "app.sln").write_text("")
    proj = tmp_path / "App"
    proj.mkdir()
    (proj / "App.csproj").write_text("")
    src = proj / "Program.cs"
    src.write_text("class C {}")
    assert _resolve(src, tmp_path) == str(tmp_path)


def test_csproj_fallback_without_sln(tmp_path):
    proj = tmp_path / "App"
    proj.mkdir()
    (proj / "App.csproj").write_text("")
    src = proj / "Program.cs"
    src.write_text("class C {}")
    assert _resolve(src, tmp_path) == str(proj)


def test_macos_resource_fork_sln_ignored(tmp_path):
    (tmp_path / "._app.sln").write_text("")
    proj = tmp_path / "App"
    proj.mkdir()
    (proj / "App.csproj").write_text("")
    src = proj / "Program.cs"
    src.write_text("class C {}")
    assert _resolve(src, tmp_path) == str(proj)


def test_unity_project_gated_off(tmp_path):
    ps = tmp_path / "ProjectSettings"
    ps.mkdir()
    (ps / "ProjectVersion.txt").write_text("m_EditorVersion: 2022.3.0f1")
    (tmp_path / "app.sln").write_text("")
    scripts = tmp_path / "Assets" / "Scripts"
    scripts.mkdir(parents=True)
    src = scripts / "Foo.cs"
    src.write_text("class Foo {}")
    assert _resolve(src, tmp_path) is None


def test_no_marker_falls_back_to_workspace(tmp_path):
    sub = tmp_path / "loose"
    sub.mkdir()
    src = sub / "Script.cs"
    src.write_text("class C {}")
    assert _resolve(src, tmp_path) == str(tmp_path)


def test_slnx_recognized(tmp_path):
    (tmp_path / "app.slnx").write_text("")
    proj = tmp_path / "App"
    proj.mkdir()
    (proj / "App.csproj").write_text("")
    src = proj / "Program.cs"
    src.write_text("class C {}")
    assert _resolve(src, tmp_path) == str(tmp_path)


def test_registry_matches_cs_files():
    from agent.lsp.servers import find_server_for_file
    srv = find_server_for_file("/some/where/Foo.cs")
    assert srv is not None
    assert srv.server_id == "csharp-ls"
