"""Tests for Swift SourceKit-LSP registration and spawn behavior."""
from __future__ import annotations

import agent.lsp.servers as srv
from agent.lsp.install import detect_status
from agent.lsp.servers import ServerContext, find_server_for_file


def test_swift_files_route_to_sourcekit_lsp():
    server = find_server_for_file("Sources/App/Main.swift")
    assert server is not None
    assert server.server_id == "sourcekit-lsp"


def test_sourcekit_install_status_is_manual_or_installed():
    assert detect_status("sourcekit-lsp") in {"manual-only", "installed"}


def test_sourcekit_root_prefers_package_manifest(tmp_path):
    package = tmp_path / "Package"
    sources = package / "Sources" / "App"
    sources.mkdir(parents=True)
    (package / "Package.swift").write_text("// swift-tools-version: 6.0\n")
    file_path = sources / "Main.swift"
    file_path.write_text("print(\"hello\")\n")

    assert srv._root_swift(str(file_path), str(tmp_path)) == str(package)


def test_sourcekit_spawn_uses_path_binary_and_config_overrides(monkeypatch, tmp_path):
    monkeypatch.setattr(srv, "_which", lambda *names: "/usr/bin/sourcekit-lsp")
    ctx = ServerContext(
        workspace_root=str(tmp_path),
        install_strategy="manual",
        env_overrides={"sourcekit-lsp": {"SOURCEKIT_LOGGING": "1"}},
        init_overrides={"sourcekit-lsp": {"backgroundIndexing": True}},
    )

    spec = srv._spawn_sourcekit_lsp(str(tmp_path), ctx)

    assert spec is not None
    assert spec.command == ["/usr/bin/sourcekit-lsp"]
    assert spec.cwd == str(tmp_path)
    assert spec.workspace_root == str(tmp_path)
    assert spec.env == {"SOURCEKIT_LOGGING": "1"}
    assert spec.initialization_options == {"backgroundIndexing": True}


def test_sourcekit_spawn_prefers_binary_override(monkeypatch, tmp_path):
    override = tmp_path / "sourcekit-lsp"
    override.write_text("")
    monkeypatch.setattr(srv, "_which", lambda *names: "/usr/bin/sourcekit-lsp")
    ctx = ServerContext(
        workspace_root=str(tmp_path),
        binary_overrides={"sourcekit-lsp": [str(override)]},
    )

    spec = srv._spawn_sourcekit_lsp(str(tmp_path), ctx)

    assert spec is not None
    assert spec.command == [str(override)]


def test_sourcekit_spawn_skips_when_binary_missing(monkeypatch, tmp_path):
    import agent.lsp.install as install

    monkeypatch.setattr(srv, "_which", lambda *names: None)
    monkeypatch.setattr(install, "try_install", lambda *args: None)
    ctx = ServerContext(workspace_root=str(tmp_path), install_strategy="manual")
    assert srv._spawn_sourcekit_lsp(str(tmp_path), ctx) is None
