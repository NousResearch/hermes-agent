"""LSP needs separate consent to run servers and install missing packages."""
import json
import os

import pytest

from agent.lsp.manager import LSPService
from agent.lsp.servers import ServerContext, find_server_for_file


@pytest.mark.parametrize("raw_config", [False, True])
@pytest.mark.parametrize(
    "config,enabled,strategy",
    [
        ({}, False, "manual"),
        ({"lsp": {}}, False, "manual"),
        ({"lsp": {"enabled": True}}, True, "manual"),
        ({"lsp": {"install_strategy": "auto"}}, False, "auto"),
        ({"lsp": {"enabled": True, "install_strategy": "auto"}}, True, "auto"),
        ({"lsp": {"enabled": False, "install_strategy": "off"}}, False, "off"),
    ],
)
def test_service_defaults_and_explicit_config(
    tmp_path, monkeypatch, raw_config, config, enabled, strategy
):
    # Exercise both the real YAML/deep-merge loader and the factory's own fallbacks.
    if raw_config:
        monkeypatch.setattr("hermes_cli.config.load_config_readonly", lambda: config)
    else:
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        (tmp_path / "config.yaml").write_text(json.dumps(config), encoding="utf-8")
    svc = LSPService.create_from_config()
    assert svc is not None
    try:
        assert svc.is_active() is enabled
        assert svc.get_status()["install_strategy"] == strategy
    finally:
        svc.shutdown()


@pytest.mark.parametrize("location", [None, "path", "managed"])
def test_standalone_context_uses_existing_binaries_without_installing(
    tmp_path, monkeypatch, location
):
    from agent.lsp import install

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    path_bin = tmp_path / "path-bin"
    path_bin.mkdir()
    monkeypatch.setenv("PATH", str(path_bin))

    def unexpected_install(*args):
        pytest.fail("a default ServerContext must not run a package installer")

    monkeypatch.setattr(install, "_do_install", unexpected_install)
    monkeypatch.setattr(install, "_install_results", {})
    binary = None
    if location is not None:
        folder = path_bin if location == "path" else install.hermes_lsp_bin_dir()
        binary = folder / ("pyright-langserver.exe" if os.name == "nt" else "pyright-langserver")
        binary.write_text("", encoding="utf-8")
        binary.chmod(0o755)
    server = find_server_for_file(str(tmp_path / "example.py"))
    spawn = server.build_spawn(str(tmp_path), ServerContext(workspace_root=str(tmp_path)))
    if binary is None:
        assert spawn is None
    else:
        assert spawn.command == [str(binary), "--stdio"]
