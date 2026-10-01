"""Config-declared LSP servers (``lsp.servers.<id>`` with ``extensions``) route files and spawn like built-ins."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from unittest.mock import patch

import pytest

from agent.lsp.manager import LSPService
from agent.lsp.servers import SERVERS, ServerContext, custom_servers, find_server_for_file, language_id_for

_MOCK = os.path.join(os.path.dirname(__file__), "_mock_lsp_server.py")


def test_custom_server_precedes_builtins_and_carries_language_id():
    cfg = {
        "pyright": {"disabled": True},  # override of a built-in id is NOT a custom server
        "blade": {"command": [sys.executable, _MOCK], "extensions": [".PHP"], "language_id": "blade"},
        "broken": {"command": "not-a-list", "extensions": [".zzz"]},  # malformed → skipped, not fatal
    }
    custom = custom_servers(cfg)
    assert [s.server_id for s in custom] == ["blade"]
    registry = [*custom, *SERVERS]
    assert find_server_for_file("/w/index.php", registry).server_id == "blade"  # custom wins over intelephense
    assert find_server_for_file("/w/index.php").server_id == "intelephense"  # built-in registry untouched
    assert language_id_for("/w/index.php", custom[0]) == "blade"
    assert language_id_for("/w/index.php") == "php"
    spec = custom[0].build_spawn("/w", ServerContext("/w", install_strategy="manual"))
    assert spec is not None and spec.command == [sys.executable, _MOCK]


@pytest.mark.timeout(60)
def test_service_gets_diagnostics_from_config_declared_server(tmp_path, monkeypatch):
    """End to end from config.yaml: the only production path from ``lsp.servers`` to a running server."""
    from tools.environments.local import LocalEnvironment
    from tools.file_operations import ShellFileOperations

    home = tmp_path / "home"
    home.mkdir()
    (home / "config.yaml").write_text(json.dumps({"lsp": {
        "install_strategy": "manual", "wait_timeout": 5.0, "idle_timeout": 0,
        "servers": {"panache": {"command": [sys.executable, _MOCK], "extensions": [".pnch"],
                                "env": {"MOCK_LSP_SCRIPT": "errors"}}},
    }}), encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    ws = tmp_path / "ws"
    ws.mkdir()
    subprocess.run(["git", "init", "-q", str(ws)], check=True)
    target = ws / "doc.pnch"
    target.write_text("hello\n", encoding="utf-8")
    (ws / "x.rs").write_text("fn main() {}\n", encoding="utf-8")
    svc = LSPService.create_from_config()
    assert svc is not None
    try:
        assert svc.enabled_for(str(target))
        diags = svc.get_diagnostics_sync(str(target))
        assert diags and diags[0]["source"] == "mock-lsp"
        # Control: a built-in extension still routes to the registry entry, not the custom server.
        assert svc._server_for(str(ws / "x.rs")).server_id == "rust-analyzer"
        # The write path captures pre-write content for the custom extension only via the service.
        fops = ShellFileOperations(LocalEnvironment())
        with patch.object(fops, "_lsp_service", return_value=svc):
            assert fops._lsp_handles_extension(".pnch")
        with patch.object(fops, "_lsp_service", return_value=None):
            assert not fops._lsp_handles_extension(".pnch") and fops._lsp_handles_extension(".rs")
    finally:
        svc.shutdown()
