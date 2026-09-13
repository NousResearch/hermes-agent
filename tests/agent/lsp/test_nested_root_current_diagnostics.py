"""``_current_diags_async`` must find a single-root client under the root it was spawned with.

``_get_or_spawn`` keys single-root servers by ``srv.resolve_root(...)`` (a nested ``package.json``
project), but the current-diagnostics lookup keyed by the enclosing workspace root, so the delta
baseline was refreshed from ``[]`` while the live client held diagnostics.
"""
from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

from agent.lsp import manager, servers

MOCK_SERVER = str(Path(__file__).parent / "_mock_lsp_server.py")


def test_nested_single_root_client_is_found_by_current_lookup(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    (repo / ".git").mkdir(parents=True)
    nested = repo / "package"
    nested.mkdir()
    (nested / "package.json").write_text("{}", encoding="utf-8")
    src = nested / "x.ts"
    src.write_text("const x = 1;\n", encoding="utf-8")
    monkeypatch.chdir(repo)

    original = next(s for s in servers.SERVERS if s.server_id == "typescript")
    assert original.resolve_root(str(src), str(repo)) == str(nested) and not original.multi_root

    def spawn(root, ctx):
        return servers.SpawnSpec(command=[sys.executable, MOCK_SERVER], workspace_root=root, cwd=root,
                                 env={"MOCK_LSP_SCRIPT": "errors"}, initialization_options={})

    mocked = dataclasses.replace(original, build_spawn=spawn)
    monkeypatch.setattr(servers, "SERVERS", [mocked if s is original else s for s in servers.SERVERS])

    svc = manager.LSPService(enabled=True, wait_mode="document", wait_timeout=5, install_strategy="manual")
    try:
        svc.snapshot_baseline(str(src))
        live = svc.get_diagnostics_sync(str(src), delta=False)
        assert live
        assert svc._loop.run(svc._current_diags_async(str(src)), timeout=5) == live
    finally:
        svc.shutdown()
