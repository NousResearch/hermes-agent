import dataclasses
import sys
from pathlib import Path


def test_nested_server_root_keeps_current_diagnostics(tmp_path, monkeypatch):
    from agent.lsp import manager, servers
    repo = tmp_path / 'repo'
    repo.mkdir()
    (repo / '.git').mkdir()
    nested = repo / 'package'
    nested.mkdir()
    (nested / 'package.json').write_text('{}')
    p = nested / 'x.ts'
    p.write_text('const x = 1;\n')
    monkeypatch.chdir(repo)
    original = next((s for s in servers.SERVERS if s.server_id == 'typescript'))
    root = original.resolve_root(str(p), str(repo))
    assert root == str(nested) and (not original.multi_root)
    mock = Path(manager.__file__).resolve().parents[2] / 'tests/agent/lsp/_mock_lsp_server.py'

    def spawn(root, ctx):
        return servers.SpawnSpec(command=[sys.executable, str(mock)], workspace_root=root, cwd=root, env={'MOCK_LSP_SCRIPT': 'errors'}, initialization_options={})
    replacement = dataclasses.replace(original, build_spawn=spawn)
    monkeypatch.setattr(servers, 'SERVERS', [replacement if s is original else s for s in servers.SERVERS])
    svc = manager.LSPService(enabled=True, wait_mode='document', wait_timeout=2, install_strategy='manual')
    try:
        svc.snapshot_baseline(str(p))
        actual = svc.get_diagnostics_sync(str(p), delta=False)
        assert actual
        current = svc._loop.run(svc._current_diags_async(str(p)), timeout=3)
        assert current == actual, 'A live single-root server must be looked up by its resolved nested root'
    finally:
        svc.shutdown()
