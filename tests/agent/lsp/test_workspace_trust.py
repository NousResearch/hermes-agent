"""Language servers must not load code a checkout ships unless the operator trusts that workspace.

A cloned repository can ship its own ``.venv/bin/python`` (pyright executes the configured
interpreter), ``node_modules/typescript`` (typescript-language-server and Vue load it),
``svelte.config.js`` and Rust build scripts.  Nothing here executes those files: the tests read
the configuration Hermes hands each server.
"""
from __future__ import annotations

import dataclasses
import json
import os
import shutil

from agent.lsp import manager
from agent.lsp.servers import ServerContext, find_server_for_file
from agent.lsp.workspace import clear_cache, is_inside_workspace

_SERVERS = {"pyright": "a.py", "typescript": "a.ts", "vue-language-server": "a.vue",
            "svelte-language-server": "a.svelte", "rust-analyzer": "a.rs"}


def _write(path, text: str = ""):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _hermes_side_tree(tmp_path, monkeypatch) -> str:
    """``<HERMES_HOME>/lsp/node_modules`` with a JS TypeScript SDK and Vue 2.x; returns a launcher inside it."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("VIRTUAL_ENV", raising=False)
    staging = tmp_path / "home" / "lsp" / "node_modules"
    _write(staging / "typescript" / "lib" / "typescript.js")
    _write(staging / "typescript" / "lib" / "tsserver.js")
    _write(staging / "@vue" / "language-server" / "package.json", json.dumps({"version": "2.2.12"}))
    return str(_write(staging / "server-launcher"))


def _checkout_shipping_its_own_toolchain(root) -> None:
    """Marker files only: an interpreter path and a TypeScript SDK the project brings itself."""
    (root / ".git").mkdir(parents=True)
    _write(root / "pyproject.toml")
    _write(root / ".venv" / "bin" / "python")
    _write(root / ".venv" / "Scripts" / "python.exe")
    _write(root / "node_modules" / "typescript" / "lib" / "typescript.js")
    _write(root / "node_modules" / "typescript" / "lib" / "tsserver.js")


def _strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for v in value.values():
            yield from _strings(v)
    elif isinstance(value, (list, tuple)):
        for v in value:
            yield from _strings(v)


def test_untrusted_workspace_config_never_points_a_server_at_the_checkouts_own_code(tmp_path, monkeypatch):
    launcher = _hermes_side_tree(tmp_path, monkeypatch)
    root = tmp_path / "cloned"
    _checkout_shipping_its_own_toolchain(root)
    ctx = ServerContext(workspace_root=str(root), install_strategy="manual",
                        binary_overrides={sid: [launcher] for sid in _SERVERS})

    untrusted = {sid: find_server_for_file(f).build_spawn(str(root), ctx) for sid, f in _SERVERS.items()}
    for sid, spec in untrusted.items():
        assert spec is not None, sid
        leaked = [s for s in _strings(spec.initialization_options) if os.path.isabs(s) and is_inside_workspace(s, str(root))]
        assert not leaked, (sid, leaked)
    # Servers that fall back to project code by themselves are told not to.
    assert os.path.isfile(os.path.join(untrusted["typescript"].initialization_options["tsserver"]["path"], "tsserver.js"))
    assert untrusted["svelte-language-server"].initialization_options["isTrusted"] is False
    rust = untrusted["rust-analyzer"].initialization_options
    assert rust["cargo"]["buildScripts"]["enable"] is False and rust["procMacro"]["enable"] is False
    # With no Hermes-side SDK, Vue is skipped rather than handed the checkout's TypeScript.
    shutil.rmtree(os.path.join(os.path.dirname(launcher), "typescript"))
    assert find_server_for_file("a.vue").build_spawn(str(root), ctx) is None

    # The same checkout, trusted, gets its own interpreter back and no restriction.
    trusted_ctx = dataclasses.replace(ctx, trusted=True)
    trusted = {sid: find_server_for_file(f).build_spawn(str(root), trusted_ctx) for sid, f in _SERVERS.items()}
    assert is_inside_workspace(trusted["pyright"].initialization_options["python"]["pythonPath"], str(root / ".venv"))
    assert all(spec.initialization_options == {} for sid, spec in trusted.items()
               if sid in {"typescript", "svelte-language-server", "rust-analyzer"})


def test_only_the_launch_worktree_and_listed_directories_are_trusted(tmp_path, monkeypatch):
    """Config → service → spawn: which checkout's interpreter pyright is handed."""
    launcher = _hermes_side_tree(tmp_path, monkeypatch)
    launch, listed = tmp_path / "launch", tmp_path / "listed" / "proj"
    nested, sibling = launch / "vendor" / "clone", tmp_path / "elsewhere" / "clone"
    for root in (launch, nested, sibling, listed):
        _checkout_shipping_its_own_toolchain(root)
    _write(tmp_path / "home" / "config.yaml", json.dumps({"lsp": {
        "trusted_workspaces": [str(tmp_path / "listed")],
        "servers": {"pyright": {"command": [launcher]}},
    }}))
    (launch / "src").mkdir()
    monkeypatch.chdir(launch / "src")
    clear_cache()

    handed = {}

    class _RecordingClient:
        def __init__(self, *, workspace_root, initialization_options, **_):
            handed[workspace_root] = initialization_options

        async def start(self):
            raise RuntimeError("recorded, not started")

    monkeypatch.setattr(manager, "LSPClient", _RecordingClient)
    svc = manager.LSPService.create_from_config()
    try:
        for root in (launch, nested, sibling, listed):
            svc._loop.run(svc._get_or_spawn(str(_write(root / "mod.py"))), timeout=10)
    finally:
        svc.shutdown()

    def interpreter(root):
        return handed[str(root)].get("python", {}).get("pythonPath", "")

    assert is_inside_workspace(interpreter(launch), str(launch / ".venv"))
    assert is_inside_workspace(interpreter(listed), str(listed / ".venv"))
    for untrusted in (nested, sibling):
        assert not is_inside_workspace(interpreter(untrusted), str(untrusted)), untrusted
