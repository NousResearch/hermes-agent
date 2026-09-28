"""Nothing a checkout ships runs unless the operator trusts that workspace.

A cloned repository can ship its own ``.venv/bin/python`` (pyright executes the configured
interpreter), ``node_modules/typescript`` (typescript-language-server and Vue load it),
``svelte.config.js``, Rust build scripts and Gradle builds (rust-analyzer, jdtls and
kotlin-language-server evaluate them), and a ``node_modules/.bin/tsc`` or ``rust-toolchain.toml``
the post-write shell linters would pick up.  Nothing here executes those files: the tests record
which servers Hermes would start, the configuration it hands them, and the shell commands it runs.
"""
from __future__ import annotations

import json
import os
import shutil
from types import SimpleNamespace

from agent.lsp import manager
from agent.lsp.servers import UNTRUSTED_SAFE_SERVERS, ServerContext, find_server_for_file
from agent.lsp.workspace import clear_cache, is_inside_workspace

_SERVERS = {"pyright": "a.py", "typescript": "a.ts", "vue-language-server": "a.vue",
            "svelte-language-server": "a.svelte", "rust-analyzer": "a.rs", "jdtls": "A.java",
            "kotlin-language-server": "a.kt"}


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
    """Marker files only: an interpreter, a TypeScript SDK and build files the project brings itself."""
    (root / ".git").mkdir(parents=True)
    _write(root / "pyproject.toml")
    _write(root / ".venv" / "bin" / "python")
    _write(root / ".venv" / "Scripts" / "python.exe")
    _write(root / "node_modules" / "typescript" / "lib" / "typescript.js")
    _write(root / "node_modules" / "typescript" / "lib" / "tsserver.js")
    _write(root / "node_modules" / ".bin" / "tsc")
    _write(root / "Cargo.toml")
    _write(root / "build.rs")
    _write(root / "rust-toolchain.toml")
    _write(root / "build.gradle.kts")


def _strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for v in value.values():
            yield from _strings(v)
    elif isinstance(value, (list, tuple)):
        for v in value:
            yield from _strings(v)


def _record_spawns(tmp_path, monkeypatch, launcher, roots, *, trusted_workspaces=()):
    """Config → service → spawn for every ``_SERVERS`` file in every root, cwd inside ``tmp/launch``.
    Returns ``{(server_id, root): initialization_options}`` for the servers Hermes would start, and the status."""
    _write(tmp_path / "home" / "config.yaml", json.dumps({"lsp": {
        "trusted_workspaces": [str(p) for p in trusted_workspaces],
        "servers": {sid: {"command": [launcher]} for sid in _SERVERS},
    }}))
    (tmp_path / "launch" / "src").mkdir(parents=True, exist_ok=True)
    monkeypatch.chdir(tmp_path / "launch" / "src")
    clear_cache()
    handed = {}

    class _RecordingClient:
        def __init__(self, *, server_id, workspace_root, initialization_options, **_):
            handed[(server_id, workspace_root)] = initialization_options

        async def start(self):
            raise RuntimeError("recorded, not started")

    monkeypatch.setattr(manager, "LSPClient", _RecordingClient)
    svc = manager.LSPService.create_from_config()
    try:
        for root in roots:
            for name in _SERVERS.values():
                svc._loop.run(svc._get_or_spawn(str(_write(root / name))), timeout=10)
        status = svc.get_status()
    finally:
        svc.shutdown()
    return handed, status


def test_untrusted_checkout_starts_only_allowlisted_servers_pinned_to_hermes_code(tmp_path, monkeypatch):
    launcher = _hermes_side_tree(tmp_path, monkeypatch)
    launch, clone = tmp_path / "launch", tmp_path / "elsewhere" / "clone"
    for root in (launch, clone):
        _checkout_shipping_its_own_toolchain(root)
    handed, status = _record_spawns(tmp_path, monkeypatch, launcher, (launch, clone))

    untrusted = {sid: init for (sid, root), init in handed.items() if root == str(clone)}
    # Deny by default: servers that evaluate build files (cargo, Gradle) never start in the clone.
    assert set(untrusted) == set(_SERVERS) & UNTRUSTED_SAFE_SERVERS
    assert {("rust-analyzer", str(clone)), ("jdtls", str(clone))} <= set(status["untrusted_skipped"])
    for sid, init in untrusted.items():
        leaked = [s for s in _strings(init) if os.path.isabs(s) and is_inside_workspace(s, str(clone))]
        assert not leaked, (sid, leaked)
    # Allowlisted servers that would fall back to project code by themselves are told not to.
    assert os.path.isfile(os.path.join(untrusted["typescript"]["tsserver"]["path"], "tsserver.js"))
    assert untrusted["svelte-language-server"]["isTrusted"] is False
    # With no Hermes-side SDK, Vue is skipped rather than handed the checkout's TypeScript.
    shutil.rmtree(os.path.join(os.path.dirname(launcher), "typescript"))
    ctx = ServerContext(workspace_root=str(clone), install_strategy="manual", binary_overrides={"vue-language-server": [launcher]})
    assert find_server_for_file("a.vue").build_spawn(str(clone), ctx) is None

    # The launch worktree is trusted: every server starts, pyright gets the project interpreter back.
    trusted = {sid: init for (sid, root), init in handed.items() if root == str(launch)}
    assert set(trusted) == set(_SERVERS)
    assert is_inside_workspace(trusted["pyright"]["python"]["pythonPath"], str(launch / ".venv"))
    assert all(trusted[sid] == {} for sid in ("typescript", "svelte-language-server", "rust-analyzer"))


def test_only_the_launch_worktree_and_listed_directories_are_trusted(tmp_path, monkeypatch):
    """The same trust decision gates the servers and the post-write shell linters that use the repo's toolchain."""
    launcher = _hermes_side_tree(tmp_path, monkeypatch)
    launch, listed = tmp_path / "launch", tmp_path / "listed" / "proj"
    nested, sibling = launch / "vendor" / "clone", tmp_path / "elsewhere" / "clone"
    roots = (launch, nested, sibling, listed)
    for root in roots:
        _checkout_shipping_its_own_toolchain(root)
    handed, _ = _record_spawns(tmp_path, monkeypatch, launcher, roots, trusted_workspaces=[tmp_path / "listed"])

    from tools.environments.local import LocalEnvironment
    from tools.file_operations import ShellFileOperations
    fops = ShellFileOperations(LocalEnvironment(cwd=str(launch / "src")))
    ran = []
    monkeypatch.setattr(fops, "_exec", lambda cmd, **_: ran.append(cmd) or SimpleNamespace(exit_code=0, stdout=""))
    monkeypatch.setattr(fops, "_run_managed_node_linter", lambda ext, path: ran.append(path) or SimpleNamespace(exit_code=0, stdout=""))
    monkeypatch.setattr(fops, "_has_command", lambda _cmd: True)
    monkeypatch.setattr(fops, "_lsp_will_handle", lambda _path: False)

    def shell_linted(root):
        ran.clear()
        for name in ("a.ts", "a.rs"):
            fops._check_lint(str(root / name))
        return len(ran)

    for root in (launch, listed):
        assert is_inside_workspace(handed[("pyright", str(root))]["python"]["pythonPath"], str(root / ".venv"))
        assert ("rust-analyzer", str(root)) in handed
        assert shell_linted(root) == 2, root
    for root in (nested, sibling):
        assert not is_inside_workspace(handed[("pyright", str(root))].get("python", {}).get("pythonPath", ""), str(root)), root
        assert ("rust-analyzer", str(root)) not in handed
        assert shell_linted(root) == 0, root
