"""Native TypeScript is an opt-in LSP, isolated from the Vue/legacy JavaScript SDK."""
from __future__ import annotations

import asyncio
import json
import os

import pytest

from agent.lsp import install
from agent.lsp.manager import LSPService
from agent.lsp.servers import ServerContext, custom_servers, find_server_for_file


def _service(**kwargs):
    return LSPService(enabled=False, wait_mode="document", wait_timeout=5,
                      install_strategy="manual", **kwargs)


def test_backend_selection_preserves_custom_and_framework_servers(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    (home / "config.yaml").write_text(json.dumps({"lsp": {"enabled": False, "typescript_backend": "native"}}))
    svc = LSPService.create_from_config()
    assert svc._server_for("a.ts").server_id == "typescript-native"
    assert svc._server_for("a.vue").server_id == "vue-language-server"
    assert svc._server_for("a.svelte").server_id == "svelte-language-server"
    assert svc._server_for("a.astro").server_id == "astro-language-server"
    assert _service()._server_for("a.ts").server_id == "typescript"
    extra = custom_servers({"custom": {"command": ["other-lsp"], "extensions": [".ts"]}})
    assert _service(typescript_backend="native", extra_servers=extra)._server_for("a.ts").server_id == "custom"
    # Native mode shares the legacy root resolver, including the Deno gate.
    (tmp_path / "tsconfig.json").write_text("{}")
    (tmp_path / "deno.json").write_text("{}")
    assert svc._server_for("a.ts").resolve_root(str(tmp_path / "a.ts"), str(tmp_path)) is None


def test_native_install_isolated_from_legacy_sdk_and_path_tsc(tmp_path, monkeypatch, capsys):
    from agent.lsp.cli import _cmd_which
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(install, "_install_results", {})
    monkeypatch.setattr(install, "_node_package_manager", lambda: "npm")
    monkeypatch.setattr(install, "find_node_executable", lambda name: "/managed/npm")
    monkeypatch.setattr(install.shutil, "which", lambda *args: "/global/tsc")
    sdk = tmp_path / "lsp" / "node_modules" / "typescript" / "lib" / "typescript.js"
    sdk.parent.mkdir(parents=True)
    sdk.write_text("legacy SDK")
    native_modules = tmp_path / "lsp" / "typescript-native" / "node_modules"
    package_metadata = native_modules / "typescript" / "package.json"
    package_metadata.parent.mkdir(parents=True)
    package_metadata.write_text('{"version":"6.0.3"}')
    stale = native_modules / ".bin" / "tsc"
    stale.parent.mkdir()
    stale.write_text("old compiler")
    stale.chmod(0o755)
    calls = []

    def installer(tool, pkg, cmd, **kwargs):
        calls.append(cmd)
        prefix = tmp_path / "lsp" / "typescript-native"
        launcher = prefix / "node_modules" / ".bin" / "tsc"
        launcher.parent.mkdir(parents=True, exist_ok=True)
        launcher.write_text("native launcher")
        launcher.chmod(0o755)
        package_metadata.write_text('{"version":"7.0.2"}')
        return True

    monkeypatch.setattr(install, "_run_installer", installer)
    assert install.try_install("typescript-native", "manual") is None
    assert install.detect_status("typescript-native") == "missing"
    srv = _service(typescript_backend="native")._server_for("a.ts")
    assert srv.build_spawn(str(tmp_path), ServerContext(str(tmp_path), install_strategy="off")) is None
    binary = install.try_install("typescript-native")
    assert calls[0][calls[0].index("--prefix") + 1] == str(tmp_path / "lsp" / "typescript-native")
    assert sdk.read_text() == "legacy SDK"
    assert not (tmp_path / "lsp" / "bin" / "tsc").exists()
    assert install.try_install("typescript-native", "off") == binary
    assert install.detect_status("typescript-native") == "installed"
    assert _cmd_which("typescript-native") == 0
    assert capsys.readouterr().out.strip() == binary
    spec = srv.build_spawn(str(tmp_path), ServerContext(str(tmp_path), install_strategy="manual"))
    assert spec.command == [binary, "--lsp", "--stdio"]
    assert not spec.seed_diagnostics_on_first_push


def test_native_command_overrides_preserve_argv_env_and_initialization(tmp_path):
    launcher = tmp_path / "native"
    launcher.write_text("#!/bin/sh\n")
    launcher.chmod(0o755)
    srv = _service(typescript_backend="native")._server_for("a.ts")
    spec = srv.build_spawn(str(tmp_path), ServerContext(str(tmp_path),
        binary_overrides={"typescript-native": [str(launcher), "--lsp", "--stdio", "--clientProcessId", "123"]},
        env_overrides={"typescript-native": {"TEST_NATIVE": "1"}},
        init_overrides={"typescript-native": {"preferences": {"quotePreference": "single"}}}))
    assert spec.command == [str(launcher), "--lsp", "--stdio", "--clientProcessId", "123"]
    assert spec.env["TEST_NATIVE"] == "1"
    assert spec.initialization_options["preferences"]["quotePreference"] == "single"


def test_native_trust_and_disabled_gates_prevent_spawning(tmp_path, monkeypatch):
    project = tmp_path / "project"
    project.mkdir()
    (project / ".git").mkdir()
    target = project / "a.ts"
    target.write_text("const x = 1;")
    monkeypatch.chdir(tmp_path)  # repository is not the operator's launch directory
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    for kwargs in ({}, {"trusted_workspaces": [str(project)], "disabled_servers": ["typescript-native"]}):
        svc = LSPService(enabled=True, wait_mode="document", wait_timeout=1,
                         install_strategy="off", typescript_backend="native", idle_timeout=0, **kwargs)
        async def forbidden(*args):
            pytest.fail("gated server must not spawn")
        monkeypatch.setattr(svc, "_spawn_client", forbidden)
        try:
            assert svc._loop.run(svc._get_or_spawn(str(target)), timeout=2) is None
        finally:
            svc.shutdown()


@pytest.mark.skipif(not os.environ.get("HERMES_TEST_TYPESCRIPT_NATIVE"), reason="requires an explicit isolated TS7 launcher")
@pytest.mark.timeout(30)
def test_real_native_open_edit_diagnostics(tmp_path, monkeypatch):
    """Opt-in integration: no downloads; exercise the supplied native launcher with Hermes's client."""
    from agent.lsp.client import LSPClient
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    project = tmp_path / "project"
    project.mkdir()
    (project / "tsconfig.json").write_text('{"compilerOptions":{"strict":true,"noEmit":true}}')
    target = project / "a.ts"
    target.write_text('const x: string = 42;\n')
    srv = _service(typescript_backend="native")._server_for(str(target))
    spec = srv.build_spawn(str(project), ServerContext(str(project), trusted=True,
        binary_overrides={"typescript-native": [os.environ["HERMES_TEST_TYPESCRIPT_NATIVE"], "--lsp", "--stdio"]}))

    async def exercise():
        client = LSPClient(server_id=srv.server_id, command=spec.command, workspace_root=str(project), cwd=str(project), env=spec.env)
        try:
            await client.start()
            version = await client.open_file(str(target), language_id="typescript")
            assert await client.wait_for_diagnostics(str(target), version, timeout=5)
            assert any(str(d.get("code")) == "2322" for d in client.diagnostics_for(str(target), fresh_only=True))
            target.write_text('const x: string = "fixed";\n')
            version = await client.open_file(str(target), language_id="typescript")
            assert await client.wait_for_diagnostics(str(target), version, timeout=5)
            assert client.diagnostics_for(str(target), fresh_only=True) == []
        finally:
            await client.shutdown()
        assert not client.is_running

    asyncio.run(exercise())
