"""Tests for the Vue server registration.

``@vue/language-server`` 3.x only works behind a client-hosted tsserver tunnel
(``tsserver/request`` notifications) that Hermes's generic client does not
run, so it never publishes diagnostics.  Hermes pins the self-hosting 2.x line
and starts it with ``vue.hybridMode`` off plus an explicit JS TypeScript SDK.
"""
from __future__ import annotations

import json
import os

import pytest

from agent.lsp.servers import ServerContext, find_server_for_file


def _stage(tmp_path, monkeypatch, vue_version: str, *, js_sdk: bool):
    """Fake ``<HERMES_HOME>/lsp`` tree: a staged launcher, the Vue package, optionally a JS TypeScript SDK."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    node_modules = tmp_path / "lsp" / "node_modules"
    pkg = node_modules / "@vue" / "language-server"
    (pkg / "bin").mkdir(parents=True)
    (pkg / "package.json").write_text(json.dumps({"version": vue_version}), encoding="utf-8")
    launcher = pkg / "bin" / "vue-language-server.js"
    launcher.write_text("#!/usr/bin/env node\n", encoding="utf-8")
    launcher.chmod(0o755)
    (tmp_path / "lsp" / "bin").mkdir()
    staged = tmp_path / "lsp" / "bin" / "vue-language-server"
    staged.write_text(launcher.read_text(encoding="utf-8"), encoding="utf-8")  # copied like the Windows path, not symlinked
    staged.chmod(0o755)
    if js_sdk:
        (node_modules / "typescript" / "lib").mkdir(parents=True)
        (node_modules / "typescript" / "lib" / "typescript.js").write_text("", encoding="utf-8")
    return str(staged)


@pytest.mark.parametrize("vue_version, js_sdk, spawns", [("2.2.12", True, True), ("3.3.11", True, False), ("2.2.12", False, False)])
def test_vue_spawns_only_a_self_hosting_server_with_a_js_typescript_sdk(tmp_path, monkeypatch, vue_version, js_sdk, spawns):
    """2.x + JS SDK → spawn with ``hybridMode`` off and ``tsdk`` pointing at that SDK; a 3.x install
    (tunnel-only) or a missing/Go-only TypeScript must be skipped instead of spawning a server that
    can never publish diagnostics."""
    _stage(tmp_path, monkeypatch, vue_version, js_sdk=js_sdk)
    root = tmp_path / "project"
    root.mkdir()
    ctx = ServerContext(workspace_root=str(root), install_strategy="manual")
    spec = find_server_for_file("App.vue").build_spawn(str(root), ctx)
    if not spawns:
        assert spec is None
        return
    assert spec.command[1:] == ["--stdio"]
    assert spec.initialization_options["vue"] == {"hybridMode": False}
    tsdk = spec.initialization_options["typescript"]["tsdk"]
    assert os.path.isfile(os.path.join(tsdk, "typescript.js"))


def test_js_toolchain_recipes_pin_a_javascript_typescript_sdk():
    """Both TypeScript-hosting recipes co-install a JS-based TypeScript line (7+ is the Go port with no
    ``tsserver.js``), and the Vue recipe stays below the tunnel-only 3.x major."""
    from agent.lsp.install import INSTALL_RECIPES

    def major(spec: str, name: str) -> int:
        assert spec.startswith(f"{name}@"), spec  # unpinned = floats onto the next major
        return int(spec[len(name) + 1:].split(".")[0])

    vue, tsls = INSTALL_RECIPES["@vue/language-server"], INSTALL_RECIPES["typescript-language-server"]
    assert major(vue["pkg"], "@vue/language-server") < 3
    for recipe in (vue, tsls):
        sdk = [p for p in recipe.get("extra_pkgs") or [] if p.split("@")[0] == "typescript"]
        assert sdk and major(sdk[0], "typescript") < 7
