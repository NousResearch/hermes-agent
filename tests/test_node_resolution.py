"""Node discovery reads PM state; legacy home-local trees are never activated."""

import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import pytest

import hermes_constants
import pm
from pm import paths
from pm.lock import Facts, Lockfile
from pm.package import Runner
from pm.registry import get_package
from pm.store import current_target, tree_digest


@pytest.fixture
def node_store(tmp_path, monkeypatch):
    node = shutil.which("node")
    if node is None:
        pytest.skip("requires an already-installed Node executable")
    assert node is not None
    monkeypatch.setenv("PATH", str(Path(node).parent))
    home = tmp_path / "home"
    home.mkdir()
    store = home / "tools"
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_RUNTIME_DIR", str(store))
    monkeypatch.setenv("HERMES_DISABLE_LAZY_INSTALLS", "1")
    lock_path = tmp_path / "lock.json"
    monkeypatch.setattr(paths, "lockfile_path", lambda: lock_path)
    lock = Lockfile(lock_path)
    target = current_target()
    package = get_package("node")
    version = subprocess.run(
        [node, "--version"], capture_output=True, text=True, check=True, timeout=10,
    ).stdout.strip().removeprefix("v")
    entry = store / package.store_entry(version, target)
    binary = package.binary(entry, target)
    assert binary is not None
    binary.parent.mkdir(parents=True)
    binary.symlink_to(node)
    digest = hashlib.sha256(Path(node).read_bytes()).hexdigest()
    lock.set_pin("node", version, {target: {"url": Path(node).as_uri(), "sha256": digest}})
    lock.save()
    Facts(paths.facts_path()).record(
        "node", version, entry.name, package.env(entry, target), store,
        target=target, artifacts=[digest], digest=tree_digest(entry),
    )
    return home, binary, node


@pytest.mark.platforms("posix")
def test_pm_node_wins_over_legacy_tree_and_composes_child_environment(node_store, monkeypatch):
    home, binary, external = node_store
    legacy = home / "node" / "bin" / "node"
    legacy.parent.mkdir(parents=True)
    legacy.symlink_to(external)
    monkeypatch.setenv("PATH", str(Path(external).parent))
    before = paths.facts_path().read_bytes()

    selected = pm.installed_package("node")
    assert selected is not None and selected.binary == binary
    resolved = hermes_constants.find_node_executable("node")
    assert resolved is not None and resolved == str(binary)
    base = {"PATH": str(Path(external).parent), "CALLER_VALUE": "preserved"}
    environment = hermes_constants.with_hermes_node_path(base)
    assert environment == pm.env_for("npm", base_env=base)
    assert shutil.which("node", path=environment["PATH"]) == str(binary)
    child = subprocess.run(
        [resolved, "-p", "process.version"],
        env=environment, capture_output=True, text=True, check=True, timeout=10,
    )
    assert child.stdout.strip().removeprefix("v") == selected.version
    assert paths.facts_path().read_bytes() == before
    assert legacy.is_file()
    assert base == {"PATH": str(Path(external).parent), "CALLER_VALUE": "preserved"}


@pytest.mark.platforms("posix")
def test_passive_discovery_never_installs_or_repairs_a_legacy_tree(tmp_path, monkeypatch):
    home = tmp_path / "home"
    legacy = home / "node" / "bin" / "node"
    legacy.parent.mkdir(parents=True)
    legacy.write_text("#!/bin/sh\nexit 1\n", encoding="utf-8")
    legacy.chmod(0o755)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_RUNTIME_DIR", str(home / "tools"))
    monkeypatch.setenv("HERMES_DISABLE_LAZY_INSTALLS", "1")
    monkeypatch.setenv("PATH", "")
    before = {p.relative_to(home): p.read_bytes() for p in home.rglob("*") if p.is_file()}
    attempts = []

    def forbidden_install(*args, **kwargs):
        attempts.append((args, kwargs))
        raise AssertionError("passive lookup must not provision anything")

    monkeypatch.setattr(pm, "ensure", forbidden_install)
    assert hermes_constants.find_node_executable("node") is None
    assert hermes_constants.find_node_executable("npm") is None
    assert hermes_constants.with_hermes_node_path({"PATH": ""}) == {"PATH": ""}
    assert attempts == []
    assert {p.relative_to(home): p.read_bytes() for p in home.rglob("*") if p.is_file()} == before
    assert not (home / "tools").exists()

    # A user-owned PATH toolchain remains usable without acquiring a PM one.
    external = tmp_path / "external"
    external.mkdir()
    npm = external / "npm"
    npm.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    npm.chmod(0o755)
    monkeypatch.setenv("PATH", str(external))
    assert hermes_constants.find_node_executable("npm") == str(npm)
    assert attempts == []
    assert not (home / "tools").exists()


@pytest.mark.platforms("posix")
def test_explicit_node_path_is_not_replaced_by_managed_name(node_store):
    _home, _binary, external = node_store
    assert hermes_constants.find_node_executable(external) == external
    assert hermes_constants.find_node_executable(str(Path(external).with_name("missing-node"))) is None
    assert hermes_constants.find_node_executable("not-a-node-command") is None


@pytest.mark.platforms("posix")
def test_npm_and_npx_use_the_paired_pm_entry(node_store, monkeypatch):
    home, node, external = node_store
    npm = Path(external).with_name("npm")
    npx = Path(external).with_name("npx")
    if not npm.is_file() or not npx.is_file():
        pytest.skip("requires already-installed npm and npx")
    version = subprocess.run(
        [str(npm), "--version"], capture_output=True, text=True, check=True, timeout=10,
    ).stdout.strip()
    package = get_package("npm")
    target = current_target()
    store = paths.store_root()
    entry = store / package.store_entry(version, target)
    binary = package.binary(entry, target)
    assert binary is not None
    binary.parent.mkdir(parents=True)
    binary.symlink_to(npm)
    companion = binary.with_name("npx")
    companion.symlink_to(npx)
    lock = Lockfile(paths.lockfile_path())
    digest = hashlib.sha256(npm.read_bytes()).hexdigest()
    lock.set_pin("npm", version, {target: {"url": npm.as_uri(), "sha256": digest}})
    lock.save()
    Facts(paths.facts_path()).record(
        "npm", version, entry.name, package.env(entry, target), store,
        target=target, artifacts=[digest], digest=tree_digest(entry),
    )
    legacy = home / "node" / "bin" / "npm"
    legacy.parent.mkdir(parents=True)
    legacy.symlink_to(npm)
    before = paths.facts_path().read_bytes()
    monkeypatch.setenv("PATH", "")

    assert hermes_constants.find_node_executable("npm") == str(binary)
    assert hermes_constants.find_node_executable("npx") == str(companion)
    environment = hermes_constants.with_hermes_node_path({"PATH": ""})
    assert shutil.which("npm", path=environment["PATH"]) == str(binary)
    assert shutil.which("node", path=environment["PATH"]) == str(node)
    for command in ("npm", "npx"):
        resolved = hermes_constants.find_node_executable(command)
        assert resolved is not None
        result = subprocess.run(
            [resolved, "--version"], env=environment,
            capture_output=True, text=True, check=True, timeout=10,
        )
        assert result.stdout.strip() == version
    assert paths.facts_path().read_bytes() == before
    companion.unlink()
    assert hermes_constants.find_node_executable("npx") is None


@pytest.mark.platforms("windows")
def test_windows_path_prefers_launchable_npm_cmd(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_RUNTIME_DIR", str(tmp_path / "absent-store"))
    monkeypatch.setenv("PATH", str(tmp_path))
    for name in ("npm", "npm.ps1", "npm.cmd"):
        (tmp_path / name).write_text("@exit /b 0\n", encoding="utf-8")
    assert hermes_constants.find_node_executable("npm") == str(tmp_path / "npm.cmd")


@pytest.fixture
def npm_probe(node_store, monkeypatch):
    home, node, _external = node_store
    store = paths.store_root()
    package = get_package("npm")
    target = current_target()
    entry = store / package.store_entry("1.0.0", target)
    binary = package.binary(entry, target)
    assert binary is not None
    binary.parent.mkdir(parents=True)
    binary.write_text(
        '#!/usr/bin/env node\n'
        'const fs = require("fs"); const path = require("path");\n'
        'const i = process.argv.indexOf("--prefix");\n'
        'const root = i < 0 ? process.cwd() : process.argv[i + 1];\n'
        'fs.mkdirSync(path.join(root, "node_modules/.bin"), {recursive:true});\n'
        'fs.writeFileSync(path.join(root, "node_modules/.bin/test-server"), "ready");\n'
        'fs.writeFileSync(path.join(root, "called.json"), JSON.stringify({argv: process.argv, env: process.env}));\n',
        encoding="utf-8",
    )
    binary.chmod(0o755)
    lock = Lockfile(paths.lockfile_path())
    lock.set_pin("npm", "1.0.0", {})
    lock.save()

    def publish():
        Facts(paths.facts_path()).record(
            "npm", "1.0.0", entry.name, package.env(entry, target), store,
        )
        return Runner("npm", pm.env_for("npm"))

    monkeypatch.setenv("PATH", "")
    return home, node, binary, publish


@pytest.fixture
def npm_consumers(npm_probe, tmp_path, monkeypatch):
    from agent.lsp.install import _install_npm
    from gateway.config import PlatformConfig
    from hermes_cli.main_platform_setup import _whatsapp_install_bridge
    from hermes_cli.web_routers.messaging import _ensure_whatsapp_bridge_dependencies
    from plugins.platforms.photon import adapter as photon, cli
    from plugins.platforms.whatsapp.adapter import WhatsAppAdapter

    home, _node, _binary, _publish = npm_probe
    bridge = tmp_path / "bridge"
    bridge.mkdir()
    (bridge / "package.json").write_text('{"name":"test-bridge"}', encoding="utf-8")
    adapter = WhatsAppAdapter(PlatformConfig(extra={"bridge_script": str(bridge / "bridge.js")}))
    monkeypatch.setattr(photon, "_sidecar_dir", lambda: bridge)
    monkeypatch.setattr(cli, "_sidecar_dir", lambda: bridge)
    return {
        "lsp": (lambda: _install_npm("test-pkg", "test-server"), home / "lsp"),
        "whatsapp": (lambda: adapter._ensure_bridge_deps(bridge), bridge),
        "photon": (photon._reinstall_sidecar_deps, bridge),
        "photon-cli": (cli._install_sidecar, bridge),
        "cli": (lambda: _whatsapp_install_bridge(bridge), bridge),
        "dashboard": (lambda: _ensure_whatsapp_bridge_dependencies(bridge), bridge),
    }


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("consumer", ["lsp", "whatsapp", "photon", "photon-cli", "cli", "dashboard"])
def test_npm_consumers_execute_with_pm_node(npm_probe, npm_consumers, consumer):
    _home, node, binary, publish = npm_probe
    publish()
    call, output_dir = npm_consumers[consumer]
    call()
    result = json.loads((output_dir / "called.json").read_text())
    assert Path(result["argv"][1]) == binary
    assert shutil.which("node", path=result["env"]["PATH"]) == str(node)
    if consumer == "lsp":
        assert result["argv"][2:] == [
            "install", "--prefix", str(output_dir), "--silent", "--no-fund", "--no-audit", "test-pkg",
        ]


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("surface", ["cli", "dashboard", "whatsapp", "photon", "photon-cli"])
def test_missing_npm_acquires_npm_closure_at_install_boundary(npm_probe, npm_consumers, monkeypatch, surface):
    _home, node, binary, publish = npm_probe
    calls = []

    def ensure(name, **kwargs):
        calls.append((name, kwargs.get("explicit", False)))
        assert name == "npm"
        return publish()

    monkeypatch.setattr(pm, "ensure", ensure)
    call, bridge = npm_consumers[surface]
    call()
    assert calls == [("npm", surface in {"cli", "dashboard", "photon-cli"})]
    result = json.loads((bridge / "called.json").read_text())
    assert Path(result["argv"][1]) == binary
    assert shutil.which("node", path=result["env"]["PATH"]) == str(node)


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("surface", ["whatsapp", "photon"])
def test_runtime_npm_refusal_never_falls_back_to_literal_npm(npm_probe, npm_consumers, monkeypatch, surface):
    call, bridge = npm_consumers[surface]
    spawns = []
    run = subprocess.run

    def forbidden_spawn(*args, **kwargs):
        if args[0][0] == "git":  # Fatal-status reporting may read the code revision.
            return run(*args, **kwargs)
        spawns.append(args)
        raise AssertionError("refused PM preparation must not spawn npm")

    monkeypatch.setattr(subprocess, "run", forbidden_spawn)
    call()
    assert spawns == []
    assert not (bridge / "called.json").exists()
    assert pm.installed_package("npm") is None


@pytest.mark.parametrize("allowed", [False, True])
def test_adapter_availability_never_provisions_missing_node(tmp_path, monkeypatch, allowed):
    from plugins.platforms.photon import adapter as photon
    from plugins.platforms.whatsapp import adapter as whatsapp

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_RUNTIME_DIR", str(tmp_path / "missing-tools"))
    monkeypatch.setenv("PATH", "")
    monkeypatch.delenv("PHOTON_NODE_BIN", raising=False)
    monkeypatch.setattr(photon, "_sidecar_dir", lambda: tmp_path)
    monkeypatch.setattr(photon, "HTTPX_AVAILABLE", True)
    monkeypatch.setattr(pm, "lazy_installs_allowed", lambda: allowed)
    installs = []

    def forbidden_ensure(*args, **kwargs):
        installs.append(args)
        raise AssertionError("availability must not install")

    monkeypatch.setattr(pm, "ensure", forbidden_ensure)
    assert whatsapp.check_whatsapp_requirements() is allowed
    assert photon.check_requirements() is allowed
    assert installs == []
    assert not (tmp_path / "missing-tools").exists()


@pytest.mark.platforms("posix")
def test_dashboard_pairing_prepares_npm_before_node_lookup(npm_probe, tmp_path, monkeypatch):
    from gateway.platforms import whatsapp_common
    from hermes_cli.web_routers.messaging import _spawn_whatsapp_pairing_process

    _home, node, _npm, publish = npm_probe
    node_facts = paths.facts_path().read_bytes()
    paths.facts_path().unlink()
    bridge = tmp_path / "bridge"
    bridge.mkdir()
    (bridge / "bridge.js").write_text('console.log(JSON.stringify({argv:process.argv, path:process.env.PATH}));\n')
    monkeypatch.setattr(whatsapp_common, "resolve_whatsapp_bridge_dir", lambda: bridge)
    installs = []

    def ensure(name, **kwargs):
        installs.append((name, kwargs.get("explicit", False)))
        paths.facts_path().write_bytes(node_facts)
        return publish()

    monkeypatch.setattr(pm, "ensure", ensure)
    child = _spawn_whatsapp_pairing_process(tmp_path / "session", "bot")
    try:
        output, _ = child.communicate(timeout=10)
        assert child.returncode == 0
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(timeout=10)
    assert installs == [("npm", True)]
    result = json.loads(output)
    assert "--pair-json" in result["argv"]
    assert shutil.which("node", path=result["path"]) == str(node)


@pytest.mark.platforms("posix")
def test_lsp_node_server_inherits_pm_runtime_and_preserves_overrides(node_store, tmp_path, monkeypatch):
    from agent.lsp.servers import ServerContext, find_server_for_file

    _home, node, _external = node_store
    script = tmp_path / "language-server"
    script.write_text('#!/usr/bin/env node\nconsole.log(JSON.stringify({path:process.env.PATH, flag:process.env.LSP_FLAG}));\n')
    script.chmod(0o755)
    monkeypatch.setenv("PATH", "")
    ctx = ServerContext(
        workspace_root=str(tmp_path), install_strategy="off",
        binary_overrides={"typescript": [str(script)]},
        env_overrides={"typescript": {"LSP_FLAG": "project-value"}},
    )
    server = find_server_for_file(str(tmp_path / "test.ts"))
    assert server is not None
    spec = server.build_spawn(str(tmp_path), ctx)
    assert spec is not None
    child = subprocess.run(spec.command, env=spec.env, capture_output=True, text=True, timeout=10)
    assert child.returncode == 0, child.stderr
    observed = json.loads(child.stdout)
    assert shutil.which("node", path=observed["path"]) == str(node)
    assert observed["flag"] == "project-value"
    ctx.env_overrides["typescript"]["PATH"] = "/explicit/project/tools"
    spec = server.build_spawn(str(tmp_path), ctx)
    assert spec is not None and spec.env["PATH"] == "/explicit/project/tools"
