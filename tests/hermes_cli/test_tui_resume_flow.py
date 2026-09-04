from argparse import Namespace
import os
from pathlib import Path
import subprocess
import sys
import textwrap
import types

import pytest


def _args(**overrides):
    base = {
        "continue_last": None,
        "model": None,
        "provider": None,
        "resume": None,
        "toolsets": None,
        "tui": True,
        "tui_dev": False,
    }
    base.update(overrides)
    return Namespace(**base)


def _raise_exit(rc):
    raise SystemExit(rc)


@pytest.fixture
def main_mod(monkeypatch):
    import hermes_cli.main as mod

    monkeypatch.setattr(mod, "_has_any_provider_configured", lambda: True)
    # Reset the idempotency guard so each test starts fresh.
    monkeypatch.setattr(mod, "_oneshot_cleanup_done", False)
    return mod
















def test_termux_skips_bundled_skill_sync_when_stamp_fresh(monkeypatch, tmp_path, main_mod):
    calls = []

    monkeypatch.setenv("TERMUX_VERSION", "1")
    monkeypatch.setattr(main_mod, "get_hermes_home", lambda: tmp_path)
    monkeypatch.setattr(main_mod, "_termux_bundled_skills_fingerprint", lambda: "fp1")
    main_mod._mark_termux_bundled_skills_synced()
    monkeypatch.setitem(
        sys.modules,
        "tools.skills_sync",
        types.SimpleNamespace(sync_skills=lambda quiet: calls.append(quiet)),
    )

    assert main_mod._sync_bundled_skills_for_startup() is False
    assert calls == []






def test_exit_after_oneshot_flushes_stdio_and_calls_os_exit(
    monkeypatch, main_mod
):
    flushed = []
    exits = []

    class FakeStream:
        def __init__(self, name):
            self.name = name

        def flush(self):
            flushed.append(self.name)

    def fake_exit(rc):
        exits.append(rc)
        raise SystemExit(rc)

    monkeypatch.setattr(main_mod.sys, "stdout", FakeStream("stdout"))
    monkeypatch.setattr(main_mod.sys, "stderr", FakeStream("stderr"))
    monkeypatch.setattr(main_mod.os, "_exit", fake_exit)
    monkeypatch.setattr("logging.shutdown", lambda: None)

    with pytest.raises(SystemExit) as exc:
        main_mod._exit_after_oneshot(17)

    assert exc.value.code == 17
    assert exits == [17]
    assert flushed == ["stdout", "stderr"]






def test_oneshot_subprocess_exits_without_teardown_abort():
    program = textwrap.dedent(
        """
        import hermes_cli.oneshot as oneshot
        from hermes_cli.main import _exit_after_oneshot

        oneshot._run_agent = lambda *args, **kwargs: ("ok", {"final_response": "ok"})
        _exit_after_oneshot(oneshot.run_oneshot("hello"))
        """
    )

    result = subprocess.run(
        [sys.executable, "-c", program],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        timeout=10,
        check=False,
    )

    assert result.returncode == 0
    assert result.stdout == b"ok\n"
    # Don't demand byte-empty stderr — an import-time warning from the heavy
    # CLI import chain shouldn't fail this. What matters is no crash traceback.
    assert b"Traceback" not in result.stderr








def _stub_plugin_discovery(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.plugins",
        types.SimpleNamespace(discover_plugins=lambda: None),
    )




def test_oneshot_wires_session_db_for_recall(monkeypatch):
    """hermes -z bypasses HermesCLI, but recall still needs SessionDB."""
    from hermes_cli.oneshot import _run_agent

    captured = {}
    sentinel_db = object()

    class FakeAgent:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.suppress_status_output = False
            self.stream_delta_callback = object()
            self.tool_gen_callback = object()

        def run_conversation(self, prompt, **_kwargs):
            captured["prompt"] = prompt
            return {"final_response": "ok", "failed": False, "partial": False}

    class FakeSessionDB:
        def __new__(cls):
            return sentinel_db

    def mod(name, **attrs):
        module = types.ModuleType(name)
        for key, value in attrs.items():
            setattr(module, key, value)
        return module

    monkeypatch.setitem(sys.modules, "run_agent", mod("run_agent", AIAgent=FakeAgent))
    monkeypatch.setitem(sys.modules, "hermes_state", mod("hermes_state", SessionDB=FakeSessionDB))
    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.config",
        mod("hermes_cli.config", load_config=lambda: {"model": {"default": "m"}}),
    )
    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.models",
        mod("hermes_cli.models", detect_provider_for_model=lambda *_args, **_kwargs: None),
    )
    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.runtime_provider",
        mod(
            "hermes_cli.runtime_provider",
            resolve_runtime_provider=lambda **_kwargs: {
                "api_key": "k",
                "base_url": "u",
                "provider": "p",
                "api_mode": "chat_completions",
                "credential_pool": None,
            },
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.tools_config",
        mod("hermes_cli.tools_config", _get_platform_tools=lambda *_args, **_kwargs: {"session_search"}),
    )

    text, result = _run_agent("recall this")
    assert text == "ok"
    assert not result.get("failed")
    assert captured["session_db"] is sentinel_db
    assert captured["enabled_toolsets"] == ["session_search"]
    assert captured["prompt"] == "recall this"


def test_launch_tui_exports_model_provider_and_toolsets(monkeypatch, main_mod):
    captured = {}
    active_path_during_call = None

    monkeypatch.setattr(
        main_mod,
        "_make_tui_argv",
        lambda tui_dir, tui_dev: (["node", "dist/entry.js"], Path(".")),
    )

    def fake_call(argv, cwd=None, env=None):
        nonlocal active_path_during_call
        captured.update({"argv": argv, "cwd": cwd, "env": env})
        active_path_during_call = Path(env["HERMES_TUI_ACTIVE_SESSION_FILE"])
        assert active_path_during_call.exists()
        return 1

    monkeypatch.setattr(main_mod.subprocess, "call", fake_call)

    with pytest.raises(SystemExit):
        main_mod._launch_tui(
            model="nous/hermes-test", provider="nous", toolsets="web, terminal"
        )

    env = captured["env"]
    assert env["HERMES_MODEL"] == "nous/hermes-test"
    assert env["HERMES_INFERENCE_MODEL"] == "nous/hermes-test"
    assert env["HERMES_TUI_PROVIDER"] == "nous"
    assert env["HERMES_INFERENCE_PROVIDER"] == "nous"
    assert env["HERMES_TUI_TOOLSETS"] == "web,terminal"
    active_path = Path(env["HERMES_TUI_ACTIVE_SESSION_FILE"])
    assert active_path.name.startswith("hermes-tui-active-session-")
    assert active_path.suffix == ".json"
    assert active_path_during_call == active_path
    assert not active_path.exists()
    assert env["NODE_ENV"] == "production"




def test_make_tui_argv_dev_prebuilds_hermes_ink(monkeypatch, main_mod, tmp_path):
    tui_dir = tmp_path / "ui-tui"
    tsx = tui_dir / "node_modules" / ".bin" / "tsx"
    ink_dir = tui_dir / "packages" / "hermes-ink"
    tsx.parent.mkdir(parents=True)
    ink_dir.mkdir(parents=True)
    tsx.write_text("#!/usr/bin/env node\n", encoding="utf-8")

    monkeypatch.setattr(main_mod, "_ensure_tui_node", lambda: None)
    monkeypatch.setattr(main_mod, "_tui_need_npm_install", lambda _tui_dir: False)
    monkeypatch.delenv("HERMES_TUI_DIR", raising=False)
    monkeypatch.setattr(main_mod.shutil, "which", lambda bin_name: f"/usr/bin/{bin_name}")

    calls = []

    def fake_run(cmd, cwd=None, **_kwargs):
        calls.append((cmd, cwd))
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(main_mod.subprocess, "run", fake_run)

    argv, cwd = main_mod._make_tui_argv(tui_dir, tui_dev=True)

    assert argv == [str(tsx), "src/entry.tsx"]
    assert cwd == tui_dir
    assert calls == [(["/usr/bin/npm", "run", "build"], str(ink_dir))]


def test_make_tui_argv_uses_writable_cached_bundle_when_source_is_read_only(
    monkeypatch, main_mod, tmp_path
):
    tui_dir = tmp_path / "readonly-install" / "ui-tui"
    source_entry = tui_dir / "dist" / "entry.js"
    source_entry.parent.mkdir(parents=True)
    source_entry.write_text("console.log('source')\n", encoding="utf-8")

    cache_dir = tmp_path / "hermes-home" / "cache" / "tui-bundle"
    cache_entry = cache_dir / "dist" / "entry.js"
    cache_entry.parent.mkdir(parents=True)
    cache_entry.write_text("console.log('cached')\n", encoding="utf-8")

    monkeypatch.setattr(main_mod, "_ensure_tui_node", lambda: None)
    monkeypatch.setattr(main_mod, "_find_bundled_tui", lambda: None)
    monkeypatch.setattr(main_mod, "_tui_need_npm_install", lambda _tui_dir: False)
    monkeypatch.setattr(
        main_mod, "_tui_workspace_writable", lambda _tui_dir: False, raising=False
    )
    monkeypatch.setattr(
        main_mod,
        "_ensure_tui_cached_bundle",
        lambda _tui_dir, *, node, npm=None: cache_dir,
        raising=False,
    )
    monkeypatch.delenv("HERMES_TUI_DIR", raising=False)
    monkeypatch.setattr(
        main_mod.shutil, "which", lambda bin_name: f"/usr/bin/{bin_name}"
    )

    argv, cwd = main_mod._make_tui_argv(tui_dir, tui_dev=False)

    assert argv == ["/usr/bin/node", "--expose-gc", str(cache_entry)]
    assert cwd == cache_dir


def test_make_tui_argv_prefers_source_workspace_when_writable(
    monkeypatch, main_mod, tmp_path
):
    tui_dir = tmp_path / "writable-install" / "ui-tui"
    source_entry = tui_dir / "dist" / "entry.js"
    source_entry.parent.mkdir(parents=True)
    source_entry.write_text("console.log('source')\n", encoding="utf-8")

    monkeypatch.setattr(main_mod, "_ensure_tui_node", lambda: None)
    monkeypatch.setattr(main_mod, "_find_bundled_tui", lambda: None)
    monkeypatch.setattr(main_mod, "_tui_need_npm_install", lambda _tui_dir: False)
    monkeypatch.setattr(
        main_mod, "_tui_workspace_writable", lambda _tui_dir: True, raising=False
    )
    monkeypatch.delenv("HERMES_TUI_DIR", raising=False)
    monkeypatch.setattr(
        main_mod.shutil, "which", lambda bin_name: f"/usr/bin/{bin_name}"
    )
    monkeypatch.setattr(
        main_mod.subprocess,
        "run",
        lambda *_args, **_kwargs: types.SimpleNamespace(
            returncode=0, stdout="", stderr=""
        ),
    )

    argv, cwd = main_mod._make_tui_argv(tui_dir, tui_dev=False)

    assert argv == ["/usr/bin/node", "--expose-gc", str(source_entry)]
    assert cwd == tui_dir


def test_ensure_tui_cached_bundle_uses_root_lockfile_and_workspace_install(
    monkeypatch, main_mod, tmp_path
):
    repo = tmp_path / "repo"
    tui_dir = repo / "ui-tui"
    (tui_dir / "src").mkdir(parents=True)
    (tui_dir / "scripts").mkdir()
    (tui_dir / "packages" / "hermes-ink" / "src").mkdir(parents=True)
    (repo / "apps" / "shared" / "src").mkdir(parents=True)
    (repo / "package.json").write_text(
        '{"private":true,"workspaces":["apps/*","ui-tui","ui-tui/packages/*"],'
        '"overrides":{"lodash":"4.18.1"}}',
        encoding="utf-8",
    )
    (repo / "package-lock.json").write_text(
        '{"name":"hermes-agent","lockfileVersion":3,"packages":{}}',
        encoding="utf-8",
    )
    (tui_dir / "package.json").write_text(
        '{"name":"hermes-tui","scripts":{"build":"node scripts/build.mjs"},'
        '"dependencies":{"@hermes/shared":"file:../apps/shared"}}',
        encoding="utf-8",
    )
    (tui_dir / "scripts" / "build.mjs").write_text(
        "console.log('build')\n", encoding="utf-8"
    )
    (tui_dir / "src" / "entry.tsx").write_text(
        "console.log('src')\n", encoding="utf-8"
    )
    (tui_dir / "packages" / "hermes-ink" / "package.json").write_text(
        '{"name":"@hermes/ink"}', encoding="utf-8"
    )
    (repo / "apps" / "shared" / "package.json").write_text(
        '{"name":"@hermes/shared","exports":{".":"./src/index.ts"}}',
        encoding="utf-8",
    )
    (repo / "apps" / "shared" / "src" / "index.ts").write_text(
        "export const shared = true\n", encoding="utf-8"
    )

    cache_dir = tmp_path / "home" / "cache" / "tui-bundle"
    build_dir = tmp_path / "home" / "cache" / "tui-bundle-build"
    monkeypatch.setattr(main_mod, "_tui_cached_bundle_dir", lambda: cache_dir)
    monkeypatch.setattr(
        main_mod, "_tui_cached_build_dir", lambda _cache_root=None: build_dir
    )
    monkeypatch.setattr(
        "hermes_constants.with_hermes_node_path",
        lambda env=None: {"PATH": "/managed-node/bin"},
    )
    monkeypatch.setenv("HERMES_QUIET", "1")

    calls = []
    colliding_tmp = build_dir.with_name(f"{build_dir.name}.{os.getpid()}.tmp")
    colliding_tmp.mkdir(parents=True)
    collision_marker = colliding_tmp / "belongs-to-another-writer"
    collision_marker.write_text("keep", encoding="utf-8")

    def fake_run(cmd, cwd=None, **kwargs):
        calls.append((cmd, Path(cwd) if cwd else None))
        assert kwargs["env"]["PATH"] == "/managed-node/bin"
        if cmd[:2] == ["npm", "install"]:
            assert cwd == str(build_dir)
            root_manifest = (build_dir / "package.json").read_text(encoding="utf-8")
            assert '"overrides"' in root_manifest
            assert (build_dir / "package-lock.json").is_file()
            assert (build_dir / "ui-tui" / "package.json").is_file()
            assert (build_dir / "apps" / "shared" / "src" / "index.ts").is_file()
        elif cmd == ["npm", "run", "build", "--workspace", "ui-tui"]:
            assert cwd == str(build_dir)
            entry = build_dir / "ui-tui" / "dist" / "entry.js"
            entry.parent.mkdir(parents=True, exist_ok=True)
            entry.write_text("console.log('cached bundle')\n", encoding="utf-8")
        elif cmd[:2] == ["node", "--check"]:
            assert cmd[2] == str(build_dir / "ui-tui" / "dist" / "entry.js")
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(main_mod.subprocess, "run", fake_run)

    result = main_mod._ensure_tui_cached_bundle(tui_dir, node="node", npm="npm")

    assert result.parent == cache_dir / "generations"
    assert main_mod._tui_cached_active_bundle_dir(cache_dir) == result
    assert (cache_dir / "current").read_text(encoding="utf-8").strip() == result.name
    assert (result / "dist" / "entry.js").read_text(encoding="utf-8") == (
        "console.log('cached bundle')\n"
    )
    assert cache_dir.with_name("tui-bundle.lock").is_file()
    assert calls[0][0] == [
        "npm",
        "install",
        "--workspace",
        "ui-tui",
        "--include=dev",
        "--silent",
        "--no-fund",
        "--no-audit",
        "--progress=false",
    ]
    assert calls[1][0] == ["npm", "run", "build", "--workspace", "ui-tui"]
    assert collision_marker.read_text(encoding="utf-8") == "keep"


def test_tui_cached_bundle_stamps_the_staged_copy(main_mod, tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    tui_dir = repo / "ui-tui"
    (tui_dir / "src").mkdir(parents=True)
    (repo / "package.json").write_text(
        '{"private":true,"workspaces":["ui-tui"]}', encoding="utf-8"
    )
    (repo / "package-lock.json").write_text(
        '{"name":"hermes-agent","lockfileVersion":3,"packages":{}}',
        encoding="utf-8",
    )
    (tui_dir / "package.json").write_text(
        '{"name":"hermes-tui","scripts":{"build":"node build.js"}}',
        encoding="utf-8",
    )
    source_file = tui_dir / "src" / "index.tsx"
    source_file.write_text("export const version = 'before'\n", encoding="utf-8")

    cache_root = tmp_path / "home" / "cache" / "tui-bundle"
    build_dir = tmp_path / "home" / "cache" / "tui-bundle-build"
    monkeypatch.setattr(main_mod, "_tui_cached_bundle_dir", lambda: cache_root)
    monkeypatch.setattr(
        main_mod, "_tui_cached_build_dir", lambda _cache_root=None: build_dir
    )
    monkeypatch.setattr(main_mod, "_tui_workspace_writable", lambda _path: False)
    monkeypatch.setattr(main_mod, "_resolve_node_runtime_npm", lambda: "/usr/bin/npm")

    original_copytree = main_mod.shutil.copytree

    def mutate_before_primary_copy(src, dst, *args, **kwargs):
        if Path(src) == tui_dir:
            source_file.write_text("export const version = 'after'\n", encoding="utf-8")
        return original_copytree(src, dst, *args, **kwargs)

    staged = {}

    def fake_run(cmd, **kwargs):
        if "cwd" not in kwargs:
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        cwd = Path(kwargs["cwd"])
        if cmd[1:2] == ["install"]:
            staged["stamp"] = main_mod._tui_bundle_stamp(cwd / "ui-tui")
        elif cmd[1:3] == ["run", "build"]:
            dist = cwd / "ui-tui" / "dist"
            dist.mkdir(parents=True, exist_ok=True)
            (dist / "entry.js").write_text("console.log('built')\n", encoding="utf-8")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(main_mod.shutil, "copytree", mutate_before_primary_copy)
    monkeypatch.setattr(main_mod.subprocess, "run", fake_run)

    initial_stamp = main_mod._tui_bundle_stamp(tui_dir)
    result = main_mod._ensure_tui_cached_bundle(tui_dir, node="node", npm="npm")

    assert staged["stamp"] != initial_stamp
    assert (result / ".hermes-tui-bundle-stamp").read_text(
        encoding="utf-8"
    ).strip() == staged["stamp"]


def test_tui_bundle_stamp_tracks_external_file_workspace(main_mod, tmp_path):
    repo = tmp_path / "repo"
    tui_dir = repo / "ui-tui"
    shared_source = repo / "apps" / "shared" / "src" / "index.ts"
    (tui_dir / "src").mkdir(parents=True)
    shared_source.parent.mkdir(parents=True)
    (repo / "package.json").write_text(
        '{"private":true,"workspaces":["apps/*","ui-tui"]}', encoding="utf-8"
    )
    (repo / "package-lock.json").write_text(
        '{"name":"hermes-agent","lockfileVersion":3,"packages":{}}',
        encoding="utf-8",
    )
    (tui_dir / "package.json").write_text(
        '{"name":"hermes-tui","dependencies":'
        '{"@hermes/shared":"file:../apps/shared"}}',
        encoding="utf-8",
    )
    (tui_dir / "src" / "entry.tsx").write_text(
        "console.log('tui')\n", encoding="utf-8"
    )
    (repo / "apps" / "shared" / "package.json").write_text(
        '{"name":"@hermes/shared"}', encoding="utf-8"
    )
    shared_source.write_text("export const version = 1\n", encoding="utf-8")

    first = main_mod._tui_bundle_stamp(tui_dir)
    shared_source.write_text("export const version = 2\n", encoding="utf-8")

    assert main_mod._tui_bundle_stamp(tui_dir) != first


def test_tui_bundle_stamp_rejects_symlinks_in_external_workspace(main_mod, tmp_path):
    repo = tmp_path / "repo"
    tui_dir = repo / "ui-tui"
    shared_dir = repo / "apps" / "shared"
    outside = tmp_path / "outside"
    (tui_dir / "src").mkdir(parents=True)
    shared_dir.mkdir(parents=True)
    outside.mkdir()
    (repo / "package.json").write_text(
        '{"private":true,"workspaces":["apps/*","ui-tui"]}', encoding="utf-8"
    )
    (repo / "package-lock.json").write_text(
        '{"name":"hermes-agent","lockfileVersion":3,"packages":{}}',
        encoding="utf-8",
    )
    (tui_dir / "package.json").write_text(
        '{"name":"hermes-tui","dependencies":'
        '{"@hermes/shared":"file:../apps/shared"}}',
        encoding="utf-8",
    )
    (outside / "secret.ts").write_text("export const secret = true\n", encoding="utf-8")
    (shared_dir / "escape").symlink_to(outside, target_is_directory=True)

    with pytest.raises(RuntimeError, match="symlink"):
        main_mod._tui_bundle_stamp(tui_dir)


def test_tui_bundle_stamp_rejects_symlinks_in_primary_workspace(main_mod, tmp_path):
    repo = tmp_path / "repo"
    tui_dir = repo / "ui-tui"
    outside = tmp_path / "outside"
    (tui_dir / "src").mkdir(parents=True)
    outside.mkdir()
    (repo / "package.json").write_text(
        '{"private":true,"workspaces":["ui-tui"]}', encoding="utf-8"
    )
    (repo / "package-lock.json").write_text(
        '{"name":"hermes-agent","lockfileVersion":3,"packages":{}}',
        encoding="utf-8",
    )
    (tui_dir / "package.json").write_text('{"name":"hermes-tui"}', encoding="utf-8")
    (outside / "secret.ts").write_text("export const secret = true\n", encoding="utf-8")
    (tui_dir / "src" / "escape").symlink_to(outside, target_is_directory=True)

    with pytest.raises(RuntimeError, match="symlink"):
        main_mod._tui_bundle_stamp(tui_dir)


def test_tui_workspace_input_iterator_rejects_symlinked_workspace_root(
    main_mod, tmp_path
):
    real_workspace = tmp_path / "real-ui-tui"
    real_workspace.mkdir()
    (real_workspace / "package.json").write_text(
        '{"name":"hermes-tui"}', encoding="utf-8"
    )
    linked_workspace = tmp_path / "linked-ui-tui"
    linked_workspace.symlink_to(real_workspace, target_is_directory=True)

    with pytest.raises(RuntimeError, match="symlink"):
        list(main_mod._iter_tui_workspace_inputs(Path("ui-tui"), linked_workspace))


def test_tui_cache_advisory_lock_serializes_builders(main_mod, tmp_path):
    lock_path = tmp_path / "tui-bundle.lock"
    first = main_mod._try_acquire_tui_cache_lock(lock_path)
    assert first is not None

    try:
        assert main_mod._try_acquire_tui_cache_lock(lock_path) is None
    finally:
        main_mod._release_tui_cache_lock(first)

    second = main_mod._try_acquire_tui_cache_lock(lock_path)
    assert second is not None
    main_mod._release_tui_cache_lock(second)


def test_tui_cache_refresh_keeps_previous_returned_generation_launchable(
    monkeypatch, main_mod, tmp_path
):
    repo = tmp_path / "repo"
    tui_dir = repo / "ui-tui"
    (tui_dir / "src").mkdir(parents=True)
    (tui_dir / "scripts").mkdir()
    (repo / "package.json").write_text(
        '{"private":true,"workspaces":["ui-tui"]}', encoding="utf-8"
    )
    (repo / "package-lock.json").write_text(
        '{"name":"hermes-agent","lockfileVersion":3,"packages":{}}',
        encoding="utf-8",
    )
    (tui_dir / "package.json").write_text(
        '{"name":"hermes-tui","scripts":{"build":"node scripts/build.mjs"}}',
        encoding="utf-8",
    )
    (tui_dir / "scripts" / "build.mjs").write_text(
        "console.log('build')\n", encoding="utf-8"
    )
    source_entry = tui_dir / "src" / "entry.tsx"
    source_entry.write_text("console.log('source-v1')\n", encoding="utf-8")

    cache_dir = tmp_path / "home" / "cache" / "tui-bundle"
    build_dir = tmp_path / "home" / "cache" / "tui-bundle-build"
    monkeypatch.setattr(main_mod, "_tui_cached_bundle_dir", lambda: cache_dir)
    monkeypatch.setattr(
        main_mod, "_tui_cached_build_dir", lambda _cache_root=None: build_dir
    )
    monkeypatch.setenv("HERMES_QUIET", "1")

    build_number = 0

    def fake_run(cmd, cwd=None, **_kwargs):
        nonlocal build_number
        if cmd == ["npm", "run", "build", "--workspace", "ui-tui"]:
            build_number += 1
            entry = build_dir / "ui-tui" / "dist" / "entry.js"
            entry.parent.mkdir(parents=True, exist_ok=True)
            entry.write_text(f"console.log('bundle-v{build_number}')\n", encoding="utf-8")
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(main_mod.subprocess, "run", fake_run)

    first = main_mod._ensure_tui_cached_bundle(tui_dir, node="node", npm="npm")
    first_entry = first / "dist" / "entry.js"
    assert first_entry.read_text(encoding="utf-8") == "console.log('bundle-v1')\n"

    source_entry.write_text("console.log('source-v2')\n", encoding="utf-8")
    second = main_mod._ensure_tui_cached_bundle(tui_dir, node="node", npm="npm")

    assert second != first
    assert first_entry.is_file()
    assert first_entry.read_text(encoding="utf-8") == "console.log('bundle-v1')\n"
    assert (second / "dist" / "entry.js").read_text(encoding="utf-8") == (
        "console.log('bundle-v2')\n"
    )


def test_tui_cached_bundle_rejects_symlinked_generations_parent(
    monkeypatch, main_mod, tmp_path
):
    cache_root = tmp_path / "home" / "cache" / "tui-bundle"
    outside = tmp_path / "outside-generations"
    evil = outside / "evil"
    (evil / "dist").mkdir(parents=True)
    (evil / "dist" / "entry.js").write_text("console.log('evil')\n", encoding="utf-8")
    cache_root.mkdir(parents=True)
    (cache_root / "generations").symlink_to(outside, target_is_directory=True)
    (cache_root / "current").write_text("evil\n", encoding="utf-8")

    assert main_mod._tui_cached_active_bundle_dir(cache_root) == cache_root

    staged = cache_root / "staged"
    staged.mkdir()
    with pytest.raises(RuntimeError, match="unsafe TUI cache generations"):
        main_mod._publish_tui_cached_generation(cache_root, staged, "deadbeef")
    main_mod._cleanup_tui_cached_generations(cache_root, evil)
    assert (evil / "dist" / "entry.js").is_file()


def test_tui_cached_bundle_rejects_symlinked_cache_root(main_mod, tmp_path):
    outside = tmp_path / "outside-cache"
    outside.mkdir()
    cache_root = tmp_path / "tui-bundle"
    cache_root.symlink_to(outside, target_is_directory=True)

    assert main_mod._tui_cached_generations_dir(cache_root, create=True) is None
    assert not (outside / "generations").exists()


def test_make_tui_argv_uses_managed_npm_outside_path(
    monkeypatch, main_mod, tmp_path
):
    tui_dir = tmp_path / "readonly-install" / "ui-tui"
    tui_dir.mkdir(parents=True)
    cache_dir = tmp_path / "home" / "cache" / "tui-bundle"
    cache_entry = cache_dir / "dist" / "entry.js"
    cache_entry.parent.mkdir(parents=True)
    cache_entry.write_text("console.log('cached')\n", encoding="utf-8")
    calls = []

    monkeypatch.delenv("HERMES_TUI_DIR", raising=False)
    monkeypatch.setattr(main_mod, "_ensure_tui_node", lambda: None)
    monkeypatch.setattr(main_mod, "_find_bundled_tui", lambda: None)
    monkeypatch.setattr(main_mod, "_tui_workspace_writable", lambda _path: False)
    monkeypatch.setattr(main_mod, "_resolve_node_runtime_npm", lambda: "/managed/npm")
    monkeypatch.setattr(main_mod.shutil, "which", lambda _name: None)
    monkeypatch.setattr(
        "hermes_constants.find_node_executable",
        lambda name: f"/managed/{name}",
    )
    monkeypatch.setattr(
        main_mod,
        "_ensure_tui_cached_bundle",
        lambda path, *, node, npm=None: calls.append((path, node, npm)) or cache_dir,
    )

    argv, cwd = main_mod._make_tui_argv(tui_dir, tui_dev=False)

    assert calls == [(tui_dir, "/managed/node", "/managed/npm")]
    assert argv == ["/managed/node", "--expose-gc", str(cache_entry)]
    assert cwd == cache_dir


def test_tui_cache_refresh_uses_managed_node_and_npm(
    monkeypatch, main_mod, tmp_path
):
    tui_dir = tmp_path / "readonly-install" / "ui-tui"
    tui_dir.mkdir(parents=True)
    cache_root = tmp_path / "home" / "cache" / "tui-bundle"
    cache_root.mkdir(parents=True)
    calls = []

    monkeypatch.setattr(main_mod, "_tui_cached_bundle_dir", lambda: cache_root)
    monkeypatch.setattr(main_mod, "_resolve_node_runtime_npm", lambda: "/managed/npm")
    monkeypatch.setattr(main_mod.shutil, "which", lambda _name: None)
    monkeypatch.setattr(
        "hermes_constants.find_node_executable",
        lambda name: f"/managed/{name}",
    )
    monkeypatch.setattr(
        main_mod,
        "_ensure_tui_cached_bundle",
        lambda path, *, node, npm=None: calls.append((path, node, npm)) or cache_root,
    )

    main_mod._refresh_tui_cached_bundle_after_update(tui_dir)

    assert calls == [(tui_dir, "/managed/node", "/managed/npm")]


def test_tui_workspace_writable_preserves_colliding_probe(
    monkeypatch, main_mod, tmp_path
):
    tui_dir = tmp_path / "ui-tui"
    tui_dir.mkdir()
    (tui_dir / "package-lock.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(main_mod.os, "getpid", lambda: 424242)
    probe = tui_dir / ".hermes-tui-write-test-424242"
    probe.write_text("belongs-to-someone-else", encoding="utf-8")

    assert main_mod._tui_workspace_writable(tui_dir) is True
    assert probe.read_text(encoding="utf-8") == "belongs-to-someone-else"


def test_tui_cache_rejects_symlinked_ancestor(main_mod, tmp_path):
    home = tmp_path / "home"
    outside = tmp_path / "outside"
    home.mkdir()
    outside.mkdir()
    (home / "cache").symlink_to(outside, target_is_directory=True)
    cache_root = home / "cache" / "tui-bundle"

    assert main_mod._tui_cached_generations_dir(cache_root, create=True) is None
    assert not (outside / "tui-bundle" / "generations").exists()


def test_tui_cache_rejects_junction_in_ancestor(
    monkeypatch, main_mod, tmp_path
):
    home = tmp_path / "home"
    cache_parent = home / "cache"
    cache_parent.mkdir(parents=True)
    cache_root = cache_parent / "tui-bundle"
    monkeypatch.setattr(
        Path,
        "is_junction",
        lambda self: self == cache_parent,
        raising=False,
    )

    assert main_mod._tui_cached_generations_dir(cache_root, create=True) is None
    assert not (cache_root / "generations").exists()


def test_tui_workspace_inputs_reject_junction_ancestor(
    monkeypatch, main_mod, tmp_path
):
    workspace = tmp_path / "ui-tui"
    junction = workspace / "src" / "junction"
    junction.mkdir(parents=True)
    (junction / "outside.ts").write_text(
        "export const outside = true\n", encoding="utf-8"
    )
    monkeypatch.setattr(
        Path,
        "is_junction",
        lambda self: self == junction,
        raising=False,
    )

    with pytest.raises(RuntimeError, match="redirect|junction"):
        list(main_mod._iter_tui_workspace_inputs(Path("ui-tui"), workspace))


def test_tui_redirect_detection_supports_python311_reparse_attributes(main_mod):
    class ReparsePath:
        @staticmethod
        def is_symlink():
            return False

        @staticmethod
        def lstat():
            return types.SimpleNamespace(st_file_attributes=0x400)

    assert main_mod._tui_path_is_redirect(ReparsePath()) is True


@pytest.mark.skipif(os.name == "nt", reason="dir_fd hardening is POSIX-only")
def test_tui_cache_paths_resolve_legitimate_symlinked_hermes_home(
    monkeypatch, main_mod, tmp_path
):
    real_home = tmp_path / "real-home"
    real_home.mkdir(mode=0o700)
    linked_home = tmp_path / "linked-home"
    linked_home.symlink_to(real_home, target_is_directory=True)
    monkeypatch.setenv("HERMES_HOME", str(linked_home))

    cache_root = main_mod._tui_cached_bundle_dir()
    replacement_home = tmp_path / "replacement-home"
    replacement_home.mkdir(mode=0o700)
    linked_home.unlink()
    linked_home.symlink_to(replacement_home, target_is_directory=True)
    build_root = main_mod._tui_cached_build_dir(cache_root)

    assert cache_root == real_home / "cache" / "tui-bundle"
    assert build_root == real_home / "cache" / "tui-bundle-build"

    main_mod._prepare_tui_cache_root(cache_root)

    assert cache_root.is_dir()
    assert not (replacement_home / "cache").exists()


@pytest.mark.skipif(os.name == "nt", reason="dir_fd hardening is POSIX-only")
def test_tui_cache_root_creation_resists_parent_swap(
    monkeypatch, main_mod, tmp_path
):
    home = tmp_path / "home"
    cache_parent = home / "cache"
    displaced_cache = home / "cache-original"
    outside = tmp_path / "outside"
    cache_parent.mkdir(parents=True, mode=0o700)
    outside.mkdir()
    cache_root = cache_parent / "tui-bundle"
    original_mkdir = main_mod.os.mkdir
    swapped = False

    def swap_before_cache_root_create(path, mode=0o777, *, dir_fd=None):
        nonlocal swapped
        if path == "tui-bundle" and dir_fd is not None and not swapped:
            cache_parent.rename(displaced_cache)
            cache_parent.symlink_to(outside, target_is_directory=True)
            swapped = True
        return original_mkdir(path, mode, dir_fd=dir_fd)

    monkeypatch.setattr(main_mod.os, "mkdir", swap_before_cache_root_create)

    with pytest.raises(RuntimeError, match="unsafe TUI cache"):
        main_mod._prepare_tui_cache_root(cache_root)

    assert swapped is True
    assert not (outside / "tui-bundle").exists()
    assert (displaced_cache / "tui-bundle").is_dir()


@pytest.mark.skipif(os.name == "nt", reason="POSIX ownership/mode gate")
def test_tui_cache_root_rejects_group_writable_home(main_mod, tmp_path):
    home = tmp_path / "home"
    home.mkdir(mode=0o2770)
    home.chmod(0o2770)
    cache_root = home / "cache" / "tui-bundle"

    with pytest.raises(RuntimeError, match="private|unsafe"):
        main_mod._prepare_tui_cache_root(cache_root)

    assert not cache_root.exists()
