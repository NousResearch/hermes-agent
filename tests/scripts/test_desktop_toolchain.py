"""Desktop preparation keeps dependency acquisition out of live installs."""
from __future__ import annotations

import importlib
import os
from pathlib import Path
import json
import subprocess
import sys
from types import SimpleNamespace

import pytest


def test_bootstrap_environment_isolates_owned_paths_without_mutating_caller(tmp_path):
    toolchain = importlib.import_module("scripts.bundles.desktop_toolchain")
    source, work, cache = (tmp_path / name for name in ("source", "work", "cache"))
    original_home = tmp_path / "live-user"
    inherited = {
        "HOME": str(original_home), "USERPROFILE": str(original_home),
        "LOCALAPPDATA": str(original_home / "AppData/Local"),
        "APPDATA": str(original_home / "AppData/Roaming"),
        "HERMES_HOME": str(original_home / "profile"),
        "HERMES_RUNTIME_DIR": str(original_home / "tools"),
        "HERMES_INSTALL_ROOT": str(original_home / "installed"),
        "HERMES_PAYLOAD_ROOT": str(original_home / "payload"),
        "HERMES_PAYLOAD_TAG": "v1.2.3", "HERMES_BUILD_COMMIT": "a" * 40,
        "HERMES_SITE": str(original_home / "site"),
        "HERMES_PYTHON_SRC_ROOT": str(original_home / "repo"),
        "HERMES_PYTHON": str(original_home / "python"),
        "HERMES_NODE": str(original_home / "node"),
        "HERMES_PROFILE": "live", "HERMES_REAL_HOME": str(original_home),
        "HERMES_BUNDLED_SKILLS": str(original_home / "skills"),
        "HERMES_OPTIONAL_MCPS": str(original_home / "mcps"),
        "VIRTUAL_ENV": str(original_home / "venv"),
        "PYTHONHOME": str(original_home / "python-home"),
        "PYTHONPATH": str(original_home / "site"),
        "UV_CACHE_DIR": str(original_home / "uv-cache"),
        "npm_config_cache": str(original_home / "npm-cache"),
        "npm_execpath": str(original_home / "foreign-npm.js"),
        "NPM_CONFIG_USERCONFIG": str(original_home / "credentials.npmrc"),
        "CARGO_HOME": str(original_home / "custom-cargo"),
        "RUSTUP_HOME": str(original_home / "custom-rustup"),
        "PATH": os.defpath, "SIGNING_TOKEN": "inherited-not-serialized",
    }
    before = inherited.copy()
    process_before = dict(os.environ)
    environment = toolchain.bootstrap_environment(source, work, cache, inherited)
    assert inherited == before
    assert dict(os.environ) == process_before
    assert environment["CARGO_HOME"] == inherited["CARGO_HOME"]
    assert environment["RUSTUP_HOME"] == inherited["RUSTUP_HOME"]
    for key in ("HOME", "USERPROFILE", "LOCALAPPDATA", "APPDATA", "HERMES_HOME",
                "XDG_CONFIG_HOME", "XDG_CACHE_HOME"):
        assert Path(environment[key]).is_relative_to(work)
    assert Path(environment["HERMES_RUNTIME_DIR"]) == cache / "tools"
    assert Path(environment["UV_CACHE_DIR"]) == cache / "python/runtime"
    assert Path(environment["npm_config_cache"]) == cache / "npm"
    assert "npm_execpath" not in environment
    assert "NPM_CONFIG_USERCONFIG" not in environment
    assert environment["npm_config_userconfig"] == os.devnull
    assert "npm_config_globalconfig" not in environment
    assert environment["HERMES_PYTHON_SRC_ROOT"] == str(source)
    for key in ("HERMES_INSTALL_ROOT", "HERMES_PAYLOAD_ROOT", "HERMES_PAYLOAD_TAG",
                "HERMES_BUILD_COMMIT", "HERMES_SITE", "HERMES_PROFILE", "HERMES_REAL_HOME",
                "HERMES_PYTHON", "HERMES_NODE", "HERMES_BUNDLED_SKILLS", "HERMES_OPTIONAL_MCPS",
                "VIRTUAL_ENV", "PYTHONHOME", "PYTHONPATH"):
        assert key not in environment
    assert environment["SIGNING_TOKEN"] == inherited["SIGNING_TOKEN"]
    assert environment["PATH"] == inherited["PATH"]
    assert not work.exists()  # Forming the child environment is pure.

    defaults = toolchain.bootstrap_environment(source, work, cache, {"HOME": str(original_home),
                                                                   "USERPROFILE": str(original_home)})
    assert defaults["CARGO_HOME"] == str(original_home / ".cargo")
    assert defaults["RUSTUP_HOME"] == str(original_home / ".rustup")

    mixed = toolchain.bootstrap_environment(source, work, cache, {
        "Home": str(original_home), "UserProfile": str(original_home), "LocalAppData": "live-local",
        "Hermes_Home": "live-profile", "Hermes_Runtime_Dir": "live-store", "Cargo_Home": "custom-cargo",
        "Rustup_Home": "custom-rustup", "NPM_CONFIG_CACHE": "live-npm", "Path": os.defpath,
    })
    assert mixed["CARGO_HOME"] == "custom-cargo" and mixed["RUSTUP_HOME"] == "custom-rustup"
    assert len([key for key in mixed if key.upper() == "HOME"]) == 1
    assert len([key for key in mixed if key.upper() == "HERMES_RUNTIME_DIR"]) == 1


@pytest.mark.parametrize("exit_code", [0, 7])
@pytest.mark.platforms("linux", "macos", "windows")
def test_run_preparation_bootstraps_before_worker_in_isolated_child(tmp_path, monkeypatch, exit_code):
    from scripts.bundles import desktop_toolchain

    source, work, cache = (tmp_path / name for name in ("source space", "work", "cache"))
    (source / "pm").mkdir(parents=True)
    (source / "pm/__init__.py").write_text("", encoding="utf-8")
    # A bootstrap double runs in a REAL child: imports must already be isolated,
    # not temporarily redirected in the parent's process around the PM call.
    (source / "pm/runtime.py").write_text(
        "import os, sys\n"
        "from pathlib import Path\n"
        "assert Path(os.environ['HOME']).name == 'home'\n"
        "assert 'HERMES_INSTALL_ROOT' not in os.environ\n"
        "assert sys.flags.isolated and sys.flags.no_site\n"
        "def runtime_command(script, args, *, cache):\n"
        "    assert cache == Path(os.environ['UV_CACHE_DIR'])\n"
        "    return [sys.executable, '-I', '-B', str(script), *args]\n",
        encoding="utf-8",
    )
    worker = source / "scripts/bundles/desktop_prepare.py"
    worker.parent.mkdir(parents=True)
    worker.write_text(
        "import json, os, sys\n"
        "from pathlib import Path\n"
        "assert sys.argv[1] == '--request' and sys.argv[3:] == ['--worker']\n"
        "import ssl\n"
        "assert ssl.SSLContext.__module__.startswith('truststore')\n"
        "request = json.loads(Path(sys.argv[2]).read_text())\n"
        "assert os.environ['HERMES_RUNTIME_DIR'] == request['tools']\n"
        "Path(request['receipt']).write_text(str(Path.cwd()))\n"
        "sys.exit(request['exit_code'])\n", encoding="utf-8",
    )
    request_file = tmp_path / "request space.json"
    receipt = tmp_path / "worker-ran"
    request_file.write_text(json.dumps({"receipt": str(receipt), "tools": str(cache / "tools"),
                                        "exit_code": exit_code}), encoding="utf-8")
    monkeypatch.setenv("HERMES_INSTALL_ROOT", str(tmp_path / "live-install"))
    before = dict(os.environ)
    assert desktop_toolchain.run_preparation(source, work, cache, request_file) == exit_code
    assert dict(os.environ) == before
    assert receipt.read_text(encoding="utf-8-sig") == str(source)


def test_bootstrap_real_pm_resolves_only_build_owned_state(tmp_path):
    from scripts.bundles.desktop_toolchain import bootstrap_environment

    source = Path(__file__).resolve().parents[2]
    work, cache = tmp_path / "work", tmp_path / "cache"
    env = bootstrap_environment(source, work, cache, os.environ)
    probe = (
        "import json, sys; from pathlib import Path; sys.path.insert(0, sys.argv[1]); "
        "from pm import paths; from pm.packages import uv_cache_dir; "
        "from hermes_cli.runtime_paths import install_state_dir; "
        "print(json.dumps([str(paths.store_root()), str(paths.partials_root()), "
        "str(uv_cache_dir()), str(install_state_dir(Path(sys.argv[1])))]))"
    )
    result = subprocess.run([sys.executable, "-I", "-S", "-B", "-c", probe, str(source)],
                            env=env, capture_output=True, text=True, check=True)
    store, partials, uv_default, state = map(Path, json.loads(result.stdout))
    assert store == cache / "tools"
    assert all(path.is_relative_to(work) for path in (partials, uv_default, state))


@pytest.mark.parametrize("target", ["darwin-arm64", "darwin-x64", "linux-x64", "win32-arm64"])
def test_packaging_preserves_keychain_home_without_retargeting_build_state(tmp_path, target):
    from scripts.bundles.desktop_inputs import packaging_environment
    from scripts.bundles.desktop_toolchain import bootstrap_environment

    source, work, cache = (tmp_path / name for name in ("source", "work", "cache"))
    login_home = str(tmp_path / "login")
    inherited = {"HOME": login_home, "CSC_KEYCHAIN": "explicit.keychain"}
    isolated = bootstrap_environment(source, work, cache, inherited)
    before = isolated.copy()
    for caller in (inherited, {**inherited, "HOME": str(work / "launcher-home"),
                               "HERMES_REAL_HOME": login_home}, {}):
        packaged = packaging_environment(isolated, caller, target)
        expected_home = (caller.get("HERMES_REAL_HOME") or caller.get("HOME") or str(Path.home())
                         if target.startswith("darwin-") else isolated["HOME"])
        assert packaged == {**isolated, "HOME": expected_home}
    assert isolated == before


@pytest.mark.platforms("linux", "macos", "windows")
def test_prepare_tools_uses_pm_native_pins_and_separate_cache(tmp_path, monkeypatch):
    import pm
    from scripts.bundles import desktop_toolchain
    from scripts.build import windows_deps

    source, work, cache = (tmp_path / name for name in ("source", "work", "cache"))
    env = desktop_toolchain.bootstrap_environment(source, work, cache, os.environ)
    monkeypatch.setenv("HERMES_RUNTIME_DIR", env["HERMES_RUNTIME_DIR"])
    acquired, native_calls = [], []
    bins = {name: cache / "tools" / name / "bin" / name for name in ("python", "node")}
    for binary in bins.values():
        binary.parent.mkdir(parents=True)
        binary.touch()

    def ensure(name, *, explicit):
        assert explicit
        acquired.append(name)

    def native(**kwargs):
        native_calls.append(kwargs)
        return {**kwargs["env"], "OPENSSL_DIR": str(cache / "native/openssl")}

    monkeypatch.setattr(pm, "ensure", ensure)
    monkeypatch.setattr(pm, "installed_package", lambda name: SimpleNamespace(binary=bins[name]))
    monkeypatch.setattr(pm, "env_for", lambda *names, base_env: {**base_env, "PATH": "pm-tools"})
    monkeypatch.setattr(windows_deps, "prepare_windows_environment", native)
    monkeypatch.setattr(desktop_toolchain, "native_cache_path", lambda cache, env: cache / "python/runtime/native-identity",
                        raising=False)
    before = dict(os.environ)
    python, node, prepared = desktop_toolchain.prepare_tools(source, work, cache, env)
    assert acquired == ["uv", "npm"]
    assert python == bins["python"] and node == bins["node"]
    assert prepared["HERMES_PYTHON"] == str(python)
    assert prepared["HERMES_NODE"] == str(node)
    assert prepared["PATH"] == "pm-tools"
    assert Path(prepared["UV_CACHE_DIR"]) == cache / "python/runtime/native-identity"
    assert dict(os.environ) == before
    if pm.current_target() == "win32-arm64":
        assert len(native_calls) == 1
        assert native_calls[0]["state"].is_relative_to(cache / "native")
        assert prepared["OPENSSL_DIR"] == str(cache / "native/openssl")
    else:
        assert native_calls == []


@pytest.mark.platforms("linux", "macos", "windows")
def test_native_cache_identity_tracks_compilers_sdk_and_openssl(tmp_path, monkeypatch):
    from scripts.bundles import desktop_toolchain

    openssl = tmp_path / "openssl"
    (openssl / "include/openssl").mkdir(parents=True)
    (openssl / "include/openssl/opensslv.h").write_bytes(b"version one")
    (openssl / "lib").mkdir()
    library = openssl / "lib/libcrypto.lib"
    library.write_bytes(b"first build")
    (openssl / "lib/libssl.lib").write_bytes(b"ssl build")
    env = {"PATH": os.defpath, "OPENSSL_DIR": str(openssl),
           "WindowsSDKVersion": "10.0.1", "VCToolsVersion": "14.1"}
    version = ["compiler one"]

    def probe(command, **kwargs):
        assert kwargs["env"] == env
        return SimpleNamespace(returncode=0, stdout=version[0], stderr="")

    # Let stdlib cache its real-host probe before substituting compiler commands.
    desktop_toolchain.platform.platform()
    monkeypatch.setattr(desktop_toolchain.subprocess, "run", probe)
    cache = tmp_path / "cache"
    first = desktop_toolchain.native_cache_path(cache, env)
    assert first.is_relative_to(cache / "python/runtime")
    assert desktop_toolchain.native_cache_path(cache, env) == first
    moved_cache = tmp_path / "relocated-cache"
    moved_openssl = moved_cache / "native/openssl"
    import shutil
    shutil.copytree(openssl, cache / "native/openssl")
    shutil.copytree(openssl, moved_openssl)
    env["OPENSSL_DIR"] = str(cache / "native/openssl")
    portable = desktop_toolchain.native_cache_path(cache, env)
    env["OPENSSL_DIR"] = str(moved_openssl)
    assert desktop_toolchain.native_cache_path(moved_cache, env).name == portable.name
    env["OPENSSL_DIR"] = str(openssl)
    version[0] = "compiler two"
    second = desktop_toolchain.native_cache_path(cache, env)
    assert second != first
    library.write_bytes(b"changed build")
    third = desktop_toolchain.native_cache_path(cache, env)
    assert third != second
    env["WindowsSDKVersion"] = "10.0.2"
    assert desktop_toolchain.native_cache_path(cache, env) != third
    monkeypatch.setattr(desktop_toolchain.subprocess, "run", lambda *a, **k: (_ for _ in ()).throw(FileNotFoundError()))
    assert desktop_toolchain.native_cache_path(cache, env) != desktop_toolchain.native_cache_path(cache, env)


@pytest.mark.parametrize("complete", [False, True])
def test_native_cache_leaves_room_for_sdist_compiler_outputs(tmp_path, monkeypatch, complete):
    from pathlib import PureWindowsPath
    from scripts.bundles import desktop_toolchain

    # Replay the Windows CI layout as path data, without faking the host OS.
    # uv builds in its cached sdist; /Fo is relative to that working directory.
    cache = tmp_path / "cache"
    desktop_toolchain.platform.platform()

    def probe(*args, **kwargs):
        if not complete:
            raise FileNotFoundError()
        return SimpleNamespace(returncode=0, stdout="compiler identity", stderr="")

    openssl = cache / "native/openssl"
    for name in ("include/openssl/opensslv.h", "lib/libcrypto.lib"):
        path = openssl / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"native input")
    monkeypatch.setattr(desktop_toolchain.subprocess, "run", probe)
    selected = desktop_toolchain.native_cache_path(cache, {
        "WindowsSDKVersion": "10.0.1", "VCToolsVersion": "14.1", "OPENSSL_DIR": str(openssl),
    })
    assert selected.is_relative_to(cache / "python/runtime")
    runner_cache = PureWindowsPath("D:/a/hermes-agent/hermes-agent/.cache/desktop-inputs")
    sdist = PureWindowsPath("sdists-v9/pypi/pilk/0.2.4/5b4cbVXt0GPuGQrp/src")
    for architecture in ("win-amd64", "win-arm64"):
        output = (runner_cache.joinpath(*selected.relative_to(cache).parts) / sdist
                  / f"build/temp.{architecture}-cpython-314/Release/src/SKP_SILK_SRC"
                  / "SKP_Silk_NLSF_VQ_rate_distortion_FIX.obj")
        assert len(str(output)) < 260, str(output)


@pytest.mark.parametrize("on_demand", [False, True])
def test_icon_environment_prepares_locked_group_before_generation(tmp_path, monkeypatch, on_demand):
    import pm
    from scripts.build import icon_environment

    source, out, cache = (tmp_path / name for name in ("source", "icons-venv", "build-cache"))
    python = out / "bin/python"
    acquired = []

    def build(**kwargs):
        acquired.append(kwargs)
        return python

    monkeypatch.setattr(pm, "build_environment", build)
    assert icon_environment.prepare_icon_environment(source, out, cache) == python
    assert acquired == [{"source": source, "out": out, "cache": cache,
                         "groups": ["icon-build"], "only_groups": True, "explicit": True}]
    calls = []

    def prepare(source, out, cache, *, explicit=True):
        calls.append((source, out, cache, explicit))
        return python

    launched = []

    def run(command, *, cwd):
        launched.append(command)
        assert calls and cwd == source
        return SimpleNamespace(returncode=9)

    monkeypatch.setattr(icon_environment, "prepare_icon_environment", prepare)
    monkeypatch.setattr(icon_environment.subprocess, "run", run)
    args = ["--source", str(source), "--out", str(tmp_path / "icons")]
    if on_demand:
        args.append("--on-demand")
    assert icon_environment.main(args) == 9
    assert calls[0][0] == source and calls[0][2] == source / ".cache/icon-build"
    assert calls[0][3] is not on_demand
    assert launched[0][:2] == [str(python), "-I"]
    assert "--on-demand" not in launched[0]
