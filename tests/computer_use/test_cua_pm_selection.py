"""CUA selection uses PM facts, never a second installer or stale vendor tree."""

import sys
from pathlib import Path

import pytest


@pytest.fixture
def cua_home(tmp_path, monkeypatch):
    from pm import paths

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    monkeypatch.setenv("HERMES_RUNTIME_DIR", str(tmp_path / "tools"))
    monkeypatch.delenv("HERMES_CUA_DRIVER_CMD", raising=False)
    monkeypatch.setenv("HERMES_DISABLE_LAZY_INSTALLS", "1")
    monkeypatch.setattr(paths, "lockfile_path", lambda: tmp_path / "lock.json")
    return tmp_path


def _record_driver(version="0.20.0"):
    from pm import Facts, Lockfile, current_target, get_package, paths
    from pm.store import tree_digest

    package = get_package("cua-driver")
    target = current_target()
    root = paths.store_root()
    entry = root / package.store_entry(version, target)
    entry.mkdir(parents=True)
    binary = package.binary(entry, target)
    assert binary is not None
    binary.parent.mkdir(parents=True, exist_ok=True)
    binary.write_bytes(Path(sys.executable).read_bytes())
    binary.chmod(0o755)
    artifact = {"url": "https://example.invalid/cua-fixture", "sha256": "a" * 64}
    lock = Lockfile(paths.lockfile_path())
    lock.set_pin(package.name, version, {target: artifact})
    lock.save()
    Facts(paths.facts_path()).record(
        package.name, version, entry.name, package.env(entry, target), root,
        target=target, artifacts=[artifact["sha256"]], digest=tree_digest(entry),
    )
    return binary


def test_pm_selection_ignores_vendor_tree_and_is_passive(cua_home, monkeypatch):
    import pm
    from pm import paths
    from tools.computer_use.cua_backend_driver import resolve_cua_driver_cmd

    def no_install(*args, **kwargs):
        pytest.fail("passive CUA lookup attempted acquisition")

    monkeypatch.setattr(pm, "ensure", no_install)
    binary = _record_driver()
    legacy = cua_home / ".local" / "bin" / binary.name
    legacy.parent.mkdir(parents=True)
    legacy.write_bytes(binary.read_bytes())
    legacy.chmod(0o755)
    monkeypatch.setenv("PATH", str(legacy.parent))
    before = paths.facts_path().read_bytes()

    assert resolve_cua_driver_cmd() == str(binary)
    assert resolve_cua_driver_cmd(str(legacy)) == str(legacy)
    assert resolve_cua_driver_cmd(str(cua_home / "missing")) is None
    assert paths.facts_path().read_bytes() == before
    lock = pm.Lockfile(paths.lockfile_path())
    lock.set_pin("cua-driver", "0.21.0", lock.pinned_artifacts("cua-driver"))
    lock.save()
    assert resolve_cua_driver_cmd() is None
    previous = pm.installed_package("cua-driver", allow_outdated=True)
    assert previous is not None and previous.binary == binary
    binary.unlink()
    assert resolve_cua_driver_cmd() is None


@pytest.mark.platforms("linux")
def test_setup_acquires_through_pm_and_validates_real_manifest(cua_home, monkeypatch):
    import pm
    from hermes_cli.tools_config_cua import install_cua_driver
    from tools.computer_use.cua_backend_driver import (
        _CUA_DRIVER_RUNTIME_CONTRACT_ARGS, cua_driver_runtime_contract_status,
    )

    monkeypatch.setenv("PATH", "")
    calls = []

    def acquire(name, *, explicit):
        calls.append((name, explicit))
        binary = _record_driver()
        # A local process fixture, not an upstream CUA installation.
        manifest = {
            "binary_version": "0.20.0",
            "mcp_invocation": {"command": str(binary), "args": ["mcp"]},
            "subcommands": [
                {"name": verb, "args": [{"name": arg} for arg in sorted(args)]}
                for verb, args in _CUA_DRIVER_RUNTIME_CONTRACT_ARGS.items()
            ],
        }
        binary.write_text(
            f"#!{sys.executable}\nimport json, sys\n"
            f"manifest = {manifest!r}\n"
            "assert sys.argv[1:] == ['manifest']\n"
            "print(json.dumps(manifest))\n",
            encoding="utf-8",
        )
        return pm.Runner(name, pm.env_for(name))

    monkeypatch.setattr(pm, "ensure", acquire)
    assert install_cua_driver(show_installer_progress=False)
    assert calls == [("cua-driver", True)]
    state = cua_driver_runtime_contract_status()
    assert state["ready"], state
    installed = pm.installed_package("cua-driver")
    assert installed is not None
    assert state["binary"] == str(installed.binary)


@pytest.mark.platforms("linux")
@pytest.mark.parametrize("permission_mode", ["bounded", "unrestricted"])
@pytest.mark.parametrize("preinstalled", [False, True], ids=["cold", "previous-selection"])
def test_private_start_uses_post_ensure_binary(cua_home, monkeypatch, permission_mode, preinstalled):
    """Real PM lookup and manifest processes; only daemon/MCP serving is stubbed."""
    import subprocess
    from types import SimpleNamespace

    import pm
    from tools.computer_use import cua_backend as backend_module
    from tools.computer_use.cua_backend_driver import _CUA_DRIVER_RUNTIME_CONTRACT_ARGS

    monkeypatch.setenv("PATH", "")
    manifest_path = cua_home / "capabilities.yaml"
    manifest_path.write_text("version: 3\n", encoding="utf-8")
    monkeypatch.setattr(backend_module, "_computer_use_cfg", lambda: {
        "capability_manifest": str(manifest_path), "no_overlay": False,
    })
    if preinstalled:
        previous = _record_driver()
        previous.write_text(f"#!{sys.executable}\nprint('{{}}')\n", encoding="utf-8")
    backend = backend_module.CuaDriverBackend(permission_mode=permission_mode)
    assert backend._embedded_daemon is not None
    versions = iter(["0.20.1", "0.20.2"])
    acquired = []

    def acquire(name, **kwargs):
        assert name == "cua-driver" and not kwargs.get("explicit", False)
        version = next(versions)
        binary = _record_driver(version)
        manifest = {
            "binary_version": version,
            "mcp_invocation": {"command": str(binary), "args": ["mcp"]},
            "subcommands": [
                {"name": verb, "args": [{"name": arg} for arg in sorted(args)]}
                for verb, args in _CUA_DRIVER_RUNTIME_CONTRACT_ARGS.items()
            ],
        }
        binary.write_text(
            f"#!{sys.executable}\nimport json, sys\n"
            f"manifest = {manifest!r}\n"
            "if sys.argv[1:] == ['manifest']:\n    print(json.dumps(manifest))\n"
            "else:\n    assert sys.argv[1] in ('status', 'stop')\n",
            encoding="utf-8",
        )
        acquired.append(str(binary))
        return pm.Runner(name, pm.env_for(name))

    monkeypatch.setattr(pm, "ensure", acquire)
    monkeypatch.setattr(pm, "ensure_import", lambda name: None)
    monkeypatch.setattr(backend._session, "start", lambda: None)
    monkeypatch.setattr(backend._session, "call_tool", lambda *args: None)
    spawn = subprocess.Popen
    launches = []

    def capture_serve(command, **kwargs):
        if command[1] == "serve":
            launches.append(command)
            return SimpleNamespace(stderr=(), poll=lambda: None, wait=lambda **kw: 0)
        return spawn(command, **kwargs)

    monkeypatch.setattr(subprocess, "Popen", capture_serve)
    # Re-start the same backend after PM publishes another version as well.
    for _ in range(2):
        try:
            backend.start()
            command = launches[-1]
            assert command[0] == acquired[-1]
            assert command[command.index("--permission-mode") + 1] == permission_mode
            assert command[command.index("--capability-manifest") + 1] == str(manifest_path)
            assert "--approve-capability-manifest" in command
            assert ("--dangerously-bypass-approvals" in command) == (permission_mode == "unrestricted")
            proxy, args = backend._embedded_daemon.proxy_invocation()
            assert proxy == acquired[-1]
            assert args[args.index("--socket") + 1] == command[command.index("--socket") + 1]
        finally:
            backend.stop()


def test_runtime_cannot_bypass_pm_lazy_install_refusal(cua_home, monkeypatch):
    from pm import InstallError
    from tools.computer_use.cua_backend import CuaDriverBackend

    monkeypatch.setenv("PATH", "")
    backend = CuaDriverBackend()
    with pytest.raises(InstallError, match="lazy installs are disabled"):
        backend.start()