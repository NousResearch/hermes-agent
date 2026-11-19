"""A sealed Linux x64 payload can carry engines that the local loader cannot run."""

import json
import os
from pathlib import Path

import pytest

import pm
from hermes_cli.local_runtime import binaries, bootstrap
from pm import cli, install, paths, store
from pm.lock import Facts, Lockfile

pytestmark = pytest.mark.platforms("linux")


@pytest.fixture
def payload(tmp_path, monkeypatch):
    root = tmp_path / "payload" / "tools"
    root.mkdir(parents=True)
    (root.parent / "manifest.json").write_text("{}")
    monkeypatch.setenv("HERMES_RUNTIME_DIR", str(root))
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr(store, "_native_machine", lambda: "x86_64")
    if pm.current_target() != "linux-x64":
        pytest.skip("fixture requires a glibc Linux userland")
    monkeypatch.setattr(bootstrap, "_detect_gpu_vendor", lambda: "nvidia")
    lock_path = tmp_path / "lock.json"
    monkeypatch.setattr(paths, "lockfile_path", lambda: lock_path)
    lock = Lockfile(lock_path)
    facts = Facts(root / "facts.json")
    for name in ("llamacpp-cpu", "llamacpp-cuda", "llamacpp-hip", "gh"):
        package = pm.get_package(name)
        entry = root / package.store_entry("123", "linux-x64")
        exe = package.binary(entry, "linux-x64")
        exe.parent.mkdir(parents=True)
        rc = 23 if name in ("llamacpp-cuda", "llamacpp-hip") else 0
        exe.write_text(f'#!/bin/sh\nprintf "%s\\n" {name} >> "$PROBE_LOG"\nexit {rc}\n')
        exe.chmod(0o755)
        lock.set_pin(name, "123", {"linux-x64": {"url": "file:///fixture", "sha256": "a" * 64}})
        facts.record(name, "123", entry.name, package.env(entry, "linux-x64"), root,
                     target="linux-x64", artifacts=["a" * 64], digest=store.tree_digest(entry))
    lock.save()
    log = tmp_path / "probes"
    monkeypatch.setenv("PROBE_LOG", str(log))
    return root, log


def _glibc(monkeypatch, version):
    confstr = os.confstr
    monkeypatch.setattr(os, "confstr", lambda key: f"glibc {version}" if key == "CS_GNU_LIBC_VERSION" else confstr(key))


@pytest.mark.parametrize("version,can_run", [("2.35", False), ("2.38", True)])
def test_doctor_checks_staged_engines_without_masking_probe_errors(payload, monkeypatch, capsys, version, can_run):
    _, log = payload
    _glibc(monkeypatch, version)
    assert cli.cmd_doctor(None) == int(can_run)
    ran = set(log.read_text().splitlines())
    assert ran == ({"gh", "llamacpp-cpu", "llamacpp-cuda", "llamacpp-hip"} if can_run else {"gh", "llamacpp-cpu"})
    output = capsys.readouterr().out
    if can_run:
        assert "exited 23" in output
    else:
        assert "staged bytes verified" in output and "2.38" in output and "2.35" in output


def _replace_arch(packages, row, entry):
    (entry / "llama-server").write_bytes(b"\x7fELF" + bytes(14) + b"\xb7\0" + bytes(44))
    row["digest"] = store.tree_digest(entry)


@pytest.mark.parametrize("damage", [
    pytest.param(lambda _, row, entry: (entry / "libggml.so").write_bytes(b"changed"), id="bytes"),
    pytest.param(lambda _, row, entry: row.update(artifacts=["b" * 64]), id="identity"),
    pytest.param(lambda _, row, entry: row.pop("digest"), id="digest"),
    pytest.param(lambda packages, row, entry: packages.pop("llamacpp-hip"), id="fact"),
    pytest.param(_replace_arch, id="architecture"),
])
def test_doctor_rejects_damaged_ineligible_engine(payload, monkeypatch, damage):
    root, log = payload
    _glibc(monkeypatch, "2.35")
    path = root / "facts.json"
    facts = json.loads(path.read_text())
    row = facts["packages"]["llamacpp-hip"]
    damage(facts["packages"], row, root / row["entry"])
    path.write_text(json.dumps(facts))
    assert cli.cmd_doctor(None) == 1
    assert set(log.read_text().splitlines()) == {"gh", "llamacpp-cpu"}


@pytest.mark.parametrize("backend", ["cuda", "hip"])
def test_explicit_engine_refuses_old_abi_even_on_cache_hit(payload, monkeypatch, backend):
    _, log = payload
    _glibc(monkeypatch, "2.35")
    name = binaries.BACKEND_PACKAGES[backend]
    with pytest.raises(pm.InstallError, match="glibc.*2.38"):
        pm.ensure(name)
    assert binaries.installed_engine(backend) is None
    with pytest.raises(binaries.BinaryResolutionError, match="glibc.*2.38"):
        binaries.resolve_backend(backend)
    with pytest.raises(pm.InstallError, match="glibc.*2.38"):
        install.ensure(name, explicit=True, verify=False)
    assert not log.exists()
    _glibc(monkeypatch, "2.38")
    assert binaries.resolve_backend(backend) == backend
    engine = binaries.installed_engine(backend)
    assert engine is not None and engine.backend == backend
    assert binaries.resolve_backend() == "cuda"


def test_native_prepare_refuses_gpu_even_when_its_cached_bytes_match(payload, monkeypatch):
    from pm.build_operations import prepare_tools

    root, log = payload
    _glibc(monkeypatch, "2.35")
    package = pm.get_package("llamacpp-cuda")
    entry = root / package.store_entry("123", "linux-x64")
    assert "glibc >= 2.38" in package.verify(entry, "linux-x64")
    with pytest.raises(pm.InstallError, match="glibc.*2.38"):
        prepare_tools([package.name], out=root, target="linux-x64")
    assert not log.exists()


def test_offline_boot_selects_cpu_and_activation_does_not_probe(payload, monkeypatch):
    root, log = payload
    _glibc(monkeypatch, "2.35")
    assert install.activate() == []
    assert all("llamacpp" not in item for item in os.environ["PATH"].split(os.pathsep) if item.startswith(str(root)))
    engine = binaries.installed_engine()
    assert engine is not None and engine.backend == "cpu"
    assert binaries.resolve_backend() == "cpu"
    assert not log.exists()
