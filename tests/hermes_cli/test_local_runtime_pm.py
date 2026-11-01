"""Backend selection uses reviewed target pins and never installs on lookup."""

from contextlib import nullcontext

import pytest

import pm
from hermes_cli.local_runtime import binaries
from pm import paths
from pm.lock import Lockfile


@pytest.mark.parametrize("target,vendor,expected", [
    ("linux-x64", "nvidia", "cuda"),
    ("linux-arm64", "nvidia", "cuda"),
    ("win32-arm64", "nvidia", "cuda"),
    ("win32-arm64", "AMD Radeon", "cpu"),
    ("win32-x64", "AMD Radeon", "vulkan"),
    ("win32-x64", None, "cpu"),
    ("darwin-arm64", None, "metal"),
])
def test_auto_backend_uses_only_compatible_pins(target, vendor, expected):
    assert binaries.resolve_backend("auto", gpu_vendor=vendor, target=target) == expected
    with pytest.raises(binaries.BinaryResolutionError):
        binaries.resolve_backend("cuda", target="darwin-arm64")


def test_linux_cuda_runtime_libraries_land_beside_the_server(tmp_path):
    """The engine's RUNPATH is $ORIGIN and its archive carries neither the CUDA
    runtime nor OpenMP, so the cudart tarball's top-level dir and the libgomp
    .deb's usr/lib/<triplet>/ library must both end up next to llama-server."""
    package = pm.get_package("llamacpp-cuda")
    staged = tmp_path / "tree"
    (staged / "llama-b1").mkdir(parents=True)
    (staged / "llama-b1" / "llama-server").write_bytes(b"engine")
    (staged / "cudart-llama-b1-bin-ubuntu-cuda-13.4-x64").mkdir()
    (staged / "cudart-llama-b1-bin-ubuntu-cuda-13.4-x64" / "libcudart.so.13").write_bytes(b"rt")
    gomp = staged / "usr" / "lib" / "x86_64-linux-gnu"
    gomp.mkdir(parents=True)
    (gomp / "libgomp.so.1.0.0").write_bytes(b"omp")
    (gomp / "libgomp.so.1").symlink_to("libgomp.so.1.0.0")

    package.stage(None, staged, "1", "linux-x64")

    assert package.binary(staged, "linux-x64").is_file()
    assert (staged / "libcudart.so.13").is_file()
    assert not list(staged.glob("cudart-*"))
    assert not (staged / "libgomp.so.1").is_symlink()
    assert (staged / "libgomp.so.1").read_bytes() == b"omp"
    assert not (staged / "usr").exists()


def test_missing_pin_is_refused_instead_of_constructing_download_url(tmp_path, monkeypatch):
    lock_path = tmp_path / "lock.json"
    monkeypatch.setattr(paths, "lockfile_path", lambda: lock_path)
    lock = Lockfile(lock_path)
    lock.set_pin("llamacpp-cpu", "123", {})
    lock.save()
    assert binaries.pinned_tag("cpu") == "b123"
    with pytest.raises(binaries.BinaryResolutionError, match="not pinned"):
        binaries.resolve_backend("cpu", target=pm.current_target())


def test_configured_binary_is_used_without_a_pm_install(tmp_path):
    """Regression for #140818: a user-built llama-server bypasses the PM engine."""
    binary = tmp_path / "llama-server"
    binary.write_text("#!/bin/sh\n", encoding="utf-8")
    binary.chmod(0o755)

    engine = binaries.configured_engine(str(binary))

    assert engine is not None
    assert engine.binary == binary
    assert engine.backend == "custom"


@pytest.mark.parametrize("path", ["relative/llama-server", "/missing/llama-server"])
def test_configured_binary_refuses_unusable_paths(path):
    with pytest.raises(binaries.BinaryResolutionError, match="binary"):
        binaries.configured_engine(path)


def test_boot_uses_configured_binary_before_the_pm_engine(tmp_path, monkeypatch):
    """Regression for #140818: configured forks retain managed lifecycle support."""
    from hermes_cli.local_runtime import bootstrap

    binary = tmp_path / "llama-server"
    binary.write_text("#!/bin/sh\n", encoding="utf-8")
    binary.chmod(0o755)
    captured = {}

    class FakeSupervisor:
        base_url = "http://127.0.0.1:8080/v1"

        def __init__(self, executable, *_args, **_kwargs):
            captured["binary"] = executable

        def start(self):
            pass

    monkeypatch.setattr(bootstrap, "_SUPERVISOR", None)
    monkeypatch.setattr(bootstrap, "adopt_legacy_models", lambda: None)
    monkeypatch.setattr(bootstrap, "_cross_process_boot_lock", nullcontext)
    monkeypatch.setattr("hermes_cli.local_runtime.endpoint._state_endpoint", lambda: None)
    monkeypatch.setattr("hermes_cli.local_runtime.bootstrap._generate_presets", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("hermes_cli.local_runtime.bootstrap._start_idle_sweeper", lambda *_args: None)
    monkeypatch.setattr("hermes_cli.local_runtime.bootstrap._stop_at_exit", lambda: None)
    monkeypatch.setattr("hermes_cli.local_runtime.binaries.installed_engine",
                        lambda *_args: pytest.fail("PM engine should not be queried"))
    monkeypatch.setattr("hermes_cli.local_runtime.supervisor.LlamaServerSupervisor", FakeSupervisor)

    supervisor = bootstrap.ensure_local_runtime({"local_runtime": {"enabled": True, "binary": str(binary)}}, force=True)

    assert supervisor is not None
    assert captured["binary"] == binary


def test_hardware_probe_uses_configured_binary_before_the_pm_engine(tmp_path, monkeypatch):
    from hermes_cli.local_runtime import hardware

    binary = tmp_path / "llama-server"
    binary.write_text("#!/bin/sh\n", encoding="utf-8")
    binary.chmod(0o755)
    monkeypatch.setattr("hermes_cli.config.load_config_readonly",
                        lambda: {"local_runtime": {"binary": str(binary)}})
    monkeypatch.setattr("hermes_cli.local_runtime.bootstrap.serving_engine", lambda: None)
    monkeypatch.setattr("hermes_cli.local_runtime.binaries.installed_engine",
                        lambda *_args: pytest.fail("PM engine should not be queried"))

    assert hardware._configured_engine().binary == binary
