"""Backend selection uses reviewed target pins and never installs on lookup."""

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


def test_custom_engine_serves_a_user_binary(tmp_path, monkeypatch):
    server = tmp_path / "llama-server.exe"
    server.write_bytes(b"fake")
    monkeypatch.setenv("HERMES_LLAMACPP_SERVER", str(server))
    engine = binaries.installed_engine("cuda")
    assert engine is not None
    assert (engine.backend, engine.tag, engine.binary) == ("custom", "user", server)


def test_custom_executable_config_value_resolves(tmp_path, monkeypatch):
    monkeypatch.delenv("HERMES_LLAMACPP_SERVER", raising=False)
    server = tmp_path / "llama-server"
    server.write_bytes(b"fake")
    engine = binaries.installed_engine("cuda", custom_executable=str(server))
    assert engine is not None
    assert (engine.backend, engine.tag, engine.binary) == ("custom", "user", server)


def test_custom_env_wins_over_config(tmp_path, monkeypatch):
    env_server = (tmp_path / "env" / "llama-server.exe")
    env_server.parent.mkdir()
    env_server.write_bytes(b"env")
    cfg_server = (tmp_path / "cfg" / "llama-server.exe")
    cfg_server.parent.mkdir()
    cfg_server.write_bytes(b"cfg")
    monkeypatch.setenv("HERMES_LLAMACPP_SERVER", str(env_server))
    engine = binaries.installed_engine("cuda", custom_executable=str(cfg_server))
    assert engine is not None
    assert engine.binary == env_server


def test_custom_executable_must_be_a_llama_server_binary(tmp_path, monkeypatch):
    monkeypatch.delenv("HERMES_LLAMACPP_SERVER", raising=False)
    not_server = tmp_path / "python.exe"
    not_server.write_bytes(b"fake")
    with pytest.raises(binaries.BinaryResolutionError, match="must be a llama-server"):
        binaries.installed_engine("cuda", custom_executable=str(not_server))


def test_missing_custom_executable_fails_loudly(tmp_path, monkeypatch):
    monkeypatch.delenv("HERMES_LLAMACPP_SERVER", raising=False)
    missing = tmp_path / "llama-server.exe"
    with pytest.raises(binaries.BinaryResolutionError, match="does not exist"):
        binaries.installed_engine("cuda", custom_executable=str(missing))
