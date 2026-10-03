"""Backend selection uses reviewed target pins and never installs on lookup."""

import pytest

import pm
from hermes_cli.local_runtime import binaries
from pm import paths
from pm.lock import Lockfile


@pytest.mark.parametrize("target,vendor,expected", [
    ("linux-x64", "nvidia", "cuda"),
    ("linux-arm64", "nvidia", "vulkan"),
    ("win32-arm64", "nvidia", "cuda"),
    ("win32-arm64", "AMD Radeon", "cpu"),
    ("win32-x64", "AMD Radeon", "vulkan"),
    ("win32-x64", None, "cpu"),
    ("darwin-arm64", None, "metal"),
])
def test_auto_backend_uses_only_compatible_pins(target, vendor, expected):
    assert binaries.resolve_backend("auto", gpu_vendor=vendor, target=target) == expected
    with pytest.raises(binaries.BinaryResolutionError):
        binaries.resolve_backend("cuda", target="linux-arm64")


def test_linux_cuda_runtime_libraries_land_beside_the_server(tmp_path):
    """The engine's RUNPATH is $ORIGIN and its archive carries no CUDA libraries,
    so the cudart tarball's own top-level dir must be hoisted next to llama-server."""
    package = pm.get_package("llamacpp-cuda")
    staged = tmp_path / "tree"
    (staged / "llama-b1").mkdir(parents=True)
    (staged / "llama-b1" / "llama-server").write_bytes(b"engine")
    (staged / "cudart-llama-b1-bin-ubuntu-cuda-13.4-x64").mkdir()
    (staged / "cudart-llama-b1-bin-ubuntu-cuda-13.4-x64" / "libcudart.so.13").write_bytes(b"rt")

    package.stage(None, staged, "1", "linux-x64")

    assert package.binary(staged, "linux-x64").is_file()
    assert (staged / "libcudart.so.13").is_file()
    assert not list(staged.glob("cudart-*"))


def test_missing_pin_is_refused_instead_of_constructing_download_url(tmp_path, monkeypatch):
    lock_path = tmp_path / "lock.json"
    monkeypatch.setattr(paths, "lockfile_path", lambda: lock_path)
    lock = Lockfile(lock_path)
    lock.set_pin("llamacpp-cpu", "123", {})
    lock.save()
    assert binaries.pinned_tag("cpu") == "b123"
    with pytest.raises(binaries.BinaryResolutionError, match="not pinned"):
        binaries.resolve_backend("cpu", target=pm.current_target())
