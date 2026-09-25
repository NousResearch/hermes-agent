"""Backend selection uses reviewed target pins and never installs on lookup."""

import pytest

import pm
from hermes_cli.local_runtime import binaries
from pm import paths
from pm.lock import Lockfile


@pytest.mark.parametrize("target,vendor,expected", [
    ("linux-x64", "nvidia", "vulkan"),
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
        binaries.resolve_backend("cuda", target="linux-x64")


def test_missing_pin_is_refused_instead_of_constructing_download_url(tmp_path, monkeypatch):
    lock_path = tmp_path / "lock.json"
    monkeypatch.setattr(paths, "lockfile_path", lambda: lock_path)
    lock = Lockfile(lock_path)
    lock.set_pin("llamacpp-cpu", "123", {})
    lock.save()
    assert binaries.pinned_tag("cpu") == "b123"
    with pytest.raises(binaries.BinaryResolutionError, match="not pinned"):
        binaries.resolve_backend("cpu", target=pm.current_target())
