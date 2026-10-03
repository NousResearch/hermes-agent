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


def test_nested_adoption_rolls_back_to_verified_source_on_facts_failure(tmp_path, monkeypatch):
    source = tmp_path / "legacy"
    nested = source / "llama-b123"
    binary = nested / "llama-server"
    binary.parent.mkdir(parents=True)
    binary.write_text("engine", encoding="utf-8")
    manifest_path = source / "manifest.json"
    manifest_path.write_text("{}", encoding="utf-8")
    root = tmp_path / "store"
    root.mkdir()

    class Package:
        name = "llamacpp-cpu"

        @staticmethod
        def store_entry(version, target):
            return "entry"

        @staticmethod
        def binary(directory, target):
            candidate = directory / "llama-server"
            return candidate if candidate.exists() else None

        @staticmethod
        def verify(directory, target):
            return None

        @staticmethod
        def env(entry, target):
            return {}

    class FailingFacts:
        def __init__(self, path):
            self.path = path

        def record(self, *args, **kwargs):
            raise RuntimeError("facts write failed")

    monkeypatch.setattr(binaries, "_legacy_artifacts", lambda *args: ["a" * 64])
    monkeypatch.setattr(pm, "Facts", FailingFacts)
    monkeypatch.setattr("pm.store.tree_digest", lambda entry: "digest")

    with pytest.raises(RuntimeError, match="facts write failed"):
        binaries._adopt(Package(), "123", "linux-x64", source, {"assets": {}}, root,
                        tmp_path / "facts.json")

    assert (source / "manifest.json").is_file()
    assert binary.is_file()
    assert not (root / "entry").exists()
