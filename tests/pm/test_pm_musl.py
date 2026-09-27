"""Regression contracts for native musl Linux PM targets (#123682)."""

from __future__ import annotations

import sysconfig
from argparse import Namespace
from pathlib import Path

import pytest

from pm.lock import Lockfile
from pm.registry import get_package
from pm.store import ALL_TARGETS

pytestmark = pytest.mark.platforms("linux")

REPO_ROOT = Path(__file__).resolve().parents[2]
_MUSL_TARGETS = ("linux-x64-musl", "linux-arm64-musl")


def test_all_targets_include_native_musl_lanes():
    for target in _MUSL_TARGETS:
        assert target in ALL_TARGETS
        assert ALL_TARGETS.count(target) == 1


def test_current_target_detects_musl_from_python_build_metadata(monkeypatch):
    from pm import store

    monkeypatch.setattr(store, "_native_machine", lambda: "x86_64")
    monkeypatch.setattr(store, "_is_bionic_libc", lambda: False)
    monkeypatch.setattr(store, "_native_linux_uses_musl", lambda: None)
    monkeypatch.setattr(
        sysconfig,
        "get_config_var",
        lambda key: "x86_64-alpine-linux-musl" if key == "HOST_GNU_TYPE" else None,
    )

    assert store.current_target() == "linux-x64-musl"


def test_elf_loader_distinguishes_native_musl_from_glibc(tmp_path):
    from pm import store

    musl = tmp_path / "musl-bin"
    musl.write_bytes(b"\x7fELF" + b"\0" * 64 + b"/lib/ld-musl-x86_64.so.1\0")
    glibc = tmp_path / "glibc-bin"
    glibc.write_bytes(b"\x7fELF" + b"\0" * 64 + b"/lib64/ld-linux-x86-64.so.2\0")

    assert store._elf_loader_is_musl(musl) is True
    assert store._elf_loader_is_musl(glibc) is False


def test_musl_detection_falls_back_to_native_userland(monkeypatch):
    from pm import store

    monkeypatch.setattr(sysconfig, "get_config_var", lambda _key: "x86_64-unknown-linux-gnu")
    monkeypatch.setattr(store, "_native_linux_uses_musl", lambda: True)

    assert store._is_musl_libc() is True


def test_native_glibc_overrides_foreign_musl_bootstrap(monkeypatch):
    from pm import store

    monkeypatch.setattr(store, "_native_linux_uses_musl", lambda: False)
    monkeypatch.setattr(sysconfig, "get_config_var", lambda _key: "x86_64-unknown-linux-musl")

    assert store._is_musl_libc() is False


def test_bionic_remains_more_specific_than_musl(monkeypatch):
    from pm import store

    monkeypatch.setattr(store, "_native_machine", lambda: "aarch64")
    monkeypatch.setattr(store, "_is_bionic_libc", lambda: True)
    monkeypatch.setattr(store, "_is_musl_libc", lambda: True)

    assert store.current_target() == "linux-arm64-bionic"


@pytest.mark.parametrize("target", _MUSL_TARGETS)
def test_required_runtime_has_native_musl_pins(target):
    """Every runtime tool PM selects by default has an artifact for musl."""
    lock = Lockfile(REPO_ROOT / "pm" / "lock.json")

    for name in ("python", "uv", "node", "npm", "ripgrep"):
        assert get_package(name).missing_reason(target) is None, name
        assert lock.artifacts(name, target), f"{name} has no {target} artifact"


@pytest.mark.parametrize("target", _MUSL_TARGETS)
def test_interpreter_and_js_runtime_never_reuse_glibc_rows(target):
    lock = Lockfile(REPO_ROOT / "pm" / "lock.json")

    for name in ("python", "uv", "node"):
        row = lock.artifacts(name, target)[0]
        assert "musl" in row["url"], f"{name} {target} fell back to {row['url']}"
        assert get_package(name).fetch_url(lock.version(name), target) == row["url"]


@pytest.mark.parametrize(
    "target,base",
    [("linux-x64-musl", "linux-x64"), ("linux-arm64-musl", "linux-arm64")],
)
def test_static_linux_tools_reuse_the_reviewed_bytes(target, base):
    lock = Lockfile(REPO_ROOT / "pm" / "lock.json")

    for name in ("ripgrep", "gh", "bws", "iron-proxy"):
        assert lock.artifacts(name, target) == lock.artifacts(name, base)


@pytest.mark.parametrize("target", _MUSL_TARGETS)
def test_musl_default_closure_excludes_incompatible_ffmpeg(monkeypatch, target):
    from pm import store
    from pm.package import InstallError
    from pm.registry import source_install_packages, tool_roots

    lock = Lockfile(REPO_ROOT / "pm" / "lock.json")
    monkeypatch.setattr(store, "current_target", lambda: target)
    assert get_package("ffmpeg").missing_reason(target)
    assert lock.artifacts("ffmpeg", target) == []
    assert "ffmpeg" not in source_install_packages(lock.names())
    assert "ffmpeg" not in tool_roots(lock.names())
    with pytest.raises(InstallError, match="unavailable on"):
        get_package("ffmpeg").fetch_url(lock.version("ffmpeg"), target)


@pytest.mark.parametrize("target", _MUSL_TARGETS)
def test_optional_incompatible_tools_are_explicit_gaps(target):
    for name in ("git", "cua-driver", "agent-browser", "chromium"):
        assert get_package(name).missing_reason(target), name


def test_portable_go_tools_resolve_the_generic_linux_archives():
    lock = Lockfile(REPO_ROOT / "pm" / "lock.json")

    for name in ("gh", "iron-proxy"):
        package = get_package(name)
        for target, base in (("linux-x64-musl", "linux-x64"), ("linux-arm64-musl", "linux-arm64")):
            assert package.missing_reason(target) is None
            assert package.fetch_url(lock.version(name), target) == lock.artifacts(name, base)[0]["url"]


def test_security_tools_only_claim_musl_where_upstream_has_an_asset():
    lock = Lockfile(REPO_ROOT / "pm" / "lock.json")

    for target, base in (("linux-x64-musl", "linux-x64"), ("linux-arm64-musl", "linux-arm64")):
        assert lock.artifacts("bws", target) == lock.artifacts("bws", base)

    tirith = get_package("tirith")
    assert tirith.missing_reason("linux-x64-musl")
    assert tirith.missing_reason("linux-arm64-musl") is None
    arm = lock.artifacts("tirith", "linux-arm64-musl")
    assert arm and "aarch64-unknown-linux-musl" in arm[0]["url"]


def test_node_musl_rows_use_the_dedicated_build_channel():
    lock = Lockfile(REPO_ROOT / "pm" / "lock.json")

    for target in _MUSL_TARGETS:
        assert lock.artifacts("node", target)[0]["url"].startswith(
            "https://unofficial-builds.nodejs.org/download/release/"
        )


def test_lock_hashes_a_shared_target_url_once(monkeypatch):
    from pm import cli

    shared = "https://example.invalid/static-linux.tar.gz"
    other = "https://example.invalid/darwin.tar.gz"

    class Package:
        name = "portable"

        def missing_reason(self, _target):
            return None

        def fetch_urls(self, _version, target):
            return [other if target == "darwin-arm64" else shared]

        def known_sha256(self, _version, _url):
            return None

    class FakeLock:
        def __init__(self):
            self.pin = None
            self.saved = False

        def set_pin(self, name, version, artifacts):
            self.pin = (name, version, artifacts)

        def save(self):
            self.saved = True

    lock = FakeLock()
    hashed = []
    monkeypatch.setattr(
        cli, "ALL_TARGETS",
        ("linux-x64", "linux-x64-musl", "darwin-arm64"),
    )
    monkeypatch.setattr(cli, "get_package", lambda _name: Package())
    monkeypatch.setattr(cli, "_lockfile", lambda: lock)
    monkeypatch.setattr(
        cli, "hash_url",
        lambda url: hashed.append(url) or ("a" * 64 if url == shared else "b" * 64),
    )

    assert cli._pin_tool(Namespace(name="portable", version="1.0")) == 0
    assert hashed.count(shared) == 1
    assert hashed.count(other) == 1
    assert lock.saved is True
    assert lock.pin[2]["linux-x64"] == lock.pin[2]["linux-x64-musl"]
