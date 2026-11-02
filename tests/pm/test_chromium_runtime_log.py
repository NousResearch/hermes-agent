"""Regression for #140042: Chromium's Windows diagnostic log is not package payload."""

from pathlib import Path

import pytest

from pm.install import _entry_verified
from pm.lock import Facts, Lockfile
from pm.packages import Chromium
from pm.store import Store, current_target, tree_digest


def _installed(tmp_path, target):
    package, store = Chromium(), Store(tmp_path / "tools")
    version = "1208+145.0.7632.6"
    entry = store.entry(package.store_entry(version, target))
    binary = entry / "chrome-win64" / "chrome.exe"
    binary.parent.mkdir(parents=True)
    binary.write_bytes(b"fixture executable")
    (entry / "INSTALLATION_COMPLETE").touch()
    fact = dict(entry=entry.name, digest=tree_digest(entry))
    return package, store, entry, fact


@pytest.mark.parametrize("target", ["win32-x64", "win32-arm64"])
def test_installed_chromium_accepts_only_its_runtime_log(tmp_path, target):
    package, store, entry, fact = _installed(tmp_path, target)
    assert _entry_verified(package, fact, store, target)
    log = entry / "chrome-win64" / "debug.log"
    log.write_bytes(b"crashpad registration error\n")
    assert _entry_verified(package, fact, store, target)
    log.write_bytes(b"another runtime diagnostic\n")
    assert _entry_verified(package, fact, store, target)
    # A same-named log elsewhere, and actual executable damage, still fail.
    other = entry / "debug.log"
    other.write_bytes(b"not a declared runtime file")
    assert not _entry_verified(package, fact, store, target)
    other.unlink()
    (log.parent / "chrome.exe").write_bytes(b"modified executable")
    assert not _entry_verified(package, fact, store, target)


@pytest.mark.parametrize("consumer", ["copy", "build", "refresh"])
def test_runtime_digest_agrees_across_packaging_consumers(tmp_path, consumer):
    from pm.build_operations import verified_tools
    from pm.install import _copy_verified_source

    target = "win32-x64"
    package, store, entry, fact = _installed(tmp_path, target)
    lock = Lockfile(tmp_path / "lock.json")
    version = "1208+145.0.7632.6"
    lock.set_pin(package.name, version, {
        target: {"url": "https://example.invalid/chromium.zip", "sha256": "a" * 64},
    })
    facts = Facts(store.root / "facts.json")
    facts.record(package.name, version, entry.name, package.env(entry, target), store.root,
                 target=target, artifacts=["a" * 64], digest=fact["digest"])
    log = entry / "chrome-win64" / "debug.log"
    log.write_bytes(b"runtime diagnostic")
    if consumer == "copy":
        staged = tmp_path / "staged"
        _copy_verified_source(package, lock, (facts, store), staged, version, target)
        assert package.digest(staged, target) == fact["digest"]
    elif consumer == "build":
        from pm.filesystem import native

        tools = verified_tools([package.name], source_store=store.root, target=target, lock=lock)
        assert tools.entries[package.name].path == Path(native(entry))
    else:
        assert facts.refresh_digests(store.root) == 1
        assert facts.get(package.name)["digest"] == fact["digest"]


def test_runtime_exclusion_does_not_hide_other_packages_or_directories(tmp_path):
    from pm.package import Package

    package, store, entry, fact = _installed(tmp_path, "win32-x64")
    log = entry / "chrome-win64" / "debug.log"
    log.mkdir()
    (log / "payload").write_bytes(b"directory contents remain payload")
    assert not _entry_verified(package, fact, store, "win32-x64")
    (log / "payload").unlink()
    log.rmdir()
    log.write_bytes(b"a regular log")
    assert Package().digest(entry, "win32-x64") != fact["digest"]


@pytest.mark.platforms("posix")
def test_runtime_exclusion_still_binds_link_targets(tmp_path):
    package, store, entry, fact = _installed(tmp_path, "win32-x64")
    log = entry / "chrome-win64" / "debug.log"
    log.symlink_to(tmp_path / "outside")
    assert not _entry_verified(package, fact, store, "win32-x64")


@pytest.mark.platforms("windows")
def test_doctor_and_install_accept_a_locked_chromium_log(tmp_path, monkeypatch, capsys):
    import ctypes
    from ctypes import wintypes
    from pm import cli, registry

    target = current_target()
    package, store, entry, fact = _installed(tmp_path, target)
    lock = Lockfile(tmp_path / "lock.json")
    lock.set_pin(package.name, "1208+145.0.7632.6", {
        target: {"url": "https://example.invalid/chromium.zip", "sha256": "a" * 64},
    })
    facts = Facts(store.root / "facts.json")
    facts.record(package.name, lock.version(package.name), entry.name, package.env(entry, target),
                 store.root, target=target, artifacts=["a" * 64], digest=fact["digest"])
    monkeypatch.setattr(cli, "_lockfile", lambda: lock)
    monkeypatch.setattr(cli, "_facts", lambda: facts)
    monkeypatch.setattr(cli, "_store", lambda: store)
    monkeypatch.setattr("pm.install._installed_location", lambda *args: (facts, store))
    monkeypatch.setitem(registry._packages, package.name, package)
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
                                  wintypes.LPVOID, wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
    kernel.CreateFileW.restype = wintypes.HANDLE
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    log = entry / "chrome-win64" / "debug.log"
    log.write_bytes(b"crashpad registration error\n")
    handle = kernel.CreateFileW(str(log), 0x40000000, 3, None, 3, 0x80, None)
    assert handle != wintypes.HANDLE(-1).value, ctypes.get_last_error()
    try:
        with pytest.raises(PermissionError):
            log.unlink()
        before = facts.path.read_bytes()
        assert cli.cmd_doctor(None) == 0
        assert _entry_verified(package, facts.get(package.name), store, target)
        assert facts.path.read_bytes() == before
        assert log.read_bytes() == b"crashpad registration error\n"
        (log.parent / "chrome.exe").write_bytes(b"modified executable")
        assert cli.cmd_doctor(None) == 1
        assert "realized bytes do not match recorded digest" in capsys.readouterr().out
    finally:
        kernel.CloseHandle(handle)
