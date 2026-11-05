"""The terminal runs the user's own tools first and Hermes's after them (pm.user_facing)."""
from __future__ import annotations

import os

import pm.install
import pm.user_facing
from pm.user_facing import user_facing_path


def test_user_tools_come_before_hermes_and_python_stops_at_the_store_stub(tmp_path, monkeypatch):
    npm, node, python, rg = (str(tmp_path / "store" / name) for name in ("npm", "node", "python", "rg"))
    stub = tmp_path / "AppData" / "Local" / "Microsoft" / "WindowsApps"
    stub.mkdir(parents=True)
    monkeypatch.setattr(pm.install, "_store_path_dirs", lambda: [npm, node, python, rg])
    monkeypatch.setattr(pm.user_facing, "_python_dirs", lambda: [python])
    path = os.pathsep.join([npm, node, python, rg, "/usr/bin", str(stub), "/home/u/bin"])

    # The stub alias would open the Store, so a machine without Python still runs ours.
    assert user_facing_path(path).split(os.pathsep) == [
        "/usr/bin", python, str(stub), "/home/u/bin", npm, node, rg]

    # A real Store Python ships python3.N.exe beside its alias: then the user's Python wins.
    (stub / "python3.13.exe").write_bytes(b"")
    assert user_facing_path(path).split(os.pathsep) == [
        "/usr/bin", str(stub), "/home/u/bin", python, npm, node, rg]

    monkeypatch.setattr(pm.install, "_store_path_dirs", list)
    assert user_facing_path(path) == path


def test_a_sealed_windows_payload_swaps_packaged_tools_for_writable_copies(tmp_path, monkeypatch):
    """cmd.exe cannot start an exe inside WindowsApps; a verified copy outside it can."""
    from pm import paths, registry
    from pm.lock import Facts, Lockfile
    from pm.packages import Nodejs, Python
    from pm.store import current_target, tree_digest
    from pm.user_facing import copies

    shipped, writable = tmp_path / "payload" / "tools", tmp_path / "home" / "tools"
    target, digest = current_target(), "1" * 64
    lock, facts = Lockfile(tmp_path / "lock.json"), Facts(shipped / "facts.json")
    bins = {}
    for package in (Nodejs(), Python()):
        package.verify = lambda entry, target: ""
        monkeypatch.setitem(registry._packages, package.name, package)
        lock.set_pin(package.name, "test", {target: {"url": "https://test.invalid", "sha256": digest}})
        entry = shipped / package.store_entry("test", target)
        binary = package.binary(entry, target)
        binary.parent.mkdir(parents=True)
        binary.write_bytes(b"pinned " + package.name.encode())
        facts.record(package.name, "test", entry.name, package.env(entry, target), shipped,
                     target=target, artifacts=[digest], digest=tree_digest(entry))
        bins[package.name] = {root: str(package.binary(root / entry.name, target).parent)
                              for root in (shipped, writable)}
    lock.save()
    monkeypatch.setattr(paths, "lockfile_path", lambda: lock.path)
    monkeypatch.setattr(paths, "store_root", lambda: shipped)
    monkeypatch.setattr(paths, "writable_store_root", lambda: writable)
    monkeypatch.setattr(pm.user_facing, "applies", lambda: True)
    path = os.pathsep.join([bins["python"][shipped], bins["node"][shipped], "/usr/bin"])

    # Lazy installs off: PM makes no copy, so the packaged dirs stay as the fallback.
    monkeypatch.setattr(pm.install, "lazy_installs_allowed", lambda: False)
    assert user_facing_path(path).split(os.pathsep) == [
        "/usr/bin", bins["python"][shipped], bins["node"][shipped]]
    assert not writable.exists()

    monkeypatch.setattr(pm.install, "lazy_installs_allowed", lambda: True)
    assert user_facing_path(path).split(os.pathsep) == [
        "/usr/bin", bins["python"][writable], bins["node"][writable]]
    assert copies([bins["node"][shipped]]) == [bins["node"][writable]]
