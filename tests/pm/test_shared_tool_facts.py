"""Regression for NousResearch/hermes-agent#149497: shared tool-store pins stay usable."""

from pm import cli, install, lock as pm_lock, paths
from pm._uv import _toolchain
from pm.lock import Facts, Lockfile
from pm.registry import get_package
from pm.store import Store, current_target


def _publish(root, name, version, sha, target):
    """A real store entry with an executable binary, recorded like an install does."""
    package = get_package(name)
    entry = root / package.store_entry(version, target)
    binary = package.binary(entry, target)
    assert binary is not None
    binary.parent.mkdir(parents=True, exist_ok=True)
    binary.write_bytes(b"fixture")
    binary.chmod(0o755)
    Facts(root / "facts.json").record(name, version, entry.name, package.env(entry, target), root,
                                      target=target, artifacts=[sha], digest="f" * 64)
    return entry, binary


def _pin(path, name, version, sha, target):
    lock = Lockfile(path)
    lock.set_pin(name, version, {target: {"url": "https://invalid.test/tool", "sha256": sha}})
    return lock


def test_two_installs_keep_independent_tool_pins_and_gc_roots(tmp_path, monkeypatch):
    root = tmp_path / "tools"
    root.mkdir()
    monkeypatch.setattr(paths, "store_root", lambda: root)
    monkeypatch.setattr(paths, "writable_store_root", lambda: root)
    target = current_target()
    locks, binaries, entries = {}, {}, set()

    # A, then B, then A record: a gateway checkout and a worktree re-running setup in turn.
    for owner, version in (("A", "1.0"), ("B", "2.0"), ("A", "1.0")):
        monkeypatch.setattr(pm_lock, "install_owner", lambda owner=owner: owner)
        sha = ("1" if version == "1.0" else "2") * 64
        lock = Lockfile(tmp_path / f"install-{owner}.json")
        for name in ("uv", "python"):
            lock.set_pin(name, version, {target: {"url": "https://invalid.test/tool", "sha256": sha}})
            entry, binaries[(version, name)] = _publish(root, name, version, sha, target)
            entries.add(entry.name)
        locks[owner] = lock

    for owner, version in (("A", "1.0"), ("B", "2.0")):
        lock = locks[owner]
        monkeypatch.setattr(install, "_lockfile", lambda lock=lock: lock)
        toolchain = _toolchain(realize=False)
        assert toolchain is not None
        assert tuple(path.resolve() for path in toolchain) == tuple(
            binaries[(version, name)].resolve() for name in ("uv", "python"))

    # Both checkouts' dependency state still exists, so gc keeps both installs' entries.
    for owner in ("A", "B"):
        (tmp_path / "installs" / owner).mkdir(parents=True)
    assert cli._gc_store(Store(root), Facts(root / "facts.json")) == (0, len(entries))

    # An identical version with a different artifact hash cannot borrow either fact,
    # and an explicit outdated read returns the store's selection, not another install's pin.
    repinned = _pin(tmp_path / "repinned.json", "uv", "1.0", "a" * 64, target)
    assert install._installed_location(get_package("uv"), repinned, target) is None
    fallback = install._installed_location(get_package("uv"), repinned, target, allow_outdated=True)
    assert fallback is not None and fallback[0].get("uv")["version"] == "1.0"


def test_gc_reclaims_pins_no_live_checkout_holds(tmp_path, monkeypatch):
    root = tmp_path / "tools"
    root.mkdir()
    target = current_target()
    store = Store(root)
    for owner in ("A", "B"):
        (tmp_path / "installs" / owner).mkdir(parents=True)

    monkeypatch.setattr(pm_lock, "install_owner", lambda: "A")
    old, _ = _publish(root, "uv", "1.0", "1" * 64, target)
    new, _ = _publish(root, "uv", "1.1", "2" * 64, target)
    monkeypatch.setattr(pm_lock, "install_owner", lambda: "B")
    other, _ = _publish(root, "uv", "2.0", "3" * 64, target)
    solo, _ = _publish(root, "python", "3.0", "4" * 64, target)

    # A's own upgrade superseded 1.0: nothing pins it any more, so gc reclaims it.
    cli._gc_store(store, Facts(root / "facts.json"))
    assert not old.exists() and new.is_dir() and other.is_dir() and solo.is_dir()

    # Once B's checkout is gone, its pins no longer hold 2.0 (though B recorded last)
    # nor the python entry B alone recorded.
    (tmp_path / "installs" / "B").rmdir()
    cli._gc_store(store, Facts(root / "facts.json"))
    assert not other.exists() and not solo.exists() and new.is_dir()
    facts = Facts(root / "facts.json")
    assert facts.get("uv")["version"] == "1.1" and facts.get("python") is None
