"""Regression for NousResearch/hermes-agent#149497: shared tool-store pins stay usable."""

import shutil

from pm import cli, environments, install, lock as pm_lock, paths
from pm._uv import _toolchain
from pm.build_operations import verified_tools
from pm.lock import Facts, Lockfile
from pm.registry import get_package
from pm.store import Store, current_target, tree_digest
from scripts.bundles.native import prune_staged_store


def _publish(root, name, version, sha, target, *, digest=False):
    """A real store entry with an executable binary, recorded like an install does."""
    package = get_package(name)
    entry = root / package.store_entry(version, target)
    if name == "chromium":  # its binary is found inside the entry, behind its completion marker
        exe = "chrome.exe" if target.startswith("win32") else "chrome"
        (entry / "chrome").mkdir(parents=True)
        (entry / "chrome" / exe).write_bytes(b"fixture")
        (entry / "INSTALLATION_COMPLETE").write_text("")
    binary = package.binary(entry, target)
    assert binary is not None
    binary.parent.mkdir(parents=True, exist_ok=True)
    binary.write_bytes(b"fixture")
    binary.chmod(0o755)
    if name == "python" and target.startswith("win32"):
        shutil.copy2(binary, entry / "python3.exe")
    Facts(root / "facts.json").record(name, version, entry.name, package.env(entry, target), root,
                                      target=target, artifacts=[sha],
                                      digest=tree_digest(entry) if digest else "f" * 64)
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
    # With the selected entry broken, the outdated read finds nothing, never B's 2.0 pin.
    binaries[("1.0", "uv")].unlink()
    assert install._installed_location(get_package("uv"), repinned, target, allow_outdated=True) is None


def test_gc_reclaims_pins_no_live_checkout_holds(tmp_path, monkeypatch):
    root = tmp_path / "tools"
    root.mkdir()
    target = current_target()
    store = Store(root)
    monkeypatch.setattr(environments, "installs_root", lambda: tmp_path / "installs")
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

    # Once B's checkout is gone (the orphan sweep removed its state), its pins no longer hold
    # 2.0 (though B recorded last) nor the python entry B alone recorded.
    shutil.rmtree(tmp_path / "installs" / "B")
    cli._gc_store(store, Facts(root / "facts.json"))
    assert not other.exists() and not solo.exists() and new.is_dir()
    facts = Facts(root / "facts.json")
    assert facts.get("uv")["version"] == "1.1" and facts.get("python") is None


def test_gc_keeps_pins_of_a_checkout_that_only_recorded_them(tmp_path, monkeypatch):
    """A checkout that never committed dependency state (a sealed or resident runtime that only
    ran ``ensure``) still owns the pin it recorded: another checkout's gc must not reclaim it."""
    root = tmp_path / "tools"
    root.mkdir()
    target = current_target()
    monkeypatch.setattr(environments, "installs_root", lambda: tmp_path / "installs")
    monkeypatch.setattr(pm_lock, "install_owner", lambda: "A")
    pinned, _ = _publish(root, "uv", "1.0", "1" * 64, target)
    monkeypatch.setattr(pm_lock, "install_owner", lambda: "B")
    _publish(root, "uv", "2.0", "2" * 64, target)

    cli._gc_store(Store(root), Facts(root / "facts.json"))
    assert pinned.is_dir()


def test_cache_hit_on_this_checkouts_pin_selects_it_for_builds(tmp_path, monkeypatch):
    """A, B, then A prepare one build store: A's cache hit on its own pin in the history must be
    what build admission and the staged-store prune read, not B's last selection."""
    root = tmp_path / "build-tools"
    root.mkdir()
    target = current_target()
    package = get_package("python")
    locks = {owner: _pin(tmp_path / f"{owner}.json", "python", version, sha * 64, target)
             for owner, version, sha in (("A", "1.0", "1"), ("B", "2.0", "2"))}
    for owner, version, sha in (("A", "1.0", "1"), ("B", "2.0", "2")):
        monkeypatch.setattr(pm_lock, "install_owner", lambda owner=owner: owner)
        _publish(root, "python", version, sha * 64, target, digest=True)

    monkeypatch.setattr(pm_lock, "install_owner", lambda: "A")
    entry = install._install(package, locks["A"], Facts(root / "facts.json"), Store(root), target)
    assert entry.name == package.store_entry("1.0", target)

    assert verified_tools(["python"], source_store=root, target=target,
                          lock=locks["A"]).entries["python"].version == "1.0"
    prune_staged_store(root, ["python"])
    assert entry.is_dir()


def test_cache_hit_on_another_checkouts_install_of_the_same_pin_claims_it(tmp_path, monkeypatch):
    """B waits on the install lock while A installs the pin both lock, then B's install hits A's
    entry: B must own that row, or A's later upgrade drops it and gc deletes B's tool."""
    root = tmp_path / "tools"
    root.mkdir()
    target = current_target()
    package = get_package("python")
    monkeypatch.setattr(paths, "store_root", lambda: root)
    monkeypatch.setattr(paths, "writable_store_root", lambda: root)
    monkeypatch.setattr(environments, "installs_root", lambda: tmp_path / "installs")
    lock = _pin(tmp_path / "shared.json", "python", "1.0", "1" * 64, target)

    monkeypatch.setattr(pm_lock, "install_owner", lambda: "A")
    entry, _ = _publish(root, "python", "1.0", "1" * 64, target, digest=True)
    monkeypatch.setattr(pm_lock, "install_owner", lambda: "B")
    assert install._install(package, lock, Facts(root / "facts.json"), Store(root), target) == entry

    monkeypatch.setattr(pm_lock, "install_owner", lambda: "A")
    _publish(root, "python", "2.0", "2" * 64, target, digest=True)
    cli._gc_store(Store(root), Facts(root / "facts.json"))
    assert entry.is_dir()


def test_claiming_a_found_pin_keeps_the_callers_view_on_its_own_pin(tmp_path, monkeypatch):
    """A reuses its pin while another checkout's newer pin is the store's selection: claiming the
    pin must not turn the view A runs from into the other checkout's selection."""
    root = tmp_path / "tools"
    root.mkdir()
    target = current_target()
    package = get_package("python")
    monkeypatch.setattr(paths, "store_root", lambda: root)
    monkeypatch.setattr(paths, "writable_store_root", lambda: root)
    for owner, version, sha in (("C", "1.0", "1"), ("B", "2.0", "2")):
        monkeypatch.setattr(pm_lock, "install_owner", lambda owner=owner: owner)
        _publish(root, "python", version, sha * 64, target, digest=True)

    monkeypatch.setattr(pm_lock, "install_owner", lambda: "A")
    lock = _pin(tmp_path / "A.json", "python", "1.0", "1" * 64, target)
    location = install._installed_location(package, lock, target)
    install._refresh_found_env(package, location, target, None)
    assert location[0].get("python")["version"] == "1.0"
    claimed = [row["version"] for row in Facts(root / "facts.json").variants("python")
               if "A" in (row.get("owners") or ())]
    assert claimed == ["1.0"]


def test_update_claims_a_default_tool_another_checkout_installed(tmp_path, monkeypatch):
    """The gateway's update finds the Chromium a worktree installed on the same pin: it must claim
    that row, or the worktree's next upgrade drops it and gc deletes the gateway's browser."""
    from hermes_cli.update_cmd_maint import _install_default_tools_after_update
    from pm import client

    root = tmp_path / "tools"
    root.mkdir()
    target = current_target()
    monkeypatch.setattr(paths, "store_root", lambda: root)
    monkeypatch.setattr(paths, "writable_store_root", lambda: root)
    monkeypatch.setattr(environments, "installs_root", lambda: tmp_path / "installs")
    monkeypatch.setattr(client, "is_runtime", lambda: True)
    monkeypatch.setattr(install, "lazy_installs_allowed", lambda: True)
    for owner in ("gateway", "worktree"):
        (tmp_path / "installs" / owner).mkdir(parents=True)
    lock = _pin(tmp_path / "gateway.json", "chromium", "1248", "1" * 64, target)
    lock.save()  # the update step reads the defaults from this checkout's lockfile on disk
    monkeypatch.setattr(install, "_lockfile", lambda: lock)
    monkeypatch.setattr(paths, "lockfile_path", lambda: tmp_path / "gateway.json")

    monkeypatch.setattr(pm_lock, "install_owner", lambda: "worktree")
    entry, _ = _publish(root, "chromium", "1248", "1" * 64, target, digest=True)
    monkeypatch.setattr(pm_lock, "install_owner", lambda: "gateway")
    _install_default_tools_after_update()

    monkeypatch.setattr(pm_lock, "install_owner", lambda: "worktree")
    _publish(root, "chromium", "1300", "2" * 64, target, digest=True)
    cli._gc_store(Store(root), Facts(root / "facts.json"))
    monkeypatch.setattr(pm_lock, "install_owner", lambda: "gateway")
    assert entry.is_dir()
    found = install.installed_package("chromium")
    assert found is not None and found.version == "1248"
