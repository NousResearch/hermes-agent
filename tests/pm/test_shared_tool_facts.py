"""Regression for NousResearch/hermes-agent#149497: shared tool-store pins stay usable."""

from pm import install, paths
from pm._uv import _toolchain
from pm.lock import Facts, Lockfile
from pm.registry import get_package
from pm.store import current_target


def test_two_installs_keep_independent_tool_pins_and_gc_roots(tmp_path, monkeypatch):
    root = tmp_path / "tools"
    root.mkdir()
    monkeypatch.setattr(paths, "store_root", lambda: root)
    monkeypatch.setattr(paths, "writable_store_root", lambda: root)
    target = current_target()
    facts = Facts(root / "facts.json")
    locks, binaries, entries = [], {}, set()

    for index, version in enumerate(("1.0", "2.0"), 1):
        lock = Lockfile(tmp_path / f"install-{index}.json")
        for name in ("uv", "python"):
            package = get_package(name)
            sha = f"{index:064x}"
            lock.set_pin(name, version, {target: {"url": "https://invalid.test/tool",
                                                  "sha256": sha}})
            entry = root / package.store_entry(version, target)
            binary = package.binary(entry, target)
            assert binary is not None
            binary.parent.mkdir(parents=True, exist_ok=True)
            binary.write_bytes(b"fixture")
            binary.chmod(0o755)
            facts.record(name, version, entry.name, package.env(entry, target), root,
                         target=target, artifacts=[sha], digest="f" * 64)
            binaries[(version, name)] = binary
            entries.add(entry.name)
        lock.save()
        locks.append(lock)

    # B recorded last, but A's exact pins still resolve without reinstalling.
    for lock, version in ((locks[0], "1.0"), (locks[1], "2.0"), (locks[0], "1.0")):
        monkeypatch.setattr(install, "_lockfile", lambda lock=lock: lock)
        toolchain = _toolchain(realize=False)
        assert toolchain is not None
        assert tuple(path.resolve() for path in toolchain) == tuple(
            binaries[(version, name)].resolve() for name in ("uv", "python")
        )
        for name in ("uv", "python"):
            location = install._installed_location(get_package(name), lock, target)
            assert location is not None
            assert location[0].get(name)["version"] == version

    # GC must not reclaim either install's still-recorded entries.
    assert Facts(root / "facts.json").entries_in_use() == entries

    # An identical version with a different artifact hash cannot borrow a fact.
    repinned = Lockfile(tmp_path / "repinned.json")
    repinned.set_pin("uv", "1.0", {target: {"url": "https://invalid.test/changed",
                                          "sha256": "a" * 64}})
    assert install._installed_location(get_package("uv"), repinned, target) is None
    # Outdated fallback remains the last selected install, not an older variant.
    fallback = install._installed_location(get_package("uv"), repinned, target,
                                           allow_outdated=True)
    assert fallback is not None
    assert fallback[0].get("uv")["version"] == "2.0"
