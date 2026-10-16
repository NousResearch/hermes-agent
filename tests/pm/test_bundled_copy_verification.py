"""Bundled copies retain identity, byte-integrity and executable checks."""
from pathlib import Path
import shutil

import pytest

from pm.install import _install
from pm.lock import Facts, Lockfile
from pm.package import InstallError
from pm.packages import Nodejs
from pm.store import Store, current_target, tree_digest


@pytest.mark.parametrize("damage", ["identity", "digest", "source", "copy", "executable"])
def test_invalid_bundled_copy_is_not_published(tmp_path, monkeypatch, damage):
    package, target, pin = Nodejs(), current_target(), "1" * 64
    source, destination = Store(tmp_path / "source"), Store(tmp_path / "destination")
    lock = Lockfile(tmp_path / "lock.json")
    lock.set_pin("node", "test", {target: {"url": "https://test.invalid", "sha256": pin}})
    entry = source.entry(package.store_entry("test", target))
    binary = package.binary(entry, target)
    binary.parent.mkdir(parents=True)
    binary.write_bytes(b"pinned node")
    facts, copied_facts = Facts(source.root / "facts.json"), Facts(destination.root / "facts.json")
    facts.record("node", "test", entry.name, package.env(entry, target), source.root,
                 target=target, artifacts=["2" * 64 if damage == "identity" else pin],
                 digest=None if damage == "digest" else tree_digest(entry))
    monkeypatch.setattr(package, "verify", lambda *_: "cannot execute" if damage == "executable" else "")
    monkeypatch.setattr(Store, "fetch_many", lambda *_a, **_kw: pytest.fail("copy downloaded bytes"))

    if damage == "source":
        binary.write_bytes(b"damaged source")
    elif damage == "copy":
        copytree = shutil.copytree

        def corrupt_copy(src, dst, **kwargs):
            result = copytree(src, dst, **kwargs)
            package.binary(Path(dst), target).write_bytes(b"damaged copy")
            return result

        monkeypatch.setattr(shutil, "copytree", corrupt_copy)

    with pytest.raises(InstallError, match="verification|copied bytes"):
        _install(package, lock, copied_facts, destination, target, copy_from=(facts, source))

    assert not destination.entry(entry.name).exists()
    assert copied_facts.get("node") is None
