"""Archive lifetime follows package publication, not transfer completion."""
from __future__ import annotations

import hashlib
import io
from pathlib import Path
import zipfile

import pytest

import pm
from pm import paths, registry
from pm.ensure import ensure, stage_only
from pm.lock import Facts, Lockfile
from pm.package import Package
from tests.pm._range_server import RangeHandler, dl_server, url  # noqa: F401


class Components(Package):
    name = "cleanup-components"

    def verify(self, entry, target):
        return "" if (entry / "engine").is_file() and (entry / "library").is_file() else "incomplete package"


@pytest.fixture
def install_case(tmp_path, monkeypatch, dl_server):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    store = tmp_path / "store"
    monkeypatch.setenv("HERMES_RUNTIME_DIR", str(store))
    lock_path = tmp_path / "lock.json"
    monkeypatch.setattr(paths, "lockfile_path", lambda: lock_path)
    package = Components()
    monkeypatch.setitem(registry._packages, package.name, package)
    pins = []
    for name in ("engine", "library"):
        data = io.BytesIO()
        with zipfile.ZipFile(data, "w") as archive:
            archive.writestr(name, name.encode())
        path = f"/{name}.zip"
        RangeHandler.payloads[path] = data.getvalue()
        pins.append({"url": url(dl_server, path), "sha256": hashlib.sha256(data.getvalue()).hexdigest()})
    lock = Lockfile(lock_path)
    lock.set_pin(package.name, "1", {pm.current_target(): pins})
    lock.save()
    other = store / ("fetch-" + "f" * 64)
    other.mkdir(parents=True)
    (other / "other.zip").write_bytes(b"another install's download")
    partial = paths.partials_root() / "unrelated.part"
    partial.parent.mkdir(parents=True, exist_ok=True)
    partial.write_bytes(b"in-progress download")
    return package, store, pins, other, partial


def install(package, mode):
    if mode == "stage":
        return stage_only(package.name, pm.current_target())
    ensure(package.name, explicit=True, base_env={})
    return pm.installed_package(package.name).path


@pytest.mark.parametrize("mode", ["install", "stage"])
def test_publication_removes_only_its_archives(install_case, mode):
    package, store, pins, other, partial = install_case
    entry = install(package, mode)
    for name in ("engine", "library"):
        assert (entry / name).read_bytes() == name.encode()
    assert not any((store / f"fetch-{pin['sha256']}").exists() for pin in pins)
    assert (other / "other.zip").read_bytes() == b"another install's download"
    assert partial.read_bytes() == b"in-progress download"
    RangeHandler.payloads.clear()
    assert install(package, mode) == entry  # Already installed works offline without archives.


@pytest.mark.parametrize(("mode", "failure"), [
    (mode, failure)
    for mode in ("install", "stage")
    for failure in ("unpack", "verify", "publish", "published-verify", "facts")
    if mode == "install" or failure != "facts"
])
def test_failure_keeps_archives_for_offline_retry(install_case, monkeypatch, mode, failure):
    package, store, pins, _, _ = install_case
    original_verify = package.verify
    original_unpack = package.unpack

    def fail(*args, **kwargs):
        raise pm.InstallError(package.name, "injected failure")

    def unpack(archive, staged, target):
        original_unpack(archive, staged, target)
        if (staged / "library").exists():
            fail()

    def verify(entry, target):
        published = entry.parent == store
        if failure == "verify" or published:
            fail()
        return original_verify(entry, target)

    with monkeypatch.context() as fault:
        if failure == "unpack":
            fault.setattr(package, "unpack", unpack)
        elif failure in ("verify", "published-verify"):
            fault.setattr(package, "verify", verify)
        elif failure == "publish":
            original_publish = pm.Store.publish

            def publish(self, staged, name):
                if name.startswith(package.name):
                    fail()
                return original_publish(self, staged, name)

            fault.setattr(pm.Store, "publish", publish)
        else:
            fault.setattr(Facts, "record", fail)
        with pytest.raises(pm.InstallError, match="injected failure"):
            install(package, mode)
    assert all((store / f"fetch-{pin['sha256']}").is_dir() for pin in pins)
    RangeHandler.payloads.clear()
    entry = install(package, mode)
    assert (entry / "engine").read_bytes() == b"engine"
    assert not any((store / f"fetch-{pin['sha256']}").exists() for pin in pins)
