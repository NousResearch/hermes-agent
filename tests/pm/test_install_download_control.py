"""Install controls reach real archive transfers without publishing partial packages."""

from __future__ import annotations

import hashlib
import io
import threading
import zipfile
from pathlib import Path

import pytest

import pm
from pm import paths, registry
from pm.downloader import DownloadPaused
from pm.ensure import ensure
from pm.lock import Facts, Lockfile
from pm.package import Package
from tests.pm._range_server import RangeHandler, dl_server, url  # noqa: F401


class ComponentPackage(Package):
    name = "download-components"


def archive(files: dict[str, bytes]) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as stream:
        for name, body in files.items():
            stream.writestr(name, body)
    return output.getvalue()


@pytest.fixture(autouse=True)
def isolate_home(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))


def test_install_pause_preserves_archives_and_resumes_the_same_pin(tmp_path, monkeypatch, dl_server):
    root = tmp_path / "store"
    lock_path = tmp_path / "lock.json"
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(paths, "store_root", lambda: root)
    monkeypatch.setattr(paths, "lockfile_path", lambda: lock_path)
    monkeypatch.setitem(registry._packages, ComponentPackage.name, ComponentPackage())
    contents = {"engine.dat": b"engine component", "runtime.dll": bytes(range(256)) * (64 * 1024)}
    payloads = [archive({name: body}) for name, body in contents.items()]
    pins = []
    for index, payload in enumerate(payloads):
        path = f"/component-{index}.zip"
        RangeHandler.payloads[path] = payload
        pins.append({"url": url(dl_server, path), "sha256": hashlib.sha256(payload).hexdigest()})
    lock = Lockfile(lock_path)
    lock.set_pin(ComponentPackage.name, "1", {pm.current_target(): pins})
    lock.save()
    pause = threading.Event()

    def progress(stage, done, total, label):
        if stage == "download" and label == "2/2" and len(payloads[0]) < done < total:
            pause.set()

    with pytest.raises(DownloadPaused):
        ensure(ComponentPackage.name, explicit=True, progress=progress, pause_event=pause)
    assert Facts(paths.facts_path()).get(ComponentPackage.name) is None
    assert list(paths.partials_root().glob("*.ranges"))
    first_requests = [request for request in RangeHandler.ranges_seen if request[0] == "/component-0.zip"]
    assert first_requests
    assert (root / f"fetch-{pins[0]['sha256']}").is_dir()

    pause.clear()
    ensure(ComponentPackage.name, explicit=True, pause_event=pause)
    fact = Facts(paths.facts_path()).get(ComponentPackage.name)
    assert fact["artifacts"] == [pin["sha256"] for pin in pins]
    for name, body in contents.items():
        assert (root / fact["entry"] / name).read_bytes() == body
    assert [request for request in RangeHandler.ranges_seen if request[0] == "/component-0.zip"] == first_requests
    assert not list(paths.partials_root().glob("*.part"))
    assert not list(root.glob("fetch-*"))


def test_install_progress_covers_all_archives_including_cache(tmp_path, monkeypatch, dl_server):
    root = tmp_path / "store"
    lock_path = tmp_path / "lock.json"
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(paths, "store_root", lambda: root)
    monkeypatch.setattr(paths, "lockfile_path", lambda: lock_path)
    monkeypatch.setitem(registry._packages, ComponentPackage.name, ComponentPackage())
    payloads = [archive({"engine.dat": b"engine"}), archive({"runtime.dll": b"runtime"})]
    pins = []
    for index, payload in enumerate(payloads):
        path = f"/component-{index}.zip"
        RangeHandler.payloads[path] = payload
        pins.append({"url": url(dl_server, path), "sha256": hashlib.sha256(payload).hexdigest()})
    lock = Lockfile(lock_path)
    lock.set_pin(ComponentPackage.name, "1", {pm.current_target(): pins})
    lock.save()
    store = pm.Store(root)
    with store.scratch() as scratch:
        store.fetch(pins[0]["url"], pins[0]["sha256"], scratch)
    ticks = []
    stages = []
    ensure(ComponentPackage.name, explicit=True,
              progress=lambda *args: stages.append(args),
              download_progress=lambda done, total, ranges: ticks.append((done, total, ranges)))
    expected = sum(map(len, payloads))
    assert ticks and all(total == expected for _, total, _ in ticks)
    assert ticks[0][0] == len(payloads[0])
    assert ticks[-1][0] == expected
    assert all(sum(end - start for rows in ranges.values() for start, end in rows) == done
               for done, _, ranges in ticks)
    assert [done for done, _, _ in ticks] == sorted(done for done, _, _ in ticks)
    assert {stage for stage, *_ in stages} >= {"download", "unpack", "verify"}
