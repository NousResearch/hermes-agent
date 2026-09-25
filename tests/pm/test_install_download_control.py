"""Install controls reach real archive transfers without publishing partial packages."""

from __future__ import annotations

import hashlib
import io
import json
from functools import partial
import threading
import zipfile

import pytest

import pm
from pm import paths, registry
from pm.downloader import DownloadPaused
from pm.install import ensure, stage_only
from pm.lock import Facts, Lockfile
from pm.package import Package
from pm.store import tree_digest
from tests.pm._range_server import RangeHandler, dl_server, url  # noqa: F401


class ComponentPackage(Package):
    name = "download-components"


def archive(files: dict[str, bytes]) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as stream:
        for name, body in files.items():
            stream.writestr(name, body)
    return output.getvalue()




@pytest.fixture(params=["install", "stage"])
def realize(request):
    if request.param == "install":
        return partial(ensure, explicit=True)
    return partial(stage_only, target="linux-arm64-bionic")


@pytest.mark.parametrize("cached", [False, True])
def test_install_pause_preserves_archives_and_resumes_the_same_pin(tmp_path, monkeypatch, dl_server, realize, cached):
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
    lock.set_pin(ComponentPackage.name, "1", {"any": pins})
    lock.save()
    if cached:
        with pm.Store(root).scratch() as scratch:
            pm.Store(root).fetch(pins[0]["url"], pins[0]["sha256"], scratch)
    other = root / ("fetch-" + "f" * 64) / "other.zip"
    other.parent.mkdir(parents=True, exist_ok=True)
    other.write_bytes(b"another install")
    unrelated = paths.partials_root() / "unrelated.part"
    unrelated.parent.mkdir(parents=True, exist_ok=True)
    unrelated.write_bytes(b"in progress")
    pause = threading.Event()
    ticks, stages = [], []

    def progress(stage, done, total, label):
        stages.append((stage, done, total, label))
        if stage == "download" and label == "2/2" and len(payloads[0]) < done < total:
            pause.set()

    with pytest.raises(DownloadPaused):
        realize(ComponentPackage.name, progress=progress, pause_event=pause,
                download_progress=lambda done, total, ranges: ticks.append((done, total, ranges)))
    expected = sum(map(len, payloads))
    assert ticks and all(total == expected for _, total, _ in ticks)
    assert ticks[0][0] == (len(payloads[0]) if cached else 0)
    assert all(sum(end - start for rows in ranges.values() for start, end in rows) == done
               for done, _, ranges in ticks)
    assert [done for done, _, _ in ticks] == sorted(done for done, _, _ in ticks)
    assert Facts(paths.facts_path()).get(ComponentPackage.name) is None
    assert list(paths.partials_root().glob("*.ranges"))
    first_requests = [request for request in RangeHandler.ranges_seen if request[0] == "/component-0.zip"]
    assert first_requests
    assert (root / f"fetch-{pins[0]['sha256']}").is_dir()

    pause.clear()
    stages.clear()
    result = realize(ComponentPackage.name, pause_event=pause,
                     progress=lambda *args: stages.append(args),
                     download_progress=lambda done, total, ranges: ticks.append((done, total, ranges)))
    assert ticks[-1][:2] == (expected, expected)
    fact = Facts(paths.facts_path()).get(ComponentPackage.name)
    if fact is None:
        entry = result
        assert json.loads((entry / ".pm-stage-pin.json").read_text()) == {
            "target": "linux-arm64-bionic", "sha256": [pin["sha256"] for pin in pins],
        }
        assert not paths.facts_path().exists()
    else:
        entry = root / fact["entry"]
        assert fact["artifacts"] == [pin["sha256"] for pin in pins]
        assert fact["digest"] == tree_digest(entry)
        assert not (entry / ".pm-stage-pin.json").exists()
    for name, body in contents.items():
        assert (entry / name).read_bytes() == body
    assert [request for request in RangeHandler.ranges_seen if request[0] == "/component-0.zip"] == first_requests
    assert [label for stage, _, _, label in stages if stage == "unpack"] == ["1/2", "2/2"]
    assert {stage for stage, *_ in stages} >= {"download", "unpack", "verify"}
    assert list(paths.partials_root().glob("*.part")) == [unrelated]
    assert list(root.glob("fetch-*")) == [other.parent]
    assert other.read_bytes() == b"another install"
    assert unrelated.read_bytes() == b"in progress"
    RangeHandler.payloads.clear()
    realize(ComponentPackage.name)  # Warm reuse needs neither server nor archives.
