"""Minor-style updates follow advertised artifacts without dropping other pins."""
from __future__ import annotations

from argparse import Namespace
import hashlib
import importlib
import io
import json
import zipfile

import pytest

from pm import cli, paths, registry
from pm.lock import Facts, Lockfile
from pm.package import Package
from pm.store import current_target
from tests.pm._range_server import RangeHandler, dl_server, url  # noqa: F401


@pytest.mark.parametrize("next_version", ["1.2.3", "1.2.4"])
def test_same_minor_refresh_updates_real_bytes_and_preserves_unresolved_targets(
    tmp_path, dl_server, monkeypatch, next_version,
):
    class RollingPackage(Package):
        name = "rolling-tool"
        version_style = "minor"

        def latest_versions(self, target, locked=None):
            return [next_version] if target == current_target() else []

        def fetch_url(self, version, target):
            assert version == next_version
            assert target == current_target()
            return url(dl_server, "/new-build.zip")

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("tool.txt", b"new rolling build")
    body = buffer.getvalue()
    RangeHandler.payloads["/new-build.zip"] = body
    original_get = RangeHandler.do_GET
    requests = []

    def record(handler):
        requests.append(handler.path)
        original_get(handler)

    monkeypatch.setattr(RangeHandler, "do_GET", record)
    target = current_target()
    held_target = "linux-arm64-bionic" if target != "linux-arm64-bionic" else "win32-arm64"
    held = {"url": "https://upstream.invalid/held.deb", "sha256": "b" * 64}
    lock = Lockfile(tmp_path / "lock.json")
    lock.set_pin("rolling-tool", "1.2.3", {
        target: {"url": url(dl_server, "/old-build.zip"), "sha256": "a" * 64},
        held_target: held,
    })
    lock.set_pin("unrelated", "2.0", {"any": held})
    lock.save()
    store = tmp_path / "tools"
    monkeypatch.setitem(registry._packages, "rolling-tool", RollingPackage())
    monkeypatch.setattr(paths, "lockfile_path", lambda: lock.path)
    monkeypatch.setattr(paths, "repo_root", lambda: tmp_path)
    monkeypatch.setattr(paths, "partials_root", lambda: tmp_path / "partials")
    monkeypatch.setenv("HERMES_RUNTIME_DIR", str(store))
    syncs = []
    engine = importlib.import_module("pm.ensure")
    monkeypatch.setattr(engine, "sync_venv", lambda **kwargs: syncs.append(kwargs))
    before = lock.path.read_bytes()
    args = Namespace(names=["rolling-tool"], target=None, check=True, uv=False, npm=False)
    assert cli.cmd_update(args) == 1
    assert lock.path.read_bytes() == before and not requests and not syncs
    assert not store.exists()

    args.check = False
    assert cli.cmd_update(args) == 0
    after = json.loads(lock.path.read_text(encoding="utf-8"))["packages"]
    original = json.loads(before)["packages"]
    assert after["unrelated"] == original["unrelated"]
    assert after["rolling-tool"]["version"] == original["rolling-tool"]["version"]
    assert after["rolling-tool"]["artifacts"][held_target] == held
    assert Lockfile(lock.path).artifacts("rolling-tool", target) == [{
        "url": url(dl_server, "/new-build.zip"), "sha256": hashlib.sha256(body).hexdigest(),
    }]
    fact = Facts(store / "facts.json").get("rolling-tool")
    assert (store / fact["entry"] / "tool.txt").read_bytes() == b"new rolling build"
    assert syncs == [{"explicit": True}]
    assert requests and all(path == "/new-build.zip" for path in requests)

    pinned = lock.path.read_bytes()
    requests.clear()
    syncs.clear()
    for check in (True, False):
        args.check = check
        assert cli.cmd_update(args) == 0
    assert lock.path.read_bytes() == pinned and not requests and not syncs
