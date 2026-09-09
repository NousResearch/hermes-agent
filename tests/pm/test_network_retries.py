"""PM retries transient HTTP failures before publishing verified bytes."""

from __future__ import annotations

import hashlib
import importlib
import io
import zipfile

import pytest

import pm.paths as paths
from pm.lock import Facts, Lockfile
from pm.package import Package
from pm.store import Store, current_target
from tests.pm._range_server import RangeHandler, dl_server, url  # noqa: F401


@pytest.mark.parametrize("failure_phase", [
    "probe", "probe-disconnect", "probe-empty-body", "ranged", "single",
    "interrupted", "single-interrupted",
])
@pytest.mark.parametrize("install_path", ["install", "stage"])
def test_install_recovers_from_transient_http_failure(tmp_path, dl_server, monkeypatch, failure_phase, install_path):
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w") as archive:
        archive.writestr("tool.txt", b"verified tool bytes")
    body = payload.getvalue()
    RangeHandler.payloads["/tool.zip"] = body
    RangeHandler.no_range = failure_phase.startswith("single")
    original_get = RangeHandler.do_GET
    requests = []
    failed = False

    def fail_first_request(handler):
        nonlocal failed
        request_range = handler.headers.get("Range")
        requests.append(request_range)
        is_probe = request_range == "bytes=0-0"
        if not failed and is_probe == failure_phase.startswith("probe"):
            failed = True
            if failure_phase in ("probe-disconnect", "probe-empty-body"):
                if failure_phase == "probe-empty-body":
                    handler.send_response(206)
                    handler.send_header("Content-Range", f"bytes 0-0/{len(body)}")
                    handler.send_header("Content-Length", "1")
                    handler.end_headers()
                handler.close_connection = True
            elif failure_phase.endswith("interrupted"):
                handler.send_response(200 if RangeHandler.no_range else 206)
                if not RangeHandler.no_range:
                    handler.send_header("Content-Range", f"bytes 0-{len(body) - 1}/{len(body)}")
                handler.send_header("Content-Length", str(len(body)))
                handler.end_headers()
                handler.wfile.write(body[: len(body) // 2])
                handler.wfile.flush()
                handler.close_connection = True
            else:
                handler.send_error(500)
            return
        original_get(handler)

    monkeypatch.setattr(RangeHandler, "do_GET", fail_first_request)
    monkeypatch.setattr(paths, "partials_root", lambda: tmp_path / "partials")
    package = Package()
    package.name = "retry-tool"
    store = Store(tmp_path / "store")
    facts = Facts(store.root / "facts.json")
    lock = Lockfile(tmp_path / "lock.json")
    lock.set_pin(package.name, "1.0", {"any": {
        "url": url(dl_server, "/tool.zip"),
        "sha256": hashlib.sha256(body).hexdigest(),
    }})
    ensure = importlib.import_module("pm.ensure")
    if install_path == "install":
        ensure._install(package, lock, facts, store, current_target())
        entry = store.entry(facts.get(package.name)["entry"])
    else:
        monkeypatch.setattr(ensure, "_lockfile", lambda: lock)
        monkeypatch.setattr(ensure, "_store", lambda: store)
        monkeypatch.setattr(ensure, "get_package", lambda _: package)
        entry = ensure.stage_only(package.name, current_target())
        assert facts.get(package.name) is None

    assert (entry / "tool.txt").read_bytes() == b"verified tool bytes"
    assert len(requests) > 1
    if failure_phase == "interrupted":
        assert requests[-1] == f"bytes={len(body) // 2}-{len(body) - 1}"
    assert not list((tmp_path / "partials").glob("*.part"))
    assert not list((tmp_path / "partials").glob("*.ranges"))


@pytest.mark.parametrize("reader_name", ["hash", "json", "text", "digests"])
@pytest.mark.parametrize("short_body", [False, True])
def test_pm_metadata_reads_retry_transient_http_failure(dl_server, monkeypatch, reader_name, short_body):
    from pm import packages, store, update

    body = b'{"assets": [{"name": "tool.zip", "digest": "sha256:abc123"}]}'
    RangeHandler.payloads["/metadata"] = body
    original_get = RangeHandler.do_GET
    requests = []

    def fail_first_request(handler):
        requests.append(handler.path)
        if len(requests) == 1:
            if short_body:
                handler.send_response(200)
                handler.send_header("Content-Length", str(len(body)))
                handler.end_headers()
                handler.wfile.write(body[: len(body) // 2])
                handler.wfile.flush()
                handler.close_connection = True
            else:
                handler.send_error(503)
            return
        original_get(handler)

    monkeypatch.setattr(RangeHandler, "do_GET", fail_first_request)
    endpoint = url(dl_server, "/metadata")
    readers = {
        "hash": (lambda: store.hash_url(endpoint), hashlib.sha256(body).hexdigest()),
        "json": (lambda: update._get_json(endpoint), {"assets": [{"name": "tool.zip", "digest": "sha256:abc123"}]}),
        "text": (lambda: update._get_text(endpoint), body.decode()),
    }
    if reader_name == "digests":
        import urllib.request
        original_urlopen = urllib.request.urlopen

        def local_release(request, **kwargs):
            return original_urlopen(urllib.request.Request(endpoint, headers=request.headers), **kwargs)

        monkeypatch.setattr(urllib.request, "urlopen", local_release)
        monkeypatch.setattr(packages, "_release_digest_cache", {})
        readers["digests"] = (lambda: packages._github_release_digests("test/repo", "test"), {"tool.zip": "abc123"})
    read, expected = readers[reader_name]
    assert read() == expected
    assert len(requests) == 2
