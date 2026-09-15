"""An update reads each successful index/artifact once, not once per target."""
from __future__ import annotations

from argparse import Namespace
from collections import Counter
import hashlib
import json
from urllib.error import HTTPError
from urllib.parse import urlsplit

import pytest

from pm import cli, downloader, packages, update
from pm.lock import Lockfile
from pm.registry import get_package
from pm.store import ALL_TARGETS
from tests.pm._fixtures import make_tar
from tests.pm._range_server import RangeHandler, dl_server, url  # noqa: F401


@pytest.fixture
def upstream(dl_server, monkeypatch):
    from hermes_cli import urllib_security

    calls = []
    failures = {}
    original_get = RangeHandler.do_GET

    def record(handler):
        calls.append((handler.path, handler.headers.get("Authorization")))
        pending = failures.get(handler.path, [])
        if pending:
            handler.send_error(pending.pop(0))
            return
        original_get(handler)

    def route(request):
        parsed = urlsplit(request.full_url)
        request.full_url = url(dl_server, parsed.path + (f"?{parsed.query}" if parsed.query else ""))
        return request

    real_open = urllib_security.open_credentialed_url
    real_artifact_open = downloader._OPENER.open
    monkeypatch.setattr(RangeHandler, "do_GET", record)
    monkeypatch.setattr(urllib_security, "open_credentialed_url", lambda req, **kw: real_open(route(req), **kw))
    monkeypatch.setattr(downloader._OPENER, "open", lambda req, **kw: real_artifact_open(route(req), **kw))
    return calls, failures


@pytest.mark.parametrize("name", ["npm", "node", "uv", "python", "llamacpp-cpu", "ffmpeg"])
def test_resolution_reuses_successful_responses_but_refreshes_next_operation(upstream, monkeypatch, tmp_path, name):
    calls, failures = upstream
    package = get_package(name)
    targets = [t for t in ALL_TARGETS if package.missing_reason(t) is None]
    for generation in (1, 2):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / f"profile-{generation}"))
        monkeypatch.setenv("HERMES_RUNTIME_DIR", str(tmp_path / f"store-{generation}"))
        monkeypatch.setenv("GH_TOKEN", f"github-{generation}")
        monkeypatch.setenv("HF_TOKEN", f"huggingface-{generation}")
        calls.clear()
        payloads = {}
        expected_targets = targets
        version = f"2.0.{generation}"
        locked = "1.0.0"
        artifacts = None
        if name == "npm":
            payloads["/-/package/npm/dist-tags"] = {"latest": version}
        elif name == "node":
            payloads["/dist/index.json"] = [{"version": f"v{version}"}, {"version": "nightly"}]
        elif name == "uv":
            # Three populated pages exercise the real pagination loop.
            for page in range(1, 4):
                payloads[f"/repos/astral-sh/uv/releases?per_page=30&page={page}"] = [
                    {"tag_name": version}, *[{"tag_name": "9.0.0", "prerelease": True}] * 29,
                ]
        elif name == "python":
            version = f"3.14.{generation}+20990102"
            locked = "3.14.0+20990101"
            # Only one architecture is advertised: sharing JSON must not share its candidates.
            expected_targets = ["linux-x64"]
            payloads["/repos/astral-sh/python-build-standalone/releases?per_page=30&page=1"] = [{
                "tag_name": "20990102", "assets": [
                    {"name": f"cpython-{version}-x86_64-unknown-linux-gnu-install_only.tar.gz"},
                    {"name": "cpython-3.15.0+20990102-aarch64-apple-darwin-install_only.tar.gz"},
                ],
            }]
        elif name == "llamacpp-cpu":
            version = str(100 + generation)
            locked = "100"
            payloads["/buckets/ggml-org/install.sh/resolve/latest"] = f"b{version}\n"
            payloads["/api/buckets/ggml-org/install.sh/tree?limit=1000&offset=0"] = []
        else:
            locked = version = "9.1.2"
            artifacts = {}
            expected_targets = [t for t in targets if t != "linux-arm64-bionic"]
            payloads["/"] = "\n".join(
                f'<a href="/download/{osname}/{arch}/{generation}_{version}/ffmpeg.zip">build</a>'
                for osname in ("linux", "macos") for arch in ("amd64", "arm64")
            )
            payloads["/repos/BtbN/FFmpeg-Builds/releases?per_page=30&page=1"] = [{
                "tag_name": f"autobuild-{generation}", "assets": [
                    {"name": f"ffmpeg-n{version}-1-gabcdef-win{arch}-gpl-9.1.zip"}
                    for arch in ("64", "arm64")
                ],
            }]
        RangeHandler.payloads = {path: (value if isinstance(value, str) else json.dumps(value)).encode()
                                 for path, value in payloads.items()}
        # A failed transfer still retries; only the eventual success is reusable.
        retry_path = next(iter(payloads))
        failures[retry_path] = [503]
        decision = update.resolve_package(package, targets, locked, artifacts=artifacts)
        assert decision.version == version
        assert decision.per_target == dict.fromkeys(expected_targets, version)
        assert decision.changed
        if name == "ffmpeg":
            assert set(decision.artifact_updates) == set(expected_targets)
            assert all(f"/{generation}_{version}/" in urls[0] or f"/autobuild-{generation}/" in urls[0]
                       for urls in decision.artifact_updates.values())
        expected = Counter(dict.fromkeys(payloads, 1))
        expected[retry_path] += 1
        assert Counter(path for path, _ in calls) == expected
        for path, auth in calls:
            token = (f"github-{generation}" if path.startswith("/repos/") else
                     f"huggingface-{generation}" if "ggml-org/install.sh" in path else None)
            assert auth == (f"Bearer {token}" if token else None)

    # Recovery semantics differ: npm propagates, while llama's optional pointer
    # falls back and must try again on the next target, not memoize its failure.
    calls.clear()
    failures[retry_path] = [404]
    if name == "npm":
        with pytest.raises(HTTPError, match="404"):
            update.resolve_package(package, targets, locked)
        assert len(calls) == 1
        assert update.resolve_package(package, targets, locked).version == version
        assert Counter(path for path, _ in calls) == {retry_path: 2}
    elif name == "llamacpp-cpu":
        fallback = "/repos/ggml-org/llama.cpp/releases?per_page=30&page=1"
        RangeHandler.payloads[fallback] = json.dumps([{"tag_name": f"b{version}"}]).encode()
        assert update.resolve_package(package, targets, locked).version == version
        assert Counter(path for path, _ in calls) == {**expected, fallback: 1}
    elif name == "uv":
        failures.clear()
        endpoint = "https://api.github.com" + retry_path
        with update.reuse_index_responses():
            for credential in ("first", "second"):
                monkeypatch.setenv("GH_TOKEN", credential)
                for _ in range(2):
                    assert update._get_json(endpoint) == payloads[retry_path]
        assert calls == [(retry_path, "Bearer first"), (retry_path, "Bearer second")]
    elif name == "ffmpeg":
        # Manual re-pinning also needs a short-lived index scope after removing
        # the old process-wide FFmpeg indexes. Pin real bytes into a real lock.
        failures.clear()
        monkeypatch.setattr(package, "gaps", {"linux-arm64-bionic": "separate supplier"})
        body = b"advertised ffmpeg artifact"
        for urls in decision.artifact_updates.values():
            RangeHandler.payloads[urlsplit(urls[0]).path] = body
        lock = Lockfile(tmp_path / "lock.json")
        monkeypatch.setattr(cli, "_lockfile", lambda: lock)
        assert cli.cmd_lock(Namespace(name=name, version=version)) == 0
        pinned = Lockfile(lock.path)
        assert pinned.version(name) == version
        for target, urls in decision.artifact_updates.items():
            assert pinned.artifacts(name, target) == [{"url": urls[0], "sha256": hashlib.sha256(body).hexdigest()}]
        assert Counter(path for path, _ in calls) == dict.fromkeys(RangeHandler.payloads, 1)


@pytest.mark.parametrize("pointer, tree, expected", [
    ("b10679\n", [{"path": "b10326/linux"}, {"path": "b10098/win"}, {"path": "b10326/mac"}, {"path": "b9733/mac"}, {"path": "latest/other"}], ["10679", "10326", "10098", "9733"]),
    (None, [], ["10362", "10217"]),
    ("malformed", [{"path": "not-a-build"}, {}], ["10362", "10217"]),
])
def test_llama_real_index_fallback(upstream, monkeypatch, pointer, tree, expected):
    calls, _ = upstream
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("GH_TOKEN", raising=False)
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    latest = "/buckets/ggml-org/install.sh/resolve/latest"
    if pointer is not None:
        RangeHandler.payloads[latest] = pointer.encode()
    RangeHandler.payloads["/api/buckets/ggml-org/install.sh/tree?limit=1000&offset=0"] = json.dumps(tree).encode()
    github = "/repos/ggml-org/llama.cpp/releases?per_page=30&page=1"
    RangeHandler.payloads[github] = b'[{"tag_name":"b10362"},{"tag_name":"b10217"}]'
    assert packages.LlamaCppCpu().latest_versions("linux-x64") == expected
    assert all(auth is None for _, auth in calls)
    assert any(path == github for path, _ in calls) is (pointer != "b10679\n")


def test_pinning_hashes_new_npm_url_once_and_keeps_existing_rows(upstream, tmp_path, monkeypatch):
    calls, failures = upstream
    monkeypatch.setenv("HERMES_RUNTIME_DIR", str(tmp_path / "store"))
    package = packages.Npm()
    version = "2.0.0"
    artifact_url = package.fetch_url(version, ALL_TARGETS[0])
    artifact_path = urlsplit(artifact_url).path
    filename, digest = make_tar(tmp_path, "npm.tgz", {"package/package.json": '{"version":"2.0.0"}'})
    RangeHandler.payloads[artifact_path] = (tmp_path / filename).read_bytes()
    old = {"url": package.fetch_url("1.0.0", ALL_TARGETS[0]), "sha256": "a" * 64}
    current = {"any": old, "unresolved-target": old}
    decision = update.Resolved("npm", "1.0.0", "semver", version,
                               per_target=dict.fromkeys(ALL_TARGETS, version))
    pinned = cli._pin_artifacts(package, decision, current)
    assert current == {"any": old, "unresolved-target": old}
    assert pinned == {**current, **dict.fromkeys(ALL_TARGETS, {"url": artifact_url, "sha256": digest})}
    assert Counter(path for path, _ in calls) == {artifact_path: 1}

    # Existing pins need no download; a separate pin invocation must not retain new hashes.
    calls.clear()
    assert cli._pin_artifacts(package, decision, pinned) == pinned
    assert not calls
    replacement = b"changed upstream bytes"
    RangeHandler.payloads[artifact_path] = replacement
    failures[artifact_path] = [404]
    with pytest.raises(HTTPError, match="404"):
        cli._pin_artifacts(package, decision, current)
    assert current == {"any": old, "unresolved-target": old}
    calls.clear()
    repinned = cli._pin_artifacts(package, decision, current)
    assert repinned[ALL_TARGETS[-1]]["sha256"] == hashlib.sha256(replacement).hexdigest()
    assert Counter(path for path, _ in calls) == {artifact_path: 1}
