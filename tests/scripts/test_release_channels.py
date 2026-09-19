"""Exercise publisher/reader through the real signed HTTP transport."""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import socket
from threading import Lock, Thread
from urllib.parse import parse_qs, urlsplit
import xml.sax.saxutils

import pytest


@contextmanager
def object_server():
    objects, headers, requests = {}, {}, []
    lock = Lock()
    faults = {"lose_put": False, "stale_public": None, "conflict": None, "bad_pagination": False}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            parsed = urlsplit(self.path)
            query = parse_qs(parsed.query)
            with lock:
                requests.append(("GET", self.path))
                if "list-type" in query:
                    prefix = query.get("prefix", [""])[0]
                    keys = sorted(k for k in objects if k.startswith(prefix))
                    start = int(query.get("continuation-token", ["0"])[0])
                    page = keys[start:start + 2]
                    more = start + 2 < len(keys)
                    body = ("<ListBucketResult>" + "".join(
                        f"<Contents><Key>{xml.sax.saxutils.escape(k)}</Key></Contents>" for k in page)
                        + f"<IsTruncated>{str(more).lower()}</IsTruncated>"
                        + (f"<NextContinuationToken>{start + 2}</NextContinuationToken>" if more else "")
                        + "</ListBucketResult>").encode()
                    if faults["bad_pagination"]:
                        body = b"<ListBucketResult><IsTruncated>true</IsTruncated></ListBucketResult>"
                    etag = None
                else:
                    key = parsed.path.removeprefix("/bucket/")
                    body = objects.get(key)
                    if not self.headers.get("Authorization") and faults["stale_public"] is not None:
                        body = faults["stale_public"]
                    etag = '"' + hashlib.sha256(body).hexdigest() + '"' if body is not None else None
                self.send_response(200 if body is not None else 404)
                if etag:
                    self.send_header("ETag", etag)
                self.end_headers()
                if body is not None:
                    self.wfile.write(body)

        def do_PUT(self):
            body = self.rfile.read(int(self.headers["Content-Length"]))
            key = self.path.removeprefix("/bucket/")
            with lock:
                requests.append(("PUT", key))
                if faults["conflict"]:
                    conflict = faults["conflict"]
                    faults["conflict"] = None
                    conflict(objects, key)
                old = objects.get(key)
                etag = '"' + hashlib.sha256(old).hexdigest() + '"' if old is not None else None
                conflict = ((self.headers.get("If-None-Match") == "*" and old is not None)
                            or (self.headers.get("If-Match") is not None and self.headers["If-Match"] != etag))
                if conflict:
                    self.send_response(412)
                    self.end_headers()
                    return
                objects[key] = body
                headers[key] = dict(self.headers)
                if faults["lose_put"]:
                    faults["lose_put"] = False
                    self.connection.shutdown(socket.SHUT_RDWR)
                    self.connection.close()
                    return
                self.send_response(200)
                self.end_headers()

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", objects, headers, requests, faults
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def publisher(url, **kwargs):
    from scripts.releases.channels import ChannelPublisher, R2ChannelStore
    store = R2ChannelStore({"access_key_id": "fixture", "secret_key": "fixture"}, url, "bucket")
    return ChannelPublisher(store, "example/hermes-agent", url + "/bucket",
                            authorize=kwargs.pop("authorize", lambda action, record: None), **kwargs)


def test_unknown_channel_created_over_http_retains_identity_and_immutable_requests():
    from hermes_cli.release_channels import ChannelReader, ChannelNotFound
    with object_server() as (url, objects, headers, requests, faults):
        pub = publisher(url)
        reader = ChannelReader(url + "/bucket", repository="example/hermes-agent")
        with pytest.raises(ChannelNotFound):
            reader.resolve("not-registered-in-code")
        record = pub.create("not-registered-in-code")
        assert reader.resolve(record["name"]).manifest is None
        one = pub.allocate(record["name"], "a" * 40, "1.2.3", {"FEATURE": "one"})
        two = pub.allocate(record["name"], "a" * 40, "1.2.3", {"FEATURE": "two"})
        assert one["identity"] == two["identity"] == record["identity"]
        assert one["buildId"] != two["buildId"]
        assert one["sequence"] < two["sequence"]
        assert pub.request(one["buildId"]) == one
        assert reader.resolve(record["name"]).terminal["nextSequence"] == two["sequence"] + 1
        assert headers[f"releases/channels/{record['name']}.json"]["Cache-Control"] == "no-store"
        assert "immutable" in headers[f"releases/channel-builds/{one['buildId']}/request.json"]["Cache-Control"]


def put_build(objects, request):
    from hermes_cli.release_channels import build_prefix, canonical_json
    prefix = build_prefix(request["buildId"])
    data = b"fixture native artifact"
    objects[prefix + "darwin/package.zip"] = data
    manifest = {"schema": 1, "receiverProtocol": 1, "request": request, "packages": [{"platform": "darwin", "arch": "arm64", "variant": "bundled",
        "artifact": {"key": prefix + "darwin/package.zip", "sha256": hashlib.sha256(data).hexdigest(), "size": len(data)},
        "version": request["version"], "identity": request["identity"]["appId"], "teamId": "ABCDEFGHIJ",
        "feed": {"key": prefix + "darwin/stable-mac.yml", "channel": "stable"}}]}
    objects[prefix + "build.json"] = canonical_json(manifest)
    return manifest


def test_concurrent_allocations_reverse_completion_retirement_and_readback():
    from concurrent.futures import ThreadPoolExecutor
    from hermes_cli.release_channels import ChannelError, canonical_json
    from scripts.releases.channels import PublicVisibilityError
    with object_server() as (url, objects, headers, requests, faults):
        with ThreadPoolExecutor(max_workers=2) as pool:
            pub = publisher(url, verify_build=lambda request, manifest: True)
            created = list(pool.map(pub.create, ["race-preview"] * 2))
        assert created[0]["identity"] == created[1]["identity"]
        with ThreadPoolExecutor(max_workers=2) as pool:
            allocated = list(pool.map(lambda _: pub.allocate("race-preview", "a" * 40, "1.0.0"), range(2)))
        one, two = sorted(allocated, key=lambda r: r["sequence"])
        assert one["sequence"] != two["sequence"], "concurrent allocations must be distinct"
        for request in (one, two):
            put_build(objects, request)
        pub.promote(two["buildId"])
        with pytest.raises(ChannelError, match="newer|stale"):
            pub.promote(one["buildId"])
        assert pub.reader.resolve("race-preview").manifest["request"] == two
        # The protected destination is seeded from accepted existing metadata, not a preview masquerade.
        pub.create("destination")
        target = pub._read("destination")[0]
        target["policy"] = "stable-release"
        stable = dict(two, channel="destination", identity=target["identity"], releaseTag="v2.0.0", version="2.0.0", windowsVersion="2.0.0.0", sourceVersion="2.0.0", buildId="e" * 32)
        stable_manifest = put_build(objects, stable)
        raw = canonical_json(stable_manifest)
        target.update(nextSequence=stable["sequence"] + 1, head={"buildId": stable["buildId"], "sequence": stable["sequence"], "manifestKey": "releases/channel-builds/" + stable["buildId"] + "/build.json", "sha256": hashlib.sha256(raw).hexdigest()})
        objects["releases/channels/destination.json"] = canonical_json(target)
        unsupported = dict(stable_manifest)
        del unsupported["receiverProtocol"]
        objects[target["head"]["manifestKey"]] = canonical_json(unsupported)
        old_target = {**target, "head": {**target["head"], "sha256": hashlib.sha256(canonical_json(unsupported)).hexdigest()}}
        objects["releases/channels/destination.json"] = canonical_json(old_target)
        with pytest.raises(ChannelError, match="receiver support"):
            pub.retire("race-preview", "destination", "2.0.0")
        assert pub._read("race-preview")[0]["state"] == "active"
        objects[target["head"]["manifestKey"]] = raw
        objects["releases/channels/destination.json"] = canonical_json(target)
        retired = pub.retire("race-preview", "destination", "2.0.0")
        assert retired["destinationHead"] == target["head"]
        assert retired["lastHead"]["buildId"] == two["buildId"]
        with pytest.raises(ChannelError, match="Retired"):
            pub.promote(two["buildId"])
        with pytest.raises(ChannelError, match="Retired"):
            pub.create("race-preview")
        pub.create("visibility")
        faults["lose_put"] = True
        lost = pub.allocate("visibility", "b" * 40, "1.0.0")
        assert pub.request(lost["buildId"]) == lost
        faults["stale_public"] = b"{}"
        with pytest.raises(PublicVisibilityError, match="Committed"):
            pub.allocate("visibility", "b" * 40, "1.0.0")
        assert json.loads(objects["releases/channels/visibility.json"])["nextSequence"] > lost["sequence"] + 1


def test_list_bootstrap_protected_roles_and_qualification_gate():
    from hermes_cli.release_channels import ChannelError
    with object_server() as (url, objects, headers, requests, faults):
        pub = publisher(url)
        main = {"schema": 1, "name": "main", "repository": "example/hermes-agent", "policy": "source-branch", "state": "active", "revision": 1, "nextSequence": 1, "identity": None, "head": None, "delivery": {"kind": "source-branch", "branch": "main"}}
        assert pub.bootstrap(main) == main and not objects
        pub.bootstrap(main, publish=True)
        for name in ("first", "second", "third"):
            pub.create(name)
        assert {r["name"] for r in pub.list()} == {"main", "first", "second", "third"}
        assert any("continuation-token" in path for method, path in requests)
        faults["bad_pagination"] = True
        with pytest.raises(ChannelError, match="pagination"):
            pub.list()
        faults["bad_pagination"] = False
        assert pub.reader.resolve("main").manifest is None
        with pytest.raises(ChannelError, match="Protected"):
            pub.allocate("main", "a" * 40, "1.0.0")
        request = pub.allocate("first", "a" * 40, "1.0.0")
        put_build(objects, request)
        with pytest.raises(ChannelError, match="qualification"):
            pub.promote(request["buildId"])
        assert pub.reader.resolve("first").terminal["head"] is None


def test_retirement_race_requires_a_new_explicit_attempt():
    from hermes_cli.release_channels import ChannelError, canonical_json
    from scripts.releases.channels import ChannelConflict
    with object_server() as (url, objects, headers, requests, faults):
        pub = publisher(url, verify_build=lambda request, manifest: True)
        pub.create("preview")
        first = pub.allocate("preview", "a" * 40, "1.0.0")
        second = pub.allocate("preview", "b" * 40, "1.0.0")
        put_build(objects, first)
        put_build(objects, second)
        pub.promote(first["buildId"])
        pub.create("stable")
        target = pub._read("stable")[0]
        target["policy"] = "stable-release"
        stable = dict(first, channel="stable", identity=target["identity"], releaseTag="v2.0.0", version="2.0.0", windowsVersion="2.0.0.0", sourceVersion="2.0.0", buildId="f" * 32)
        manifest = put_build(objects, stable)
        target.update(nextSequence=2, head={"buildId": stable["buildId"], "sequence": 1, "manifestKey": "releases/channel-builds/" + stable["buildId"] + "/build.json", "sha256": hashlib.sha256(canonical_json(manifest)).hexdigest()})
        objects["releases/channels/stable.json"] = canonical_json(target)

        def race(store, object_key):
            record = json.loads(store[object_key])
            record["revision"] += 1
            record["head"] = {"buildId": second["buildId"], "sequence": second["sequence"], "manifestKey": "releases/channel-builds/" + second["buildId"] + "/build.json", "sha256": hashlib.sha256(store["releases/channel-builds/" + second["buildId"] + "/build.json"]).hexdigest()}
            store[object_key] = canonical_json(record)
        faults["conflict"] = race
        with pytest.raises(ChannelConflict):
            pub.retire("preview", "stable", "2.0.0")
        assert pub.reader.resolve("preview").manifest["request"] == second
        with pytest.raises(ChannelError, match="cycle"):
            pub.retire("preview", "preview", "2.0.0")
        assert pub.retire("preview", "stable", "2.0.0")["lastHead"]["buildId"] == second["buildId"]


def test_retire_derives_receiver_kind_from_channel_identity_match():
    """The pinned kind is derived from identity comparison, never caller-asserted."""
    from hermes_cli.release_channels import canonical_json
    with object_server() as (url, objects, headers, requests, faults):
        pub = publisher(url, verify_build=lambda request, manifest: True)
        for name in ("mainline-preview", "suffixed-preview", "stable"):
            pub.create(name)
        # A mainline-like prerelease shares the destination stable identity.
        preview = pub._read("mainline-preview")[0]
        first = pub.allocate("mainline-preview", "a" * 40, "1.0.0")
        put_build(objects, first)
        pub.promote(first["buildId"])
        target = pub._read("stable")[0]
        target["policy"] = "stable-release"
        target["identity"] = preview["identity"]
        stable = dict(first, channel="stable", identity=target["identity"], releaseTag="v2.0.0", version="2.0.0", windowsVersion="2.0.0.0", sourceVersion="2.0.0", buildId="e" * 32)
        manifest = put_build(objects, stable)
        target.update(nextSequence=stable["sequence"] + 1, head={"buildId": stable["buildId"], "sequence": stable["sequence"], "manifestKey": "releases/channel-builds/" + stable["buildId"] + "/build.json", "sha256": hashlib.sha256(canonical_json(manifest)).hexdigest()})
        objects["releases/channels/stable.json"] = canonical_json(target)
        in_place = pub.retire("mainline-preview", "stable", "2.0.0")
        assert in_place["receiver"] == {"kind": "in-place"}
        assert pub.reader.resolve("mainline-preview").requested["receiver"] == {"kind": "in-place"}
        # A suffixed channel identity can never match stable's.
        second = pub.allocate("suffixed-preview", "b" * 40, "1.0.0")
        put_build(objects, second)
        pub.promote(second["buildId"])
        discontinued = pub.retire("suffixed-preview", "stable", "2.0.0")
        assert discontinued["receiver"] == {"kind": "discontinued"}
        assert pub.reader.resolve("suffixed-preview").requested["receiver"] == {"kind": "discontinued"}


def test_mutable_read_loss_recovery_never_clones_another_allocation():
    from scripts.releases.channels import ChannelConflict
    from hermes_cli.release_channels import canonical_json
    with object_server() as (url, objects, headers, requests, faults):
        pub = publisher(url)
        pub.create("nonce-check")
        initial = pub._read("nonce-check")[0]
        def compete(store, key):
            winner = dict(initial, revision=2, nextSequence=2,
                          lastAllocation={"buildId": "b" * 32, "sequence": 1})
            store[key] = canonical_json(winner)
        faults["conflict"] = compete
        request = pub.allocate("nonce-check", "a" * 40, "1.0.0")
        assert request["sequence"] == 2
        key = "releases/channel-builds/" + request["buildId"] + "/request.json"
        with pytest.raises(ChannelConflict):
            pub.store.put(key, canonical_json(dict(request, commit="b" * 40)))
        assert pub.request(request["buildId"]) == request


def test_protected_releases_bootstrap_retry_and_refuse_late_or_ungated_promotion():
    from hermes_cli.release_channels import ChannelError, canonical_json
    from scripts.releases.channels import preview_identity
    with object_server() as (url, objects, headers, requests, faults):
        pub = publisher(url, verify_build=lambda request, manifest: True)
        accepted = True
        gate = lambda request: accepted
        identity = preview_identity("official", "1" * 16)

        def allocate(version, commit):
            return pub.allocate_protected(
                "official", commit, version, release_tag="v" + version,
                version=version, windows_version=version + ".0", identity=identity,
                policy="stable-release", release_gate=gate)

        # A failed request PUT is not recovered by sequence: retrying the same
        # release adopts a fresh sequence gap, while the deterministic build ID
        # keeps the immutable request idempotent across retries.
        original_write = pub._write
        def lose_request(key, value, etag=None):
            if key.endswith("request.json"):
                raise OSError("request upload interrupted")
            return original_write(key, value, etag)
        pub._write = lose_request
        with pytest.raises(OSError):
            allocate("1.0.0", "a" * 40)
        pub._write = original_write
        allocate("0.5.0", "e" * 40)
        first = allocate("1.0.0", "a" * 40)
        assert allocate("1.0.0", "a" * 40) == first
        assert len(first["buildId"]) == 32
        assert pub.reader.resolve("official").manifest is None
        put_build(objects, first)
        accepted = False
        with pytest.raises(ChannelError, match="release gate"):
            pub.promote_protected(first["buildId"], policy="stable-release", release_gate=gate)
        assert pub.reader.resolve("official").manifest is None
        accepted = True
        pub.promote_protected(first["buildId"], policy="stable-release", release_gate=gate)
        second = allocate("2.0.0", "b" * 40)
        late = allocate("1.5.0", "c" * 40)
        for request in (second, late):
            put_build(objects, request)
        pub.promote_protected(second["buildId"], policy="stable-release", release_gate=gate)
        assert pub.promote_protected(second["buildId"], policy="stable-release", release_gate=gate)["head"]["buildId"] == second["buildId"]
        with pytest.raises(ChannelError, match="version|newer|stale"):
            pub.promote_protected(late["buildId"], policy="stable-release", release_gate=gate)
        with pytest.raises(ChannelError, match="Protected"):
            pub.promote(second["buildId"])
        assert pub.reader.resolve("official").manifest["request"] == second
        # A competing allocation must not turn a protected promotion into an overwrite.
        def contend(store, key):
            record = json.loads(store[key])
            record["revision"] += 1
            record["nextSequence"] += 1
            store[key] = canonical_json(record)
        third = allocate("3.0.0", "d" * 40)
        put_build(objects, third)
        faults["conflict"] = contend
        pub.promote_protected(third["buildId"], policy="stable-release", release_gate=gate)
        assert pub.reader.resolve("official").manifest["request"] == third


def test_accepted_release_receipts_feed_the_protected_head_without_rebuilding(tmp_path, monkeypatch):
    from scripts.releases import channel_releases
    from hermes_cli.release_channels import ChannelError, canonical_json
    from scripts.releases.channels import preview_identity
    from scripts.releases.handoff import receipt_name
    from copy import deepcopy
    import zipfile
    identity = preview_identity("released", "2" * 16)
    tag, commit = "v2.0.0", "d" * 40
    with object_server() as (url, objects, headers, requests, faults):
        pub = publisher(url)
        base = pub.public_base
        prefix = f"releases/tag/{tag}/"
        for platform in ("darwin", "win32"):
            for arch in ("arm64", "x64"):
                native = "macos" if platform == "darwin" else "windows"
                version = "2.0.0" if platform == "darwin" else "2.0.0.0"
                metadata = {"platform": native, "arch": arch, "tag": tag, "commit": commit,
                            "version": version, "identity": identity["appId" if platform == "darwin" else "msixAppIdWithOrg"]}
                files = {}
                if platform == "darwin":
                    metadata["teamId"] = "ABCDEFGHIJ"
                    for suffix in ("zip", "dmg", "zip.blockmap", "dmg.blockmap"):
                        name = f"{identity['artifactNamePascal']}-2.0.0-mac-{arch}.{suffix}"
                        files[name] = (name + " fixture bytes").encode()
                    metadata["filename"] = f"{identity['artifactNamePascal']}-2.0.0-mac-{arch}.zip"
                else:
                    metadata.update(publisher="CN=Fixture", applicationId=identity["appNamePascal"])
                    files[f"{identity['artifactNamePascal']}-2.0.0-win-{arch}.msix"] = b"fixture msix"
                files[f"metadata-{native}-{arch}.json"] = canonical_json(metadata)
                rows = []
                for name, body in files.items():
                    objects[prefix + name] = body
                    (tmp_path / name).write_bytes(body)
                    rows.append({"path": name, "size": len(body), "sha256": hashlib.sha256(body).hexdigest()})
                receipt = {"schema": 1, "tag": tag, "commit": commit, "name": f"{platform}-{arch}", "files": rows}
                (tmp_path / receipt_name(receipt["name"])).write_bytes(canonical_json(receipt))
                objects[prefix + receipt_name(receipt["name"])] = canonical_json(receipt)
        bundle_name = f"{identity['artifactNamePascal']}-2.0.0.0-win.msixbundle"
        with zipfile.ZipFile(tmp_path / bundle_name, "w") as archive:
            archive.writestr("AppxMetadata/AppxBundleManifest.xml", f'<Bundle><Identity Name="{identity["msixAppIdWithOrg"]}" Publisher="CN=Fixture" Version="2.0.0.0"/><Packages><Package Type="application" Architecture="arm64"/><Package Type="application" Architecture="x64"/></Packages></Bundle>')
        body = (tmp_path / bundle_name).read_bytes()
        objects[prefix + bundle_name] = body
        receipt = {"schema": 1, "tag": tag, "commit": commit, "name": "windows-universal", "files": [{"path": bundle_name, "size": len(body), "sha256": hashlib.sha256(body).hexdigest()}]}
        (tmp_path / receipt_name(receipt["name"])).write_bytes(canonical_json(receipt))
        objects[prefix + receipt_name(receipt["name"])] = canonical_json(receipt)
        # An R2 policy record chooses this name, not the legacy default selector.
        record = {"schema": 1, "name": "released", "repository": pub.repository, "policy": "stable-release", "state": "active", "revision": 1, "nextSequence": 1, "head": None, "identity": identity}
        objects["releases/channels/released.json"] = canonical_json(record)
        assert channel_releases.select_channel(pub, "stable-release") == "released"
        native = channel_releases.read_native_receipts(tmp_path, tag, commit)
        request = pub.allocate_protected("released", commit, "2.0.0", release_tag=tag, version="2.0.0", windows_version="2.0.0.0", identity=identity, policy="stable-release", release_gate=lambda request: True)
        from scripts.bundles.channel_artifacts import assemble
        manifest, feeds = assemble(request, native, tmp_path, artifact_prefix=prefix)
        assert {p["arch"] for p in manifest["packages"]} == {"arm64", "x64"}
        assert all(p["artifact"]["key"].startswith(prefix) for p in manifest["packages"])
        assert all(f.is_file() for f in feeds)
        accepted = {"packages": []}
        for row in manifest["packages"]:
            accepted["packages"].append({"platform": "macos" if row["platform"] == "darwin" else "windows", "arch": row["arch"], "identity": row["identity"], "version": row["version"], "artifact": {"url": base + "/" + row["artifact"]["key"], "sha256": row["artifact"]["sha256"]}, **{k: row[k] for k in ("teamId", "publisher") if k in row}})
            if row["platform"] == "win32":
                accepted["packages"][-1]["applicationId"] = identity["appNamePascal"]
        channel_releases.match_accepted_packages(manifest, accepted)
        wrong = deepcopy(accepted)
        wrong["packages"][0]["artifact"]["sha256"] = "0" * 64
        with pytest.raises(ChannelError, match="accepted"):
            channel_releases.match_accepted_packages(manifest, wrong)
        with pytest.raises(ChannelError, match="every native"):
            channel_releases.match_accepted_packages(dict(manifest, packages=manifest["packages"][:1]), accepted)
        # Exercise the real controller, HTTP receipt downloader, immutable feeds,
        # manifest and final CAS; only GitHub admission and generated product facts
        # are fixture inputs (no native signature acceptance is claimed here).
        from scripts.releases import r2, commit_build
        monkeypatch.setattr(channel_releases, "admit_transaction", lambda policy, env: (tag, commit))
        monkeypatch.setattr(channel_releases, "accepted_stable", lambda *args: accepted)
        monkeypatch.setattr(channel_releases, "product_identity", lambda tag: dict(identity))
        monkeypatch.setattr(commit_build, "version_at", lambda *args: "2.0.0")
        monkeypatch.setattr(r2, "credentials", lambda: (pub.store.creds, url, "bucket"))
        monkeypatch.setattr(r2, "public_base_url", lambda: base)
        def put(**kwargs):
            pub.store.put(kwargs["key"], __import__("pathlib").Path(kwargs["file"]).read_bytes())
        monkeypatch.setattr(r2, "put", put)
        result = channel_releases.publish_release("stable-release", {"GITHUB_REPOSITORY": pub.repository}, tmp_path / "downloaded")
        assert result["name"] == "released"
        assert pub.reader.resolve("released").manifest == manifest
        before = dict(objects)
        assert channel_releases.publish_release("stable-release", {"GITHUB_REPOSITORY": pub.repository}, tmp_path / "retry") == result
        assert objects == before
        (tmp_path / bundle_name).write_bytes(b"corrupt")
        with pytest.raises(ChannelError, match="receipt"):
            channel_releases.read_native_receipts(tmp_path, tag, commit)


def test_protected_transaction_refuses_custom_workflow_failed_gate_and_unpublished_release():
    from scripts.releases import channel_releases
    from hermes_cli.release_channels import ChannelError
    tag, commit = "v2.0.0", "a" * 40
    env = {"GITHUB_ACTIONS": "true", "GITHUB_EVENT_NAME": "workflow_dispatch",
           "GITHUB_REPOSITORY": "example/hermes-agent", "GITHUB_SHA": commit,
           "GITHUB_REF": "refs/tags/" + tag, "RELEASE_TAG": tag,
           "GITHUB_WORKFLOW_REF": "example/hermes-agent/.github/workflows/stable-release.yml@refs/tags/" + tag,
           "RELEASE_NEEDS": json.dumps({key: {"result": "success"} for key in channel_releases.STABLE_NEEDS})}
    published = {"tagName": tag, "isDraft": False, "isPrerelease": False}

    def run(command):
        if command[:3] == ["gh", "release", "view"]:
            return json.dumps(published)
        if command[:2] == ["git", "ls-remote"]:
            return commit + "\trefs/tags/" + tag
        if command[:2] == ["git", "rev-parse"]:
            return commit
        return ""

    assert channel_releases.admit_transaction("stable-release", env, run=run) == (tag, commit)
    published["isDraft"] = True
    with pytest.raises(ChannelError, match="published"):
        channel_releases.admit_transaction("stable-release", env, run=run)
    published["isDraft"] = False
    with pytest.raises(ChannelError, match="workflow"):
        channel_releases.admit_transaction("stable-release", dict(env, GITHUB_WORKFLOW_REF="custom.yml"), run=run)
    with pytest.raises(ValueError, match="blocked"):
        channel_releases.admit_transaction("stable-release", dict(env, RELEASE_NEEDS="{}"), run=run)
    from hermes_cli.release_channels import canonical_json
    with object_server() as (url, objects, headers, requests, faults):
        pub = publisher(url)
        # Exercise HTTPS authority validation through the loopback transport.
        pub.public_base = "https://releases.example"
        candidate = {"schema": 2, "tag": tag, "commit": commit,
                     "smoke_results": {job: {"result": "success"} for job in channel_releases.stable.SMOKE_JOBS},
                     "packages": []}
        for platform in ("macos", "windows"):
            for arch in ("arm64", "x64"):
                candidate["packages"].append({"platform": platform, "arch": arch, "tag": tag, "commit": commit,
                    "version": "2.0.0" if platform == "macos" else "2.0.0.0", "identity": "fixture.identity",
                    "teamId": "ABCDEFGHIJ", "publisher": "CN=Fixture", "applicationId": "Fixture",
                    "artifact": {"url": f"{pub.public_base}/releases/tag/{tag}/fixture-{arch}." + ("zip" if platform == "macos" else "msixbundle"), "sha256": "d" * 64}})
        raw = canonical_json(candidate)
        key = f"releases/tag/{tag}/release-candidates.json"
        objects[key] = raw
        candidate_env = {"CANDIDATE_MANIFEST_SHA256": hashlib.sha256(raw).hexdigest(), "CANDIDATE_MANIFEST_URL": pub.public_base + "/" + key}
        with pytest.raises(ChannelError, match="transaction"):
            channel_releases.accepted_stable(pub, candidate_env, tag, commit)
        objects["releases/stable/release-candidates.json"] = raw
        assert channel_releases.accepted_stable(pub, candidate_env, tag, commit) == candidate
        faults["stale_public"] = b"{}"
        with pytest.raises(ChannelError):
            channel_releases.accepted_stable(pub, candidate_env, tag, commit)


def test_request_inputs_are_rejected_before_allocating():
    from hermes_cli.release_channels import ChannelError
    with object_server() as (url, objects, headers, requests, faults):
        pub = publisher(url)
        pub.create("validation")
        before = dict(objects)
        with pytest.raises(ValueError):
            pub.allocate("validation", "a" * 40, "1.0.0", [])
        assert objects == before
        with pytest.raises(ChannelError):
            pub.allocate("validation", "not-a-sha", "1.0.0")
        assert objects == before
