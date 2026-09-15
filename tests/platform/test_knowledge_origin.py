"""Mirroring a corpus from a bucket.

An S3 origin is the first time a corpus's contents arrive from somewhere NOVA's operator
does not fully control. Object keys come from the bucket, and a key is a string — so the
properties worth pinning are the ones that hold when the bucket is hostile: a key that
climbs out of the corpus root writes nothing, a key the corpus does not accept is refused
and *named*, and an object over the declared cap never lands.

The second half is about honesty of the mirror. A mirrored corpus is not a place to put
documents: an upload into one survives exactly until the next sync, so the upload route
refuses rather than quietly losing the file later.

boto3 is stubbed. The point here is NOVA's handling of what a bucket returns, and a test
that needed real AWS credentials would be a test nobody runs.
"""

from __future__ import annotations

import base64
import json
import shutil
import sys
import threading
import types
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from nova.audit import AuditLog
from nova.control import ControlAPI
from nova.control.auth import Principal
from nova.errors import SpecError
from nova.knowledge.origin import Origin, sync
from nova.knowledge.sources import KnowledgeSource
from nova.runtime import get_runtime
from nova.spec import load_bundle

ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "nova" / "examples" / "acme"
ADMIN = Principal(name="ops", role="admin")
SOURCE = "company-handbook"


class FakeS3:
    """The two calls :func:`nova.knowledge.origin.sync` makes, over an in-memory bucket."""

    def __init__(self, objects, *, fail=None):
        # {key: bytes}
        self.objects = dict(objects)
        self.fail = fail
        self.downloads: list[str] = []

    # boto3's paginator protocol, reduced to what sync uses.
    def get_paginator(self, name):
        assert name == "list_objects_v2"
        return self

    def paginate(self, *, Bucket, Prefix=None):
        if self.fail:
            raise self.fail
        contents = [
            {"Key": key, "Size": len(body), "ETag": f'"{_etag(body)}"'}
            for key, body in sorted(self.objects.items())
            if Prefix is None or key.startswith(Prefix)
        ]
        # Two pages, so pagination is actually exercised.
        yield {"Contents": contents[:1]}
        yield {"Contents": contents[1:]}

    def download_file(self, bucket, key, target):
        self.downloads.append(key)
        Path(target).write_bytes(self.objects[key])


def _etag(body: bytes) -> str:
    import hashlib

    return hashlib.md5(body).hexdigest()  # noqa: S324 — matching S3's ETag, not hashing a secret


@pytest.fixture
def stub_boto3(monkeypatch):
    """Install a fake ``boto3`` and hand the test the client it will hand NOVA."""
    holder = {}

    def install(client):
        holder["client"] = client
        module = types.ModuleType("boto3")
        module.client = lambda service, **kwargs: (  # noqa: ARG005
            holder.setdefault("kwargs", kwargs) is not None and client
        ) and client
        monkeypatch.setitem(sys.modules, "boto3", module)
        return client

    install.holder = holder  # type: ignore[attr-defined]
    return install


def _source(tmp_path, **origin_kwargs):
    root = tmp_path / "mirror"
    return KnowledgeSource(
        id="mirrored", root=root, include=("**/*.md",),
        max_file_bytes=1024,
        origin=Origin(kind="s3", bucket="acme-docs", **origin_kwargs),
    )


# -- declaration ---------------------------------------------------------------


def test_a_corpus_without_an_origin_is_unchanged(tmp_path):
    bundle = load_bundle(EXAMPLE)
    assert [s.origin for s in bundle.knowledge.sources] == [None, None]


def test_an_origin_round_trips_through_the_declaration():
    from nova._fields import Doc

    doc = Doc({"type": "s3", "bucket": "acme-docs", "prefix": "handbook/", "region": "eu-west-2"})
    origin = Origin.parse(doc)
    assert origin.location == "s3://acme-docs/handbook"
    assert origin.to_dict()["region"] == "eu-west-2"
    # prune defaults on: a mirror that only ever added would keep answering from a
    # document the customer deleted.
    assert origin.prune is True


def test_an_origin_must_name_a_bucket():
    from nova._fields import Doc

    with pytest.raises(SpecError):
        Origin.parse(Doc({"type": "s3"}))


def test_an_unknown_origin_kind_is_refused():
    from nova._fields import Doc

    with pytest.raises(SpecError):
        Origin.parse(Doc({"type": "dropbox", "bucket": "x"}))


def test_a_declared_origin_survives_a_bundle_round_trip(tmp_path):
    import yaml

    root = tmp_path / "bundle"
    shutil.copytree(EXAMPLE, root)
    path = root / "knowledge.yaml"
    data = yaml.safe_load(path.read_text())
    data["sources"][0]["origin"] = {"type": "s3", "bucket": "acme-docs", "prefix": "hb/"}
    path.write_text(yaml.safe_dump(data))
    bundle = load_bundle(root)
    assert bundle.knowledge.get(SOURCE).origin.bucket == "acme-docs"


# -- syncing -------------------------------------------------------------------


def test_a_sync_downloads_what_the_bucket_holds(tmp_path, stub_boto3):
    client = stub_boto3(FakeS3({"a.md": b"# A\n", "b.md": b"# B\n"}))
    report = sync(_source(tmp_path))
    assert report.ok and report.downloaded == 2
    assert (tmp_path / "mirror" / "a.md").read_bytes() == b"# A\n"
    assert sorted(client.downloads) == ["a.md", "b.md"]


def test_a_second_sync_does_not_refetch_unchanged_objects(tmp_path, stub_boto3):
    client = stub_boto3(FakeS3({"a.md": b"# A\n"}))
    sync(_source(tmp_path))
    client.downloads.clear()
    report = sync(_source(tmp_path))
    assert report.unchanged == 1 and report.downloaded == 0
    assert client.downloads == []


def test_a_changed_object_is_refetched(tmp_path, stub_boto3):
    client = stub_boto3(FakeS3({"a.md": b"# A\n"}))
    sync(_source(tmp_path))
    client.objects["a.md"] = b"# A revised\n"
    report = sync(_source(tmp_path))
    assert report.downloaded == 1
    assert (tmp_path / "mirror" / "a.md").read_bytes() == b"# A revised\n"


def test_a_locally_deleted_file_is_restored(tmp_path, stub_boto3):
    stub_boto3(FakeS3({"a.md": b"# A\n"}))
    sync(_source(tmp_path))
    (tmp_path / "mirror" / "a.md").unlink()
    # The marker still records the ETag, so only the missing file on disk can catch this.
    report = sync(_source(tmp_path))
    assert report.downloaded == 1 and (tmp_path / "mirror" / "a.md").is_file()


def test_pruning_removes_what_the_bucket_no_longer_has(tmp_path, stub_boto3):
    client = stub_boto3(FakeS3({"a.md": b"# A\n", "b.md": b"# B\n"}))
    sync(_source(tmp_path))
    del client.objects["b.md"]
    report = sync(_source(tmp_path))
    assert report.removed == 1
    assert not (tmp_path / "mirror" / "b.md").exists()


def test_pruning_can_be_declined(tmp_path, stub_boto3):
    client = stub_boto3(FakeS3({"a.md": b"# A\n", "b.md": b"# B\n"}))
    sync(_source(tmp_path, prune=False))
    del client.objects["b.md"]
    report = sync(_source(tmp_path, prune=False))
    assert report.removed == 0
    assert (tmp_path / "mirror" / "b.md").is_file()


def test_a_prefix_is_stripped_from_the_local_path(tmp_path, stub_boto3):
    stub_boto3(FakeS3({"handbook/a.md": b"# A\n", "other/b.md": b"# B\n"}))
    report = sync(_source(tmp_path, prefix="handbook/"))
    assert report.downloaded == 1
    assert (tmp_path / "mirror" / "a.md").is_file()
    assert not (tmp_path / "mirror" / "handbook").exists()


def test_nested_keys_keep_their_shape(tmp_path, stub_boto3):
    stub_boto3(FakeS3({"policies/refunds.md": b"# R\n"}))
    sync(_source(tmp_path))
    assert (tmp_path / "mirror" / "policies" / "refunds.md").is_file()


def test_a_dry_run_writes_nothing(tmp_path, stub_boto3):
    client = stub_boto3(FakeS3({"a.md": b"# A\n"}))
    report = sync(_source(tmp_path), dry_run=True)
    assert report.downloaded == 1
    assert client.downloads == []
    assert not (tmp_path / "mirror" / "a.md").exists()
    assert not (tmp_path / "mirror" / ".nova-sync.json").exists()


# -- refusals ------------------------------------------------------------------


def test_a_key_that_climbs_out_of_the_root_is_refused(tmp_path, stub_boto3):
    stub_boto3(FakeS3({"../../escape.md": b"owned\n"}))
    report = sync(_source(tmp_path))
    assert report.downloaded == 0
    assert any("outside" in row["reason"] for row in report.skipped)
    assert not (tmp_path / "escape.md").exists()
    assert not (tmp_path.parent / "escape.md").exists()


def test_an_absolute_key_does_not_escape(tmp_path, stub_boto3):
    stub_boto3(FakeS3({"/etc/cron.d/owned.md": b"x\n"}))
    sync(_source(tmp_path))
    # Whatever it did, it did not write outside the corpus.
    assert not Path("/etc/cron.d/owned.md").exists()


def test_an_object_the_corpus_does_not_accept_is_named(tmp_path, stub_boto3):
    stub_boto3(FakeS3({"a.md": b"# A\n", "ledger.xlsx": b"binary\n"}))
    report = sync(_source(tmp_path))
    assert report.downloaded == 1
    assert [row["document"] for row in report.skipped] == ["ledger.xlsx"]
    assert not (tmp_path / "mirror" / "ledger.xlsx").exists()


def test_an_oversized_object_is_refused_before_it_is_downloaded(tmp_path, stub_boto3):
    client = stub_boto3(FakeS3({"huge.md": b"x" * 2048}))
    report = sync(_source(tmp_path))
    assert client.downloads == []
    assert any("max_file_bytes" in row["reason"] for row in report.skipped)


def test_a_bucket_failure_is_a_report_not_a_crash(tmp_path, stub_boto3):
    stub_boto3(FakeS3({}, fail=RuntimeError("AccessDenied")))
    report = sync(_source(tmp_path))
    assert not report.ok and "AccessDenied" in report.error


def test_a_missing_boto3_names_itself(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "boto3", None)
    report = sync(_source(tmp_path))
    assert not report.ok and "boto3" in report.error


def test_syncing_a_corpus_with_no_origin_refuses(tmp_path):
    source = KnowledgeSource(id="local", root=tmp_path / "docs")
    with pytest.raises(SpecError):
        sync(source)


# -- through the Control Centre ------------------------------------------------


@pytest.fixture
def live(tmp_path, monkeypatch):
    import yaml

    root = tmp_path / "bundle"
    shutil.copytree(EXAMPLE, root)
    path = root / "knowledge.yaml"
    data = yaml.safe_load(path.read_text())
    data["sources"][0]["origin"] = {"type": "s3", "bucket": "acme-docs", "prefix": "hb/"}
    path.write_text(yaml.safe_dump(data))

    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("NOVA_HOME", str(home))
    monkeypatch.setenv("HERMES_HOME", str(home))
    bundle = load_bundle(root)
    runtime = get_runtime("hermes", home=home, tenant_id=bundle.tenant_id)
    audit = AuditLog.for_home(home, tenant_id=bundle.tenant_id, actor="test")
    return {"api": ControlAPI(bundle, runtime, audit=audit), "home": home, "root": root}


def test_the_sync_route_mirrors_and_indexes(live, stub_boto3):
    stub_boto3(FakeS3({"hb/refunds.md": b"# Refunds\n\nWithin 14 days.\n"}))
    response = live["api"].write(f"/platform/v1/knowledge/{SOURCE}/sync", ADMIN, {})
    assert response.status == 200, response.body
    assert response.body["sync"]["downloaded"] == 1
    # Downloaded is not the same claim as searchable, so both are reported.
    assert response.body["index"]["ok"] is True


def test_a_viewer_cannot_sync(live, stub_boto3):
    stub_boto3(FakeS3({"hb/a.md": b"# A\n"}))
    response = live["api"].write(
        f"/platform/v1/knowledge/{SOURCE}/sync", Principal(name="w", role="viewer"), {}
    )
    assert response.status == 403


def test_uploading_into_a_mirror_is_refused(live, stub_boto3):
    response = live["api"].write(f"/platform/v1/knowledge/{SOURCE}/upload", ADMIN, {
        "filename": "note.md", "data": base64.b64encode(b"# Note\n").decode(),
    })
    assert response.status == 409
    assert "s3://acme-docs/hb" in response.body["error"]["message"]


def test_removing_from_a_mirror_is_refused(live):
    response = live["api"].write(f"/platform/v1/knowledge/{SOURCE}/remove", ADMIN,
                                 {"name": "refunds.md"})
    assert response.status == 409


def test_syncing_a_local_corpus_says_what_is_missing(live):
    response = live["api"].write("/platform/v1/knowledge/product-docs/sync", ADMIN, {})
    assert response.status == 409
    assert "origin" in response.body["error"]["message"]


def test_the_corpus_screen_reports_its_origin(live):
    body = live["api"].handle(f"/platform/v1/knowledge/{SOURCE}/documents").body
    assert body["origin"]["bucket"] == "acme-docs"
    assert live["api"].handle(
        "/platform/v1/knowledge/product-docs/documents"
    ).body["origin"] is None


def test_a_sync_is_audited_intent_then_committed(live, stub_boto3):
    stub_boto3(FakeS3({"hb/a.md": b"# A\n"}))
    live["api"].write(f"/platform/v1/knowledge/{SOURCE}/sync", ADMIN, {})
    lines = [json.loads(line) for line in
             (live["home"] / "nova" / "audit.jsonl").read_text().splitlines()]
    synced = [row for row in lines if row["kind"] == "knowledge.synced"]
    assert [row["phase"] for row in synced] == ["intent", "committed"]
    assert synced[0]["detail"]["actor"] == "ops"


def test_a_failed_sync_is_audited_as_failed(live, stub_boto3):
    stub_boto3(FakeS3({}, fail=RuntimeError("AccessDenied")))
    response = live["api"].write(f"/platform/v1/knowledge/{SOURCE}/sync", ADMIN, {})
    assert response.body["ok"] is False
    lines = [json.loads(line) for line in
             (live["home"] / "nova" / "audit.jsonl").read_text().splitlines()]
    synced = [row for row in lines if row["kind"] == "knowledge.synced"]
    assert synced[-1]["phase"] == "failed"


# -- NOVA's own bookkeeping is not a customer document -------------------------


def test_the_sync_marker_is_never_ingested(tmp_path, stub_boto3):
    """The marker records which object was downloaded at which ETag, so it has to live in
    the corpus root. That makes it the one file in there that is not a document, and a
    corpus declaring the permissive ``**/*`` would otherwise index NOVA's bookkeeping and
    let an agent cite it."""
    from nova.knowledge.sources import iter_documents
    from nova.knowledge.store import list_documents

    stub_boto3(FakeS3({"a.md": b"# A\n"}))
    source = KnowledgeSource(
        id="mirrored", root=tmp_path / "mirror", include=("**/*",), max_file_bytes=1024,
        origin=Origin(kind="s3", bucket="acme-docs"),
    )
    sync(source)
    assert (tmp_path / "mirror" / ".nova-sync.json").is_file()
    assert [p.name for p in iter_documents(source).files] == ["a.md"]
    assert [row["name"] for row in list_documents(source)] == ["a.md"]


def test_the_marker_is_not_reported_as_a_skipped_file(tmp_path, stub_boto3):
    """It is not something the customer put there, so naming it in an ingest report would
    be NOVA telling an operator about NOVA."""
    from nova.knowledge.sources import iter_documents

    stub_boto3(FakeS3({"a.md": b"# A\n"}))
    source = KnowledgeSource(
        id="mirrored", root=tmp_path / "mirror", include=("**/*",), max_file_bytes=1024,
        origin=Origin(kind="s3", bucket="acme-docs"),
    )
    sync(source)
    assert not [s for s in iter_documents(source).skipped if ".nova-sync" in str(s.path)]


def test_a_local_corpus_still_indexes_its_own_dotfiles(tmp_path):
    """The exclusion is by name, not by a dot-file rule.

    A customer whose corpus legitimately holds ``.eslintrc.md`` should get it indexed; a
    rule broad enough to catch NOVA's marker would quietly drop theirs too.
    """
    from nova.knowledge.sources import iter_documents

    root = tmp_path / "docs"
    root.mkdir()
    (root / ".eslintrc.md").write_text("# Config\n")
    source = KnowledgeSource(id="local", root=root, include=("**/*",))
    assert [p.name for p in iter_documents(source).files] == [".eslintrc.md"]


# -- against the real protocol -------------------------------------------------
#
# Everything above stubs boto3, which proves NOVA's handling of what a bucket returns and
# proves nothing about whether the calls are shaped the way boto3 expects. These drive a
# real boto3 client against a minimal in-process S3 server, so a wrong keyword argument or
# a misread paginator fails here instead of in a customer's deployment.

boto3 = pytest.importorskip("boto3", reason="the S3 origin is an optional capability")


class _S3Protocol(BaseHTTPRequestHandler):
    """ListObjectsV2 and GetObject, enough for a real client. Signatures are ignored:
    what is under test is NOVA, not AWS's authentication."""

    objects: dict[str, bytes] = {}

    def log_message(self, *args):  # noqa: A003 — silence the default stderr logging
        pass

    def do_GET(self):  # noqa: N802 — BaseHTTPRequestHandler's naming
        from urllib.parse import parse_qs, unquote, urlparse
        from xml.sax.saxutils import escape

        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)
        parts = [p for p in parsed.path.split("/") if p]
        if query.get("list-type") == ["2"]:
            prefix = (query.get("prefix") or [""])[0]
            rows = "".join(
                f"<Contents><Key>{escape(key)}</Key><Size>{len(body)}</Size>"
                f"<ETag>&quot;{_etag(body)}&quot;</ETag></Contents>"
                for key, body in sorted(self.objects.items()) if key.startswith(prefix)
            )
            payload = (
                '<?xml version="1.0" encoding="UTF-8"?>'
                '<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">'
                "<Name>acme</Name><MaxKeys>1000</MaxKeys>"
                f"<Prefix>{escape(prefix)}</Prefix><KeyCount>{len(self.objects)}</KeyCount>"
                f"<IsTruncated>false</IsTruncated>{rows}</ListBucketResult>"
            ).encode()
        else:
            key = unquote("/".join(parts[1:]))
            if key not in self.objects:
                self.send_response(404)
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            payload = self.objects[key]
        self.send_response(200)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    do_HEAD = do_GET


@pytest.fixture
def bucket(monkeypatch):
    """A real S3 endpoint on loopback, and the Origin pointing at it."""
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "test")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test")
    handler = type("Handler", (_S3Protocol,), {"objects": {}})
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    port = server.server_address[1]
    try:
        yield handler.objects, Origin(
            kind="s3", bucket="acme", prefix="published/", region="us-east-1",
            endpoint_url=f"http://127.0.0.1:{port}",
        )
    finally:
        server.shutdown()


def test_a_real_client_mirrors_the_bucket(tmp_path, bucket):
    objects, origin = bucket
    objects.update({
        "published/refunds.md": b"# Refunds\n\nWithin 14 days.\n",
        "published/policies/escalation.md": b"# Escalation\n\nTier 2 after 2 hours.\n",
        "published/ledger.xlsx": b"binary",
        "published/../../escape.md": b"owned",
    })
    source = KnowledgeSource(
        id="handbook", root=tmp_path / "docs", include=("**/*.md",),
        max_file_bytes=4096, origin=origin,
    )
    report = sync(source)
    assert report.ok and report.downloaded == 2, report.to_dict()
    assert (tmp_path / "docs" / "refunds.md").exists()
    assert (tmp_path / "docs" / "policies" / "escalation.md").exists()
    # The two refusals, against a real listing rather than a hand-built one.
    reasons = {row["document"]: row["reason"] for row in report.skipped}
    assert "outside" in reasons["../../escape.md"]
    assert "not accepted" in reasons["ledger.xlsx"]
    assert not (tmp_path / "escape.md").exists()
    assert not (tmp_path.parent / "escape.md").exists()


def test_a_real_client_skips_unchanged_objects_and_prunes(tmp_path, bucket):
    objects, origin = bucket
    objects.update({"published/a.md": b"# A\n", "published/b.md": b"# B\n"})
    source = KnowledgeSource(
        id="handbook", root=tmp_path / "docs", include=("**/*.md",),
        max_file_bytes=4096, origin=origin,
    )
    assert sync(source).downloaded == 2
    assert sync(source).unchanged == 2

    del objects["published/b.md"]
    report = sync(source)
    assert report.removed == 1
    assert not (tmp_path / "docs" / "b.md").exists()
