"""F5 provenance contracts: trusted inputs, prepare receipt, exception policy,
raw blob extraction through `cat-file --batch` (NOT_RUN in the authoring lane).
Git is a fake boundary; everything else runs for real against tmp_path."""

import hashlib
import io
import json

import pytest

from isolated_tests import inputs, main, policy, prepare

# inputs imports pwd and relies on POSIX uid/mode bits (0600 owner checks, chmod).
pytestmark = pytest.mark.platforms("posix")

BASE = "b" * 40
RUN_TESTS = b"#!/bin/sh\nexit 0\n"
RUNNER_FILE = b"print('runner')\n"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _ls_tree(files: dict) -> bytes:
    out = b""
    for rel, body in files.items():
        mode = "100755" if rel.endswith(".sh") else "100644"
        out += f"{mode} blob {prepare._blob_id(body)}\t{rel}".encode() + b"\0"
    return out


def _batch_object(oid: str, body: bytes) -> bytes:
    return f"{oid} blob {len(body)}\n".encode() + body + b"\n"


class FakeGit:
    """ls-tree and cat-file --batch with git's wire formats; `responses` replaces the
    batch output for one oid with raw bytes, `tree` replaces the ls-tree output."""

    def __init__(self, files: dict, tree: bytes | None = None, responses: dict | None = None):
        self.files = files
        self.tree = tree
        self.responses = responses or {}
        self.by_oid = {prepare._blob_id(b): b for b in files.values()}

    def ls_tree(self, commit):
        assert commit == BASE
        return self.tree if self.tree is not None else _ls_tree(self.files)

    def cat_file_batch(self, oids):
        return io.BytesIO(b"".join(
            self.responses.get(oid) or _batch_object(oid, self.by_oid[oid]) for oid in oids))

    def apply(self, workdir, patch):  # writes what the patch's manifest pins
        (workdir / "app.py").chmod(0o600)
        (workdir / "app.py").write_bytes(b"patched\n")


BASE_FILES = {"scripts/run_tests.sh": RUN_TESTS, "app.py": b"orig\n", "web/.npmrc": b"registry\n"}


def _trusted(tmp_path, *, patches=(), exceptions=None, max_count=4, runner=None):
    doc = {"schema": inputs.TRUSTED_INPUTS_SCHEMA, "base_commit": BASE,
           "candidate_patches": list(patches),
           "runner_allowlist": runner if runner is not None else
           {"scripts/isolated_tests/new.py": _sha(RUNNER_FILE)},
           "exception_policy": {"allowed": exceptions if exceptions is not None else
                                {"web/.npmrc": _sha(b"registry\n")}, "max_count": max_count}}
    path = tmp_path / "trusted.json"
    path.write_text(json.dumps(doc))
    path.chmod(0o600)
    return path


def _runner_src(tmp_path):
    src = tmp_path / "runner-src"
    (src / "scripts" / "isolated_tests").mkdir(parents=True)
    (src / "scripts" / "isolated_tests" / "new.py").write_bytes(RUNNER_FILE)
    (src / "scripts" / "isolated_tests" / "extra.py").write_bytes(b"not allowlisted\n")
    return src


def _prepare(tmp_path, git=None, **kw):
    trusted = _trusted(tmp_path, **kw)
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    out = tmp_path / "receipts"
    out.mkdir()
    receipt = prepare.prepare(str(trusted), git or FakeGit(dict(BASE_FILES)),
                              str(_runner_src(tmp_path)), str(scratch), str(out / "r.json"))
    return trusted, out / "r.json", receipt


def test_prepare_builds_snapshot_bound_to_trusted_inputs(tmp_path):
    trusted_path, receipt_path, receipt = _prepare(tmp_path)
    trusted = inputs.load_trusted_inputs(str(trusted_path))
    doc = inputs.read_receipt(str(receipt_path), trusted)
    snap = inputs.verify_snapshot(receipt["snapshot_root"], doc["applied_exceptions"])
    assert inputs.verify_receipt(doc, trusted, snap)["no_candidate"] is True
    assert doc["applied_exceptions"] == {"web/.npmrc": _sha(b"registry\n")}
    assert "scripts/isolated_tests/extra.py" not in json.loads(
        (snap.root / inputs.MANIFEST_NAME).read_bytes())["files"]


def test_receipt_from_other_inputs_is_invalid(tmp_path):
    _, receipt_path, _ = _prepare(tmp_path)
    other = tmp_path / "other"
    other.mkdir()
    changed = inputs.load_trusted_inputs(str(_trusted(other, max_count=5)))
    with pytest.raises(inputs.ProvenanceInvalid, match="does not match the trusted inputs"):
        inputs.read_receipt(str(receipt_path), changed)


def test_receipt_manifest_mismatch_is_invalid(tmp_path):
    trusted_path, receipt_path, receipt = _prepare(tmp_path)
    trusted = inputs.load_trusted_inputs(str(trusted_path))
    doc = inputs.read_receipt(str(receipt_path), trusted)
    snap = inputs.verify_snapshot(receipt["snapshot_root"], doc["applied_exceptions"])
    with pytest.raises(inputs.ProvenanceInvalid, match="manifest_sha256"):
        inputs.verify_receipt({**doc, "manifest_sha256": "0" * 64}, trusted, snap)


@pytest.mark.parametrize("edit", [
    {"no_candidate": False},
    {"result": "PARTIAL"},
    {"applied_exceptions": {"web/.npmrc": "0" * 64}},
    {"applied_exceptions": {"other/.npmrc": "1" * 64}},
    {"schema": "hermes-isolated-receipt/1"},
])
def test_receipt_tampering_is_invalid(tmp_path, edit):
    trusted_path, receipt_path, _ = _prepare(tmp_path)
    doc = json.loads(receipt_path.read_bytes())
    receipt_path.chmod(0o600)
    receipt_path.write_text(json.dumps({**doc, **edit}))
    with pytest.raises(inputs.ProvenanceInvalid):
        inputs.read_receipt(str(receipt_path), inputs.load_trusted_inputs(str(trusted_path)))


@pytest.mark.parametrize("raw", [b"{not json", b"[]", b'{"schema": "x"}'])
def test_malformed_receipt_is_invalid_not_empty(tmp_path, raw):
    trusted = inputs.load_trusted_inputs(str(_trusted(tmp_path)))
    bad = tmp_path / "receipt.json"
    bad.write_bytes(raw)
    bad.chmod(0o600)
    with pytest.raises(inputs.ProvenanceInvalid):
        inputs.read_receipt(str(bad), trusted)


@pytest.mark.parametrize("rel", [".env", "keys/id_rsa", "certs/server.pem", "a/auth.json",
                                 "data/state.db", ".git/config", "x/.env.local"])
def test_policy_cannot_except_forbidden_patterns(tmp_path, rel):
    with pytest.raises(inputs.ProvenanceInvalid, match="never-exceptable"):
        inputs.load_trusted_inputs(str(_trusted(tmp_path, exceptions={rel: "a" * 64})))


@pytest.mark.parametrize("rel", ["home/.netrc", ".pypirc", "cfg/.docker/config.json",
                                 "keys/id_dsa", "KEYS/ID_RSA", "x/.ENV"])
def test_policy_cannot_except_credential_stores(tmp_path, rel):
    with pytest.raises(inputs.ProvenanceInvalid, match="never-exceptable"):
        inputs.load_trusted_inputs(str(_trusted(tmp_path, exceptions={rel: "a" * 64})))


def test_env_example_stays_exceptable(tmp_path):
    rel = "web/.env.example"
    allowed = inputs.load_trusted_inputs(str(_trusted(tmp_path, exceptions={rel: "a" * 64})))
    assert dict(allowed.exceptions) == {rel: "a" * 64}


def test_blob_id_known_answers():
    """Literal git blob ids, independent of the implementation under test."""
    assert prepare._blob_id(b"") == "e69de29bb2d1d6434b8b29ae775ad8c2e48c5391"
    assert prepare._blob_id(b"hello\n") == "ce013625030ba8dba906f756967f9e9ca394464a"


def test_canonical_digest_known_answer():
    """sha256 of '{"a":["x"],"b":1}': keys sorted, no whitespace."""
    assert inputs._canon_sha({"b": 1, "a": ["x"]}) == (
        "9bfce180614e8aaffe8a52084e7be3ee3b4c2e740cae2793550177f953f11c04")


def test_policy_over_max_count_is_invalid(tmp_path):
    with pytest.raises(inputs.ProvenanceInvalid, match="max_count"):
        inputs.load_trusted_inputs(str(_trusted(tmp_path, max_count=0)))


def test_denied_file_outside_policy_refuses_prepare(tmp_path):
    with pytest.raises(inputs.Refused, match="denied name outside the approved policy"):
        _prepare(tmp_path, exceptions={})


def test_denied_file_with_wrong_hash_refuses_prepare(tmp_path):
    with pytest.raises(inputs.Refused, match="denied name outside the approved policy"):
        _prepare(tmp_path, exceptions={"web/.npmrc": "0" * 64})


@pytest.mark.parametrize("rel", ["/etc/passwd", "../escape.py", "a/../../escape.py"])
def test_unsafe_tree_path_refused(tmp_path, rel):
    files = {**BASE_FILES, rel: b"x"}
    with pytest.raises(inputs.Refused, match="unsafe path"):
        _prepare(tmp_path, git=FakeGit(files))
    # The snapshot root is scratch/snapshot-*/, so "../escape.py" would land in scratch.
    assert not (tmp_path / "scratch" / "escape.py").exists()
    assert not (tmp_path / "escape.py").exists()


@pytest.mark.parametrize("record", [
    "120000 blob {oid}\tlink",
    "160000 commit {oid}\tsubmodule",
    "040000 tree {oid}\tdir",
    "100664 blob {oid}\tgroup_writable.py",
])
def test_link_gitlink_and_other_modes_refused(tmp_path, record):
    tree = _ls_tree(BASE_FILES) + record.format(oid="a" * 40).encode() + b"\0"
    with pytest.raises(inputs.Refused, match="non-regular entry"):
        _prepare(tmp_path, git=FakeGit(dict(BASE_FILES), tree=tree))


def test_blob_bytes_not_matching_oid_refused(tmp_path):
    oid = prepare._blob_id(b"orig\n")
    git = FakeGit(dict(BASE_FILES), responses={oid: _batch_object(oid, b"oriG\n")})
    with pytest.raises(inputs.Refused, match="bytes differ from git object id"):
        _prepare(tmp_path, git=git)


def test_missing_object_refused(tmp_path):
    oid = prepare._blob_id(b"orig\n")
    git = FakeGit(dict(BASE_FILES), responses={oid: f"{oid} missing\n".encode()})
    with pytest.raises(inputs.Refused, match="missing object"):
        _prepare(tmp_path, git=git)


@pytest.mark.parametrize("header", ["{other} blob 5\n", "{oid} tree 5\n", "{oid} blob -5\n",
                                    "{oid} blob\n", "{oid} blob 5"])
def test_header_oid_or_type_mismatch_refused(tmp_path, header):
    oid = prepare._blob_id(b"orig\n")
    raw = header.format(oid=oid, other="c" * 40).encode() + b"orig\n\n"
    with pytest.raises(inputs.Refused, match="unexpected header|truncated or oversized header"):
        _prepare(tmp_path, git=FakeGit(dict(BASE_FILES), responses={oid: raw}))


def test_truncated_object_refused(tmp_path):
    oid = prepare._blob_id(b"orig\n")
    git = FakeGit(dict(BASE_FILES), responses={oid: f"{oid} blob 50\n".encode() + b"orig\n"})
    with pytest.raises(inputs.Refused):
        _prepare(tmp_path, git=git)


def test_blob_over_file_size_limit_refused_before_body_read(tmp_path, monkeypatch):
    monkeypatch.setattr(prepare, "PREPARE_LIMITS", {**policy.PREPARE_LIMITS, "max_file_bytes": 4})
    body = b"12345"
    entries = [("a.py", False, prepare._blob_id(body))]
    stream = io.BytesIO(_batch_object(entries[0][2], body))
    with pytest.raises(inputs.Refused, match="too large"):
        prepare.extract_blobs(entries, stream, tmp_path, prepare._Budget())
    assert not (tmp_path / "a.py").exists()


def test_blobs_over_total_bytes_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(prepare, "PREPARE_LIMITS", {**policy.PREPARE_LIMITS, "max_total_bytes": 6})
    entries = [(f"{n}.py", False, prepare._blob_id(b"1234")) for n in "ab"]
    stream = io.BytesIO(b"".join(_batch_object(oid, b"1234") for _, _, oid in entries))
    with pytest.raises(inputs.Refused, match="total bytes"):
        prepare.extract_blobs(entries, stream, tmp_path, prepare._Budget())


def test_trailing_batch_output_refused(tmp_path):
    body = b"x\n"
    entries = [("a.py", False, prepare._blob_id(body))]
    stream = io.BytesIO(_batch_object(entries[0][2], body) + b"extra")
    with pytest.raises(inputs.Refused, match="trailing output"):
        prepare.extract_blobs(entries, stream, tmp_path, prepare._Budget())


def test_snapshot_holds_raw_blob_bytes_regardless_of_gitattributes(tmp_path):
    """A `*.ps1 text eol=crlf` attribute in the tree does not convert anything: the
    snapshot file is the blob's LF bytes, and the exec bit follows the tree mode."""
    files = {**BASE_FILES, ".gitattributes": b"*.ps1 text eol=crlf\n",
             "activate.ps1": b"Write-Host one\nWrite-Host two\n"}
    _, _, receipt = _prepare(tmp_path, git=FakeGit(files))
    snap = tmp_path / "scratch" / receipt["snapshot_root"].rsplit("/", 1)[1]
    assert (snap / "activate.ps1").read_bytes() == b"Write-Host one\nWrite-Host two\n"
    assert (snap / "scripts" / "run_tests.sh").stat().st_mode & 0o111
    assert not (snap / "app.py").stat().st_mode & 0o111


def test_runner_allowlist_missing_file_refused(tmp_path):
    with pytest.raises(inputs.Refused):
        _prepare(tmp_path, runner={"scripts/isolated_tests/absent.py": _sha(b"x")})


def test_runner_allowlist_hash_mismatch_refused(tmp_path):
    with pytest.raises(inputs.Refused, match="sha256 mismatch"):
        _prepare(tmp_path, runner={"scripts/isolated_tests/new.py": "0" * 64})


def test_runner_allowlist_cannot_overwrite_base(tmp_path):
    with pytest.raises(inputs.Refused, match="already exists"):
        _prepare(tmp_path, runner={"app.py": _sha(b"orig\n")})


def _patch_file(tmp_path, text: str):
    path = tmp_path / "candidate.patch"
    path.write_text(text)
    path.chmod(0o600)
    return {"path": str(path), "sha256": _sha(text.encode()),
            "manifest": {"app.py": _sha(b"patched\n")}}


PATCH = "diff --git a/app.py b/app.py\n--- a/app.py\n+++ b/app.py\n@@ -1 +1 @@\n-orig\n+patched\n"


def test_patch_limited_to_manifest_and_hash(tmp_path):
    _, _, receipt = _prepare(tmp_path, patches=[_patch_file(tmp_path, PATCH)])
    assert receipt["no_candidate"] is False
    assert len(receipt["candidate_patches_sha256"]) == 1


def test_patch_touching_unlisted_path_refused(tmp_path):
    text = PATCH + PATCH.replace("app.py", "other.py")
    with pytest.raises(inputs.Refused, match="touched paths != manifest"):
        _prepare(tmp_path, patches=[_patch_file(tmp_path, text)])


def test_patch_bytes_not_matching_pinned_sha_refused(tmp_path):
    entry = {**_patch_file(tmp_path, PATCH), "sha256": "0" * 64}
    with pytest.raises(inputs.Refused, match="sha256 mismatch"):
        _prepare(tmp_path, patches=[entry])


def test_main_refuses_unparseable_receipt_as_invalid(tmp_path, monkeypatch):
    """F5-2: a receipt parse error is INVALID before any snapshot walk."""
    trusted = _trusted(tmp_path)
    bad = tmp_path / "receipt.json"
    bad.write_bytes(b"{broken")
    bad.chmod(0o600)
    walked = []
    monkeypatch.setattr(inputs, "verify_snapshot", lambda *a, **k: walked.append(a))
    rc = main.main(["plan", "--runtime", "docker", "--docker", "/usr/bin/docker",
                    "--socket", "/s.sock", "--image", "sha256:" + "a" * 64,
                    "--python", "/usr/bin/python3", "--snapshot", str(tmp_path),
                    "--suite", "isolated-selftests", "--out", str(tmp_path / "out"),
                    "--inputs", str(trusted), "--receipt", str(bad)])
    assert rc == policy.EXIT_INVALID
    assert walked == []
