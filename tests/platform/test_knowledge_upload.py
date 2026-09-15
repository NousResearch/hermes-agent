"""Putting documents into a corpus from the Control Centre.

Until now the only way a document reached a knowledge source was somebody putting a file on
the host's disk. The risk in changing that is not subtle — a control plane that writes
caller-named files into a directory agents read is a path-traversal question and a
what-may-be-stored question at the same time — so most of what follows is about those two.

The other property worth pinning is that an upload **indexes**. A document on disk that the
index has never seen is invisible to every agent, which is the state an upload that skipped
reindexing would leave behind while reporting success.
"""

from __future__ import annotations

import base64
import json
import shutil
from pathlib import Path

import pytest

from nova.audit import AuditLog
from nova.control import ControlAPI
from nova.control.auth import Principal
from nova.errors import SpecError
from nova.knowledge.store import accepts, list_documents, safe_name, store_document
from nova.runtime import get_runtime
from nova.spec import load_bundle

ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "nova" / "examples" / "acme"
ADMIN = Principal(name="ops", role="admin")
VIEWER = Principal(name="watcher", role="viewer")
SOURCE = "company-handbook"

DOC = b"# Returns\n\nRefunds are issued within 14 days of receipt.\n"


@pytest.fixture
def live(tmp_path, monkeypatch):
    root = tmp_path / "bundle"
    shutil.copytree(EXAMPLE, root)
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("NOVA_HOME", str(home))
    monkeypatch.setenv("HERMES_HOME", str(home))
    bundle = load_bundle(root)
    runtime = get_runtime("hermes", home=home, tenant_id=bundle.tenant_id)
    audit = AuditLog.for_home(home, tenant_id=bundle.tenant_id, actor="test")
    return {
        "api": ControlAPI(bundle, runtime, audit=audit),
        "bundle": bundle, "root": root, "home": home,
        "corpus": root / "knowledge" / "handbook",
    }


def _upload(api, name, data=DOC, **extra):
    return api.write(f"/platform/v1/knowledge/{SOURCE}/upload", ADMIN, {
        "filename": name, "data": base64.b64encode(data).decode(), **extra,
    })


# -- naming --------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("handbook.md", "handbook.md"),
        ("Refund Policy (v2).md", "Refund-Policy-v2.md"),
        # Transliterated rather than punched full of hyphens.
        ("ünïcode.md", "unicode.md"),
        # A browser can send either of these; both reduce to a basename.
        ("../../etc/passwd.md", "passwd.md"),
        ("C:\\evil\\payload.md", "payload.md"),
        (".hidden.md", "hidden.md"),
    ],
)
def test_a_filename_is_rebuilt_not_trusted(raw, expected):
    assert safe_name(raw) == expected


@pytest.mark.parametrize("raw", ["", "..", ".", "/", "   ", "---"])
def test_a_filename_with_nothing_usable_in_it_is_refused(raw):
    with pytest.raises(SpecError):
        safe_name(raw)


def test_a_traversing_name_lands_inside_the_corpus(live, tmp_path):
    """The name is flattened to a basename, so the file goes where uploads go. Asserted
    against the filesystem rather than against the returned name."""
    escape = tmp_path / "pwned.md"
    response = _upload(live["api"], "../../../../" + str(escape.name))
    assert response.status == 200
    assert not escape.exists(), "a write escaped the corpus"
    assert (live["corpus"] / "pwned.md").is_file()


# -- what a corpus accepts -----------------------------------------------------


def test_the_corpus_decides_what_it_accepts(live):
    """Not a list of dangerous extensions kept here — the tenant already declared what
    belongs in each corpus, and a second list would eventually disagree with the first."""
    for name in ("payload.exe", "script.sh", "archive.zip", "notes.txt"):
        response = _upload(live["api"], name, b"whatever")
        assert response.status == 400, f"{name} was accepted into a **/*.md corpus"
        assert "accepts" in response.body["error"]["message"]

    assert _upload(live["api"], "allowed.md").status == 200


def test_the_corpus_size_limit_is_the_corpus_own(live):
    source = next(s for s in live["bundle"].knowledge.sources if s.id == SOURCE)
    response = _upload(live["api"], "huge.md", b"x" * (source.max_file_bytes + 1))
    assert response.status == 400
    assert "max_file_bytes" in response.body["error"]["message"]


def test_an_empty_document_is_refused(live):
    assert _upload(live["api"], "blank.md", b"").status == 400


def test_a_duplicate_is_refused_unless_replacement_is_explicit(live):
    api = live["api"]
    assert _upload(api, "policy.md").status == 200
    again = _upload(api, "policy.md", b"# Different\n")
    assert again.status == 400
    assert "already in" in again.body["error"]["message"]
    assert (live["corpus"] / "policy.md").read_bytes() == DOC, "the original was overwritten"

    assert _upload(api, "policy.md", b"# Different\n", replace=True).status == 200
    assert b"Different" in (live["corpus"] / "policy.md").read_bytes()


# -- indexing ------------------------------------------------------------------


def test_an_upload_is_indexed_so_an_agent_can_find_it(live):
    """A document on disk that the index has never seen is invisible to every agent."""
    response = _upload(live["api"], "returns.md")
    assert response.status == 200, response.body
    index = response.body["index"]
    assert index["ok"] is True
    assert index["documents"] >= 1
    assert index["chunks"] >= 1

    listed = live["api"].handle(f"/platform/v1/knowledge/{SOURCE}/documents").body
    assert "returns.md" in [d["name"] for d in listed["documents"]]
    assert listed["indexed"]["documents"] >= 1


def test_the_document_is_searchable_after_an_upload(live):
    """The end of the chain. Everything before this is plumbing if the text cannot be
    found by the query path an agent actually uses."""
    from nova.knowledge import KnowledgeIndex

    _upload(live["api"], "returns.md", b"# Returns\n\nA distinctive phrase: quokka refunds.\n")

    with KnowledgeIndex.open(live["api"].runtime.knowledge_index_path, create=False) as index:
        hits = index.search("quokka", source_ids=[SOURCE], limit=5)
    assert hits, "the uploaded document was not searchable"
    assert any("quokka" in hit.text.lower() for hit in hits)


def test_saved_and_indexed_are_reported_separately(live):
    """A document stored but not indexed is a real state, and one combined tick would let
    it read as done."""
    response = _upload(live["api"], "returns.md")
    assert response.body["saved"] is True
    assert "index" in response.body and "ok" in response.body["index"]


def test_removing_a_document_takes_it_out_of_the_index(live):
    api = live["api"]
    _upload(api, "returns.md")
    before = api.handle(f"/platform/v1/knowledge/{SOURCE}/documents").body["indexed"]["documents"]

    response = api.write(f"/platform/v1/knowledge/{SOURCE}/remove", ADMIN, {"name": "returns.md"})
    assert response.status == 200, response.body
    assert not (live["corpus"] / "returns.md").exists()

    after = api.handle(f"/platform/v1/knowledge/{SOURCE}/documents").body["indexed"]["documents"]
    assert after == before - 1, "the index still holds a document that is gone from disk"


def test_reindex_picks_up_a_file_added_on_the_host(live):
    """An operator who added files by hand should not have to upload them again."""
    (live["corpus"] / "manual.md").write_text("# Added on the host\n\nBy hand.\n")
    response = live["api"].write(f"/platform/v1/knowledge/{SOURCE}/reindex", ADMIN, {})
    assert response.status == 200
    assert response.body["index"]["ok"] is True
    listed = live["api"].handle(f"/platform/v1/knowledge/{SOURCE}/documents").body
    assert "manual.md" in [d["name"] for d in listed["documents"]]


# -- authorisation and shape ---------------------------------------------------


def test_a_viewer_may_not_change_a_corpus(live):
    for action, payload in (
        ("upload", {"filename": "x.md", "data": base64.b64encode(DOC).decode()}),
        ("remove", {"name": "refunds.md"}),
        ("reindex", {}),
    ):
        response = live["api"].write(f"/platform/v1/knowledge/{SOURCE}/{action}", VIEWER, payload)
        assert response.status == 403, f"a viewer reached {action}"


def test_an_unknown_source_is_a_404(live):
    assert _upload(live["api"], "x.md").status == 200
    response = live["api"].write("/platform/v1/knowledge/nope/upload", ADMIN,
                                 {"filename": "x.md", "data": "eA=="})
    assert response.status == 404
    assert live["api"].handle("/platform/v1/knowledge/nope/documents").status == 404


def test_a_malformed_upload_is_refused(live):
    api = live["api"]
    base = f"/platform/v1/knowledge/{SOURCE}/upload"
    assert api.write(base, ADMIN, {"filename": "x.md"}).status == 400
    assert api.write(base, ADMIN, {"filename": "x.md", "data": 42}).status == 400
    assert api.write(base, ADMIN, {"filename": "x.md", "data": "not base64!!"}).status == 400
    assert api.write(base, ADMIN, {"filename": "", "data": "eA=="}).status == 400


def test_an_unknown_action_is_unroutable(live):
    assert live["api"].write(
        f"/platform/v1/knowledge/{SOURCE}/purge", ADMIN, {}
    ).status == 404


def test_the_document_list_reports_what_the_corpus_admits(live):
    """A file sitting in the directory that the corpus does not admit must not be listed as
    though an agent could read it."""
    (live["corpus"] / "notes.txt").write_text("not markdown")
    source = next(s for s in live["bundle"].knowledge.sources if s.id == SOURCE)
    assert accepts(source, "refunds.md") is True
    assert accepts(source, "notes.txt") is False
    assert "notes.txt" not in [row["name"] for row in list_documents(source)]


def test_an_upload_is_audited_without_the_document(live):
    _upload(live["api"], "returns.md", b"# Secret\n\nCONFIDENTIAL-PHRASE-42\n")
    text = (live["home"] / "nova" / "audit.jsonl").read_text()
    records = [json.loads(line) for line in text.splitlines() if line.strip()]
    added = [r for r in records if r.get("kind") == "knowledge.document_added"]
    assert [r["phase"] for r in added] == ["intent", "committed"]
    assert {r["actor"] for r in added} == {"ops"}
    assert "returns.md" in json.dumps(added), "the audit does not say what was added"
    assert "CONFIDENTIAL-PHRASE-42" not in text, "a document's contents reached the audit log"


def test_the_store_writes_atomically(live):
    """A half-written document is one the ingester would index as truncated."""
    source = next(s for s in live["bundle"].knowledge.sources if s.id == SOURCE)
    store_document(source, filename="atomic.md", data=DOC)
    leftovers = [p.name for p in live["corpus"].iterdir() if p.name.startswith(".nova-doc-")]
    assert not leftovers, f"a temporary file was left behind: {leftovers}"
