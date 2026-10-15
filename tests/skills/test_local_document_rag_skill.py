"""Contracts for the optional local-document-rag skill script (no network, no model download)."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "optional-skills" / "research" / "local-document-rag" / "scripts" / "doc_rag.py"


@pytest.fixture
def rag(monkeypatch):
    spec = importlib.util.spec_from_file_location("doc_rag_under_test", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    def fake_embed(texts):
        # Deterministic bag-of-words vectors: shared words -> high cosine similarity.
        out = []
        for t in texts:
            v = [0.0] * mod.DIM
            for w in t.lower().split():
                v[int(hashlib.md5(w.encode()).hexdigest(), 16) % mod.DIM] += 1.0
            v[0] += 1e-3  # never a zero vector
            out.append(v)
        return out

    monkeypatch.setattr(mod, "embed", fake_embed)
    monkeypatch.setattr(mod, "USAGE_LOG", "")
    return mod


def test_chunking_overlaps_and_normalises_whitespace(rag):
    chunks = rag.chunk_page("a  b\n" * 400, size=100, overlap=20)
    assert all(len(c) <= 100 for c in chunks)
    assert chunks[0][80:] == chunks[1][:20]
    assert "\n" not in "".join(chunks)
    assert rag.chunk_page("   ") == []


def test_fts_query_uses_prefixes_and_drops_stopwords(rag):
    q = rag.fts_query("Jaki jest okres gwarancji dla serwerów?")
    assert '"gwaran"*' in q and '"serwer"*' in q
    assert "jaki" not in q and '"dla"' not in q


def test_rrf_rewards_agreement_between_rankings(rag):
    fused = [cid for cid, _ in rag.rrf([1, 2, 3], [3, 4])]
    assert fused[0] == 3


def test_ingest_query_roundtrip_with_citation_and_dedup(rag, tmp_path, capsys):
    pytest.importorskip("sqlite_vec")
    doc = tmp_path / "spec.md"
    doc.write_text("The servers carry a warranty of 60 months.\n\nNetwork switches need 48 ports.",
                   encoding="utf-8")
    db = str(tmp_path / "rag.db")

    assert rag.main(["ingest", str(doc), "--db", db, "--copy", "--project", "demo"]) == 0
    copies = list((tmp_path / "assets").glob("*-spec.md"))
    assert len(copies) == 1 and copies[0].name.endswith("-spec.md")
    assert rag.main(["ingest", str(doc), "--db", db]) == 0
    assert "already indexed" in capsys.readouterr().out

    assert rag.main(["query", "warranty servers", "--db", db, "-k", "1", "--json"]) == 0
    hits = json.loads(capsys.readouterr().out)
    assert hits[0]["source"] == "spec.md" and hits[0]["page"] == 1
    assert "warranty" in hits[0]["text"]
    assert Path(hits[0]["path"]) == copies[0]


def test_copy_disambiguates_colliding_basenames(rag, tmp_path, capsys):
    """Two docs with the same file name but different content must not share one asset."""
    pytest.importorskip("sqlite_vec")
    dir_a, dir_b = tmp_path / "a", tmp_path / "b"
    dir_a.mkdir()
    dir_b.mkdir()
    doc_a = dir_a / "spec.md"
    doc_b = dir_b / "spec.md"
    doc_a.write_text("Server warranty is 60 months for project alpha.", encoding="utf-8")
    doc_b.write_text("Server warranty is 12 months for project beta.", encoding="utf-8")
    db = str(tmp_path / "rag.db")

    assert rag.main(["ingest", str(doc_a), "--db", db, "--copy"]) == 0
    assert rag.main(["ingest", str(doc_b), "--db", db, "--copy"]) == 0

    copies = sorted((tmp_path / "assets").glob("*-spec.md"))
    assert len(copies) == 2, "colliding basenames must produce two distinct assets"
    assert copies[0].read_bytes() != copies[1].read_bytes()

    for needle in ("alpha", "beta"):
        capsys.readouterr()
        assert rag.main(["query", needle, "--db", db, "-k", "1", "--json"]) == 0
        hit = json.loads(capsys.readouterr().out)[0]
        # The hit's own asset file must actually contain the text that was matched.
        assert needle in Path(hit["path"]).read_text(encoding="utf-8")
        assert needle in hit["text"]

    # Re-ingesting identical content must reuse the existing asset, not duplicate it.
    assert rag.main(["ingest", str(doc_a), "--db", db, "--copy"]) == 0
    assert sorted((tmp_path / "assets").glob("*-spec.md")) == copies


def test_query_project_filter_restricts_recall(rag, tmp_path, capsys):
    """--project must filter candidates before ranking, not just the final top-k."""
    pytest.importorskip("sqlite_vec")
    doc_a = tmp_path / "alpha.md"
    doc_b = tmp_path / "beta.md"
    doc_a.write_text("The warranty period for alpha servers is 60 months.", encoding="utf-8")
    doc_b.write_text("The warranty period for beta servers is 60 months.", encoding="utf-8")
    db = str(tmp_path / "rag.db")

    assert rag.main(["ingest", str(doc_a), "--db", db, "--project", "A"]) == 0
    capsys.readouterr()
    assert rag.main(["ingest", str(doc_b), "--db", db, "--project", "B"]) == 0
    warn = capsys.readouterr().err
    assert "project" in warn.lower()

    assert rag.main(
        ["query", "warranty period servers", "--db", db, "-k", "5", "--json", "--project", "A"]) == 0
    hits = json.loads(capsys.readouterr().out)
    assert hits and all(h["source"] == "alpha.md" for h in hits)

    assert rag.main(
        ["query", "warranty period servers", "--db", db, "-k", "5", "--json", "--project", "B"]) == 0
    hits = json.loads(capsys.readouterr().out)
    assert hits and all(h["source"] == "beta.md" for h in hits)

    assert rag.main(
        ["query", "warranty period servers", "--db", db, "-k", "5", "--json", "--project", "nope"]) == 0
    assert capsys.readouterr().out.strip() == "(no hits)"


def test_chunk_page_rejects_overlap_not_smaller_than_size(rag):
    with pytest.raises(ValueError):
        rag.chunk_page("abcdef", size=10, overlap=10)


def test_ingest_reports_unusable_files_with_exit_code(rag, tmp_path, capsys):
    pytest.importorskip("sqlite_vec")
    bad = tmp_path / "data.xlsx"
    bad.write_bytes(b"x")
    db = str(tmp_path / "rag.db")
    assert rag.main(["ingest", str(bad), str(tmp_path / "missing.pdf"), "--db", db]) == 1
    out = capsys.readouterr().out
    assert "unsupported file type" in out and "missing file" in out
    assert not (tmp_path / "assets").exists()
