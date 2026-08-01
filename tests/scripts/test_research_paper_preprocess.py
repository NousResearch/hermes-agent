#!/usr/bin/env python3
"""Tests for research_paper_preprocess.py — scoring, arXiv parsing, dedup, main output.

Uses fixture JSON for network-heavy paths; no real API calls in tests.
"""
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import research_paper_preprocess as rpp


# ── Fixtures ────────────────────────────────────────────────────────────────

def _paper(arxiv_id="2607.05744", title="Unicode sanitization for MCP servers",
           summary="We present a method to sanitize MCP tool descriptions against prompt injection.",
           published="2026-07-20T00:00:00Z", score=None, hf_featured=False):
    p = {
        "arxiv_id": arxiv_id,
        "title": title,
        "summary": summary,
        "published": published,
        "score": score if score is not None else rpp.tight_score(title, summary),
        "source": "arxiv",
        "url": f"https://arxiv.org/abs/{arxiv_id}",
    }
    if hf_featured:
        p["hf_featured"] = True
        p["hf_upvotes"] = 60
    return p


# ── Scoring ─────────────────────────────────────────────────────────────────

class TestTightScore:
    def test_score5_mcp_phrase(self):
        assert rpp.tight_score("MCP server hardening", "a study") == 5

    def test_score4_adjacent_phrase(self):
        assert rpp.tight_score("RAG pipeline", "a study") == 4

    def test_score3_broader_ai(self):
        assert rpp.tight_score("Transformers in medicine", "attention mechanism review") == 3

    def test_score1_unrelated(self):
        assert rpp.tight_score("Quantum chemistry", "protein folding") == 1

    def test_phrase_in_summary_counts(self):
        assert rpp.tight_score("Title without keywords", "uses agent memory for planning") == 5


# ── arXiv ID parsing ────────────────────────────────────────────────────────

class TestParseArxivId:
    def test_raw_id(self):
        assert rpp.parse_arxiv_id("2606.00467") == "2606.00467"

    def test_raw_id_with_version(self):
        assert rpp.parse_arxiv_id("2606.00467v2") == "2606.00467"

    def test_abs_url(self):
        assert rpp.parse_arxiv_id("https://arxiv.org/abs/2606.00467") == "2606.00467"

    def test_abs_url_versioned(self):
        assert rpp.parse_arxiv_id("https://arxiv.org/abs/2606.00467v3") == "2606.00467"

    def test_pdf_url(self):
        assert rpp.parse_arxiv_id("https://arxiv.org/pdf/2606.00467.pdf") == "2606.00467"

    def test_invalid(self):
        assert rpp.parse_arxiv_id("not-an-id") is None


# ── Final scoring / thresholds ──────────────────────────────────────────────

class TestApplyFinalScore:
    def test_write_now_high_quality(self):
        p = _paper(score=5, hf_featured=True, published="2026-07-30T00:00:00Z")
        p["pwc_repo"] = "https://github.com/example/repo"
        p["pwc_stars"] = 600
        out = rpp.apply_final_score(p)
        assert out["action"] == "write_now"
        assert out["final_score"] >= 5.5

    def test_low_quality_file(self):
        p = _paper(score=2, published="2026-05-01T00:00:00Z")
        out = rpp.apply_final_score(p)
        assert out["action"] in ("file", "skip")

    def test_recency_boost_applied(self):
        p = _paper(score=4, published="2026-07-31T00:00:00Z")
        out = rpp.apply_final_score(p)
        assert out["quality_weight"] >= 0.7

    def test_missing_published_does_not_crash(self):
        p = _paper(score=3)
        p["published"] = "garbage-date"
        out = rpp.apply_final_score(p)
        assert "action" in out


# ── Dedup / HF cross-ref ────────────────────────────────────────────────────

class TestHfDailyCrossRef:
    def test_existing_paper_marked_featured(self, monkeypatch):
        papers = [_paper(arxiv_id="2607.05744", score=4)]
        fake = [{"paper": {"id": "2607.05744", "title": "x", "summary": "y"}, "upvotes": 80}]
        monkeypatch.setattr(rpp, "fetch_url", lambda url, **kw: json.dumps(fake))
        out = rpp.fetch_hf_daily(papers)
        assert out[0]["hf_featured"] is True
        assert out[0]["hf_upvotes"] == 80
        assert len(out) == 1

    def test_new_paper_added(self, monkeypatch):
        papers = [_paper(arxiv_id="2607.00001", score=4)]
        fake = [{"paper": {"id": "2607.99999", "title": "New agent memory paper",
                           "summary": "long-term agent memory study"}, "upvotes": 10}]
        monkeypatch.setattr(rpp, "fetch_url", lambda url, **kw: json.dumps(fake))
        out = rpp.fetch_hf_daily(papers)
        assert len(out) == 2
        added = [p for p in out if p["arxiv_id"] == "2607.99999"][0]
        assert added["source"] == "hf-daily"

    def test_bad_json_no_crash(self, monkeypatch):
        papers = [_paper(score=4)]
        monkeypatch.setattr(rpp, "fetch_url", lambda url, **kw: "not-json")
        out = rpp.fetch_hf_daily(papers)
        assert len(out) == 1


# ── Main pipeline (offline fixture mode) ────────────────────────────────────

class TestMainOffline:
    def test_main_outputs_json(self, monkeypatch, capsys, tmp_path):
        # Stub network + PwC so main runs without real API calls
        stub_papers = [
            _paper("2607.10001", "Agent memory graph", "long-term agent memory graph study", score=5),
            _paper("2607.10002", "MCP tool safety", "mcp server tool calling safety", score=4),
            _paper("2607.10003", "Quantum chemistry", "protein folding unrelated", score=1),
        ]
        monkeypatch.setattr(rpp, "fetch_arxiv", lambda: stub_papers)
        monkeypatch.setattr(rpp, "fetch_hf_daily", lambda papers: papers)
        monkeypatch.setattr(rpp, "fetch_pwc", lambda papers: papers)
        monkeypatch.setattr(rpp, "OUTPUT_PATH", str(tmp_path / "out.json"))

        rpp.main()
        out = capsys.readouterr().out
        # stdout should contain JSON envelope
        payload = json.loads(out)
        assert "candidates" in payload
        assert payload["total_fetched"] == 3
        # skip actions excluded
        assert all(c["action"] != "skip" for c in payload["candidates"])
        # file written
        assert (tmp_path / "out.json").exists()
