"""paper-literature-search skill tests (no network)."""

from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SKILL = REPO / "skills" / "research" / "paper-literature-search" / "SKILL.md"
RANK = REPO / "skills" / "research" / "paper-literature-search" / "scripts" / "paper_search_rank.py"


def test_skill_description_length():
    text = SKILL.read_text(encoding="utf-8")
    m = re.search(r"^description:\s*(.+)$", text, re.MULTILINE)
    assert m
    desc = m.group(1).strip().strip('"')
    assert len(desc) <= 60, len(desc)


def test_rank_papers_offline():
    import importlib.util

    spec = importlib.util.spec_from_file_location("paper_search_rank", RANK)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(mod)

    papers = [
        {
            "title": "Alpha",
            "year": 2024,
            "abstract": "multimodal large language",
            "citation_count": 100,
            "influential_citation_count": 10,
            "relevance": 0.9,
            "arxiv_id": "2401.00001",
            "pdf_url": "https://arxiv.org/pdf/2401.00001",
        },
        {
            "title": "Beta old",
            "year": 2018,
            "abstract": "unrelated topic",
            "citation_count": 50,
            "influential_citation_count": 5,
            "relevance": 0.2,
            "arxiv_id": "",
        },
    ]
    ranked = mod.rank_papers("multimodal", papers)
    assert ranked[0]["title"] == "Alpha"
    assert "scores" in ranked[0]
    assert ranked[0]["scores"]["display"] >= ranked[1]["scores"]["display"]


def test_dedupe_arxiv():
    import importlib.util

    spec = importlib.util.spec_from_file_location("paper_search_rank", RANK)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    papers = [
        {"title": "A", "arxiv_id": "2402.03300"},
        {"title": "A duplicate", "arxiv_id": "2402.03300"},
    ]
    assert len(mod.dedupe_papers(papers)) == 1


def test_format_deliver():
    import importlib.util

    path = REPO / "skills" / "research" / "paper-literature-search" / "scripts" / "paper_search_feishu_deliver.py"
    spec = importlib.util.spec_from_file_location("deliver", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    body = mod.format_top_list({
        "query": "RLHF",
        "candidate_count": 20,
        "papers": [{
            "title": "Test Paper",
            "year": 2024,
            "citation_count": 10,
            "influential_citation_count": 2,
            "arxiv_abs": "https://arxiv.org/abs/2402.03300",
            "scores": {"display": 88},
        }],
    })
    assert "文献检索" in body
    assert "2402.03300" in body
