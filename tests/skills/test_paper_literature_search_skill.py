"""paper-literature-search skill tests (no network)."""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SKILL = REPO / "skills" / "research" / "paper-literature-search" / "SKILL.md"
RANK = REPO / "skills" / "research" / "paper-literature-search" / "scripts" / "paper_search_rank.py"
DELIVER = REPO / "skills" / "research" / "paper-literature-search" / "scripts" / "paper_search_feishu_deliver.py"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(mod)
    return mod


def test_skill_description_length():
    text = SKILL.read_text(encoding="utf-8")
    m = re.search(r"^description:\s*(.+)$", text, re.MULTILINE)
    assert m
    desc = m.group(1).strip().strip('"')
    assert len(desc) <= 60, len(desc)


def test_skill_has_verification_section():
    text = SKILL.read_text(encoding="utf-8")
    assert "## Verification" in text
    assert "paper_search_lark_e2e.sh" in text


def test_rank_papers_offline():
    mod = _load(RANK, "paper_search_rank")
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


def test_min_citations_allows_arxiv_zero():
    """arXiv backfill papers with citation_count=0 must NOT be filtered by min_citations."""
    mod = _load(RANK, "paper_search_rank")
    papers = [
        {
            "title": "ArXiv New",
            "year": 2024,
            "abstract": "gnn",
            "citation_count": 0,
            "influential_citation_count": 0,
            "relevance": 0.8,
            "arxiv_id": "2401.00099",
            "pdf_url": "https://arxiv.org/pdf/2401.00099",
            "source": "arxiv",
        },
        {
            "title": "S2 Low",
            "year": 2022,
            "abstract": "gnn",
            "citation_count": 5,
            "influential_citation_count": 0,
            "relevance": 0.7,
            "arxiv_id": "2201.00001",
            "source": "semantic_scholar",
        },
    ]
    ranked = mod.rank_papers("gnn", papers, min_citations=50)
    # ArXiv paper (cc=0) passes; S2 paper (cc=5 < 50) is filtered out
    titles = [p["title"] for p in ranked]
    assert "ArXiv New" in titles
    assert "S2 Low" not in titles


def test_min_citations_filters_s2_below_threshold():
    """S2 papers with cc > 0 and cc < min_citations must be excluded."""
    mod = _load(RANK, "paper_search_rank")
    papers = [
        {"title": "HighCite", "year": 2020, "abstract": "x", "citation_count": 200,
         "influential_citation_count": 20, "relevance": 0.5, "arxiv_id": "1900.00001"},
        {"title": "LowCite", "year": 2022, "abstract": "x", "citation_count": 3,
         "influential_citation_count": 0, "relevance": 0.9, "arxiv_id": "2200.00002"},
    ]
    ranked = mod.rank_papers("x", papers, min_citations=10)
    titles = [p["title"] for p in ranked]
    assert "HighCite" in titles
    assert "LowCite" not in titles


def test_year_floor_filter():
    mod = _load(RANK, "paper_search_rank")
    papers = [
        {"title": "Old", "year": 2019, "abstract": "x", "citation_count": 500,
         "influential_citation_count": 50, "relevance": 0.9, "arxiv_id": "1900.00001"},
        {"title": "New", "year": 2023, "abstract": "x", "citation_count": 10,
         "influential_citation_count": 1, "relevance": 0.5, "arxiv_id": "2301.00001"},
    ]
    ranked = mod.rank_papers("x", papers, year_floor=2022)
    titles = [p["title"] for p in ranked]
    assert "New" in titles
    assert "Old" not in titles


def test_recency_score_ml_profile():
    mod = _load(RANK, "paper_search_rank")
    score_current = mod._recency_score(2026, profile="ml")
    score_old = mod._recency_score(2015, profile="ml")
    assert score_current >= score_old
    assert score_current == 1.0
    assert score_old == 0.35


def test_recency_score_survey_profile():
    mod = _load(RANK, "paper_search_rank")
    score_recent = mod._recency_score(2025, profile="survey")
    score_decade = mod._recency_score(2016, profile="survey")
    assert score_recent > score_decade


def test_log1p_norm_all_zero():
    mod = _load(RANK, "paper_search_rank")
    result = mod._log1p_norm([0, 0, 0])
    assert result == [0.0, 0.0, 0.0]


def test_log1p_norm_single_positive():
    mod = _load(RANK, "paper_search_rank")
    result = mod._log1p_norm([0, 100])
    assert result[0] == 0.0
    assert result[1] == 1.0


def test_keyword_rel_match():
    mod = _load(RANK, "paper_search_rank")
    score = mod._keyword_rel("graph neural network", "Graph Neural Network Survey", "")
    assert score > 0.5


def test_parse_query_spec_boolean_groups():
    mod = _load(RANK, "paper_search_rank")
    spec = mod.parse_query_spec(
        '(01.AI OR "Yi-Pi" OR "Pi-0.5") AND ("large language model" OR multimodal) '
        'AND NOT ("Raspberry Pi" OR "pi theorem")'
    )
    assert spec["positive_groups"][0][:2] == ["01.AI", "Yi-Pi"]
    assert "Raspberry Pi" in spec["negative_terms"]
    assert "large language model" in spec["search_query"]


def test_build_candidate_queries_splits_aliases():
    mod = _load(RANK, "paper_search_rank")
    spec = mod.parse_query_spec(
        '(01.AI OR "Yi-Pi" OR "Pi-0.5") AND ("large language model" OR multimodal)'
    )
    queries = mod.build_candidate_queries(spec)
    assert queries[0].startswith("01.AI Yi-Pi")
    assert any(q.startswith("Yi-Pi ") for q in queries)
    assert any(q.startswith("Pi-0.5 ") for q in queries)


def test_query_spec_filters_generic_llm_false_positive():
    mod = _load(RANK, "paper_search_rank")
    spec = mod.parse_query_spec(
        '(01.AI OR "Yi-Pi" OR "Pi-0.5") AND ("large language model" OR multimodal) '
        'AND NOT ("Raspberry Pi")'
    )
    keep = {
        "title": "Yi-Pi: Edge Multimodal Large Language Model from 01.AI",
        "abstract": "Yi-Pi is a multimodal large language model for edge deployment.",
    }
    drop = {
        "title": "A Survey of Large Language Models",
        "abstract": "Generic large language model survey without vendor-specific aliases.",
    }
    drop_neg = {
        "title": "Raspberry Pi Multimodal Assistant",
        "abstract": "A large language model deployed on Raspberry Pi.",
    }
    assert mod._paper_matches_query_spec(keep, spec) is True
    assert mod._paper_matches_query_spec(drop, spec) is False
    assert mod._paper_matches_query_spec(drop_neg, spec) is False


def test_search_openalex_maps_results(monkeypatch):
    mod = _load(RANK, "paper_search_rank")

    def _fake_get_json(_url, timeout=45, retries=2, headers=None):
        return {
            "results": [
                {
                    "id": "https://openalex.org/W123",
                    "display_name": "Graph Neural Networks",
                    "publication_year": 2024,
                    "cited_by_count": 42,
                    "abstract_inverted_index": {"Graph": [0], "Neural": [1], "Networks": [2]},
                    "ids": {"arxiv": "https://arxiv.org/abs/2401.00001"},
                    "authorships": [{"author": {"display_name": "A. Author"}}],
                    "primary_location": {
                        "landing_page_url": "https://openalex.org/W123",
                        "source": {"display_name": "ICLR"},
                    },
                    "best_oa_location": {"pdf_url": "https://arxiv.org/pdf/2401.00001"},
                }
            ]
        }

    monkeypatch.setattr(mod, "_get_json", _fake_get_json)
    papers = mod.search_openalex("graph neural network", limit=5)
    assert papers[0]["source"] == "openalex"
    assert papers[0]["arxiv_id"] == "2401.00001"
    assert papers[0]["venue"] == "ICLR"


def test_openalex_search_url_includes_key_and_mailto(monkeypatch):
    mod = _load(RANK, "paper_search_rank")
    monkeypatch.setenv("OPENALEX_API_KEY", "test-key")
    monkeypatch.setenv("CROSSREF_MAILTO", "user@example.com")
    url = mod._openalex_search_url("graph neural network", limit=7)
    assert "api_key=test-key" in url
    assert "mailto=user%40example.com" in url


def test_run_search_uses_openalex_fallback(monkeypatch):
    mod = _load(RANK, "paper_search_rank")
    monkeypatch.setattr(mod, "search_semantic_scholar", lambda *_a, **_kw: [])
    monkeypatch.setattr(mod, "search_openalex", lambda *_a, **_kw: [{
        "source": "openalex",
        "paper_id": "W123",
        "title": "Fallback Paper",
        "year": 2024,
        "abstract": "graph neural network",
        "citation_count": 10,
        "influential_citation_count": 0,
        "relevance": 0.0,
        "url": "https://openalex.org/W123",
        "arxiv_id": "",
        "arxiv_abs": "",
        "pdf_url": "",
        "venue": "OpenAlex",
        "authors": [],
    }])
    monkeypatch.setattr(mod, "search_arxiv_backfill", lambda *_a, **_kw: [])
    result = mod.run_search("graph neural network", top=3)
    assert result["candidate_count"] == 1
    assert result["papers"][0]["title"] == "Fallback Paper"


def test_run_search_boolean_query_filters_off_topic_results(monkeypatch):
    mod = _load(RANK, "paper_search_rank")
    monkeypatch.setattr(mod, "search_semantic_scholar", lambda *_a, **_kw: [
        {
            "source": "semantic_scholar",
            "paper_id": "good",
            "title": "Pi-0.5: A multimodal edge large language model from 01.AI",
            "year": 2025,
            "abstract": "Pi-0.5 is a multimodal large language model for edge deployment.",
            "citation_count": 8,
            "influential_citation_count": 0,
            "relevance": 0.0,
            "url": "https://example.com/good",
            "arxiv_id": "",
            "arxiv_abs": "",
            "pdf_url": "",
            "venue": "arXiv",
            "authors": [],
        },
        {
            "source": "semantic_scholar",
            "paper_id": "bad",
            "title": "A Survey of Large Language Models",
            "year": 2026,
            "abstract": "Generic large language model survey.",
            "citation_count": 500,
            "influential_citation_count": 10,
            "relevance": 0.0,
            "url": "https://example.com/bad",
            "arxiv_id": "",
            "arxiv_abs": "",
            "pdf_url": "",
            "venue": "Journal",
            "authors": [],
        },
    ])
    monkeypatch.setattr(mod, "search_openalex", lambda *_a, **_kw: [])
    monkeypatch.setattr(mod, "search_arxiv_backfill", lambda *_a, **_kw: [])
    result = mod.run_search(
        '(01.AI OR "Yi-Pi" OR "Pi-0.5") AND ("large language model" OR multimodal) '
        'AND NOT ("Raspberry Pi")',
        top=3,
    )
    assert result["candidate_count"] == 1
    assert result["papers"][0]["paper_id"] == "good"


def test_boost_recency_shifts_weights():
    """boost_recency must shift composite scores to favor newer papers."""
    import copy
    mod = _load(RANK, "paper_search_rank")
    papers = [
        {"title": "Classic", "year": 2018, "abstract": "gnn", "citation_count": 5000,
         "influential_citation_count": 300, "relevance": 0.8, "arxiv_id": "1801.00001"},
        {"title": "Fresh", "year": 2025, "abstract": "gnn", "citation_count": 2,
         "influential_citation_count": 0, "relevance": 0.7, "arxiv_id": "2501.00001"},
    ]
    ranked_normal = mod.rank_papers("gnn", copy.deepcopy(papers), boost_recency=False)
    ranked_boosted = mod.rank_papers("gnn", copy.deepcopy(papers), boost_recency=True)
    # Classic wins without boost (high citations)
    assert ranked_normal[0]["title"] == "Classic"
    # boost_recency increases w["rec"] by 0.05 and decreases w["cite"] by 0.05
    # so the composite gap between Classic and Fresh must shrink with boost
    def gap(ranked):
        scores = {p["title"]: p["scores"]["composite"] for p in ranked}
        return scores["Classic"] - scores["Fresh"]
    assert gap(ranked_normal) > gap(ranked_boosted)


def test_dedupe_arxiv():
    mod = _load(RANK, "paper_search_rank")
    papers = [
        {"title": "A", "arxiv_id": "2402.03300"},
        {"title": "A duplicate", "arxiv_id": "2402.03300"},
    ]
    assert len(mod.dedupe_papers(papers)) == 1


def test_dedupe_title_fallback():
    """Papers without arxiv_id but same title should be deduped."""
    mod = _load(RANK, "paper_search_rank")
    papers = [
        {"title": "Attention Is All You Need", "arxiv_id": ""},
        {"title": "Attention Is All You Need", "arxiv_id": ""},
    ]
    assert len(mod.dedupe_papers(papers)) == 1


def test_format_deliver():
    mod = _load(DELIVER, "deliver")
    body = mod.format_top_list({
        "query": "RLHF",
        "candidate_count": 20,
        "papers": [{
            "title": "A Survey of Large Language Models",
            "year": 2024,
            "citation_count": 10,
            "influential_citation_count": 2,
            "abstract": "This survey studies large language model systems.",
            "arxiv_abs": "https://arxiv.org/abs/2402.03300",
            "scores": {"display": 88},
        }],
    })
    assert "文献检索" in body
    assert "2402.03300" in body
    assert "EN:" in body
    assert "中纲:" in body
    assert "大语言模型" in body


def test_format_deliver_empty_papers():
    mod = _load(DELIVER, "deliver")
    body = mod.format_top_list({
        "query": "empty",
        "candidate_count": 0,
        "papers": [],
    })
    assert "文献检索" in body
    assert "候选 0" in body
    assert "未检索到同时满足当前条件的候选论文" in body


def test_build_delivery_summary_marks_no_hit():
    mod = _load(DELIVER, "deliver")
    summary = mod.build_delivery_summary("q", {"candidate_count": 0, "papers": []})
    assert summary["no_hit"] is True
    assert summary["next_action"] == "stop_no_manual_fallback"


def test_format_deliver_collapses_tail_to_compact_list():
    mod = _load(DELIVER, "deliver")
    papers = []
    for i in range(7):
        papers.append({
            "title": f"Paper {i}",
            "year": 2024,
            "citation_count": 20 + i,
            "influential_citation_count": i,
            "abstract": "Benchmark study for graph neural network models.",
            "arxiv_abs": "",
            "scores": {"display": 80 - i},
        })
    body = mod.format_top_list({
        "query": "gnn",
        "candidate_count": 14,
        "papers": papers,
    })
    assert "IM 导读：前 5 篇双语摘要" in body
    assert "延伸候选 2 篇（简表）" in body
    assert "6. [75] Paper 5 (2024)" in body


def test_deliver_dry_run(tmp_path, monkeypatch):
    """dry_run should print the top list and return 0 without calling lark-cli."""
    import sys
    mod = _load(DELIVER, "deliver_dry")
    result = {
        "query": "gnn",
        "candidate_count": 5,
        "papers": [{"title": "P", "year": 2024, "citation_count": 0,
                    "influential_citation_count": 0, "arxiv_abs": "", "scores": {"display": 70}}],
    }
    captured: list[str] = []
    monkeypatch.setattr("builtins.print", lambda *a, **kw: captured.append(" ".join(str(x) for x in a)))
    body = mod.format_top_list(result)
    assert "文献检索" in body
    res = mod.send_text("oc_test", body, dry_run=True)
    assert res.get("dry_run") is True
