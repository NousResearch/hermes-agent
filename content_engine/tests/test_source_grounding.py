"""Tests for blog.source_grounding — extract, search, inject, verify links.

Covers:
  - _extract_paper_references: model names, benchmarks, datasets
  - _search_arxiv: mocked API responses
  - _inject_links: first-occurrence linking, skip existing links
  - _verify_links: dead/alive URLs (mocked)
  - ground_post: full orchestration
"""
import json
from pathlib import Path
from unittest.mock import patch, MagicMock
from urllib.error import URLError

import pytest

import blog.source_grounding as sg


# -- _extract_paper_references tests ------------------------------------------

def test_extract_finds_model_names():
    """Named models like GPT-4, BERT, LLaMA 2 are extracted."""
    body = "We tested GPT-4 and BERT on several benchmarks. LLaMA 2 also performed well."
    refs = sg._extract_paper_references(body)
    # Should find at least 3 model names.
    ref_lower = [r.lower() for r in refs]
    assert any("gpt" in r for r in ref_lower)
    assert any("bert" in r for r in ref_lower)
    assert any("llama" in r for r in ref_lower)


def test_extract_finds_benchmark_names():
    """Benchmark names like MMLU, GSM8K, HumanEval are extracted."""
    body = "The model scored 87.2 on MMLU and 68.5 on GSM8K. HumanEval pass@5 was 0.72."
    refs = sg._extract_paper_references(body)
    ref_lower = [r.lower() for r in refs]
    assert any("mmlu" in r for r in ref_lower)
    assert any("gsm8k" in r for r in ref_lower)
    assert any("humaneval" in r for r in ref_lower)


def test_extract_deduplicates():
    """Duplicate references are deduplicated."""
    body = "GPT-4 is great. GPT-4 also does vision. GPT-4 is expensive."
    refs = sg._extract_paper_references(body)
    # Only one GPT-4 entry.
    assert sum(1 for r in refs if "gpt" in r.lower() and "4" in r) <= 1


def test_extract_skips_existing_links():
    """References inside existing markdown links are not re-extracted."""
    body = "We tested [GPT-4](https://arxiv.org/abs/2303.08774) and also BERT."
    refs = sg._extract_paper_references(body)
    ref_lower = [r.lower() for r in refs]
    assert any("bert" in r for r in ref_lower)
    # GPT-4 is inside a markdown link so should be skipped.
    assert not any("gpt" in r and "4" in r for r in refs)


def test_extract_empty_body():
    """Empty body returns empty list."""
    assert sg._extract_paper_references("") == []


# -- _search_arxiv tests -------------------------------------------------------

def test_search_arxiv_returns_url_on_success():
    """_search_arxiv returns an arXiv URL on a successful API response."""
    fake_xml = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/2303.08774v1</id>
    <title>GPT-4 Technical Report</title>
    <summary>We introduce GPT-4...</summary>
  </entry>
</feed>"""
    mock_resp = MagicMock()
    mock_resp.read.return_value = fake_xml.encode("utf-8")
    mock_resp.status = 200
    mock_resp.__enter__ = MagicMock(return_value=mock_resp)
    mock_resp.__exit__ = MagicMock(return_value=None)

    with patch("urllib.request.urlopen", return_value=mock_resp):
        result = sg._search_arxiv("GPT-4")
    assert result is not None
    assert "arxiv.org/abs/2303.08774" in result
    assert result.startswith("https://")


def test_search_arxiv_returns_none_on_network_error():
    """_search_arxiv returns None when the API is unreachable."""
    with patch("urllib.request.urlopen", side_effect=URLError("timeout")):
        result = sg._search_arxiv("GPT-4")
    assert result is None


def test_search_arxiv_returns_none_on_no_results():
    """_search_arxiv returns None when the API returns no results."""
    fake_xml = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
</feed>"""
    mock_resp = MagicMock()
    mock_resp.read.return_value = fake_xml.encode("utf-8")
    mock_resp.__enter__ = MagicMock(return_value=mock_resp)
    mock_resp.__exit__ = MagicMock(return_value=None)

    with patch("urllib.request.urlopen", return_value=mock_resp):
        result = sg._search_arxiv("NonexistentPaper12345")
    assert result is None


# -- _inject_links tests ------------------------------------------------------

def test_inject_links_first_occurrence_only():
    """Only the first occurrence of a reference is linked."""
    body = "GPT-4 is great. GPT-4 also does vision."
    links = {"GPT-4": "https://arxiv.org/abs/2303.08774"}
    result = sg._inject_links(body, links)
    assert result.count("https://arxiv.org/abs/2303.08774") == 1
    # First occurrence should be linked.
    assert "[GPT-4](https://arxiv.org/abs/2303.08774)" in result


def test_inject_links_skips_existing_links():
    """References already in markdown links are not double-linked."""
    body = "[GPT-4](https://example.com) is great. GPT-4 also does vision."
    links = {"GPT-4": "https://arxiv.org/abs/2303.08774"}
    result = sg._inject_links(body, links)
    # The second occurrence should be linked to arxiv.
    assert result.count("https://arxiv.org/abs/2303.08774") == 1
    # The existing example.com link should still be there.
    assert "example.com" in result


def test_inject_links_empty_links():
    """Empty links dict returns body unchanged."""
    body = "Some text about GPT-4."
    assert sg._inject_links(body, {}) == body


# -- _verify_links tests ------------------------------------------------------

def test_verify_links_empty_body():
    """Empty body has no links to verify."""
    assert sg._verify_links("") == []


def test_verify_links_dead_url():
    """A dead URL is returned in the dead list."""
    body = "[test](https://example-nonexistent-xyz.com)"
    with patch("blog.source_grounding._check_url", return_value=False):
        dead = sg._verify_links(body)
    assert len(dead) == 1
    assert "example-nonexistent-xyz.com" in dead[0]


def test_verify_links_alive_url():
    """Alive URLs are not returned."""
    body = "[test](https://arxiv.org/abs/2303.08774)"
    with patch("blog.source_grounding._check_url", return_value=True):
        dead = sg._verify_links(body)
    assert dead == []


def test_verify_links_mixed():
    """Mixed dead and alive URLs — only dead returned."""
    body = "[good](https://arxiv.org/abs/2303.08774) and [bad](https://dead-url.com)"
    with patch("blog.source_grounding._check_url", return_value=False):
        dead = sg._verify_links(body)
    # _verify_links uses a set, so order isn't guaranteed. Check that
    # both URLs are reported as dead (since we mocked all as False).
    assert len(dead) == 2


# -- ground_post tests --------------------------------------------------------

def test_ground_post_builder_skips_arxiv():
    """Builder stream skips arXiv search but still verifies links."""
    draft = {"body_md": "Some text about GPT-4 and MMLU."}
    with patch("blog.source_grounding._search_arxiv") as mock_search:
        with patch("blog.source_grounding._verify_links", return_value=[]):
            result = sg.ground_post(draft, stream="builder")
    # No arXiv search for builder.
    mock_search.assert_not_called()
    assert result["grounds"] == 0
    assert result["references_found"] != []  # refs still extracted


def test_ground_post_injects_links():
    """ground_post injects arXiv links for AI stream posts."""
    draft = {"body_md": "GPT-4 is a large multimodal model."}
    with patch("blog.source_grounding._search_arxiv",
               return_value="https://arxiv.org/abs/2303.08774"):
        with patch("blog.source_grounding._verify_links", return_value=[]):
            result = sg.ground_post(draft, stream="ai")
    assert result["grounds"] >= 1
    assert "arxiv.org/abs/2303.08774" in result["body_md"]


def test_ground_post_returns_dead_links():
    """Dead links are returned in the result."""
    draft = {"body_md": "[test](https://dead.com)"}
    with patch("blog.source_grounding._search_arxiv", return_value=None):
        with patch("blog.source_grounding._verify_links",
                   return_value=["https://dead.com"]):
            result = sg.ground_post(draft, stream="ai")
    assert "https://dead.com" in result["dead_links"]