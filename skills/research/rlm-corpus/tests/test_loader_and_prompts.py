"""Tests for the corpus loader and prompt builders."""
from __future__ import annotations

import json
from pathlib import Path

from corpus_loader import corpus_summary, is_cache_dir, load_corpus
from ingestion import ingest_directory
from prompts import build_root_system_prompt, wrap_sub_llm_prompt


FIXTURES = Path(__file__).parent / "fixtures" / "tiny-corpus"


def test_load_corpus_round_trip(tmp_path):
    cache = tmp_path / "cache"
    ingest_directory(FIXTURES, cache)
    assert is_cache_dir(cache)

    corpus = load_corpus(cache)
    assert len(corpus) >= 3
    for name, doc in corpus.items():
        assert "full_text" in doc
        assert "sections" in doc
        assert doc["metadata"]["title"]

    summary = corpus_summary(corpus)
    assert summary["num_docs"] == len(corpus)
    assert summary["total_chars"] > 0


def test_is_cache_dir_false_for_empty(tmp_path):
    assert not is_cache_dir(tmp_path)


def test_is_cache_dir_ignores_underscore_files(tmp_path):
    (tmp_path / "_manifest.json").write_text("{}")
    assert not is_cache_dir(tmp_path)


def test_root_system_prompt_mentions_query_and_counts():
    corpus = {
        "a.md": {"metadata": {"source_type": "md"}, "stats": {"char_count": 100}},
        "b.pdf": {"metadata": {"source_type": "pdf"}, "stats": {"char_count": 200}},
    }
    prompt = build_root_system_prompt(corpus, "why does X happen?")
    assert "why does X happen?" in prompt
    assert "2 documents" in prompt
    assert "300 total characters" in prompt
    assert "1 md" in prompt
    assert "1 pdf" in prompt


def test_root_system_prompt_disables_sub_calls_note():
    prompt = build_root_system_prompt({}, "q", enable_sub_calls=False)
    assert "DISABLED" in prompt


def test_wrap_sub_llm_prompt_contains_user_text():
    out = wrap_sub_llm_prompt("what is the main claim?")
    assert "what is the main claim?" in out
    assert "ONLY on the" in out and "provided text" in out
