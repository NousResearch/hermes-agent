"""Tests for Block 7: Original Framework Generation.

Covers:
  - frameworks.jsonl seed file structure
  - _framework_prompt_builder: name, levels, diagram, mapping table
  - Framework prompt structure and content
"""
import json
from pathlib import Path

import pytest

from blog.blog_generator import _framework_prompt_builder
from blog.blog_streams import STREAMS


_SEED_PATH = Path(__file__).parent.parent / "blog_topics" / "frameworks.jsonl"


def test_frameworks_jsonl_exists():
    """The frameworks seed file exists at the expected path."""
    assert _SEED_PATH.exists(), f"Missing {_SEED_PATH}"


def test_frameworks_jsonl_has_6_plus_seeds():
    """The seed file has at least 6 framework seeds."""
    lines = [l for l in _SEED_PATH.read_text().splitlines() if l.strip()]
    assert len(lines) >= 6, f"Only {len(lines)} seeds found"


def test_framework_seeds_have_required_fields():
    """Each seed has topic_id, title_hint, tags, priority, domain."""
    for line in _SEED_PATH.read_text().splitlines():
        if not line.strip():
            continue
        seed = json.loads(line)
        assert "topic_id" in seed
        assert "title_hint" in seed
        assert "tags" in seed
        assert "priority" in seed
        assert "domain" in seed


def test_framework_prompt_mentions_name():
    """Framework prompt instructs the LLM to name the framework."""
    plan = {"topic_id": "fw-001", "title_hint": "token efficiency", "tags": ["ai"],
            "signals": [{"signal_id": "s1", "summary": "token frontier"}]}
    prompts = _framework_prompt_builder("ai", plan, "context", [])
    assert "name" in prompts["system"].lower()
    assert "2-4 word" in prompts["system"].lower()


def test_framework_prompt_mentions_levels():
    """Framework prompt instructs 3-5 levels."""
    plan = {"topic_id": "fw-001", "title_hint": "t", "tags": [],
            "signals": [{"signal_id": "s1", "summary": "s"}]}
    prompts = _framework_prompt_builder("ai", plan, "ctx", [])
    assert "3-5 level" in prompts["system"].lower() or "3-5 stages" in prompts["system"].lower()


def test_framework_prompt_mentions_mermaid():
    """Framework prompt instructs a Mermaid diagram."""
    plan = {"topic_id": "fw-001", "title_hint": "t", "tags": [],
            "signals": [{"signal_id": "s1", "summary": "s"}]}
    prompts = _framework_prompt_builder("ai", plan, "ctx", [])
    assert "mermaid" in prompts["system"].lower()


def test_framework_prompt_mentions_mapping_table():
    """Framework prompt instructs a mapping table."""
    plan = {"topic_id": "fw-001", "title_hint": "t", "tags": [],
            "signals": [{"signal_id": "s1", "summary": "s"}]}
    prompts = _framework_prompt_builder("ai", plan, "ctx", [])
    assert "mapping table" in prompts["system"].lower()


def test_framework_prompt_includes_title_hint():
    """Framework prompt includes the seed title hint."""
    plan = {"topic_id": "fw-001", "title_hint": "Token Frontier Framework", "tags": [],
            "signals": [{"signal_id": "s1", "summary": "s"}]}
    prompts = _framework_prompt_builder("ai", plan, "ctx", [])
    assert "Token Frontier Framework" in prompts["user"]


def test_framework_prompt_includes_retry_feedback():
    """Retry feedback is threaded into framework prompt."""
    plan = {"topic_id": "fw-001", "title_hint": "t", "tags": [],
            "signals": [{"signal_id": "s1", "summary": "s"}]}
    prompts = _framework_prompt_builder("ai", plan, "ctx", [],
                                        retry_feedback="missing levels")
    assert "missing levels" in prompts["system"]