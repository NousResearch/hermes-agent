"""Prompt-contract tests: reader-facing copy + named-source citation rules."""

from blog.blog_generator import build_blog_prompt


def _system(prompts):
    return prompts["system"] if isinstance(prompts, dict) else str(prompts)


def test_essay_prompt_forbids_requester_facing_meta_notes():
    plan = {"topic_id": "t", "title_hint": "T", "signals": [{"summary": "s"}]}
    system = _system(build_blog_prompt("builder", plan, "ctx", []))
    assert "reader-facing copy" in system
    assert "Never address the requester" in system


def test_essay_prompt_requires_named_citations():
    plan = {"topic_id": "t", "title_hint": "T", "signals": [{"summary": "s"}]}
    system = _system(build_blog_prompt("builder", plan, "ctx", []))
    assert "cite it by name" in system
    assert "Never invent" in system


def test_research_prompt_reuses_shared_rules():
    plan = {"topic_id": "t", "title_hint": "T", "signals": [{"summary": "s"}]}
    system = _system(build_blog_prompt("research", plan, "ctx", []))
    assert "reader-facing copy" in system
    assert "cite it by name" in system
