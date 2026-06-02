"""Regression test for title generation with reasoning models.

Reasoning models (Qwen3.6, DeepSeek-R1, etc.) return ``content=None``
with the actual output in ``reasoning_content``. The title generator
used to read ``.content`` directly, so every session backed by a
reasoning model stayed untitled. It now routes through
``extract_content_or_reasoning`` which falls back to the structured
reasoning fields. See agent/title_generator.py (generate_title).
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

from agent.title_generator import generate_title


def _response(content=None, reasoning_content=None, reasoning=None):
    msg = SimpleNamespace(
        content=content,
        reasoning_content=reasoning_content,
        reasoning=reasoning,
    )
    return SimpleNamespace(choices=[SimpleNamespace(message=msg)])


def test_title_from_reasoning_content_when_content_null():
    """content=None + reasoning_content set → title comes from reasoning."""
    resp = _response(content=None, reasoning_content="Database Migration Plan")
    with patch("agent.title_generator.call_llm", return_value=resp):
        title = generate_title("migrate the db", "here's the plan")
    assert title == "Database Migration Plan"


def test_title_from_content_when_present():
    """Plain content path is unaffected by the fix."""
    resp = _response(content="Weekend Trip Ideas")
    with patch("agent.title_generator.call_llm", return_value=resp):
        title = generate_title("trip?", "some ideas")
    assert title == "Weekend Trip Ideas"


def test_title_cleanup_runs_on_reasoning_sourced_text():
    """The existing cleanup (Title: prefix; surrounding quotes) still runs
    on text sourced from reasoning_content, not just from content."""
    # Quote stripping on a quoted reasoning title.
    resp = _response(content=None, reasoning_content='"Quarterly Report"')
    with patch("agent.title_generator.call_llm", return_value=resp):
        assert generate_title("q report", "done") == "Quarterly Report"

    # "Title:" prefix stripping on a reasoning title.
    resp2 = _response(content=None, reasoning_content="Title: Sprint Planning")
    with patch("agent.title_generator.call_llm", return_value=resp2):
        assert generate_title("plan", "done") == "Sprint Planning"


def test_no_title_when_both_empty():
    """content=None and no reasoning → no title (falls back to None/empty)."""
    resp = _response(content=None, reasoning_content=None)
    with patch("agent.title_generator.call_llm", return_value=resp):
        title = generate_title("hi", "hello")
    assert not title
