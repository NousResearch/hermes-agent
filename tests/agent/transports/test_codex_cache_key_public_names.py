"""Public names for the Codex transport's prompt-cache key helpers.

A plugin that routes requests (or reports cache hit rates) needs to compute the
same ``prompt_cache_key`` the transport sends: the logical scope for a session
id, the content-addressed key, and the 64-character bound. Each public name is
the SAME object as the private spelling, so the transport is unchanged.
"""

import pytest

from agent.transports import codex


@pytest.mark.parametrize(
    ("public", "private"),
    [
        ("cache_scope_from_session_id", "_cache_scope_from_session_id"),
        ("bounded_prompt_cache_key", "_bounded_prompt_cache_key"),
        ("content_cache_key", "_content_cache_key"),
    ],
)
def test_public_name_is_the_private_helper(public, private):
    assert getattr(codex, public) is getattr(codex, private)


def test_cache_scope_folds_cron_fires_onto_the_job():
    assert codex.cache_scope_from_session_id("cron_abc_20260101_120000") == "cron_abc"
    assert codex.cache_scope_from_session_id("chat-1") == "chat-1"


def test_bounded_prompt_cache_key_caps_length():
    assert codex.bounded_prompt_cache_key("short") == "short"
    assert len(codex.bounded_prompt_cache_key("x" * 200)) <= 64
    assert codex.bounded_prompt_cache_key("  ") is None


def test_content_cache_key_is_tool_order_independent():
    tools = [{"name": "b"}, {"name": "a"}]
    assert (codex.content_cache_key("sys", tools, "s")
            == codex.content_cache_key("sys", list(reversed(tools)), "s"))
    assert codex.content_cache_key("", None) is None
