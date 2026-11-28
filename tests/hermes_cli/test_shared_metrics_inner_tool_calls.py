"""Inner tool calls in the shared metrics: a nested call now arrives with ``<owner>#<n>`` and no
``api_request_id``. It must open and close exactly one lifecycle instead of landing in the
unidentified bucket, and it must not collide with the turn's top-level calls."""

from __future__ import annotations

import pytest

from hermes_cli import lifecycle

from tests.hermes_cli import test_relay_shared_metrics_runtime as runtime_tests

_BASE = {"session_id": "s1", "task_id": "t1", "turn_id": "turn-1", "api_request_id": "r1", "platform": "cli"}


@pytest.fixture
def fake_relay(tmp_path, monkeypatch):
    """The runtime tests' fake Relay, so the real metrics runtime consumes the hook payloads."""
    yield from runtime_tests.direct_runtime.__wrapped__(tmp_path, monkeypatch)


def test_an_inner_call_opens_and_closes_one_tool_lifecycle(fake_relay):
    lifecycle.invoke_hook("pre_llm_call", **_BASE)
    # The outer batch call, then two inner calls as the leaf now dispatches them: ids present,
    # api_request_id deliberately empty (PR 1 does not forward it).
    outer = {**_BASE, "tool_call_id": "B1", "tool_name": "execute_code", "args": {}}
    lifecycle.invoke_hook("pre_tool_call", **outer)
    for n in (1, 2):
        inner = {**_BASE, "api_request_id": "", "tool_call_id": f"B1#{n}",
                 "tool_name": "web_search", "args": {"query": f"q{n}"}}
        lifecycle.invoke_hook("pre_tool_call", **inner)
        lifecycle.invoke_hook("post_tool_call", **inner, result='{"ok": 1}', status="ok")
    lifecycle.invoke_hook("post_tool_call", **outer, result='{"ok": 1}', status="ok")

    ends = [event[2] for event in fake_relay.events if event[0] == "tool.call_end"]
    assert len(ends) == 3, ends
    assert [end["tool_name"] for end in ends].count("web_search") == 2, ends
    assert all(end["outcome"] == "success" for end in ends), ends
