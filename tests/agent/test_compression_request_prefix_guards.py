"""Prefix replay must retain compaction's safety and session isolation boundaries."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from threading import Barrier

import pytest

from agent.context_compressor import ContextCompressor
from agent.context_compressor_cache import SummaryPrefix, select_summary_input, summary_main_runtime, summary_prefix_scope


def _prefix(session="one"):
    return SummaryPrefix(
        system_prompt=f"Coding project {session}.", tools=[], reasoning_config=None, context_length=128000,
        runtime={"provider": "custom", "model": "gpt-4.1", "base_url": "http://127.0.0.1:9999/v1",
                 "api_mode": "codex_responses", "session_id": session},
        messages=[{"role": "user", "content": "Review the implementation."},
                  {"role": "assistant", "content": "The implementation is ready for review."}],
    )


def _compressor():
    return ContextCompressor(model="gpt-4.1", config_context_length=128000, quiet_mode=True)


@pytest.mark.parametrize("change,request_model,request_base", [
    pytest.param(lambda p: replace(p, messages=[{"role": "user", "content": "password: private-test-password"}]),
                 None, None, id="secret"),
    pytest.param(lambda p: replace(p, tools=[{"password": "private-test-password"}]), None, None, id="tool-secret"),
    pytest.param(lambda p: replace(p, messages=[{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "x"}}]}]),
                 None, None, id="image"),
    pytest.param(lambda p: replace(p, messages=[{"role": "assistant", "content": "<think>private thought</think>Answer."}]),
                 None, None, id="reasoning"),
    pytest.param(lambda p: replace(p, context_length=100), None, None, id="context"),
    pytest.param(lambda p: replace(p, runtime={**p.runtime, "api_mode": "anthropic_messages"}), None, None, id="wire"),
    pytest.param(lambda p: p, "other-model", None, id="model"),
    pytest.param(lambda p: p, None, "http://127.0.0.1:9998/v1", id="endpoint"),
])
def test_replay_falls_back_at_safety_and_route_boundaries(change, request_model, request_base):
    prefix = change(_prefix())
    bounded = [{"role": "user", "content": "Redacted, bounded summary input."}]
    with summary_prefix_scope(prefix, _compressor(), 2000, None, "", True):
        messages, tools, reasoning = select_summary_input(
            "compression", "custom", request_model or "gpt-4.1", request_base or prefix.runtime["base_url"],
            "codex_responses", bounded, None, None,
        )
    assert messages is bounded
    assert tools is None and reasoning is None


def test_overlapping_summary_scopes_do_not_share_prefixes_or_runtime():
    barrier = Barrier(2)
    compressor = _compressor()

    def request_for(session):
        prefix = _prefix(session)
        with summary_prefix_scope(prefix, compressor, 2000, None, "", True):
            barrier.wait(timeout=5)
            messages, _, _ = select_summary_input(
                "compression", "custom", "gpt-4.1", prefix.runtime["base_url"], "codex_responses", [], None, None,
            )
            runtime = summary_main_runtime({})
        assert summary_main_runtime({}) == {}
        return messages[0]["content"], runtime["session_id"]

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(request_for, ["one", "two"]))
    assert results == [("Coding project one.", "one"), ("Coding project two.", "two")]
    assert summary_main_runtime({}) == {}
