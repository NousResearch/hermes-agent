"""Hermes usage extras (cache/cost split, real per-turn context size) on the
OpenAI-compat chat usage block. A gateway-backed WebUI client has no
in-process agent/ContextCompressor of its own and no other channel for them.
"""

from types import SimpleNamespace

from gateway.platforms.api_server import _chat_usage_payload


class TestChatUsagePayloadExtras:
    """_chat_usage_payload carries the OpenAI trio always, Hermes extras only
    when the caller's usage dict actually has them (an older/other caller's
    usage dict must not gain fabricated zero fields)."""

    def test_openai_trio_present_without_extras(self):
        usage = {"input_tokens": 10, "output_tokens": 2, "total_tokens": 12}
        payload = _chat_usage_payload(usage)
        assert payload == {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12}

    def test_extras_pass_through_when_present(self):
        usage = {
            "input_tokens": 52451, "output_tokens": 900, "total_tokens": 53351,
            "cache_read_tokens": 4000, "cache_write_tokens": 100,
            "estimated_cost": 0.42, "last_prompt_tokens": 26257, "threshold_tokens": 500000,
        }
        payload = _chat_usage_payload(usage)
        assert payload["prompt_tokens"] == 52451, "OpenAI trio still reflects the billing total"
        assert payload["last_prompt_tokens"] == 26257, (
            "the real per-turn prompt size must survive alongside the billing total, "
            "not be overwritten by it - that's the whole point of the extra field"
        )
        assert payload["threshold_tokens"] == 500000
        assert payload["cache_read_tokens"] == 4000
        assert payload["cache_write_tokens"] == 100
        assert payload["estimated_cost"] == 0.42


class TestFinishTurnResultUsageExtras:
    """_finish_turn_result must source last_prompt_tokens/threshold_tokens from the
    agent's own ContextCompressor (the single most recent real prompt), not from
    the cumulative session counters it also reports."""

    def _make_api_server(self):
        from gateway.platforms.api_server import APIServerAdapter

        return object.__new__(APIServerAdapter)

    def test_compressor_fields_clamp_the_post_compaction_sentinel(self):
        api = self._make_api_server()
        agent = SimpleNamespace(
            session_prompt_tokens=52451, session_completion_tokens=900, session_total_tokens=53351,
            session_cache_read_tokens=4000, session_cache_write_tokens=100, session_estimated_cost_usd=0.42,
            # -1 is ContextCompressor's post-compaction sentinel; a raw pass-through
            # would render a negative ring on the client.
            context_compressor=SimpleNamespace(last_prompt_tokens=-1, threshold_tokens=500000),
            session_id="sess-1", _last_compaction_in_place=False,
        )
        _, usage = api._finish_turn_result(
            agent, {"final_response": "ok"}, "sess-1",
            route=None, requested_runtime=None, route_source="global", confirmed_runtime_lock=False,
        )
        assert usage["input_tokens"] == 52451, "billing total is unaffected by the sentinel clamp"
        assert usage["last_prompt_tokens"] == 0
        assert usage["threshold_tokens"] == 500000

    def test_real_prompt_differs_from_the_cumulative_billing_total(self):
        api = self._make_api_server()
        agent = SimpleNamespace(
            session_prompt_tokens=52451, session_completion_tokens=900, session_total_tokens=53351,
            session_cache_read_tokens=0, session_cache_write_tokens=0, session_estimated_cost_usd=0.0,
            context_compressor=SimpleNamespace(last_prompt_tokens=26257, threshold_tokens=500000),
            session_id="sess-1", _last_compaction_in_place=False,
        )
        _, usage = api._finish_turn_result(
            agent, {"final_response": "ok"}, "sess-1",
            route=None, requested_runtime=None, route_source="global", confirmed_runtime_lock=False,
        )
        assert usage["input_tokens"] != usage["last_prompt_tokens"], (
            "input_tokens is a 2-call billing total (52451); last_prompt_tokens is the "
            "single most recent real prompt (26257) - a client using the former as a "
            "context-ring numerator over-reports by 2x on this turn"
        )
