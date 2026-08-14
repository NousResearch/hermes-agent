"""Per-advisor MoA metrics crossing the plugin-hook boundary.

MoA runs N advisor models before its aggregator and returns only the
aggregator's response, so an observability plugin sees one generation for the
whole fan-out. ``_RefAccounting`` already computes each advisor's usage and
dollars — advisors routinely run on a different provider than the aggregator,
so their spend cannot be priced at the aggregator's rate. These tests pin the
bridge that carries it out: the renderer, the non-consuming accessor, and the
conversation-loop helper that reads it.
"""

from agent.moa_loop import _RefAccounting
from agent.moa_trace import _slot_trace, slot_metrics
from agent.usage_pricing import CanonicalUsage


def _acct():
    return _RefAccounting(
        CanonicalUsage(input_tokens=100, output_tokens=50, reasoning_tokens=7),
        0.0042,
        "ok",
        "pricing_table",
        messages=[{"role": "user", "content": "x" * 10000}],
        output="advice",
        model="claude-sonnet-4-6",
        provider="anthropic",
        temperature=0.7,
    )


class TestSlotMetrics:
    def test_carries_model_provider_usage_and_cost(self):
        m = slot_metrics(_acct(), "anthropic:claude-sonnet-4-6")

        assert m["label"] == "anthropic:claude-sonnet-4-6"
        assert m["model"] == "claude-sonnet-4-6"
        assert m["provider"] == "anthropic"
        assert m["cost_usd"] == 0.0042
        assert m["cost_status"] == "ok"
        assert m["cost_source"] == "pricing_table"
        assert m["usage"]["input_tokens"] == 100
        assert m["usage"]["output_tokens"] == 50
        assert m["usage"]["reasoning_tokens"] == 7

    def test_drops_input_messages(self):
        # input_messages is the bulk of a trace record and would cross the hook
        # boundary for every advisor on every turn.
        assert "input_messages" in _slot_trace(_acct(), "label")
        assert "input_messages" not in slot_metrics(_acct(), "label")

    def test_output_override_wins(self):
        # The privacy-redacted advisor text lives alongside the accounting, not
        # on it, so the caller supplies the output.
        m = slot_metrics(_acct(), "label", output="[redacted]")
        assert m["output"] == "[redacted]"

    def test_missing_accounting_does_not_raise(self):
        m = slot_metrics(None, "label")
        assert m["label"] == "label"
        assert m["usage"] == {}


class TestLastReferenceMetricsAccessor:
    def _client(self):
        from agent.moa_loop import MoAClient

        return MoAClient("closed")

    def test_defaults_to_none_off_the_fanout_path(self):
        assert self._client().last_reference_metrics() is None

    def test_read_does_not_consume(self):
        client = self._client()
        payload = [slot_metrics(_acct(), "label")]
        client.chat.completions._last_reference_metrics = payload

        # post_api_request fires on a different branch than
        # consume_and_save_trace, so a consuming read would race it.
        assert client.last_reference_metrics() is payload
        assert client.last_reference_metrics() is payload

    def test_read_does_not_disturb_usage_accounting(self):
        client = self._client()
        client.chat.completions._last_reference_metrics = [slot_metrics(_acct(), "label")]
        client.chat.completions._pending_reference_usage = CanonicalUsage(input_tokens=5)
        client.chat.completions._pending_reference_cost = 0.05

        client.last_reference_metrics()

        usage, cost = client.consume_reference_usage()
        assert usage.input_tokens == 5
        assert cost == 0.05


class TestConversationLoopHelper:
    def test_returns_none_for_a_non_moa_client(self):
        from agent.conversation_loop import _moa_reference_metrics_for_hook

        class _Agent:
            client = object()

        assert _moa_reference_metrics_for_hook(_Agent()) is None

    def test_returns_none_when_there_is_no_client(self):
        from agent.conversation_loop import _moa_reference_metrics_for_hook

        class _Agent:
            client = None

        assert _moa_reference_metrics_for_hook(_Agent()) is None

    def test_returns_metrics_for_a_moa_client(self):
        from agent.conversation_loop import _moa_reference_metrics_for_hook

        payload = [slot_metrics(_acct(), "label")]

        class _Client:
            def last_reference_metrics(self):
                return payload

        class _Agent:
            client = _Client()

        assert _moa_reference_metrics_for_hook(_Agent()) is payload

    def test_a_raising_accessor_is_swallowed(self):
        from agent.conversation_loop import _moa_reference_metrics_for_hook

        class _Client:
            def last_reference_metrics(self):
                raise RuntimeError("boom")

        class _Agent:
            client = _Client()

        # Observability must never break a turn.
        assert _moa_reference_metrics_for_hook(_Agent()) is None
