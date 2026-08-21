"""Tests for LLMExecutionBlocked (#64662).

Covers: the exception's own contract, direct-raise propagation, downstream
propagation through a pass-through middleware, multi-level chains, and
regressions proving normal (non-blocking) middleware behavior is unaffected.
"""

import pytest

from hermes_cli import middleware as middleware_module
from hermes_cli.middleware import LLMExecutionBlocked, run_llm_execution_middleware


def _set_callbacks(monkeypatch, callbacks):
    monkeypatch.setattr(
        middleware_module,
        "_get_middleware_callbacks",
        lambda kind: list(callbacks),
    )


class TestLLMExecutionBlockedContract:
    def test_reason_and_metadata(self):
        exc = LLMExecutionBlocked("budget exceeded", metadata={"budget_usd": 5.0})
        assert exc.reason == "budget exceeded"
        assert exc.metadata == {"budget_usd": 5.0}
        assert str(exc) == "budget exceeded"

    def test_metadata_defaults_to_empty_dict(self):
        exc = LLMExecutionBlocked("blocked")
        assert exc.metadata == {}

    def test_is_a_plain_exception(self):
        exc = LLMExecutionBlocked("blocked")
        assert isinstance(exc, Exception)
        with pytest.raises(Exception):
            raise exc


class TestDirectRaise:
    def test_direct_raise_propagates_without_calling_provider(self, monkeypatch):
        provider_called = []

        def blocking_middleware(request, next_call, **kw):
            raise LLMExecutionBlocked("budget exceeded", metadata={"budget_usd": 5.0})

        _set_callbacks(monkeypatch, [blocking_middleware])

        def terminal(request):
            provider_called.append(request)
            return {"ok": True}

        with pytest.raises(LLMExecutionBlocked) as excinfo:
            run_llm_execution_middleware({"model": "x"}, terminal)

        assert excinfo.value.reason == "budget exceeded"
        # The runner backfills checked_by from the raising callback's own
        # name when the plugin didn't set it — original metadata is preserved
        # alongside it, not replaced.
        assert excinfo.value.metadata == {
            "budget_usd": 5.0,
            "checked_by": "blocking_middleware",
        }
        assert provider_called == [], "terminal provider call must not happen when blocked"

    def test_next_call_result_discarded_if_middleware_still_raises(self, monkeypatch):
        """A middleware that calls next_call() and then raises must still
        block — the successful downstream result is not silently returned."""

        def calls_then_blocks(request, next_call, **kw):
            next_call(request)
            raise LLMExecutionBlocked("blocked after downstream ran")

        _set_callbacks(monkeypatch, [calls_then_blocks])

        with pytest.raises(LLMExecutionBlocked, match="blocked after downstream ran"):
            run_llm_execution_middleware({"model": "x"}, lambda r: {"ok": True})


class TestDownstreamPropagation:
    def test_propagates_through_outer_pass_through_middleware(self, monkeypatch):
        """An outer middleware that only calls next_call() and returns its
        result must not swallow a block raised further down the chain."""

        def pass_through(request, next_call, **kw):
            return next_call(request)

        def blocks(request, next_call, **kw):
            raise LLMExecutionBlocked("blocked downstream")

        _set_callbacks(monkeypatch, [pass_through, blocks])

        with pytest.raises(LLMExecutionBlocked, match="blocked downstream"):
            run_llm_execution_middleware({"model": "x"}, lambda r: {"ok": True})

    def test_three_deep_chain_block_at_first_stops_all_downstream(self, monkeypatch):
        called = []

        def first(request, next_call, **kw):
            raise LLMExecutionBlocked("stopped at first")

        def second(request, next_call, **kw):
            called.append("second")
            return next_call(request)

        def third(request, next_call, **kw):
            called.append("third")
            return next_call(request)

        _set_callbacks(monkeypatch, [first, second, third])

        provider_called = []
        with pytest.raises(LLMExecutionBlocked, match="stopped at first"):
            run_llm_execution_middleware(
                {"model": "x"}, lambda r: provider_called.append(r) or {"ok": True}
            )

        assert called == [], "middleware after the block must never run"
        assert provider_called == [], "terminal provider must never run"

    def test_block_from_terminal_call_propagates_through_chain(self, monkeypatch):
        """The block can also originate at the terminal provider call itself
        (e.g. a governance check inside the provider wrapper), not just from
        a registered middleware callback."""

        def pass_through(request, next_call, **kw):
            return next_call(request)

        def terminal(request):
            raise LLMExecutionBlocked("blocked at terminal")

        _set_callbacks(monkeypatch, [pass_through])

        with pytest.raises(LLMExecutionBlocked, match="blocked at terminal"):
            run_llm_execution_middleware({"model": "x"}, terminal)


class TestRegressionNormalBehaviorUnaffected:
    def test_normal_exception_without_next_call_still_falls_through(self, monkeypatch):
        """Pre-existing behavior: an ordinary Exception (not
        LLMExecutionBlocked) from a middleware that never called next_call
        must still fall through to the next callback, unchanged."""

        def broken(request, next_call, **kw):
            raise ValueError("unrelated failure")

        def recovers(request, next_call, **kw):
            return next_call(request)

        _set_callbacks(monkeypatch, [broken, recovers])

        result = run_llm_execution_middleware({"model": "x"}, lambda r: {"ok": True})
        assert result == {"ok": True}

    def test_llm_execution_blocked_not_treated_as_generic_exception(self, monkeypatch):
        """Regression guard: LLMExecutionBlocked must take the explicit
        re-raise path, never the general fallthrough — i.e. it must not
        result in the next callback being invoked."""

        called = []

        def blocks(request, next_call, **kw):
            raise LLMExecutionBlocked("blocked")

        def would_recover(request, next_call, **kw):
            called.append("would_recover")
            return next_call(request)

        _set_callbacks(monkeypatch, [blocks, would_recover])

        with pytest.raises(LLMExecutionBlocked):
            run_llm_execution_middleware({"model": "x"}, lambda r: {"ok": True})

        assert called == [], "fallthrough must not occur for LLMExecutionBlocked"

    def test_no_callbacks_fast_path_unaffected(self, monkeypatch):
        _set_callbacks(monkeypatch, [])
        result = run_llm_execution_middleware({"model": "x"}, lambda r: {"ok": True, "req": r})
        assert result == {"ok": True, "req": {"model": "x"}}

    def test_metadata_accessible_on_caught_exception(self, monkeypatch):
        def blocks(request, next_call, **kw):
            raise LLMExecutionBlocked(
                "cost budget exceeded",
                metadata={"budget_usd": 5.0, "session_id": "abc123"},
            )

        _set_callbacks(monkeypatch, [blocks])

        try:
            run_llm_execution_middleware({"model": "x"}, lambda r: {"ok": True})
            assert False, "expected LLMExecutionBlocked"
        except LLMExecutionBlocked as caught:
            assert caught.metadata["budget_usd"] == 5.0
            assert caught.metadata["session_id"] == "abc123"


class TestCheckedByBackfill:
    """checked_by envelope convention (docs/plugins/hook-taxonomy.md): required
    on the deny-path, but a plugin never has to set it itself — the runner
    backfills it from the raising callback's own registered name."""

    def test_checked_by_backfilled_when_omitted(self, monkeypatch):
        def budget_guard(request, next_call, **kw):
            raise LLMExecutionBlocked("budget exceeded")

        _set_callbacks(monkeypatch, [budget_guard])

        with pytest.raises(LLMExecutionBlocked) as excinfo:
            run_llm_execution_middleware({"model": "x"}, lambda r: {"ok": True})

        assert excinfo.value.metadata["checked_by"] == "budget_guard"

    def test_checked_by_not_overwritten_when_plugin_sets_it(self, monkeypatch):
        def budget_guard(request, next_call, **kw):
            raise LLMExecutionBlocked(
                "budget exceeded", metadata={"checked_by": "amp-governance"}
            )

        _set_callbacks(monkeypatch, [budget_guard])

        with pytest.raises(LLMExecutionBlocked) as excinfo:
            run_llm_execution_middleware({"model": "x"}, lambda r: {"ok": True})

        # The plugin's own attribution wins — the runner only fills a gap,
        # it never overrides an explicit value.
        assert excinfo.value.metadata["checked_by"] == "amp-governance"

    def test_checked_by_reflects_raising_callback_in_a_multi_middleware_chain(
        self, monkeypatch
    ):
        def observer_like(request, next_call, **kw):
            return next_call(request)

        def budget_guard(request, next_call, **kw):
            raise LLMExecutionBlocked("budget exceeded")

        _set_callbacks(monkeypatch, [observer_like, budget_guard])

        with pytest.raises(LLMExecutionBlocked) as excinfo:
            run_llm_execution_middleware({"model": "x"}, lambda r: {"ok": True})

        assert excinfo.value.metadata["checked_by"] == "budget_guard"
