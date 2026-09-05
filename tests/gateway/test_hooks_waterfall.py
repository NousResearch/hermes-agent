"""Tests for emit_waterfall — Cordis-style around-middleware dispatch.

Covers the waterfall contract on gateway/hooks.py::HookRegistry:

- waterfall participants receive ``(event_type, value, context, next_fn)``;
- calling ``next_fn(new_value=...)`` delegates downstream (optionally
  rewriting the value for later listeners);
- returning without calling ``next_fn`` short-circuits the chain;
- legacy two-argument observers still run (in order, return ignored) and
  cannot rewrite or short-circuit;
- a throwing participant fails closed (chain stops), a throwing observer
  is contained (chain continues);
- async participants and observers both work.
"""

import asyncio

import pytest

from gateway.hooks import HookRegistry


class TestEmitWaterfall:
    """Direct registry tests (no filesystem discovery needed)."""

    @pytest.mark.asyncio
    async def test_delegates_value_downstream(self):
        reg = HookRegistry()
        seen = []

        async def first(event_type, value, context, next_fn):
            seen.append(("first", value))
            return await next_fn()

        def second(event_type, value, context, next_fn):
            seen.append(("second", value))
            return next_fn()

        reg._handlers["chain"] = [first, second]

        result = await reg.emit_waterfall("chain", "initial", {"k": "v"})
        assert seen == [("first", "initial"), ("second", "initial")]
        assert result == "initial"

    @pytest.mark.asyncio
    async def test_rewrite_value_propagates(self):
        reg = HookRegistry()
        seen = []

        async def rewrite(event_type, value, context, next_fn):
            seen.append(("rewrite", value))
            return await next_fn(new_value="rewritten")

        def consumer(event_type, value, context, next_fn):
            seen.append(("consumer", value))
            return next_fn()

        reg._handlers["chain"] = [rewrite, consumer]

        result = await reg.emit_waterfall("chain", "initial", {})
        assert seen == [("rewrite", "initial"), ("consumer", "rewritten")]
        assert result == "rewritten"

    @pytest.mark.asyncio
    async def test_short_circuit_after_delegation_runs_once(self):
        """Regression (triage #85370): a downstream short-circuiting handler
        must execute exactly ONCE even when upstream handlers delegated to it.

        The shared mutable ``index`` used to let each delegating ancestor
        frame resume its while-loop at the SAME un-advanced index, so with
        [A, B, C] where A and B call next_fn() and C returns without
        delegating, C ran three times (and the repeats saw the prior run's
        return value as input). Dispatch must be [A, B, C], not
        [A, B, C, C, C].
        """
        reg = HookRegistry()
        seen = []

        def a(event_type, value, context, next_fn):
            seen.append(("A", value))
            return next_fn()

        def b(event_type, value, context, next_fn):
            seen.append(("B", value))
            return next_fn()

        def c(event_type, value, context, next_fn):
            seen.append(("C", value))
            return {"decision": "deny"}  # owns the decision — no delegation

        reg._handlers["policy"] = [a, b, c]

        result = await reg.emit_waterfall("policy", "original", {})

        assert seen == [("A", "original"), ("B", "original"), ("C", "original")], (
            f"each handler must run exactly once, got {seen}"
        )
        assert result == {"decision": "deny"}
        # C must have received the ORIGINAL input, not a prior run's return.
        assert seen[2] == ("C", "original")

    @pytest.mark.asyncio
    async def test_short_circuit_after_two_delegations_three_handlers(self):
        """Triage repro: 'I don't...' exact shape — three delegating handlers
        then a short-circuiting owner. No handler may run more than once."""
        reg = HookRegistry()
        seen = []

        async def a(event_type, value, context, next_fn):
            seen.append("A")
            return await next_fn()

        async def b(event_type, value, context, next_fn):
            seen.append("B")
            return await next_fn()

        async def owner(event_type, value, context, next_fn):
            seen.append("OWNER")
            return {"decision": "allow"}

        reg._handlers["chain"] = [a, b, owner]

        result = await reg.emit_waterfall("chain", "x", {})
        assert seen == ["A", "B", "OWNER"], f"got {seen}"
        assert result == {"decision": "allow"}

    @pytest.mark.asyncio
    async def test_throwing_participant_after_delegation_runs_once(self):
        """Regression (triage #85370): a throwing participant's error was
        logged once per delegating ancestor; the handler itself re-ran for
        each ancestor frame. It must execute exactly once."""
        reg = HookRegistry()
        seen = []

        async def a(event_type, value, context, next_fn):
            seen.append("A")
            return await next_fn()

        def boom(event_type, value, context, next_fn):
            seen.append("BOOM")
            raise RuntimeError("policy crash")

        reg._handlers["policy"] = [a, boom]

        result = await reg.emit_waterfall("policy", "v", {})
        assert seen == ["A", "BOOM"], f"got {seen}"
        assert result == "v"  # fail-closed: value unchanged

    @pytest.mark.asyncio
    async def test_short_circuit_stops_chain(self):
        reg = HookRegistry()
        seen = []

        def owner(event_type, value, context, next_fn):
            seen.append("owner")
            return {"decision": "deny"}  # no next_fn() call → owns decision

        def later(event_type, value, context, next_fn):
            seen.append("later")
            return next_fn()

        reg._handlers["policy"] = [owner, later]

        result = await reg.emit_waterfall("policy", "initial", {})
        assert seen == ["owner"]
        assert result == {"decision": "deny"}

    @pytest.mark.asyncio
    async def test_observer_runs_and_cannot_short_circuit(self):
        reg = HookRegistry()
        seen = []

        def observer(event_type, context):  # legacy 2-arg signature
            seen.append(("observer", context.get("n")))

        def owner(event_type, value, context, next_fn):
            seen.append(("owner", value))
            return {"decision": "deny"}

        reg._handlers["policy"] = [observer, owner]

        result = await reg.emit_waterfall("policy", "v", {"n": 1})
        assert seen == [("observer", 1), ("owner", "v")]
        assert result == {"decision": "deny"}

    @pytest.mark.asyncio
    async def test_observer_return_ignored_chain_continues(self):
        reg = HookRegistry()
        seen = []

        def observer(event_type, context):
            return {"decision": "deny"}  # observer return must be ignored

        def owner(event_type, value, context, next_fn):
            seen.append("owner")
            return next_fn(new_value="final")

        reg._handlers["policy"] = [observer, owner]

        result = await reg.emit_waterfall("policy", "start", {})
        assert seen == ["owner"]
        assert result == "final"

    @pytest.mark.asyncio
    async def test_throwing_participant_fails_closed(self):
        reg = HookRegistry()
        seen = []

        def boom(event_type, value, context, next_fn):
            raise RuntimeError("policy crash")

        def later(event_type, value, context, next_fn):
            seen.append("later")
            return next_fn()

        reg._handlers["policy"] = [boom, later]

        result = await reg.emit_waterfall("policy", "v", {})
        assert seen == []  # chain stopped after the crash
        assert result == "v"  # value unchanged

    @pytest.mark.asyncio
    async def test_throwing_observer_is_contained(self):
        reg = HookRegistry()
        seen = []

        def boom_observer(event_type, context):
            raise RuntimeError("observer crash")

        def owner(event_type, value, context, next_fn):
            seen.append("owner")
            return next_fn()

        reg._handlers["policy"] = [boom_observer, owner]

        result = await reg.emit_waterfall("policy", "v", {})
        assert seen == ["owner"]
        assert result == "v"

    @pytest.mark.asyncio
    async def test_async_observer_runs(self):
        reg = HookRegistry()
        seen = []

        async def async_observer(event_type, context):
            seen.append("async_observer")

        def owner(event_type, value, context, next_fn):
            seen.append("owner")
            return next_fn()

        reg._handlers["policy"] = [async_observer, owner]

        await reg.emit_waterfall("policy", "v", {})
        assert seen == ["async_observer", "owner"]

    @pytest.mark.asyncio
    async def test_no_handlers_returns_value(self):
        reg = HookRegistry()
        result = await reg.emit_waterfall("none", "v", {})
        assert result == "v"

    @pytest.mark.asyncio
    async def test_wildcard_matching(self):
        reg = HookRegistry()
        seen = []

        def wildcard_owner(event_type, value, context, next_fn):
            seen.append(event_type)
            return next_fn(new_value="w")

        reg._handlers["command:*"] = [wildcard_owner]

        result = await reg.emit_waterfall("command:reset", "start", {})
        assert seen == ["command:reset"]
        assert result == "w"

    @pytest.mark.asyncio
    async def test_next_fn_called_twice_warns_and_skips_nothing(self, capsys):
        """A handler that calls next_fn() more than once must not silently
        skip or re-run downstream handlers.

        Regression for the Enough1122 review on #85370: ``next_fn()`` is a
        one-shot — the second call is a contract violation. The dispatcher
        warns and returns the current value instead of advancing the shared
        index again (which would walk past a downstream handler).
        """
        reg = HookRegistry()
        seen = []

        async def a(event_type, value, context, next_fn):
            seen.append(("A", value))
            first = await next_fn()  # real delegation — runs downstream
            second = next_fn()       # contract violation — must be ignored
            if asyncio.iscoroutine(second):
                second = await second
            return second

        def b(event_type, value, context, next_fn):
            seen.append(("B", value))
            return next_fn()

        reg._handlers["chain"] = [a, b]

        result = await reg.emit_waterfall("chain", "v0", {})
        captured = capsys.readouterr().out
        assert "more than once" in captured
        # B still ran exactly once with the original input — the second call
        # did not advance past it, and no handler re-ran.
        assert seen == [("A", "v0"), ("B", "v0")]
        assert result == "v0"

    @pytest.mark.asyncio
    async def test_sync_handler_returning_next_fn_is_awaited(self):
        """A sync handler that returns next_fn() (an un-awaited coroutine)
        delegates correctly — the dispatcher awaits the returned coroutine.

        Documented contract: sync handlers must ``return next_fn()``.
        Without the await-on-return path, the un-awaited coroutine would be
        mistaken for the handler's short-circuit result.
        """
        reg = HookRegistry()
        seen = []

        def a(event_type, value, context, next_fn):
            seen.append(("A", value))
            return next_fn(new_value="v1")  # returns coroutine, NOT awaited here

        async def b(event_type, value, context, next_fn):
            seen.append(("B", value))
            return await next_fn()

        reg._handlers["chain"] = [a, b]

        result = await reg.emit_waterfall("chain", "v0", {})
        assert seen == [("A", "v0"), ("B", "v1")]
        assert result == "v1"


class TestEmitWaterfallDiscovery:
    """Waterfall participants discovered from HOOK.yaml + handler.py files."""

    def _create_hook(self, hooks_dir, hook_name, events, handler_code):
        hook_dir = hooks_dir / hook_name
        hook_dir.mkdir(parents=True)
        (hook_dir / "HOOK.yaml").write_text(
            f"name: {hook_name}\n"
            f"description: Test hook\n"
            f"events: {events}\n"
        )
        (hook_dir / "handler.py").write_text(handler_code)
        return hook_dir

    @pytest.mark.asyncio
    async def test_discovered_waterfall_handler(self, tmp_path, monkeypatch):
        self._create_hook(
            tmp_path,
            "waterfall-hook",
            '["policy:check"]',
            (
                "def handle(event_type, value, context, next_fn):\n"
                "    return next_fn(new_value='from-hook')\n"
            ),
        )

        reg = HookRegistry()
        monkeypatch.setattr("gateway.hooks.HOOKS_DIR", tmp_path)
        reg.discover_and_load()

        assert len(reg.loaded_hooks) == 1
        result = await reg.emit_waterfall("policy:check", "start", {})
        assert result == "from-hook"
