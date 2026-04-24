"""Unit tests for agent_bus.middleware — S3 named middleware chain framework.

Spec: ~/wiki/concepts/hermes-openclaw-deerflow-integration-plan.md §S3
Covers registry, ordered execution, env-var gating, abort short-circuit,
hook correctness, error safety.
"""

from __future__ import annotations

import os
from unittest import mock

import pytest

from agent_bus.middleware import (
    BaseMiddleware,
    HOOKS,
    MiddlewareChain,
    MiddlewareContext,
    all_entries,
    clear_registry,
    register,
    registered,
)


@pytest.fixture(autouse=True)
def reset_registry():
    clear_registry()
    yield
    clear_registry()


# -------- Basic plumbing --------
class TestRegistry:
    def test_register_and_list(self):
        @register(order=10)
        class MW(BaseMiddleware):
            name = "test-mw"

        assert len(registered()) == 1
        assert registered()[0].name == "test-mw"

    def test_order_matters(self):
        @register(order=30)
        class C(BaseMiddleware):
            name = "c"

        @register(order=10)
        class A(BaseMiddleware):
            name = "a"

        @register(order=20)
        class B(BaseMiddleware):
            name = "b"

        names = [mw.name for mw in registered()]
        assert names == ["a", "b", "c"]

    def test_missing_name_attr_raises(self):
        with pytest.raises(TypeError):

            @register(order=10)
            class NoName(BaseMiddleware):
                pass

            # force access (registry invokes cls())

    def test_all_entries_includes_disabled(self):
        with mock.patch.dict(os.environ, {"HERMES_MW_FOO": "off"}):

            @register(order=10, env_var="HERMES_MW_FOO")
            class MW(BaseMiddleware):
                name = "foo"

            entries = all_entries()
            assert len(entries) == 1
            assert not entries[0].is_enabled()
            assert registered() == []


# -------- Hook execution --------
class TestHookExecution:
    def test_hooks_exist(self):
        assert set(HOOKS) == {
            "before_model",
            "after_model",
            "before_tool",
            "after_tool",
            "on_session_end",
        }

    def test_unknown_hook_raises(self):
        chain = MiddlewareChain.build()
        with pytest.raises(ValueError):
            chain.run("not-a-hook", MiddlewareContext())

    def test_mutation_propagates_through_chain(self):
        @register(order=10)
        class Append1(BaseMiddleware):
            name = "append1"

            def after_model(self, ctx):
                ctx.metadata.setdefault("trace", []).append("append1")
                return ctx

        @register(order=20)
        class Append2(BaseMiddleware):
            name = "append2"

            def after_model(self, ctx):
                ctx.metadata["trace"].append("append2")
                return ctx

        chain = MiddlewareChain.build()
        ctx = chain.run("after_model", MiddlewareContext())
        assert ctx.metadata["trace"] == ["append1", "append2"]

    def test_other_hooks_do_not_fire_on_single_hook(self):
        calls = []

        @register(order=10)
        class AllHooks(BaseMiddleware):
            name = "spy"

            def before_model(self, ctx):
                calls.append("before_model")
                return ctx

            def after_model(self, ctx):
                calls.append("after_model")
                return ctx

        chain = MiddlewareChain.build()
        chain.run("before_model", MiddlewareContext())
        assert calls == ["before_model"]
        chain.run("after_model", MiddlewareContext())
        assert calls == ["before_model", "after_model"]


# -------- Env-var gating --------
class TestEnvGating:
    def test_master_switch_off_skips_all(self):
        calls = []

        @register(order=10)
        class Spy(BaseMiddleware):
            name = "spy"

            def after_model(self, ctx):
                calls.append("ran")
                return ctx

        with mock.patch.dict(os.environ, {"HERMES_MIDDLEWARE_CHAIN": "off"}):
            chain = MiddlewareChain.build()
            chain.run("after_model", MiddlewareContext())
        assert calls == []

    def test_per_middleware_env_off(self):
        calls = []

        @register(order=10, env_var="HERMES_MW_OFF")
        class A(BaseMiddleware):
            name = "off-one"

            def after_model(self, ctx):
                calls.append("off-one")
                return ctx

        @register(order=20)
        class B(BaseMiddleware):
            name = "stays"

            def after_model(self, ctx):
                calls.append("stays")
                return ctx

        with mock.patch.dict(os.environ, {"HERMES_MW_OFF": "off"}):
            chain = MiddlewareChain.build()
            chain.run("after_model", MiddlewareContext())
        assert calls == ["stays"]

    def test_default_is_core_on(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("HERMES_MIDDLEWARE_CHAIN", None)

            @register(order=10)
            class MW(BaseMiddleware):
                name = "default-on"

                def after_model(self, ctx):
                    ctx.metadata["ran"] = True
                    return ctx

            chain = MiddlewareChain.build()
            ctx = chain.run("after_model", MiddlewareContext())
            assert ctx.metadata.get("ran") is True


# -------- Abort --------
class TestAbort:
    def test_aborted_short_circuits(self):
        calls = []

        @register(order=10)
        class Early(BaseMiddleware):
            name = "early"

            def after_model(self, ctx):
                calls.append("early")
                ctx.aborted = True
                return ctx

        @register(order=20)
        class Late(BaseMiddleware):
            name = "late"

            def after_model(self, ctx):
                calls.append("late")
                return ctx

        chain = MiddlewareChain.build()
        ctx = chain.run("after_model", MiddlewareContext())
        assert calls == ["early"]
        assert ctx.aborted
        # abort decision is logged
        assert any(d["action"] == "abort" for d in ctx.decisions)


# -------- Error safety --------
class TestErrorSafety:
    def test_non_critical_exception_continues(self):
        @register(order=10, critical=False)
        class Buggy(BaseMiddleware):
            name = "buggy"

            def after_model(self, ctx):
                raise RuntimeError("oops")

        @register(order=20)
        class Good(BaseMiddleware):
            name = "good"

            def after_model(self, ctx):
                ctx.metadata["good_ran"] = True
                return ctx

        chain = MiddlewareChain.build()
        ctx = chain.run("after_model", MiddlewareContext())
        assert ctx.metadata.get("good_ran") is True
        assert any(d["action"] == "error" and d["middleware"] == "buggy" for d in ctx.decisions)

    def test_critical_exception_raises(self):
        @register(order=10, critical=True)
        class Critical(BaseMiddleware):
            name = "critical"

            def after_model(self, ctx):
                raise RuntimeError("fatal")

        chain = MiddlewareChain.build()
        with pytest.raises(RuntimeError):
            chain.run("after_model", MiddlewareContext())


# -------- Decisions log --------
class TestDecisionsLog:
    def test_record_accumulates(self):
        ctx = MiddlewareContext()
        ctx.record("foo", "after_model", "rewrite", "normalized")
        ctx.record("bar", "before_tool", "allow", "")
        assert len(ctx.decisions) == 2
        assert ctx.decisions[0]["middleware"] == "foo"
        assert ctx.decisions[1]["action"] == "allow"
