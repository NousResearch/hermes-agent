"""Tests for S9 GuardrailMiddleware (pre-tool-call authorization)."""

from __future__ import annotations

import os
from unittest import mock

import pytest

from agent_bus.middleware import (
    MiddlewareChain,
    MiddlewareContext,
    clear_registry,
)


@pytest.fixture(autouse=True)
def reset_registry():
    clear_registry()
    yield
    clear_registry()


class TestGuardrail:
    def _reg(self):
        from agent_bus.middlewares import register_defaults
        register_defaults()

    def test_no_pending_tool_call_passthrough(self):
        self._reg()
        chain = MiddlewareChain.build()
        ctx = MiddlewareContext()
        ctx = chain.run("before_tool", ctx)
        # No tool call → no guardrail decision
        assert not any(d["middleware"] == "guardrail" for d in ctx.decisions)

    def test_default_mode_denylist_allows_unknown(self):
        self._reg()
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("HERMES_GUARDRAIL_MODE", None)
            os.environ.pop("HERMES_TOOL_DENYLIST", None)
            chain = MiddlewareChain.build()
            ctx = MiddlewareContext(
                pending_tool_call={"id": "c1", "name": "wiki_search"},
            )
            ctx = chain.run("before_tool", ctx)
            assert any(d["middleware"] == "guardrail" and d["action"] == "allow"
                       for d in ctx.decisions)
            # pending still present → caller should dispatch
            assert ctx.pending_tool_call is not None

    def test_denylist_blocks(self):
        self._reg()
        with mock.patch.dict(os.environ, {
            "HERMES_GUARDRAIL_MODE": "denylist",
            "HERMES_TOOL_DENYLIST": "bash,rm",
        }):
            chain = MiddlewareChain.build()
            ctx = MiddlewareContext(
                pending_tool_call={"id": "c1", "name": "bash"},
            )
            ctx = chain.run("before_tool", ctx)
            assert any(d["middleware"] == "guardrail" and d["action"] == "deny"
                       for d in ctx.decisions)
            # pending cleared
            assert ctx.pending_tool_call is None
            # denial recorded in metadata + synthetic tool message injected
            assert ctx.metadata["guardrail_denials"]
            deny_msgs = [m for m in ctx.messages if m.get("_guardrail_denied")]
            assert len(deny_msgs) == 1
            assert deny_msgs[0]["tool_call_id"] == "c1"

    def test_allowlist_mode(self):
        self._reg()
        with mock.patch.dict(os.environ, {
            "HERMES_GUARDRAIL_MODE": "allowlist",
            "HERMES_TOOL_ALLOWLIST": "wiki_search,memory_recall",
        }):
            chain = MiddlewareChain.build()
            # Allowed
            ctx1 = MiddlewareContext(
                pending_tool_call={"id": "c1", "name": "wiki_search"},
            )
            chain.run("before_tool", ctx1)
            assert ctx1.pending_tool_call is not None
            # Not allowed
            ctx2 = MiddlewareContext(
                pending_tool_call={"id": "c2", "name": "bash"},
            )
            chain.run("before_tool", ctx2)
            assert ctx2.pending_tool_call is None

    def test_off_mode_always_allows(self):
        self._reg()
        with mock.patch.dict(os.environ, {
            "HERMES_GUARDRAIL_MODE": "off",
            "HERMES_TOOL_DENYLIST": "bash",
        }):
            chain = MiddlewareChain.build()
            ctx = MiddlewareContext(
                pending_tool_call={"id": "c1", "name": "bash"},
            )
            chain.run("before_tool", ctx)
            assert ctx.pending_tool_call is not None

    def test_custom_provider_via_metadata(self):
        self._reg()

        class OnlyAllowReads:
            def should_allow(self, tc):
                name = tc.get("name", "")
                if "read" in name or "search" in name:
                    return True, ""
                return False, f"only read/search allowed, got `{name}`"

        chain = MiddlewareChain.build()
        ctx = MiddlewareContext(
            pending_tool_call={"id": "c1", "name": "write_file"},
            metadata={"guardrail_provider": OnlyAllowReads()},
        )
        chain.run("before_tool", ctx)
        assert ctx.pending_tool_call is None
        denials = ctx.metadata["guardrail_denials"]
        assert "only read/search" in denials[0]["reason"]

    def test_provider_exception_defaults_to_deny(self):
        self._reg()

        class BuggyProvider:
            def should_allow(self, tc):
                raise RuntimeError("oops")

        chain = MiddlewareChain.build()
        ctx = MiddlewareContext(
            pending_tool_call={"id": "c1", "name": "anything"},
            metadata={"guardrail_provider": BuggyProvider()},
        )
        chain.run("before_tool", ctx)
        # Buggy provider → safe default = deny
        assert ctx.pending_tool_call is None
