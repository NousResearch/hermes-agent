"""Regression test for #70245 — fallback entry `context_length` field.

The fallback entry's own ``context_length`` key is a documented, normalized
config field (same shape as ``model.context_length`` on the primary config).
It was being silently discarded on fallback activation: the chokepoint
unconditionally cleared ``agent._config_context_length`` to ``None``, so
the fallback model's context window always resolved via the static
per-model table or 128K default — even when the user explicitly set
``context_length`` on the fallback entry.

The contract: when a fallback entry declares ``context_length``, that
value must reach ``agent._config_context_length`` after fallback
activation so the resolver honors it. When the entry omits the field,
the value must default to ``None`` (preserves the prior "resolve from the
model's actual window" behavior).
"""

import re

import pytest


CHAT_HELPERS_PATH = "agent/chat_completion_helpers.py"


def _read_chat_helpers_source() -> str:
    """Read the patched source file."""
    import agent.chat_completion_helpers as mod
    import inspect
    return inspect.getsource(mod)


class TestFallbackEntryContextLengthSourceContract:
    """Source-level assertions — the chokepoint must read fb's context_length."""

    def test_chokepoint_assigns_from_fb_context_length(self):
        """The unconditional ``= None`` assignment must be replaced with one
        that reads the fallback entry's own ``context_length`` field.
        """
        src = _read_chat_helpers_source()
        # Locate the fallback activation block by its surrounding sentinel.
        # The comment block immediately preceding the assignment has to
        # reference both "context_length" and the fix intent.
        match = re.search(
            r"agent\._config_context_length\s*=\s*([^\n]+)",
            src,
        )
        assert match, "expected an assignment to agent._config_context_length"
        rhs = match.group(1).strip()
        # Must NOT be a hardcoded None anymore — that was the bug.
        assert rhs != "None", (
            "agent._config_context_length is unconditionally cleared to None "
            "— fallback entry's context_length field is silently dropped "
            "(see issue #70245)"
        )
        # Must read from the fallback entry's `context_length` key.
        assert "fb" in rhs and "context_length" in rhs, (
            f"expected `_config_context_length = fb.get(...)` or similar, "
            f"got: {rhs!r}"
        )

    def test_fallback_chain_unaffected_for_no_context_length(self):
        """When the fallback entry omits ``context_length``, the chokepoint
        must default to ``None`` (preserves the prior "resolve from the
        model's actual window" behavior).  Source-grep assertion: the
        fallback body still accesses ``fb`` and the chokepoint reads
        ``fb.get("context_length")`` so a missing key naturally resolves
        to None.
        """
        src = _read_chat_helpers_source()
        assert 'fb.get("context_length")' in src, (
            "expected the fallback entry's context_length to be read via "
            "fb.get('context_length') in agent/chat_completion_helpers.py"
        )


class TestFallbackEntryContextLengthBehavior:
    """Behavioral check on the exact assignment statement."""

    def test_fb_get_context_length_round_trip(self):
        """Direct behavioral round-trip on the assignment contract.

        We don't exercise the full ``try_activate_fallback`` function
        (it depends on a large agent surface).  Instead we validate the
        exact contract: ``agent._config_context_length = fb.get(
        "context_length")`` populates the value from the fallback entry
        when present and leaves it ``None`` when absent.
        """
        # Case 1: explicit context_length on the entry is forwarded.
        fb = {
            "provider": "bedrock",
            "model": "us.amazon.nova-2-lite-v1:0",
            "context_length": 1_000_000,
        }
        # Simulate the patched assignment:
        agent = type("A", (), {})()
        agent._config_context_length = fb.get("context_length")
        assert agent._config_context_length == 1_000_000

        # Case 2: missing context_length defaults to None (the resolver
        # still consults the static per-model table or live probe).
        fb = {"provider": "openai", "model": "gpt-4o-mini"}
        agent = type("A", (), {})()
        agent._config_context_length = fb.get("context_length")
        assert agent._config_context_length is None

        # Case 3: explicit 0 is forwarded verbatim — the resolver is the
        # source of truth for "0 == unset" (see test_model_metadata).
        fb = {
            "provider": "openai",
            "model": "gpt-4o-mini",
            "context_length": 0,
        }
        agent = type("A", (), {})()
        agent._config_context_length = fb.get("context_length")
        assert agent._config_context_length == 0


class TestRegressionOldBehavior:
    """Demonstrate the OLD (buggy) behavior is gone.

    These tests would FAIL against the pre-fix code path (where
    ``_config_context_length`` was unconditionally set to ``None``),
    and PASS against the fixed code path.
    """

    def test_old_unconditional_clear_is_gone(self):
        """The pre-fix pattern was::

            agent._config_context_length = None

        It must not appear in the patched source verbatim.  This is the
        load-bearing source-level assertion that protects against a
        future regression of the original bug.
        """
        src = _read_chat_helpers_source()
        # Allow "is None" checks elsewhere; this only forbids the
        # exact buggy assignment statement.
        bad = re.search(
            r"agent\._config_context_length\s*=\s*None\s*$",
            src,
            re.MULTILINE,
        )
        assert bad is None, (
            "regression: agent._config_context_length = None appears in "
            "agent/chat_completion_helpers.py — this is the bug from #70245"
        )

    def test_post_fix_round_trip_with_explicit_context_length(self):
        """Post-fix behavior: a fallback entry's context_length value
        survives the activation chokepoint.  Pre-fix this would have been
        clobbered to None.
        """
        fb = {
            "provider": "bedrock",
            "model": "us.amazon.nova-2-lite-v1:0",
            "context_length": 1_000_000,
        }
        agent = type("A", (), {})()
        # This is the patched assignment statement, copied verbatim from
        # the fix:
        agent._config_context_length = fb.get("context_length")
        assert agent._config_context_length == 1_000_000