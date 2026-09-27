"""Contract tests for the gateway's collectable text-filter hooks.

A subscribed hook may replace the user text the agent sees
(``agent:message:filter``) or the assistant text that goes out
(``agent:response:filter``). Both call sites funnel through the shared
``apply_collectable_text_filter`` helper, so these tests pin its contract:
the first valid replacement wins, malformed returns are ignored, and a
subscriber that raises never breaks the turn (fail-open).
"""

import pytest

from gateway.run_turn import apply_collectable_text_filter


class _FakeHooks:
    """Minimal stand-in for the hook registry: only ``emit_collect`` is used."""

    def __init__(self, results=(), raises=None):
        self._results = results
        self._raises = raises
        self.calls = []

    async def emit_collect(self, event_type, context):
        self.calls.append((event_type, context))
        if self._raises is not None:
            raise self._raises
        return self._results


@pytest.mark.asyncio
async def test_no_hooks_leaves_text_untouched():
    """No subscribers (empty result list) must change neither text path."""
    hooks = _FakeHooks(results=[])

    message = await apply_collectable_text_filter(
        hooks, "agent:message:filter", {"platform": "telegram"}, "message", "original message",
    )
    response = await apply_collectable_text_filter(
        hooks, "agent:response:filter", {"platform": "telegram"}, "response", "original response",
    )

    assert message == "original message"
    assert response == "original response"


@pytest.mark.asyncio
async def test_message_filter_replacement_is_applied():
    """A ``{"message": ...}`` result replaces the text the agent sees."""
    hooks = _FakeHooks(results=[{"message": "redacted message"}])

    result = await apply_collectable_text_filter(
        hooks, "agent:message:filter", {"session_id": "s1"}, "message", "secret message",
    )

    assert result == "redacted message"


@pytest.mark.asyncio
async def test_first_valid_result_wins():
    """With several subscribers, the first valid replacement is the one applied."""
    hooks = _FakeHooks(results=[{"response": "first"}, {"response": "second"}])

    result = await apply_collectable_text_filter(
        hooks, "agent:response:filter", {"session_id": "s1"}, "response", "original response",
    )

    assert result == "first"


@pytest.mark.asyncio
async def test_malformed_hook_results_are_ignored():
    """Non-dict returns, wrong value types, and missing keys are skipped silently."""
    hooks = _FakeHooks(results=[
        "not a dict",
        {"response": 123},          # present but not a string
        {"other_key": "value"},     # key absent
        None,
    ])

    result = await apply_collectable_text_filter(
        hooks, "agent:response:filter", {}, "response", "untouched",
    )

    assert result == "untouched"


@pytest.mark.asyncio
async def test_raising_hook_is_fail_open():
    """A subscriber blowing up returns the original text instead of propagating."""
    hooks = _FakeHooks(raises=RuntimeError("plugin exploded"))

    result = await apply_collectable_text_filter(
        hooks, "agent:message:filter", {}, "message", "untouched",
    )

    assert result == "untouched"
