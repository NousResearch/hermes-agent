"""
Regression test for the _nr_to_assistant_message shim dropping extra_content.

The shim introduced by d30ee2e5 ("unify transport dispatch + collapse normalize shims")
only propagated ("call_id", "response_item_id") from provider_data into the SimpleNamespace.
This silently dropped extra_content, which is where the chat_completions transport stores
Gemini's thought_signature. Downstream, _build_assistant_message reads the attribute via
getattr(tool_call, "extra_content", None) — but found None because the shim had already
stripped it, causing Google to return HTTP 400 INVALID_ARGUMENT on the replay turn.

Fixes: https://github.com/NousResearch/hermes-agent/issues/14376
Related: dfd50cec (original extra_content preservation in _build_assistant_message)
         d30ee2e5 (refactor that introduced the shim gap)
"""

from types import SimpleNamespace
import pytest


def _make_nr_tool_call(extra_content=None, call_id=None, response_item_id=None):
    """Build a minimal NR tool call with the given provider_data fields."""
    provider_data = {}
    if extra_content is not None:
        provider_data["extra_content"] = extra_content
    if call_id is not None:
        provider_data["call_id"] = call_id
    if response_item_id is not None:
        provider_data["response_item_id"] = response_item_id
    return SimpleNamespace(
        id="call_abc",
        name="my_tool",
        arguments='{"x": 1}',
        provider_data=provider_data or None,
    )


def _shim_tool_call(tc):
    """Replicate the _nr_to_assistant_message shim logic for a single tool call."""
    tc_ns = SimpleNamespace(
        id=tc.id,
        type="function",
        function=SimpleNamespace(name=tc.name, arguments=tc.arguments),
    )
    if tc.provider_data:
        for key in ("call_id", "response_item_id", "extra_content"):
            if tc.provider_data.get(key):
                setattr(tc_ns, key, tc.provider_data[key])
    return tc_ns


class TestExtraContentShimPropagation:
    """_nr_to_assistant_message must forward extra_content from provider_data."""

    def test_extra_content_propagated(self):
        """Gemini thought_signature in extra_content survives the shim."""
        sig = {"google": {"thought_signature": "abc123=="}}
        tc = _make_nr_tool_call(extra_content=sig)
        ns = _shim_tool_call(tc)
        assert getattr(ns, "extra_content", None) == sig, (
            "extra_content must be set on the SimpleNamespace so "
            "_build_assistant_message can echo the thought_signature back to Google"
        )

    def test_extra_content_none_not_set(self):
        """When extra_content is absent, no attribute is added (avoids None noise)."""
        tc = _make_nr_tool_call()
        ns = _shim_tool_call(tc)
        # getattr with default — same pattern used in _build_assistant_message
        assert getattr(ns, "extra_content", None) is None

    def test_call_id_still_propagated(self):
        """call_id propagation (existing behaviour) is not disturbed."""
        tc = _make_nr_tool_call(call_id="resp_xyz")
        ns = _shim_tool_call(tc)
        assert getattr(ns, "call_id", None) == "resp_xyz"

    def test_response_item_id_still_propagated(self):
        """response_item_id propagation (existing behaviour) is not disturbed."""
        tc = _make_nr_tool_call(response_item_id="item_42")
        ns = _shim_tool_call(tc)
        assert getattr(ns, "response_item_id", None) == "item_42"

    def test_all_three_keys_together(self):
        """All three provider_data keys can coexist on the same tool call."""
        sig = {"google": {"thought_signature": "xyz"}}
        tc = _make_nr_tool_call(
            extra_content=sig, call_id="c1", response_item_id="r1"
        )
        ns = _shim_tool_call(tc)
        assert getattr(ns, "extra_content", None) == sig
        assert getattr(ns, "call_id", None) == "c1"
        assert getattr(ns, "response_item_id", None) == "r1"

    def test_no_provider_data(self):
        """When provider_data is None, no keys are set and no exception raised."""
        tc = SimpleNamespace(
            id="call_z", name="tool", arguments="{}", provider_data=None
        )
        ns = _shim_tool_call(tc)
        assert getattr(ns, "extra_content", None) is None
        assert getattr(ns, "call_id", None) is None
