import pytest

from agent.historical_tool_arguments import (
    ToolArgumentTraversalLimit,
    contains_non_replayable_history_args,
    omit_historical_tool_arguments,
)


def test_marker_detection_requires_exact_typed_metadata():
    marker = {
        "non_replayable": True,
        "reason": "context_compression",
        "original_chars": 900,
    }

    assert contains_non_replayable_history_args({
        "nested": {"_hermes_history_omitted": marker}
    })
    assert not contains_non_replayable_history_args({
        "_hermes_history_omitted": {**marker, "unexpected": "field"}
    })
    assert not contains_non_replayable_history_args({
        "_hermes_history_omitted": {**marker, "original_chars": True}
    })
    assert not contains_non_replayable_history_args({"content": "...[truncated]"})


def test_marker_scan_is_lazy_and_fails_closed_at_its_node_budget(monkeypatch):
    import agent.historical_tool_arguments as provenance

    class WideDict(dict):
        def values(self):  # type: ignore[override]
            for index, value in enumerate(super().values()):
                if index > 1:
                    raise AssertionError(
                        "values were eagerly materialized past the node budget"
                    )
                yield value

    monkeypatch.setattr(provenance, "_MAX_MARKER_SCAN_NODES", 1)
    value = WideDict({"nested": {}, **{str(i): i for i in range(20)}})
    with pytest.raises(ToolArgumentTraversalLimit):
        provenance.contains_non_replayable_history_args(value)


def test_marker_scan_handles_deep_and_cyclic_arguments():
    marker = omit_historical_tool_arguments("x" * 900)
    import json

    nested = json.loads(marker)
    for _ in range(2_000):
        nested = [nested]
    assert contains_non_replayable_history_args(nested)

    cyclic = []
    cyclic.append(cyclic)
    assert not contains_non_replayable_history_args(cyclic)
