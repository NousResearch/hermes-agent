"""Configurable Langfuse payload depth regression for #116609."""

import importlib
import json

import pytest


@pytest.fixture(autouse=True)
def _fresh_max_depth_cache():
    """``_resolve_max_depth`` is an ``lru_cache`` keyed on the raw env string, so its
    once-per-value warning persists for the whole process. Clear it around every test
    or an earlier resolution of the same invalid value (another test, a repeat run)
    silently turns the "exactly one warning" assertions into 0 warnings."""
    plugin = importlib.import_module("plugins.observability.langfuse")
    plugin._resolve_max_depth.cache_clear()
    yield
    plugin._resolve_max_depth.cache_clear()


@pytest.mark.parametrize("mode", ["sanitized", "full"])
@pytest.mark.parametrize("configured_depth", [None, "", "0", "4", "5", " 10 "])
def test_capture_respects_configured_depth_for_tool_inputs_and_outputs(
    monkeypatch, mode, configured_depth,
):
    plugin = importlib.import_module("plugins.observability.langfuse")
    monkeypatch.setenv("HERMES_LANGFUSE_CAPTURE", mode)
    if configured_depth is None:
        monkeypatch.delenv("HERMES_LANGFUSE_MAX_DEPTH", raising=False)
    else:
        monkeypatch.setenv("HERMES_LANGFUSE_MAX_DEPTH", configured_depth)
    max_depth = int(configured_depth) if configured_depth else 4

    # Each dict value/list element adds one level; scalars at the limit survive.
    for leaf_depth in range(7):
        payload = 7
        expected = 7 if leaf_depth <= max_depth else "<max-depth>"
        for level in reversed(range(leaf_depth)):
            payload = {"data": payload} if level % 2 == 0 else [payload]
            if level <= max_depth:
                expected = {"data": expected} if level % 2 == 0 else [expected]
        assert plugin._capture_content(payload) == expected
        assert plugin._capture_content(
            payload, tool_result_of=("example", {}),
        ) == expected
        assert plugin._capture_content(
            json.dumps(payload), tool_result_of=("example", {}),
        ) == (str(expected) if leaf_depth == 0 else expected)


@pytest.mark.parametrize("invalid_depth", ["nope", "1.5", "-1"])
def test_invalid_max_depth_warns_and_preserves_default_capture(monkeypatch, caplog, invalid_depth):
    plugin = importlib.import_module("plugins.observability.langfuse")
    monkeypatch.setenv("HERMES_LANGFUSE_CAPTURE", "full")
    monkeypatch.delenv("HERMES_LANGFUSE_MAX_DEPTH", raising=False)
    payload = {"result": {"data": {"results": [{"index": 0}]}}}
    default_capture = plugin._capture_content(payload)
    monkeypatch.setenv("HERMES_LANGFUSE_MAX_DEPTH", invalid_depth)

    assert plugin._capture_content(payload) == default_capture
    assert len(caplog.records) == 1
    assert "HERMES_LANGFUSE_MAX_DEPTH" in caplog.text
    assert "non-negative integer" in caplog.text


def test_invalid_max_depth_warns_once_per_value_not_per_payload(monkeypatch, caplog):
    """A bad ``HERMES_LANGFUSE_MAX_DEPTH`` must not log one warning per captured prompt/tool payload
    for the life of the process; a changed (still bad) value gets its own single warning."""
    plugin = importlib.import_module("plugins.observability.langfuse")
    monkeypatch.setenv("HERMES_LANGFUSE_CAPTURE", "full")
    monkeypatch.setenv("HERMES_LANGFUSE_MAX_DEPTH", "abc")
    payload = {"result": {"data": [{"index": 0}]}}

    for _ in range(5):
        plugin._capture_content(payload)
        plugin._capture_content(payload, tool_result_of=("example", {}))
    assert sum("HERMES_LANGFUSE_MAX_DEPTH" in r.getMessage() for r in caplog.records) == 1

    monkeypatch.setenv("HERMES_LANGFUSE_MAX_DEPTH", "-7")
    for _ in range(3):
        plugin._capture_content(payload)
    assert sum("HERMES_LANGFUSE_MAX_DEPTH" in r.getMessage() for r in caplog.records) == 2
