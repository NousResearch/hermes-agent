"""Tests for the bundled Langfuse observability plugin.

Covers ``plugins/observability/langfuse/__init__.py``:

  * session_id/tags propagation — the ``propagate_attributes()`` context is
    held open for the trace's whole lifetime so child observations (LLM-call
    generations, tool spans) inherit session_id/tags, and it is closed
    exactly once on both the normal-finish and eviction paths.
  * ``channel:<platform>`` trace tag derived from the hook's platform value.
  * error visibility — observations still open at turn end are closed with
    ``level=ERROR`` + status message; a generation superseded by a retry of
    the same ``api_call_count`` is closed with ``level=WARNING``; successful
    calls keep the default level with no status message.

All Langfuse SDK surface is faked; no network, no real SDK required.
"""

import importlib.util
import sys
import types
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Fakes for the Langfuse SDK surface the plugin touches
# ---------------------------------------------------------------------------


class FakeObservation:
    """Records update()/end() calls; spawns child observations."""

    def __init__(self, name="root"):
        self.name = name
        self.updates = []
        self.ended = False
        self.children = []
        self.trace_io = {}

    def start_observation(self, *, name, as_type, input, metadata=None,
                          model=None, model_parameters=None):
        child = FakeObservation(name=name)
        self.children.append(child)
        return child

    def update(self, **kwargs):
        self.updates.append(kwargs)

    def end(self):
        self.ended = True

    def set_trace_io(self, **kwargs):
        self.trace_io.update(kwargs)


class FakeRootCtx:
    """Stands in for start_as_current_observation()'s context manager."""

    def __init__(self, span):
        self.span = span

    def __enter__(self):
        return self.span

    def __exit__(self, *exc):
        return False


class FakePropagateCtx:
    """Records enter/exit so tests can assert on context lifetime."""

    def __init__(self, kwargs):
        self.kwargs = kwargs
        self.entered = 0
        self.exited = 0

    def __enter__(self):
        self.entered += 1
        return self

    def __exit__(self, *exc):
        self.exited += 1
        return False


class FakeClient:
    def __init__(self):
        self.root_spans = []
        self.flushed = 0

    def create_trace_id(self, seed=""):
        return f"trace-{len(self.root_spans)}"

    def start_as_current_observation(self, *, trace_context, name, as_type,
                                     input, metadata=None, end_on_exit=False):
        span = FakeObservation(name=name)
        span.trace_context = trace_context
        self.root_spans.append(span)
        return FakeRootCtx(span)

    def flush(self):
        self.flushed += 1


# ---------------------------------------------------------------------------
# Plugin loading
# ---------------------------------------------------------------------------


def _load_plugin():
    """Import a fresh instance of the plugin module from the repo path."""
    repo_root = Path(__file__).resolve().parents[2]
    path = repo_root / "plugins" / "observability" / "langfuse" / "__init__.py"
    name = "test_langfuse_plugin_instance"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        # Fresh instance per test; drop the registration either way.
        sys.modules.pop(name, None)
    return module


@pytest.fixture()
def plugin():
    module = _load_plugin()
    module._LANGFUSE_CLIENT = FakeClient()

    propagate_ctxs = []

    def fake_propagate_attributes(**kwargs):
        ctx = FakePropagateCtx(kwargs)
        propagate_ctxs.append(ctx)
        return ctx

    module.propagate_attributes = fake_propagate_attributes
    module._test_propagate_ctxs = propagate_ctxs
    return module


def _assistant_message(content="OK", tool_calls=None):
    msg = types.SimpleNamespace()
    msg.content = content
    msg.tool_calls = tool_calls or []
    return msg


TURN = dict(
    task_id="task-1",
    session_id="session-abc",
    turn_id="turn-1",
    api_request_id="req-1",
    platform="telegram",
    model="test-model",
    provider="test-provider",
    api_mode="test",
)


def _start_request(plugin_module, api_call_count=1):
    plugin_module.on_pre_llm_request(
        **TURN,
        api_call_count=api_call_count,
        request_messages=[{"role": "user", "content": "hello"}],
    )


def _complete_request(plugin_module, api_call_count=1, content="done"):
    plugin_module.on_post_llm_call(
        task_id=TURN["task_id"],
        session_id=TURN["session_id"],
        turn_id=TURN["turn_id"],
        api_request_id=TURN["api_request_id"],
        provider=TURN["provider"],
        model=TURN["model"],
        api_call_count=api_call_count,
        assistant_message=_assistant_message(content=content),
        finish_reason="stop",
        usage=None,
    )


def _task_key(plugin_module):
    return plugin_module._trace_key(
        TURN["task_id"],
        TURN["session_id"],
        turn_id=TURN["turn_id"],
        api_request_id=TURN["api_request_id"],
    )


# ---------------------------------------------------------------------------
# session_id / tags propagation
# ---------------------------------------------------------------------------


def test_propagate_context_held_open_until_finish(plugin):
    _start_request(plugin)

    (ctx,) = plugin._test_propagate_ctxs
    assert ctx.entered == 1
    # The whole point of the fix: the context must still be open after the
    # root-trace hook returns, so observations created by later hook
    # invocations inherit session_id/tags.
    assert ctx.exited == 0
    assert ctx.kwargs["session_id"] == "session-abc"

    # Completing the turn (content, no tool calls) finishes the trace and
    # closes the context exactly once.
    _complete_request(plugin)
    assert ctx.exited == 1
    assert not plugin._TRACE_STATE


def test_channel_tag_from_platform(plugin):
    _start_request(plugin)
    (ctx,) = plugin._test_propagate_ctxs
    assert "channel:telegram" in ctx.kwargs["tags"]
    assert "hermes" in ctx.kwargs["tags"]


def test_eviction_closes_propagate_context(plugin, monkeypatch):
    monkeypatch.setattr(plugin, "_MAX_TRACE_STATE", 1)
    _start_request(plugin)
    first_ctx = plugin._test_propagate_ctxs[0]

    # A second turn evicts the first (cap is 1).
    plugin.on_pre_llm_request(
        **{**TURN, "turn_id": "turn-2", "api_request_id": "req-2"},
        api_call_count=1,
        request_messages=[{"role": "user", "content": "again"}],
    )
    assert first_ctx.exited == 1


# ---------------------------------------------------------------------------
# error visibility
# ---------------------------------------------------------------------------


def test_unfinished_generation_marked_error_on_finish(plugin):
    _start_request(plugin)
    root_span = plugin._LANGFUSE_CLIENT.root_spans[0]
    (generation,) = root_span.children

    # Turn ends without post_api_request ever firing (retries exhausted).
    plugin._finish_trace(_task_key(plugin), output=None)

    assert generation.ended
    merged = {k: v for update in generation.updates for k, v in update.items()}
    assert merged.get("level") == "ERROR"
    assert "without a recorded response" in merged.get("status_message", "")


def test_unfinished_tool_marked_error_on_finish(plugin):
    _start_request(plugin)
    plugin.on_pre_tool_call(
        tool_name="terminal",
        args={"command": "true"},
        task_id=TURN["task_id"],
        session_id=TURN["session_id"],
        turn_id=TURN["turn_id"],
        tool_call_id="call-1",
    )
    root_span = plugin._LANGFUSE_CLIENT.root_spans[0]
    tool_obs = root_span.children[-1]

    plugin._finish_trace(_task_key(plugin), output=None)

    merged = {k: v for update in tool_obs.updates for k, v in update.items()}
    assert merged.get("level") == "ERROR"
    assert "tool call's result" in merged.get("status_message", "")


def test_retry_superseded_generation_marked_warning(plugin):
    _start_request(plugin, api_call_count=1)
    root_span = plugin._LANGFUSE_CLIENT.root_spans[0]
    first_generation = root_span.children[0]

    # Same api_call_count fires again: a retry superseding the first attempt.
    _start_request(plugin, api_call_count=1)
    second_generation = root_span.children[1]

    merged = {k: v for update in first_generation.updates for k, v in update.items()}
    assert first_generation.ended
    assert merged.get("level") == "WARNING"
    assert "Retried" in merged.get("status_message", "")

    # The retry that completes normally stays at the default level.
    _complete_request(plugin, api_call_count=1)
    for update in second_generation.updates:
        assert "level" not in update
        assert "status_message" not in update


def test_successful_call_keeps_default_level(plugin):
    _start_request(plugin)
    _complete_request(plugin)

    root_span = plugin._LANGFUSE_CLIENT.root_spans[0]
    (generation,) = root_span.children
    assert generation.ended
    for update in generation.updates:
        assert "level" not in update
        assert "status_message" not in update
