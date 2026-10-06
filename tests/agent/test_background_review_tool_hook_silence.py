"""A detached background-review fork must not publish tool lifecycle hooks under the parent's
session_id (#133603) — the tool-hook mirror of the ``_persist_disabled`` guards the session/LLM
hooks already have (#107062). Status consumers map ``pre_tool_call``/``post_tool_call`` to a
busy state and ``post_llm_call`` to done, so fork tool calls arriving after the parent turn's
closing event leave the session stuck on "working" until the next user message.
"""

import pytest


@pytest.fixture
def lifecycle_recorder(monkeypatch):
    """Record ``pre_tool_call``/``post_tool_call`` dispatches through ``hermes_cli.lifecycle``
    (both hook paths import it lazily at call time, so the patch always takes effect)."""
    import hermes_cli.lifecycle as lifecycle
    calls = {"pre": [], "post": []}

    def fake_invoke_hook(name, **kwargs):
        if name == "pre_tool_call":
            calls["pre"].append(kwargs)
            return [{"action": "block", "message": f"plugin blocked {kwargs.get('tool_name')}"}]
        if name == "post_tool_call":
            calls["post"].append(kwargs)
        return []

    def fake_has_hook(name):
        return name in ("pre_tool_call", "post_tool_call")

    monkeypatch.setattr(lifecycle, "invoke_hook", fake_invoke_hook)
    monkeypatch.setattr(lifecycle, "has_hook", fake_has_hook)
    return calls


def test_detached_thread_publishes_no_pre_tool_call(lifecycle_recorder):
    from hermes_cli.plugins import (
        _dispatch_pre_tool_call_hooks,
        clear_thread_detached_tool_hooks,
        set_thread_detached_tool_hooks,
    )

    block_msg, _modified = _dispatch_pre_tool_call_hooks("read_file", {"path": "x"}, session_id="parent")
    assert block_msg == "plugin blocked read_file"
    assert len(lifecycle_recorder["pre"]) == 1

    set_thread_detached_tool_hooks()
    try:
        block_msg, modified = _dispatch_pre_tool_call_hooks(
            "read_file", {"path": "x"}, session_id="parent")
        assert block_msg is None and modified is None
        assert len(lifecycle_recorder["pre"]) == 1  # no new dispatch from the fork
    finally:
        clear_thread_detached_tool_hooks()

    block_msg, _modified = _dispatch_pre_tool_call_hooks("read_file", {"path": "x"}, session_id="parent")
    assert block_msg == "plugin blocked read_file"
    assert len(lifecycle_recorder["pre"]) == 2


def test_detached_thread_keeps_the_whitelist_fence(lifecycle_recorder):
    """The thread whitelist is the fork's own safety fence and must keep blocking even while
    observer hooks are silenced."""
    from hermes_cli.plugins import (
        _dispatch_pre_tool_call_hooks,
        clear_thread_detached_tool_hooks,
        clear_thread_tool_whitelist,
        set_thread_detached_tool_hooks,
        set_thread_tool_whitelist,
    )

    set_thread_tool_whitelist({"skill_view"})
    set_thread_detached_tool_hooks()
    try:
        block_msg, _modified = _dispatch_pre_tool_call_hooks("terminal", {}, session_id="parent")
        assert block_msg is not None and "denied" in block_msg
        assert not lifecycle_recorder["pre"]  # the fence blocks without consulting hooks
    finally:
        clear_thread_tool_whitelist()
        clear_thread_detached_tool_hooks()


def test_detached_thread_emits_no_post_tool_call(lifecycle_recorder):
    from hermes_cli.plugins import clear_thread_detached_tool_hooks, set_thread_detached_tool_hooks
    from model_tools import _emit_post_tool_call_hook

    _emit_post_tool_call_hook(
        function_name="read_file", function_args={}, result="ok", session_id="parent")
    assert len(lifecycle_recorder["post"]) == 1

    set_thread_detached_tool_hooks()
    try:
        _emit_post_tool_call_hook(
            function_name="read_file", function_args={}, result="ok", session_id="parent")
        assert len(lifecycle_recorder["post"]) == 1  # no new emission from the fork
    finally:
        clear_thread_detached_tool_hooks()

    _emit_post_tool_call_hook(
        function_name="read_file", function_args={}, result="ok", session_id="parent")
    assert len(lifecycle_recorder["post"]) == 2


def test_run_review_fork_marks_detached_and_clears_on_exit(monkeypatch):
    """``_run_review_fork`` must hold the detached marker for the whole fork phase (including
    when the fork raises) and clear it in the finally block."""
    import agent.background_review as br
    from hermes_cli.plugins import thread_tool_hooks_detached

    marker_states = {}

    class _ForkStub:
        def run_conversation(self, **_kwargs):
            marker_states["during"] = thread_tool_hooks_detached()
            raise RuntimeError("fork exploded after its tool call")

    def _fake_build(_agent, _task_cfg, max_iterations=None):
        return _ForkStub(), {"routed": False}, False

    monkeypatch.setattr(br, "build_cache_parity_fork", _fake_build)
    monkeypatch.setattr(br, "_track_review_fork", lambda *a, **k: None)
    monkeypatch.setattr(br, "_review_tool_whitelist", lambda *a, **k: ({"skill_view"}, ()))
    monkeypatch.setattr(br, "_snapshot_review_usage", lambda _fork: {})
    monkeypatch.setattr(br, "_record_review_usage_to_parent", lambda *a, **k: None)
    monkeypatch.setattr(br, "finish_background_review_run", lambda *a, **k: None)
    monkeypatch.setattr(br, "_release_fork_clients", lambda _fork: None)

    class _ParentStub:
        session_id = "parent"

    with pytest.raises(RuntimeError):
        br._run_review_fork(_ParentStub(), [], "review prompt", None, None, br._ReviewForkState())

    assert marker_states["during"] is True
    assert thread_tool_hooks_detached() is False
