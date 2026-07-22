"""Tests for the ``pre_agent_dispatch`` plugin hook and ``dispatch_pre_agent`` helper."""

import hermes_cli.plugins as plugins_mod


# ── Helper ────────────────────────────────────────────────────────────

_ACCEPTED_HOOK = "pre_agent_dispatch"


def _call_dispatch(monkeypatch, *, hook_results=None):
    """Call ``dispatch_pre_agent`` with a mocked hook pipeline.

    ``hook_results`` is either a list that ``invoke_hook`` returns, or
    ``None`` to use the real (empty) pipeline.
    """
    monkeypatch.setattr(plugins_mod, "_plugin_manager", plugins_mod.PluginManager())
    if hook_results is not None:
        monkeypatch.setattr(plugins_mod, "invoke_hook", lambda _name, **_kw: hook_results)
    return plugins_mod.dispatch_pre_agent(
        message="test message",
        session_key="test-session",
    )


# ── Baseline ──────────────────────────────────────────────────────────

def test_allow_when_no_plugins(monkeypatch):
    """With no plugins loaded, dispatch_pre_agent returns allow."""
    result = _call_dispatch(monkeypatch, hook_results=[])
    assert result == {"action": "allow"}


def test_allow_when_hooks_return_none(monkeypatch):
    """None returns from hooks are skipped."""
    result = _call_dispatch(monkeypatch, hook_results=[None])
    assert result == {"action": "allow"}


def test_non_dict_returns_are_skipped(monkeypatch):
    """Non-dict hook returns are ignored."""
    result = _call_dispatch(monkeypatch, hook_results=[123, "string", {"action": "skip"}])
    assert result == {"action": "skip"}


# ── Action: skip ──────────────────────────────────────────────────────

def test_skip_action(monkeypatch):
    """A hook returning skip drops the message."""
    result = _call_dispatch(monkeypatch, hook_results=[{"action": "skip"}])
    assert result == {"action": "skip"}


def test_skip_wins_over_allow(monkeypatch):
    """First actionable result wins — skip before allow."""
    result = _call_dispatch(
        monkeypatch,
        hook_results=[{"action": "skip"}, {"action": "allow"}],
    )
    assert result == {"action": "skip"}


# ── Action: route ─────────────────────────────────────────────────────

def test_route_action(monkeypatch):
    """A hook returning route bypasses the agent."""
    result = _call_dispatch(
        monkeypatch,
        hook_results=[{"action": "route", "result": "orchestrator says hi"}],
    )
    assert result == {"action": "route", "result": "orchestrator says hi"}


def test_route_with_streamed_flag(monkeypatch):
    """The streamed flag is passed through so callers can avoid double-streaming."""
    result = _call_dispatch(
        monkeypatch,
        hook_results=[{"action": "route", "result": "hi", "streamed": True}],
    )
    assert result == {"action": "route", "result": "hi", "streamed": True}


def test_route_without_streamed_flag(monkeypatch):
    """When streamed is not set, the result dict does not include it."""
    result = _call_dispatch(
        monkeypatch,
        hook_results=[{"action": "route", "result": "hi"}],
    )
    assert "streamed" not in result


def test_route_result_empty_string_is_preserved(monkeypatch):
    """An empty string result is kept (not coerced to '' redundantly)."""
    result = _call_dispatch(
        monkeypatch,
        hook_results=[{"action": "route", "result": ""}],
    )
    assert result == {"action": "route", "result": ""}


def test_route_missing_result_defaults_to_empty(monkeypatch):
    """A route action without a result key gets an empty string."""
    result = _call_dispatch(
        monkeypatch,
        hook_results=[{"action": "route"}],
    )
    assert result == {"action": "route", "result": ""}


# ── Action: rewrite ───────────────────────────────────────────────────

def test_rewrite_action(monkeypatch):
    """A hook returning rewrite provides replacement text."""
    result = _call_dispatch(
        monkeypatch,
        hook_results=[{"action": "rewrite", "text": "rewritten message"}],
    )
    assert result == {"action": "rewrite", "text": "rewritten message"}


def test_rewrite_empty_text_is_noop(monkeypatch):
    """An empty rewrite text is treated as a no-op — falls through to allow."""
    result = _call_dispatch(
        monkeypatch,
        hook_results=[{"action": "rewrite", "text": ""}],
    )
    assert result == {"action": "allow"}


def test_rewrite_non_string_text_is_noop(monkeypatch):
    """A non-string rewrite text is treated as a no-op."""
    result = _call_dispatch(
        monkeypatch,
        hook_results=[{"action": "rewrite", "text": 123}],
    )
    assert result == {"action": "allow"}


def test_rewrite_then_allow(monkeypatch):
    """An empty rewrite is skipped; the next hook's result is used."""
    result = _call_dispatch(
        monkeypatch,
        hook_results=[
            {"action": "rewrite", "text": ""},
            {"action": "allow"},
        ],
    )
    assert result == {"action": "allow"}


# ── Error handling ────────────────────────────────────────────────────

def test_hook_exception_falls_back_to_allow(monkeypatch):
    """If invoke_hook raises, dispatch_pre_agent returns allow (fail-closed)."""
    def _raise(_name, **_kw):
        raise RuntimeError("plugin crash")

    monkeypatch.setattr(plugins_mod, "invoke_hook", _raise)
    result = plugins_mod.dispatch_pre_agent(message="test")
    assert result == {"action": "allow"}


# ── Kwarg forwarding ──────────────────────────────────────────────────

def test_kwargs_forwarded_to_hooks(monkeypatch):
    """All expected kwargs are forwarded to invoke_hook."""
    captured = {}

    def _hook(_name, **kwargs):
        captured.update(kwargs)
        return []

    monkeypatch.setattr(plugins_mod, "invoke_hook", _hook)

    result = plugins_mod.dispatch_pre_agent(
        message="hello",
        session_key="sess-1",
        source="test-source",
        gateway="test-gateway",
        history=[{"role": "user", "content": "prev"}],
        stream_callback=lambda text: None,
    )
    assert result == {"action": "allow"}
    assert captured["message"] == "hello"
    assert captured["session_key"] == "sess-1"
    assert captured["source"] == "test-source"
    assert captured["gateway"] == "test-gateway"
    assert captured["history"] == [{"role": "user", "content": "prev"}]
    assert "stream_callback" in captured


def test_stream_callback_not_forwarded_when_none(monkeypatch):
    """When stream_callback is None, it is not forwarded to hooks."""
    captured = {}

    def _hook(_name, **kwargs):
        captured.update(kwargs)
        return []

    monkeypatch.setattr(plugins_mod, "invoke_hook", _hook)

    plugins_mod.dispatch_pre_agent(message="test", stream_callback=None)
    assert "stream_callback" not in captured
