"""``post_compaction`` fires once per committed compaction, never for an attempt that changed nothing,
and a raising plugin never changes what the compaction returns (#118382)."""

from unittest.mock import patch

import threading

import pytest

from hermes_cli import plugins as plugins_mod
from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest
from tests.agent.test_compaction_events import _Agent, _compressor, _messages
from agent.conversation_compression import compress_context


@pytest.fixture
def ctx(monkeypatch):
    mgr = PluginManager()
    mgr._discovered = True
    monkeypatch.setattr(plugins_mod, "_plugin_manager", mgr)
    monkeypatch.setattr(plugins_mod, "_plugin_managers_by_home", {})
    return PluginContext(PluginManifest(name="compaction-observer", source="user"), mgr)


@pytest.mark.parametrize("summary, committed", [("SANITIZED SUMMARY", True), (None, False)])
def test_post_compaction_fires_once_per_committed_compaction(ctx, summary, committed):
    fired = []

    def boom(**_kw):
        raise RuntimeError("plugin bug")

    ctx.register_hook("post_compaction", lambda **kw: fired.append(kw))
    ctx.register_hook("post_compaction", boom)
    agent = _Agent(_compressor(abort_on_summary_failure=True))

    with patch.object(agent.context_compressor, "_generate_summary", return_value=summary):
        compressed, _ = compress_context(agent, _messages(), "system prompt", approx_tokens=80_000, trigger="pre_api")

    assert (len(compressed) < len(_messages())) is committed
    if not committed:
        assert fired == []
        return
    [kw] = fired
    assert (kw["session_id"], kw["old_session_id"], kw["kind"], kw["reason"]) == (
        "session-events-test", "", "summarize", "pre_api",
    )
    assert kw["messages_removed"] == len(_messages()) - len(compressed) > 0
    assert kw["before_tokens"] > kw["after_tokens"] > 0


def test_a_slow_callback_for_one_session_does_not_skip_another_sessions_compaction(ctx):
    """The bounded dispatcher skips a callback still running for the same call; two sessions are two calls."""
    entered, release, seen = threading.Event(), threading.Event(), []

    def slow(session_id, **_kw):
        seen.append(session_id)
        if session_id == "a":
            entered.set()
            release.wait(10)

    ctx.register_hook("post_compaction", slow)
    first = threading.Thread(target=ctx._manager.invoke_hook, args=("post_compaction",), kwargs={"session_id": "a"})
    first.start()
    assert entered.wait(10)
    try:
        ctx._manager.invoke_hook("post_compaction", session_id="b")
    finally:
        release.set()
        first.join(10)

    assert sorted(seen) == ["a", "b"]
