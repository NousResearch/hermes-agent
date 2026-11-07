"""``post_background_review`` fires once per completed review, after the summary.

The review is the pass that writes memories and skills; a plugin that needs to act on
"the review just changed something" had no public surface. These tests pin the event:
one fire per completion, the same action strings the user saw, success=False on the
failure path, and a raising callback that cannot affect the review.
"""

from __future__ import annotations

import contextlib

import pytest

import agent.background_review as br
from hermes_cli import plugins as plugins_mod
from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest


@pytest.fixture
def ctx(monkeypatch):
    mgr = PluginManager()
    mgr._discovered = True
    monkeypatch.setattr(plugins_mod, "_plugin_manager", mgr)
    monkeypatch.setattr(plugins_mod, "_plugin_managers_by_home", {})
    return PluginContext(PluginManifest(name="review-observer", source="user"), mgr)


class _Agent:
    memory_notifications = "on"
    session_id = "session-review-test"
    provider = "stub"
    model = "stub"

    def __init__(self):
        self.calls = []

    def _safe_print(self, *_a, **_k):
        pass

    def background_review_callback(self, message):
        self.calls.append(message)

    def _emit_auxiliary_failure(self, *_a, **_k):
        pass


def _wire(monkeypatch, *, fork_raises: bool = False):
    monkeypatch.setattr(br, "_parent_can_emit_tool_calls", lambda a: True)
    monkeypatch.setattr(br, "thread_scoped_silence", lambda: contextlib.nullcontext())
    monkeypatch.setattr(br, "_track_review_fork", lambda *a, **k: None)
    monkeypatch.setattr(br, "_release_fork_clients", lambda *a, **k: None)
    monkeypatch.setattr(br, "finish_background_review_run", lambda *a, **k: None)
    monkeypatch.setattr(br, "_set_thread_approval_callback", lambda *a, **k: None)
    monkeypatch.setattr(br, "_log_review_completion", lambda *a, **k: None)
    monkeypatch.setattr(br, "summarize_background_review_actions",
                        lambda *a, **k: ["skill_manage updated a skill"])

    def fake_fork(agent, messages_snapshot, prompt, task_cfg, review_run, st, review_memory, explicit):
        if fork_raises:
            raise RuntimeError("review exploded")
        st.review_messages = [{"role": "tool", "content": "ok"}]
        st.review_usage = {}

    monkeypatch.setattr(br, "_run_review_fork", fake_fork)


def test_fires_once_with_the_review_payload(ctx, monkeypatch):
    _wire(monkeypatch)
    fired = []
    ctx.register_hook("post_background_review", lambda **kw: fired.append(kw))
    agent = _Agent()

    br._run_review_in_thread(agent, [{"role": "user", "content": "do it"}], "prompt", {}, None, False, True)

    [kw] = fired
    assert kw["session_id"] == "session-review-test"
    assert kw["actions"] == ["skill_manage updated a skill"]
    assert kw["review_memory"] is False
    assert kw["explicit"] is True
    assert kw["success"] is True


def test_raising_callback_does_not_change_the_review(ctx, monkeypatch):
    _wire(monkeypatch)
    ctx.register_hook("post_background_review", lambda **_kw: (_ for _ in ()).throw(RuntimeError("plugin bug")))
    agent = _Agent()

    br._run_review_in_thread(agent, [{"role": "user", "content": "do it"}], "prompt", {}, None, False, True)

    assert any("Self-improvement review" in c for c in agent.calls)


def test_failure_path_reports_success_false(ctx, monkeypatch):
    _wire(monkeypatch, fork_raises=True)
    fired = []
    ctx.register_hook("post_background_review", lambda **kw: fired.append(kw))
    agent = _Agent()

    br._run_review_in_thread(agent, [{"role": "user", "content": "do it"}], "prompt", {}, None, False, True)

    [kw] = fired
    assert kw["success"] is False
    assert kw["actions"] == []
