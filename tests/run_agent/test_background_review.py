"""Regression tests for background review agent cleanup."""

from __future__ import annotations

import run_agent as run_agent_module
from run_agent import AIAgent


def _bare_agent() -> AIAgent:
    agent = object.__new__(AIAgent)
    agent.model = "fake-model"
    agent.platform = "telegram"
    agent.provider = "openai"
    agent.base_url = ""
    agent.api_key = ""
    agent.api_mode = ""
    agent.session_id = "test-session"
    agent._parent_session_id = ""
    agent._credential_pool = None
    agent._memory_store = object()
    agent._memory_enabled = True
    agent._user_profile_enabled = False
    agent._cached_system_prompt = "test-cached-system-prompt"
    import datetime as _dt
    agent.session_start = _dt.datetime(2026, 1, 1, 12, 0, 0)
    agent._MEMORY_REVIEW_PROMPT = "review memory"
    agent._SKILL_REVIEW_PROMPT = "review skills"
    agent._COMBINED_REVIEW_PROMPT = "review both"
    agent.background_review_callback = None
    agent.status_callback = None
    agent._safe_print = lambda *_args, **_kwargs: None
    return agent


class ImmediateThread:
    def __init__(self, *, target, daemon=None, name=None):
        self._target = target

    def start(self):
        self._target()


def test_background_review_shuts_down_memory_provider_before_close(monkeypatch):
    events = []

    class FakeReviewAgent:
        def __init__(self, **kwargs):
            events.append(("init", kwargs))
            self._session_messages = []

        def run_conversation(self, **kwargs):
            events.append(("run_conversation", kwargs))

        def shutdown_memory_provider(self):
            events.append(("shutdown_memory_provider", None))

        def close(self):
            events.append(("close", None))

    monkeypatch.setattr(run_agent_module, "AIAgent", FakeReviewAgent)
    monkeypatch.setattr(run_agent_module.threading, "Thread", ImmediateThread)

    agent = _bare_agent()

    AIAgent._spawn_background_review(
        agent,
        messages_snapshot=[{"role": "user", "content": "hello"}],
        review_memory=True,
    )

    assert [name for name, _payload in events] == [
        "init",
        "run_conversation",
        "shutdown_memory_provider",
        "close",
    ]


def test_background_review_fork_opts_out_of_session_finalization(monkeypatch):
    """The review fork shares the parent's live session_id, so it must set
    ``_end_session_on_close = False``. Otherwise close() (now finalizing owned
    session rows) would end the still-active parent session mid-conversation
    every time the review fires (~every 10 turns). Regression for #12029.
    """
    seen = {}

    class FakeReviewAgent:
        def __init__(self, **kwargs):
            self._session_messages = []
            # Default matches AIAgent.__init__ (agent_init.py): owns its row.
            self._end_session_on_close = True

        def __setattr__(self, name, value):
            object.__setattr__(self, name, value)
            if name == "_end_session_on_close":
                seen["end_session_on_close"] = value

        def run_conversation(self, **kwargs):
            # By the time the fork runs, the opt-out must already be applied.
            seen["at_run_time"] = self._end_session_on_close

        def shutdown_memory_provider(self):
            pass

        def close(self):
            pass

    monkeypatch.setattr(run_agent_module, "AIAgent", FakeReviewAgent)
    monkeypatch.setattr(run_agent_module.threading, "Thread", ImmediateThread)

    agent = _bare_agent()

    AIAgent._spawn_background_review(
        agent,
        messages_snapshot=[{"role": "user", "content": "hello"}],
        review_memory=True,
    )

    assert seen.get("end_session_on_close") is False
    assert seen.get("at_run_time") is False


def test_background_review_summarizer_receives_captured_messages_after_close(monkeypatch):
    """The action summarizer must see review messages even after close cleanup.

    Regression for the bug where ``review_messages`` was snapshot AFTER
    ``review_agent.close()``. close() is allowed to clean per-session state
    (including ``_session_messages``), so the summarizer would receive an
    empty list and the user-visible self-improvement summary would silently
    disappear. The fix snapshots ``_session_messages`` before teardown.
    """
    import json
    import agent.background_review as bg_review

    review_tool_message = {
        "role": "tool",
        "tool_call_id": "call_bg",
        "content": json.dumps(
            {"success": True, "message": "Entry added", "target": "memory"}
        ),
    }
    captured: dict = {}
    events: list[str] = []

    class FakeReviewAgent:
        def __init__(self, **kwargs):
            self._session_messages = []

        def run_conversation(self, **kwargs):
            events.append("run_conversation")
            self._session_messages = [review_tool_message]

        def shutdown_memory_provider(self):
            events.append("shutdown_memory_provider")

        def close(self):
            events.append("close")
            # close() is allowed to clean _session_messages — the fix
            # must have snapshot them before this runs.
            self._session_messages = []

    def fake_summarize(review_messages, prior_snapshot, notification_mode="on"):
        events.append("summarize")
        captured["review_messages"] = list(review_messages)
        captured["prior_snapshot"] = list(prior_snapshot)
        captured["notification_mode"] = notification_mode
        return []

    monkeypatch.setattr(run_agent_module, "AIAgent", FakeReviewAgent)
    monkeypatch.setattr(run_agent_module.threading, "Thread", ImmediateThread)
    monkeypatch.setattr(
        bg_review,
        "summarize_background_review_actions",
        fake_summarize,
    )

    messages_snapshot = [{"role": "user", "content": "hi"}]
    agent = _bare_agent()

    AIAgent._spawn_background_review(
        agent,
        messages_snapshot=messages_snapshot,
        review_memory=True,
    )

    assert events == [
        "run_conversation",
        "shutdown_memory_provider",
        "close",
        "summarize",
    ]
    assert captured["review_messages"] == [review_tool_message]
    assert captured["prior_snapshot"] == messages_snapshot
    assert captured["notification_mode"] == "on"


def test_background_review_installs_auto_deny_approval_callback(monkeypatch):
    """Regression guard for #15216.

    The background review thread must install a non-interactive approval
    callback. If it doesn't, any dangerous-command guard the review agent
    trips falls back to input() on a daemon thread, which deadlocks against
    the parent's prompt_toolkit TUI.
    """
    import tools.terminal_tool as tt

    observed: dict = {"during_run": "<unread>", "after_finally": "<unread>"}

    class FakeReviewAgent:
        def __init__(self, **kwargs):
            self._session_messages = []

        def run_conversation(self, **kwargs):
            # Capture what the callback looks like mid-run. It must be
            # a callable (the auto-deny) -- not None.
            observed["during_run"] = tt._get_approval_callback()

        def shutdown_memory_provider(self):
            pass

        def close(self):
            pass

    monkeypatch.setattr(run_agent_module, "AIAgent", FakeReviewAgent)
    monkeypatch.setattr(run_agent_module.threading, "Thread", ImmediateThread)

    # Start from a clean slot.
    tt.set_approval_callback(None)
    agent = _bare_agent()

    AIAgent._spawn_background_review(
        agent,
        messages_snapshot=[{"role": "user", "content": "hello"}],
        review_memory=True,
    )

    observed["after_finally"] = tt._get_approval_callback()

    assert callable(observed["during_run"]), (
        "Background review did not install an approval callback on its "
        "worker thread; dangerous-command prompts will deadlock against "
        "the parent TUI (#15216)."
    )
    # The installed callback must deny (it's a safety gate, not a prompt).
    assert observed["during_run"]("rm -rf /", "test") == "deny"

    assert observed["after_finally"] is None, (
        "Background review leaked its approval callback into the worker "
        "thread's TLS slot; a recycled thread-id could reuse it."
    )


def test_background_review_summary_is_attributed_to_self_improvement_loop(monkeypatch):
    """The CLI/gateway emission must identify the self-improvement loop.

    Users who miss the line in their terminal have no way to tell that the
    background review was what modified their skill/memory stores. The
    summary prefix ``💾 Self-improvement review: …`` makes the origin
    explicit so both the CLI and gateway deliveries are unambiguous.
    """
    import json

    captured_prints: list = []
    captured_bg_callback: list = []

    class FakeReviewAgent:
        def __init__(self, **kwargs):
            # Simulate a review that successfully updated memory so
            # _summarize_background_review_actions returns a real action.
            self._session_messages = [
                {
                    "role": "tool",
                    "tool_call_id": "call_bg",
                    "content": json.dumps(
                        {"success": True, "message": "Entry added", "target": "memory"}
                    ),
                }
            ]

        def run_conversation(self, **kwargs):
            pass

        def shutdown_memory_provider(self):
            pass

        def close(self):
            pass

    monkeypatch.setattr(run_agent_module, "AIAgent", FakeReviewAgent)
    monkeypatch.setattr(run_agent_module.threading, "Thread", ImmediateThread)

    agent = _bare_agent()
    agent._safe_print = lambda *a, **kw: captured_prints.append(" ".join(str(x) for x in a))
    agent.background_review_callback = lambda msg: captured_bg_callback.append(msg)

    AIAgent._spawn_background_review(
        agent,
        messages_snapshot=[{"role": "user", "content": "hi"}],
        review_memory=True,
    )

    # Two emissions are expected: the issue #28976 start notice, then
    # the original end summary. Both must identify the self-improvement
    # review explicitly so users can correlate start/end pairs.
    assert len(captured_prints) == 2, captured_prints
    start_printed, end_printed = captured_prints
    assert "Self-improvement review: starting" in start_printed, start_printed
    assert "Self-improvement review" in end_printed, end_printed
    assert "Memory updated" in end_printed, end_printed

    # Gateway path gets the same prefix on both notices.
    assert len(captured_bg_callback) == 2, captured_bg_callback
    assert all(
        m.startswith("💾 Self-improvement review:") for m in captured_bg_callback
    ), captured_bg_callback
    assert "starting" in captured_bg_callback[0]
    assert "Memory updated" in captured_bg_callback[1]


class _NoOpThread:
    """Thread stub that skips the worker — start-notice tests don't need
    the real review to run, and skipping it removes any chance of the
    end-notice racing into our capture lists.
    """

    def __init__(self, *, target=None, daemon=None, name=None):
        pass

    def start(self):
        pass


def _capture_spawn_notices(monkeypatch, *, review_memory: bool, review_skills: bool):
    """Run ``_spawn_background_review`` against a thread stub and capture
    everything that landed on either notification surface before the
    worker would have started.
    """
    monkeypatch.setattr(run_agent_module.threading, "Thread", _NoOpThread)

    captured_prints: list = []
    captured_bg_callback: list = []

    agent = _bare_agent()
    agent._safe_print = lambda *a, **kw: captured_prints.append(" ".join(str(x) for x in a))
    agent.background_review_callback = lambda msg: captured_bg_callback.append(msg)

    AIAgent._spawn_background_review(
        agent,
        messages_snapshot=[{"role": "user", "content": "hi"}],
        review_memory=review_memory,
        review_skills=review_skills,
    )
    return captured_prints, captured_bg_callback


def test_background_review_emits_start_notice_for_memory_only(monkeypatch):
    """Issue #28976: the user must see a start notice when the background
    review is launched, not only an end notice that fires only when
    something actually changed. The scope tag tells them which stores
    are being reviewed so they can predict the kind of end-update to
    expect.
    """
    prints, bg = _capture_spawn_notices(monkeypatch, review_memory=True, review_skills=False)
    assert len(prints) == 1, prints
    assert "Self-improvement review: starting (memory)" in prints[0], prints[0]
    assert len(bg) == 1
    assert bg[0] == "💾 Self-improvement review: starting (memory)…", bg[0]


def test_background_review_emits_start_notice_for_skills_only(monkeypatch):
    prints, bg = _capture_spawn_notices(monkeypatch, review_memory=False, review_skills=True)
    assert len(prints) == 1, prints
    assert "Self-improvement review: starting (skills)" in prints[0], prints[0]
    assert bg == ["💾 Self-improvement review: starting (skills)…"], bg


def test_background_review_emits_start_notice_for_both_scopes(monkeypatch):
    prints, bg = _capture_spawn_notices(monkeypatch, review_memory=True, review_skills=True)
    assert len(prints) == 1, prints
    assert "Self-improvement review: starting (memory + skills)" in prints[0], prints[0]
    assert bg == ["💾 Self-improvement review: starting (memory + skills)…"], bg


def test_background_review_start_notice_survives_callback_exception(monkeypatch):
    """A misbehaving gateway callback must not prevent the worker from
    being scheduled, the same contract the end-notice path already
    relies on (see ``agent.background_review`` line ~507). Without this
    guard, a failing TUI consumer would silently disable background
    review entirely.
    """
    monkeypatch.setattr(run_agent_module.threading, "Thread", _NoOpThread)

    started = []

    class _RecordingThread(_NoOpThread):
        def start(self):
            started.append(True)

    monkeypatch.setattr(run_agent_module.threading, "Thread", _RecordingThread)

    def _boom(_msg):
        raise RuntimeError("gateway down")

    agent = _bare_agent()
    agent._safe_print = lambda *a, **kw: None
    agent.background_review_callback = _boom

    AIAgent._spawn_background_review(
        agent,
        messages_snapshot=[{"role": "user", "content": "hi"}],
        review_memory=True,
    )

    assert started == [True], "thread must still start after callback raise"


def test_background_review_start_notice_skipped_when_no_callback(monkeypatch):
    """The CLI path has ``background_review_callback = None``; the start
    notice must still print on ``_safe_print`` and must not crash trying
    to invoke a None callback.
    """
    monkeypatch.setattr(run_agent_module.threading, "Thread", _NoOpThread)

    captured_prints: list = []

    agent = _bare_agent()
    agent._safe_print = lambda *a, **kw: captured_prints.append(" ".join(str(x) for x in a))
    agent.background_review_callback = None

    AIAgent._spawn_background_review(
        agent,
        messages_snapshot=[{"role": "user", "content": "hi"}],
        review_memory=True,
    )

    assert len(captured_prints) == 1, captured_prints
    assert "starting (memory)" in captured_prints[0]


def test_background_review_fork_skips_external_memory_plugins(monkeypatch):
    """The background review fork must NOT touch external memory plugins.

    Without skip_memory=True on the fork constructor, AIAgent.__init__
    rebuilds its own _memory_manager from config, scoped to the parent's
    session_id.  The review fork's run_conversation() then leaks the
    harness prompt into the user's real memory namespace via three
    ingestion sites: on_turn_start (cadence + turn message),
    prefetch_all (recall query), and sync_all (harness prompt + review
    output recorded as a (user, assistant) turn pair).  The fix is a
    single kwarg on the fork constructor — this test guards it.
    """
    captured_kwargs: dict = {}

    class FakeReviewAgent:
        def __init__(self, **kwargs):
            captured_kwargs.update(kwargs)
            self._session_messages = []

        def run_conversation(self, **kwargs):
            pass

        def shutdown_memory_provider(self):
            pass

        def close(self):
            pass

    monkeypatch.setattr(run_agent_module, "AIAgent", FakeReviewAgent)
    monkeypatch.setattr(run_agent_module.threading, "Thread", ImmediateThread)

    agent = _bare_agent()

    AIAgent._spawn_background_review(
        agent,
        messages_snapshot=[{"role": "user", "content": "hello"}],
        review_memory=True,
    )

    assert captured_kwargs.get("skip_memory") is True, (
        "Background review fork must be constructed with skip_memory=True "
        "so AIAgent.__init__ does not rebuild a _memory_manager wired to "
        "external plugins (honcho, mem0, supermemory, ...).  Without this "
        "the fork leaks harness prompts into the user's real memory "
        "namespace via on_turn_start / prefetch_all / sync_all."
    )


# ---------------------------------------------------------------------------
# memory_notifications mode: off | on | verbose
# ---------------------------------------------------------------------------

import json as _json

from agent.background_review import summarize_background_review_actions


def _memory_add_review():
    """A minimal review transcript: one memory add (assistant call + tool result)."""
    return [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call_mem1",
                    "function": {
                        "name": "memory",
                        "arguments": _json.dumps(
                            {
                                "action": "add",
                                "target": "memory",
                                "content": "User prefers terse replies",
                            }
                        ),
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_mem1",
            "content": _json.dumps(
                {"success": True, "message": "Entry added.", "target": "memory"}
            ),
        },
    ]


def _skill_patch_review():
    return [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call_skill1",
                    "function": {
                        "name": "skill_manage",
                        "arguments": _json.dumps(
                            {"action": "patch", "name": "demo", "old_string": "a", "new_string": "b"}
                        ),
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_skill1",
            "content": _json.dumps(
                {
                    "success": True,
                    "message": "Patched SKILL.md in skill 'demo' (1 replacement).",
                    "_change": {"old": "a", "new": "b"},
                }
            ),
        },
    ]


def test_memory_notifications_off_returns_nothing():
    actions = summarize_background_review_actions(
        _memory_add_review(), [], notification_mode="off"
    )
    assert actions == []


def test_memory_notifications_on_returns_generic_line():
    actions = summarize_background_review_actions(
        _memory_add_review(), [], notification_mode="on"
    )
    assert actions == ["Memory updated"]


def test_memory_notifications_verbose_includes_content_preview():
    actions = summarize_background_review_actions(
        _memory_add_review(), [], notification_mode="verbose"
    )
    assert len(actions) == 1
    # Verbose surfaces the actual content that was saved.
    assert "User prefers terse replies" in actions[0]
    assert actions[0] != "Memory updated"


def test_memory_notifications_default_is_on():
    """No mode passed → behaves like 'on' (generic line, not empty/verbose)."""
    actions = summarize_background_review_actions(_memory_add_review(), [])
    assert actions == ["Memory updated"]


def test_skill_patch_off_silent_verbose_shows_diff():
    assert (
        summarize_background_review_actions(
            _skill_patch_review(), [], notification_mode="off"
        )
        == []
    )
    verbose = summarize_background_review_actions(
        _skill_patch_review(), [], notification_mode="verbose"
    )
    assert len(verbose) == 1
    assert "demo" in verbose[0] and "→" in verbose[0]
