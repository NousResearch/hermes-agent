"""The CLI's own compaction must be visible on the claude-agent-sdk lane.

Hermes never compacts here (the CLI owns context), so a turn can stall for a
compaction with nothing to show for it. The SDK's PreCompact hook is the only
honest signal, and it must surface through the SHARED status wording -- the
gateway's noise filter is built from those same template constants (#69550),
so a re-inlined string would be silently dropped on chat surfaces.
"""

from __future__ import annotations

import asyncio
import sys
from types import ModuleType

import pytest

from agent.conversation_compression import (
    COMPACTION_DONE_STATUS,
    COMPACTION_STATUS,
    COMPACTION_STATUS_KEY,
)
from agent.transports import claude_agent_sdk_session as M


@pytest.fixture(autouse=True)
def _optional_sdk_stand_in(monkeypatch):
    """Hook tests must not require the optional provider extra in CI."""
    module = ModuleType("claude_agent_sdk")

    class HookMatcher:
        def __init__(self, *, hooks):
            self.hooks = hooks

    module.HookMatcher = HookMatcher
    monkeypatch.setitem(sys.modules, "claude_agent_sdk", module)


def _session(on_compaction):
    s = M.ClaudeAgentSdkSession.__new__(M.ClaudeAgentSdkSession)
    s._on_compaction = on_compaction
    return s


def test_hook_is_registered_even_when_no_status_callback_is_wired():
    """Contract change: the hook no longer exists only to emit a status.

    Its primary job is now telling _TurnWatch that the silence between
    PreCompact and compact_boundary is legitimate -- without that the quiet
    watchdog interrupts the CLI mid-compaction and kills the turn. Skipping
    the hook because the *status* callback is unset would leave the turn
    killable for no benefit, so it is wired unconditionally and the callback
    is what stays optional.
    """
    hooks = _session(None)._build_compaction_hooks()

    assert hooks is not None
    assert set(hooks) == {"PreCompact"}


def test_unwired_hook_runs_clean_without_a_status_callback():
    """It must not call None, and must still return the empty hook result."""
    callback = _session(None)._build_compaction_hooks()["PreCompact"][0].hooks[0]

    assert asyncio.run(callback({"trigger": "auto"}, None, {"signal": None})) == {}


def test_registers_a_precompact_hook_when_wired():
    hooks = _session(lambda trigger: None)._build_compaction_hooks()

    assert hooks is not None
    assert set(hooks) == {"PreCompact"}
    matchers = hooks["PreCompact"]
    assert len(matchers) == 1
    assert len(matchers[0].hooks) == 1


@pytest.mark.parametrize("trigger", ["auto", "manual"])
def test_hook_forwards_the_sdk_trigger_verbatim(trigger):
    seen = []
    hooks = _session(seen.append)._build_compaction_hooks()
    callback = hooks["PreCompact"][0].hooks[0]

    result = asyncio.run(callback({"trigger": trigger}, None, {"signal": None}))

    assert seen == [trigger]
    assert result == {}


def test_missing_trigger_defaults_to_auto():
    """A payload without `trigger` is still a real compaction — announce it."""
    seen = []
    hooks = _session(seen.append)._build_compaction_hooks()
    callback = hooks["PreCompact"][0].hooks[0]

    asyncio.run(callback({}, None, {"signal": None}))

    assert seen == ["auto"]


def test_a_raising_callback_never_fails_the_hook():
    """Refusing a hook can block the compaction itself — always return {}."""

    def _boom(trigger):
        raise RuntimeError("status pipeline down")

    hooks = _session(_boom)._build_compaction_hooks()
    callback = hooks["PreCompact"][0].hooks[0]

    assert asyncio.run(callback({"trigger": "auto"}, None, {"signal": None})) == {}


def test_status_wording_is_single_sourced():
    """Pin the reuse: these are the constants the gateway filter is built from."""
    assert "Compacting context" in COMPACTION_STATUS
    assert "compaction complete" in COMPACTION_DONE_STATUS.lower()


# ── Observability ───────────────────────────────────────────────────────────
# The gateway logs no outbound sends (`grep -c outbound gateway.log` -> 0), so
# before these lines the only witness that a compaction notice reached the user
# was the user happening to watch the screen. Confirmed unanswerable in
# production 2026-08-16: a real auto-compaction fired (preTokens=697,652) and
# whether the completion notice followed could not be determined from disk.


class _AgentWithCallback:
    def __init__(self):
        self.seen = []
        self.status_callback = lambda kind, text: self.seen.append((kind, text))


def test_completion_emit_is_logged(caplog):
    from agent.conversation_compression import _emit_compaction_done

    agent = _AgentWithCallback()
    with caplog.at_level("INFO", logger="agent.conversation_compression"):
        _emit_compaction_done(agent)

    # Start and completion deliberately share COMPACTION_STATUS_KEY rather
    # than using distinct "compaction"/"compacted" keys. That is the point of
    # the typed-key change: a keyed consumer (the Telegram adapter) replaces
    # the existing bubble in place instead of stacking a second one, so the
    # user sees one notice that resolves rather than two that accumulate.
    assert agent.seen == [(COMPACTION_STATUS_KEY, COMPACTION_DONE_STATUS)]
    assert "completion notice emitted" in caplog.text


def test_missing_callback_is_logged_not_silent(caplog):
    """The silent-return path is the one that loses the notice — it must log."""
    from agent.conversation_compression import _emit_compaction_done

    class _NoCallback:
        status_callback = None

    with caplog.at_level("INFO", logger="agent.conversation_compression"):
        _emit_compaction_done(_NoCallback())

    assert "not emitted" in caplog.text


# The START-side log lives in a closure nested inside run_claude_agent_sdk_turn,
# which cannot be exercised without standing up a full turn. It is deliberately
# left to production verification rather than covered by a source-text grep --
# a test that asserts on source is not evidence the line ever runs.


# ── Completion edge: compact_boundary ───────────────────────────────────────
# The SDK has no PostCompact hook, so the completion edge was originally
# deferred to the end of the turn. That put it exactly where end-of-turn
# progress cleanup deletes non-durable statuses -- emitted and destroyed in the
# same instant. The CLI does announce completion, as a `system` message with
# subtype `compact_boundary` that the SDK's generic subtype fallback passes
# through mid-turn. Measured 2026-08-16: boundary at 07:41:16, deferred emit at
# 07:42:17 -- 61s late and invisible.


class SystemMessage:
    """Stands in for the SDK's SystemMessage.

    The handler dispatches on ``type(message).__name__`` rather than isinstance
    (the SDK is an optional import), so this class must genuinely BE named
    SystemMessage -- that name is the contract under test.
    """

    def __init__(self, subtype, data=None, session_id="sess-1"):
        self.subtype = subtype
        self.data = data
        self.session_id = session_id


def _boundary_session(on_compact_boundary):
    s = M.ClaudeAgentSdkSession.__new__(M.ClaudeAgentSdkSession)
    s._on_compact_boundary = on_compact_boundary
    s._session_id = "sess-1"
    return s


def _boundary(trigger="auto"):
    return SystemMessage(
        "compact_boundary",
        {"compact_metadata": {"trigger": trigger, "preTokens": 697652}},
    )


def test_boundary_fires_the_completion_callback():
    seen = []
    _boundary_session(seen.append)._handle_compact_boundary(_boundary("auto"))
    assert seen == ["auto"]


def test_boundary_forwards_the_manual_trigger():
    """The runtime decides what to announce; the transport reports the truth."""
    seen = []
    _boundary_session(seen.append)._handle_compact_boundary(_boundary("manual"))
    assert seen == ["manual"]


def test_boundary_without_metadata_defaults_to_auto():
    seen = []
    msg = SystemMessage("compact_boundary", {})
    _boundary_session(seen.append)._handle_compact_boundary(msg)
    assert seen == ["auto"]


@pytest.mark.parametrize(
    "message",
    [
        SystemMessage("init", {}),
        SystemMessage("task_started", {}),
    ],
)
def test_other_system_subtypes_are_ignored(message):
    seen = []
    _boundary_session(seen.append)._handle_compact_boundary(message)
    assert seen == []


def test_non_system_messages_are_ignored():
    class ResultMessage:
        subtype = "compact_boundary"  # right subtype, wrong message type
        data = {}

    seen = []
    _boundary_session(seen.append)._handle_compact_boundary(ResultMessage())
    assert seen == []


def test_unwired_boundary_callback_is_safe():
    """No callback wired must not raise on the live message path."""
    _boundary_session(None)._handle_compact_boundary(_boundary())


def test_a_raising_boundary_callback_never_breaks_the_turn():
    """This runs inside the message loop of a turn that is still streaming."""

    def _boom(trigger):
        raise RuntimeError("status pipeline down")

    _boundary_session(_boom)._handle_compact_boundary(_boundary())


def test_boundary_is_logged(caplog):
    with caplog.at_level("INFO", logger="agent.transports.claude_agent_sdk_session"):
        _boundary_session(None)._handle_compact_boundary(_boundary("auto"))

    assert "compact_boundary" in caplog.text
    assert "trigger=auto" in caplog.text
