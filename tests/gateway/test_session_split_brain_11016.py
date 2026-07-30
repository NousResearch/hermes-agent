"""Regression tests for issue #11016 — Telegram sessions trapped in
repeated 'Interrupting current task...' while /stop reports no active task.

Covers three layers of the fix:

1. Adapter-side task ownership (_session_tasks map): /stop, /new, /reset
   actually cancel the in-flight adapter task and release the guard in
   order, so follow-up messages reach the new session.

2. Adapter-side on-entry self-heal: if _active_sessions still has an
   entry but the recorded owner task is already done/cancelled, clear it
   on the next inbound message rather than trapping the user.

3. Runner-side generation guard: a stale async run can't promote itself
   into _running_agents after /stop/ /new bumped the generation, and
   can't clear a newer run's slot on the way out.
"""

import asyncio
from unittest.mock import MagicMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
)
from gateway.run import GatewayRunner
from gateway.session import SessionSource, build_session_key


# ---------------------------------------------------------------------------
# Adapter helpers
# ---------------------------------------------------------------------------


class _StubAdapter(BasePlatformAdapter):
    async def connect(self, *, is_reconnect: bool = False):
        pass

    async def disconnect(self):
        pass

    async def send(self, chat_id, text, **kwargs):
        pass

    async def get_chat_info(self, chat_id):
        return {}


def _make_adapter():
    config = PlatformConfig(enabled=True, token="test-token")
    adapter = _StubAdapter(config, Platform.TELEGRAM)
    adapter._busy_text_mode = ""
    adapter.sent_responses = []

    async def _mock_send_retry(chat_id, content, **kwargs):
        adapter.sent_responses.append(content)

    adapter._send_with_retry = _mock_send_retry
    return adapter


def _make_event(text="hello", chat_id="12345"):
    source = SessionSource(
        platform=Platform.TELEGRAM, chat_id=chat_id, chat_type="dm"
    )
    return MessageEvent(text=text, message_type=MessageType.TEXT, source=source)


def _session_key(chat_id="12345"):
    source = SessionSource(
        platform=Platform.TELEGRAM, chat_id=chat_id, chat_type="dm"
    )
    return build_session_key(source)


# ---------------------------------------------------------------------------
# Runner helpers
# ---------------------------------------------------------------------------


def _make_runner():
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="***")}
    )
    runner.adapters = {}
    runner._running_agents = {}
    runner._running_agents_ts = {}
    runner._session_run_generation = {}
    runner._pending_messages = {}
    runner._draining = False
    runner._update_runtime_status = MagicMock()
    return runner


# ===========================================================================
# Layer 1: Adapter-side session cancellation on /stop /new /reset
# ===========================================================================


class TestAdapterSessionCancellation:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("command_text", ["/stop", "/new", "/reset"])
    async def test_command_cancels_active_task_and_unblocks_follow_up(
        self, command_text
    ):
        """/stop /new /reset must cancel the adapter task and let follow-ups through."""
        adapter = _make_adapter()
        sk = _session_key()
        processing_started = asyncio.Event()
        processing_cancelled = asyncio.Event()
        blocked_first_message = True

        async def _handler(event):
            nonlocal blocked_first_message
            cmd = event.get_command()
            if cmd in {"stop", "new", "reset", "model"}:
                return f"handled:{cmd}"

            if blocked_first_message:
                blocked_first_message = False
                processing_started.set()
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    processing_cancelled.set()
                    raise
            return f"handled:text:{event.text}"

        adapter._message_handler = _handler

        await adapter.handle_message(_make_event("hello world"))
        await processing_started.wait()
        await asyncio.sleep(0)

        assert sk in adapter._active_sessions
        assert sk in adapter._session_tasks

        await adapter.handle_message(_make_event(command_text))

        assert processing_cancelled.is_set(), (
            f"{command_text} did not cancel the active processing task"
        )
        assert sk not in adapter._active_sessions
        assert sk not in adapter._pending_messages
        assert sk not in adapter._session_tasks
        expected = command_text.lstrip("/")
        assert any(f"handled:{expected}" in r for r in adapter.sent_responses)

        # Follow-up must go through normally now that the session is clean.
        await adapter.handle_message(
            _make_event("/model xiaomi/mimo-v2-pro --provider nous")
        )
        await asyncio.sleep(0)
        await asyncio.sleep(0)

        assert any("handled:model" in r for r in adapter.sent_responses), (
            f"follow-up /model stayed blocked after {command_text}"
        )
        assert sk not in adapter._pending_messages


# ===========================================================================
# Layer 2: Adapter-side on-entry self-heal for stale session locks
# ===========================================================================


class TestStaleSessionLockSelfHeal:
    @pytest.mark.asyncio
    async def test_stale_lock_with_done_task_is_healed_on_next_message(self):
        """A split-brain guard (owner task done but entry still live) heals on next inbound."""
        adapter = _make_adapter()
        sk = _session_key()

        # Simulate the production split-brain: an _active_sessions entry
        # remains AND a recorded owner task, but that task is already done.
        async def _done():
            return None

        done_task = asyncio.create_task(_done())
        await done_task
        assert done_task.done()

        adapter._active_sessions[sk] = asyncio.Event()
        adapter._session_tasks[sk] = done_task

        assert adapter._session_task_is_stale(sk)

        async def _handler(event):
            return f"handled:{event.get_command() or 'text'}"

        adapter._message_handler = _handler

        # An ordinary message should heal the stale lock, then fall through
        # to normal dispatch.  User gets a reply instead of a busy ack.
        await adapter.handle_message(_make_event("hello"))
        # Drain any spawned background tasks.
        for _ in range(5):
            await asyncio.sleep(0)

        assert any("handled:text" in r for r in adapter.sent_responses), (
            "stale lock trapped a normal message — split-brain not healed"
        )


    @pytest.mark.asyncio
    async def test_cleanup_releases_and_deletes_when_guard_matches(self):
        """Positive path for #48300: when the guard still matches (normal
        completion), the helper releases the guard AND drops the task entry —
        the release-then-conditional-delete must not strand a healthy session."""
        adapter = _make_adapter()
        sk = _session_key()

        event_a = asyncio.Event()

        async def _done():
            return None

        done_task = asyncio.create_task(_done())
        await done_task

        adapter._active_sessions[sk] = event_a
        adapter._session_tasks[sk] = done_task

        # No guard swap → _release_session_guard matches event_a and releases.
        adapter._cleanup_finished_session_task(sk, event_a)

        assert sk not in adapter._active_sessions, "guard must be released on match"
        assert sk not in adapter._session_tasks, "task entry must be dropped after release"


# ===========================================================================
# Layer 3: Runner-side generation guard on slot promotion + release
# ===========================================================================


class TestRunnerSessionGenerationGuard:


    def test_release_with_stale_generation_blocks(self):
        runner = _make_runner()
        sk = "agent:main:telegram:dm:12345"
        stale_gen = runner._begin_session_run_generation(sk)
        # /stop bumps the generation — stale run's generation is no longer current.
        runner._invalidate_session_run_generation(sk, reason="stop")
        # The fresh run lands next; imagine it has its own state installed.
        runner._running_agents[sk] = "fresh_agent"
        runner._running_agents_ts[sk] = 2.0

        # Stale run's unwind MUST NOT clobber the fresh run's state.
        released = runner._release_running_agent_state(sk, run_generation=stale_gen)

        assert released is False
        assert runner._running_agents[sk] == "fresh_agent"
        assert runner._running_agents_ts[sk] == 2.0


# ===========================================================================
# Layer 1 (regression): old task's finally must NOT delete a newer guard
# ===========================================================================


class TestOldTaskCannotClobberNewerGuard:
    """Direct regression for the unconditional-delete bug.

    Before the guard-match fix, a task in its finally would delete
    ``_active_sessions[session_key]`` unconditionally — even if a
    /stop/ /new command had already swapped in its own command_guard
    (which then gets clobbered, opening a race for follow-up messages).
    """

    def test_release_session_guard_matches_on_event_identity(self):
        adapter = _make_adapter()
        sk = _session_key()

        old_guard = asyncio.Event()
        new_guard = asyncio.Event()
        # Command swapped in a newer guard.
        adapter._active_sessions[sk] = new_guard

        # Old task tries to release using its captured (stale) guard.
        adapter._release_session_guard(sk, guard=old_guard)

        # The newer guard survives.
        assert adapter._active_sessions.get(sk) is new_guard

        # Now the command itself releases using the matching guard.
        adapter._release_session_guard(sk, guard=new_guard)
        assert sk not in adapter._active_sessions

