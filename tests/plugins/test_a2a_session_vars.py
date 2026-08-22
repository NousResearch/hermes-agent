"""A2A inbound binds session-context vars before agent dispatch.

Regression test for the task-completion push-back gap: the
A2A adapter never set the session-context vars that kanban
``_maybe_auto_subscribe`` reads (``HERMES_SESSION_PLATFORM`` /
``HERMES_SESSION_CHAT_ID`` / ``HERMES_SESSION_THREAD_ID``), so cards
created from an A2A session got zero ``kanban_notify_subs`` rows and the
notifier never pushed completions back to the A2A peer.

The vars must ride the dispatched task's context via ContextVars (NOT a
process-global ``os.environ`` write — that is last-writer-wins across
concurrent A2A contexts and leaks into sibling sessions):
- ``_prepare_task`` binds via ``set_session_vars`` on the HTTP worker
  thread; ``asyncio.run_coroutine_threadsafe`` snapshots that thread's
  context when it constructs the Task, so the whole dispatch chain sees
  the identity.
- ``_forward_to_profile`` carries the identity to the forwarded profile's
  CLI subprocess via the child env (``get_session_env`` falls back to
  ``os.environ`` in a CLI process).
"""

from __future__ import annotations

import asyncio
import subprocess
import threading
import time

import pytest

from gateway.config import PlatformConfig
from plugins.platforms.a2a import protocol
from plugins.platforms.a2a.adapter import A2AAdapter


def _bare_adapter() -> A2AAdapter:
    return A2AAdapter(PlatformConfig(enabled=True))


def test_prepare_task_binds_session_vars_into_dispatched_context(monkeypatch, tmp_path):
    """_prepare_task must bind platform/chat_id/thread_id ContextVars before
    dispatching, visible inside the agent-side message handler even when the
    os.environ fallback is empty (the old os.environ mechanism failed this)."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    # Clear the process-global fallback so ONLY ContextVars can supply the
    # identity — this is what distinguishes the fix from the old os.environ
    # mechanism.
    monkeypatch.delenv("HERMES_SESSION_PLATFORM", raising=False)
    monkeypatch.delenv("HERMES_SESSION_CHAT_ID", raising=False)
    monkeypatch.delenv("HERMES_SESSION_THREAD_ID", raising=False)

    from gateway.session_context import (
        _UNSET,
        reset_session_vars,
    )

    captured: list[dict] = []

    adapter = _bare_adapter()
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()
    try:
        adapter._loop = loop

        async def fake_handle_message(event):
            from gateway.session_context import (
                _UNSET,
                _VAR_MAP,
                get_session_env,
            )

            captured.append(
                {
                    # The mechanism contract: the vars must be bound as
                    # ContextVars (the old os.environ write leaves these
                    # _UNSET and only masks a same-process read via the
                    # fallback — invisible to the tool process under the
                    # subprocess-env bridge).
                    "platform_cv": _VAR_MAP["HERMES_SESSION_PLATFORM"].get(),
                    "chat_id_cv": _VAR_MAP["HERMES_SESSION_CHAT_ID"].get(),
                    "thread_id_cv": _VAR_MAP["HERMES_SESSION_THREAD_ID"].get(),
                    # The value contract: what the tool process reads.
                    "platform": get_session_env("HERMES_SESSION_PLATFORM", ""),
                    "chat_id": get_session_env("HERMES_SESSION_CHAT_ID", ""),
                    "thread_id": get_session_env("HERMES_SESSION_THREAD_ID", ""),
                    "event_chat_id": getattr(event.source, "chat_id", None),
                    "event_message_id": getattr(event, "message_id", None),
                }
            )

        adapter.handle_message = fake_handle_message  # type: ignore[method-assign]
        adapter._message_handler = object()  # type: ignore[assignment]

        params = {
            "message": {
                "parts": [{"text": "hello from peer-a"}],
                "contextId": "peer-a2a-test-ctx",
            },
        }
        _, pending = adapter._prepare_task(params, "ip:127.0.0.1")
        assert pending is not None
        task_id = pending["task_id"]

        deadline = time.time() + 5
        while not captured and time.time() < deadline:
            time.sleep(0.01)
        assert captured, "dispatched handler never ran"

        got = captured[0]
        assert got["platform_cv"] is not _UNSET and got["platform_cv"] == "a2a"
        assert got["chat_id_cv"] is not _UNSET and got["chat_id_cv"] == "peer-a2a-test-ctx"
        assert got["thread_id_cv"] is not _UNSET and got["thread_id_cv"] == task_id
        assert got["platform"] == "a2a"
        assert got["chat_id"] == "peer-a2a-test-ctx"
        assert got["thread_id"] == task_id
        assert got["event_chat_id"] == "peer-a2a-test-ctx"
        assert got["event_message_id"] == task_id
    finally:
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=5)
        try:
            loop.close()
        except Exception:
            pass
        adapter._unregister_adapter()
        # Restore the pristine per-task context for any later tests in this
        # process (pytest reuses the thread; a lingering binding would mask
        # the os.environ fallback for sibling tests).
        reset_session_vars()


def test_worker_thread_context_is_reset_after_dispatch(monkeypatch, tmp_path):
    """After scheduling the dispatch, the worker thread's own context must
    be reset so the bindings don't linger on the threadpool thread."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    from gateway.session_context import get_session_env, reset_session_vars

    adapter = _bare_adapter()
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()
    try:
        adapter._loop = loop
        adapter.handle_message = lambda event: _noop_coro()  # type: ignore[method-assign]
        adapter._message_handler = object()  # type: ignore[assignment]

        params = {
            "message": {
                "parts": [{"text": "hello"}],
                "contextId": "peer-a2a-reset-ctx",
            },
        }
        _, pending = adapter._prepare_task(params, "ip:127.0.0.1")
        assert pending is not None

        # Give the scheduled task a moment to run, then confirm THIS thread
        # (standing in for the worker thread) no longer has the bindings.
        time.sleep(0.2)
        assert get_session_env("HERMES_SESSION_PLATFORM", "") == ""
        assert get_session_env("HERMES_SESSION_CHAT_ID", "") == ""
    finally:
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=5)
        try:
            loop.close()
        except Exception:
            pass
        adapter._unregister_adapter()
        reset_session_vars()


async def _noop_coro():
    return None


def test_forward_to_profile_carries_session_vars_in_child_env(monkeypatch, tmp_path):
    """The forwarded-profile subprocess env must carry the A2A session
    identity (platform/chat_id/thread_id) so the CLI agent in the child
    process sees them via get_session_env's os.environ fallback."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    captured: dict = {}

    def fake_run(cmd, **kw):
        captured["cmd"] = cmd
        captured["env"] = kw["env"]
        return subprocess.CompletedProcess(cmd, 0, stdout="ok", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    adapter = _bare_adapter()
    try:
        # Keep the test hermetic: never touch the real profile's state.db.
        adapter._lookup_forward_session = lambda *a, **k: ""  # type: ignore[method-assign]
        adapter._latest_a2a_session = lambda *a, **k: ""  # type: ignore[method-assign]

        agent = {"slug": "worker-a", "profile": "worker-a", "local": False, "timeout": 30}
        reply, state = adapter._forward_to_profile(
            agent, "ip:127.0.0.1", "worker-a2a-fwd-ctx", "hello", "task-fwd-1"
        )
        assert state == protocol.STATE_COMPLETED
        assert reply == "ok"

        env = captured["env"]
        assert env["HERMES_SESSION_PLATFORM"] == "a2a"
        assert env["HERMES_SESSION_CHAT_ID"] == "worker-a2a-fwd-ctx"
        assert env["HERMES_SESSION_THREAD_ID"] == "task-fwd-1"
        assert env["HERMES_A2A_PEER"] == "ip:127.0.0.1"
    finally:
        adapter._unregister_adapter()
