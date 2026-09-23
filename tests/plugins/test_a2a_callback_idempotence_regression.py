"""Regression: callback-enabled terminal idempotence after pending completion.

Proves that consuming the in-memory push callback after the first
post-commit delivery does not turn a repeated terminal finalization
into a persistence conflict, and that the callback is delivered exactly
once.

This exercises the normal pending path:
  task_routing._durable_complete_pending → _finalize_task(pop_pending=False)
  → post-commit _send_push_notification (pop_push_url)
  → blocked HTTP path calls _finalize_task again

Not a manually seeded TaskStore helper — the task is created via the
real _prepare_task → publish_durable(WORKING) seam.
"""

from __future__ import annotations

import asyncio
import json
import pathlib
from unittest import mock

import pytest

from plugins.platforms.a2a import protocol
from plugins.platforms.a2a.adapter import A2AAdapter
from gateway.config import PlatformConfig


def test_pending_callback_idempotence_delivers_once_and_duplicate_succeeds(monkeypatch, tmp_path):
    """Callback is sent once; duplicate _finalize_task after pop returns idempotent success."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("A2A_PUSH_SECRET", "test-secret")

    # Allow any callback URL in test
    from plugins.platforms.a2a import security as sec_mod

    monkeypatch.setattr(sec_mod, "is_safe_callback_url", lambda url, localhost_mode=True: True)

    adapter = A2AAdapter(PlatformConfig(enabled=True, extra={"port": 0}))
    # Provide a running-capable shim for _prepare_task's pending dispatch.
    # _prepare_task schedules handle_message via run_coroutine_threadsafe; we
    # must not leave a real coroutine unawaited (which surfaces as a
    # RuntimeWarning). Replace the scheduling with a no-op future.
    dummy_loop = object()
    adapter._loop = dummy_loop  # type: ignore
    adapter._message_handler = object()
    monkeypatch.setattr(
        "asyncio.run_coroutine_threadsafe",
        lambda coro, loop: _close_coro_and_return_dummy_future(coro),  # noqa: ARG005
    )

    def _close_coro_and_return_dummy_future(coro):
        # Prevent RuntimeWarning: coroutine was never awaited
        try:
            coro.close()
        except Exception:
            pass
        fut = mock.MagicMock()
        fut.done.return_value = False
        return fut
    # Mock gateway ready and security context for push
    adapter._gateway_ready = mock.MagicMock(return_value=True)
    adapter._security_context = mock.MagicMock()
    adapter._security_context.localhost_only.return_value = True
    adapter._security_context.sign_push_payload.return_value = "sig"

    ledger = tmp_path / "a2a_task_ledger.json"

    # Track push deliveries without real HTTP
    push_calls: list[tuple[str, str, str, str]] = []

    def tracked_push(task_id: str, context_id: str, reply: str, state: str):
        push_calls.append((task_id, context_id, reply, state))
        # Pop the URL exactly as real _send_push_notification does, consuming memory
        url = adapter.tasks.pop_push_url(task_id)
        if url:
            protocol.metrics.push_sent += 1
        return

    monkeypatch.setattr(adapter, "_send_push_notification", tracked_push)

    hook_url = "http://127.0.0.1:8765/callback"

    try:
        params = {
            "message": protocol.text_message(protocol.ROLE_USER, "hello with push", context_id="ctx-cb-idem-001", sender="user1"),
            "configuration": {
                "taskPushNotificationConfig": {
                    "url": hook_url,
                }
            },
        }
        terminal, pending = adapter._prepare_task(params, "peer1", agent=adapter._agents[""])
        assert terminal is None, "expected pending, not immediate terminal"
        assert pending is not None
        task_id = pending["task_id"]
        context_id = pending["context_id"]

        # WORKING record must carry push_url durably and in memory
        assert ledger.exists(), "ledger should exist after WORKING publish"
        data = json.loads(ledger.read_text(encoding="utf-8"))
        assert data[task_id].get("push_url") == hook_url
        assert data[task_id].get("state") == protocol.STATE_WORKING
        assert adapter.tasks.get(task_id).get("push_url") == hook_url

        # Normal pending flow: agent reply via _durable_complete_pending (pop_pending=False → push)
        ok, err = adapter._durable_complete_pending(task_id, context_id, "agent reply text", "msg-1")
        assert ok, f"_durable_complete_pending should succeed, got err={err!r}"
        assert len(push_calls) == 1, f"exactly one callback expected after first completion, got {len(push_calls)}"
        assert push_calls[0][0] == task_id
        assert push_calls[0][2] == "agent reply text"

        rec_done = adapter.tasks.get(task_id)
        assert rec_done is not None
        assert rec_done.get("state") == protocol.STATE_COMPLETED
        assert rec_done.get("reply") == "agent reply text"
        # Post-commit consumption clears in-memory push_url, disk retains it
        assert rec_done.get("push_url") == "", "in-memory push_url must be consumed after delivery"
        data2 = json.loads(ledger.read_text(encoding="utf-8"))
        assert data2[task_id].get("push_url") == hook_url, "disk must still retain push_url"
        assert data2[task_id].get("state") == protocol.STATE_COMPLETED

        # Blocked HTTP retry: same pending dict, same terminal state/reply, must be idempotent
        # Simulate the HTTP thread's second _finalize_task call (pop_pending default True)
        state2, reply2 = adapter._finalize_task(pending, protocol.STATE_COMPLETED, "agent reply text")
        assert state2 == protocol.STATE_COMPLETED
        assert reply2 == "agent reply text"
        assert len(push_calls) == 1, "duplicate finalization must not resend callback"

        # Reconstructed pending (HTTP handler would build a fresh dict from request context)
        pending2 = {
            "task_id": task_id,
            "context_id": context_id,
            "peer": pending["peer"],
            "started": pending["started"],
            "created_iso": pending["created_iso"],
        }
        state3, reply3 = adapter._finalize_task(pending2, protocol.STATE_COMPLETED, "agent reply text")
        assert state3 == protocol.STATE_COMPLETED
        assert reply3 == "agent reply text"
        assert len(push_calls) == 1, "second duplicate must still not resend"

        # Verify ledger still authoritative and unchanged
        data3 = json.loads(ledger.read_text(encoding="utf-8"))
        assert data3[task_id].get("state") == protocol.STATE_COMPLETED
        assert data3[task_id].get("reply") == "agent reply text"
        assert data3[task_id].get("push_url") == hook_url

        # Duplicate with different reply must still be terminal conflict (not idempotent)
        with pytest.raises(protocol.DurablePublishError) as exc:
            adapter._finalize_task(pending2, protocol.STATE_COMPLETED, "different reply")
        assert "terminal conflict" in str(exc.value).lower() or exc.value.durable_state == protocol.STATE_COMPLETED
        assert len(push_calls) == 1, "rejected duplicate must not trigger push"

        # A new push_url on duplicate must not become authoritative nor trigger push
        # Build a candidate that tries to set a different callback; idempotence should still succeed
        # but not overwrite disk and not push.
        # We simulate by directly calling publish_durable with a different push_url and same state/reply
        # via the store — it should be treated as idempotent (state+reply match) and return disk record.
        fresh = protocol.TaskStore()
        fresh.restore(ledger)
        rec_disk = fresh.get(task_id)
        assert rec_disk is not None
        cand_try_new_url = dict(rec_disk)
        cand_try_new_url["push_url"] = "http://evil.example/cb"
        cand_try_new_url["push_config_id"] = "cfg-evil000000"
        outcome = fresh.publish_durable(ledger, task_id, cand_try_new_url)
        assert outcome.published and not outcome.newly_published, "different push_url with same terminal state/reply must be idempotent"
        assert outcome.record.get("push_url") == hook_url, "disk push_url must not be overwritten by duplicate"
        data4 = json.loads(ledger.read_text(encoding="utf-8"))
        assert data4[task_id].get("push_url") == hook_url

    finally:
        pass
