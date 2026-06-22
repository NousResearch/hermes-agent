"""Regression test for cross-channel Slack misroute (2026-06-19).

Incident timeline (UTC):
    11:20:58  Inbound from C0AH3RY3DK6 (WorldArchitect channel).
    11:22:27  Orphan reply `1781868147.039389` posted to C0AJQ5M0A0Y
              (home channel) — 5 seconds BEFORE the correct reply.
    11:22:32  Correct reply posted to C0AH3RY3DK6.

Root cause class: the gateway's outbound path sent to a chat_id that
was NOT derived from the inbound that triggered the response.

This test pins down the new OutboundGuard behavior so that:
  1. A chat_id pinned by `enter()` survives verify_send() calls.
  2. A verify_send with a DIFFERENT chat_id is recorded as a violation.
  3. A verify_send with the SAME chat_id passes.
  4. enter()/reset() round-trip restores the previous pinned chat_id.
  5. The contextvar is task-local — concurrent asyncio tasks do not
     contaminate each other (proven with a shared guard instance).
  6. allowed_extra_destinations opts a destination out of the check.
  7. When no chat_id is pinned (non-handler-triggered send), verify_send
     passes regardless of the destination.
  8. A misroute simulation matching the exact incident pattern
     (pin A, send to B, send to A) records exactly one violation.
  9. chat_id=None while inbound is pinned is treated as a violation
     (no silent bypass through the guard).
 10. Module-level singleton pin_inbound/unpin_inbound/verify_outbound
     helpers propagate the same pinned value across the process.
 11. A real SlackAdapter.send call with an aligned chat_id succeeds
     without recording any violations; a misaligned one fails fast.
 12. pin_inbound(None) marks the handler as active and refuses every
     subsequent verify_send (handler is alive but its inbound chat_id
     is unknown — we cannot verify alignment with no destination).
 13. The same sentinel semantics propagate through the module-level
     singleton (pin_inbound(None) → verify_outbound(... ) refused).
 14. `active_chat_id` returns None for both "no handler" and "handler
     pinned None" — the two are distinguished only via `is_active`.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

# Add the repo root to sys.path so `gateway.outbound_guard` is importable
# from any working directory.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from gateway.outbound_guard import (  # noqa: E402
    OutboundGuard,
    get_global_guard,
    pin_inbound,
    unpin_inbound,
    verify_outbound,
)


C_WORLDARCH = "C0AH3RY3DK6"
C_HOME = "C0AJQ5M0A0Y"


def test_enter_then_verify_same_chat_id_passes():
    guard = OutboundGuard()
    token = guard.enter(C_WORLDARCH)
    try:
        assert guard.active_chat_id == C_WORLDARCH
        assert guard.verify_send(C_WORLDARCH) is True
        assert guard.violation_count == 0
    finally:
        guard.reset(token)


def test_verify_send_to_wrong_chat_id_records_violation():
    guard = OutboundGuard()
    token = guard.enter(C_WORLDARCH)
    try:
        # The exact incident pattern: pinned C_WORLDARCH, sent to C_HOME.
        assert guard.verify_send(C_HOME, operation="chat.postMessage") is False
        assert len(guard.violations) == 1
        v = guard.violations[0]
        assert v["active_inbound_chat_id"] == C_WORLDARCH
        assert v["outbound_chat_id"] == C_HOME
        assert v["operation"] == "chat.postMessage"
    finally:
        guard.reset(token)


def test_reset_restores_previous_pinned_chat_id():
    guard = OutboundGuard()
    token_outer = guard.enter(C_WORLDARCH)
    token_inner = guard.enter(C_HOME)
    try:
        assert guard.active_chat_id == C_HOME
    finally:
        guard.reset(token_inner)
    assert guard.active_chat_id == C_WORLDARCH
    guard.reset(token_outer)
    assert guard.active_chat_id is None


def test_no_pinned_chat_id_means_unrestricted_send():
    """Startup notifications and shutdown pings happen before/after any
    handler is active — they must NOT be blocked."""
    guard = OutboundGuard()
    # No enter() call — no pinned chat_id.
    assert guard.active_chat_id is None
    assert guard.verify_send(C_HOME) is True
    assert guard.verify_send(C_WORLDARCH) is True
    assert guard.violation_count == 0


def test_allowed_extra_destinations_opt_out():
    """Call sites like the home-channel startup notification may send
    to a destination different from the active inbound. They opt out
    by passing `allowed_extra_destinations`."""
    guard = OutboundGuard()
    token = guard.enter(C_WORLDARCH)
    try:
        # Without opt-out: violation.
        assert guard.verify_send(C_HOME) is False
        assert len(guard.violations) == 1

        # With opt-out: passes.
        assert (
            guard.verify_send(
                C_HOME, allowed_extra_destinations=[C_HOME]
            )
            is True
        )
        # Still exactly the one violation from the un-opted send.
        assert len(guard.violations) == 1
    finally:
        guard.reset(token)


def test_clear_violations_resets_list():
    guard = OutboundGuard()
    token = guard.enter(C_WORLDARCH)
    try:
        guard.verify_send(C_HOME)
        guard.verify_send(C_HOME)
        assert guard.violation_count == 2
        guard.clear_violations()
        assert guard.violation_count == 0
    finally:
        guard.reset(token)


def test_incident_repro_exact_pattern():
    """Pin A, send to B, send to A — exactly one violation recorded."""
    guard = OutboundGuard()
    token = guard.enter(C_WORLDARCH)
    try:
        # Misroute (the orphan): pinned A but sent to B.
        assert guard.verify_send(C_HOME) is False
        # Correct: pinned A and sent to A.
        assert guard.verify_send(C_WORLDARCH) is True

        assert len(guard.violations) == 1
        v = guard.violations[0]
        assert v["active_inbound_chat_id"] == C_WORLDARCH
        assert v["outbound_chat_id"] == C_HOME
    finally:
        guard.reset(token)


@pytest.mark.asyncio
async def test_contextvar_is_task_local():
    """Two concurrent tasks each pin a different chat_id via a SHARED
    guard — neither sees the other's pinned value. This proves the
    ContextVar provides task-local isolation at the *contextvar*
    layer (not just at the instance layer), which is what production
    relies on for stream-consumer tasks spawned by the handler."""

    guard = OutboundGuard()  # Shared instance across both tasks

    async def pin_and_check(chat_id: str, expected_other: str, results: dict):
        token = guard.enter(chat_id)
        try:
            # Yield to let the other task run its enter().
            await asyncio.sleep(0.01)
            # Our pinned chat_id must still be ours, not the other's.
            results[chat_id] = guard.active_chat_id
            # Verify_send against our own chat_id passes.
            results[f"{chat_id}_own_pass"] = guard.verify_send(chat_id)
            # Verify_send against the other chat_id fails (and we record it).
            results[f"{chat_id}_other_fail"] = guard.verify_send(expected_other)
        finally:
            guard.reset(token)

    results: dict = {}
    await asyncio.gather(
        pin_and_check(C_WORLDARCH, C_HOME, results),
        pin_and_check(C_HOME, C_WORLDARCH, results),
    )

    assert results[C_WORLDARCH] == C_WORLDARCH
    assert results[C_HOME] == C_HOME
    assert results[f"{C_WORLDARCH}_own_pass"] is True
    assert results[f"{C_HOME}_own_pass"] is True
    assert results[f"{C_WORLDARCH}_other_fail"] is False
    assert results[f"{C_HOME}_other_fail"] is False


def test_none_chat_id_while_inbound_pinned_records_violation():
    """Regression for the codex-connector P1 + coderabbit Major finding.

    When an inbound is pinned and `verify_send(None)` is called, this
    MUST be recorded as a violation — not silently bypassed. The
    incident class includes sends with no destination at all because
    upstream code forgot to thread `source.chat_id` through; the
    guard has to refuse, not allow.
    """
    guard = OutboundGuard()
    token = guard.enter(C_WORLDARCH)
    try:
        assert guard.verify_send(None) is False
        assert len(guard.violations) == 1
        v = guard.violations[0]
        assert v["active_inbound_chat_id"] == C_WORLDARCH
        assert v["outbound_chat_id"] is None
        assert v["reason"] == "chat_id is None while inbound is pinned"
    finally:
        guard.reset(token)


def test_module_level_singleton_pins_and_verifies():
    """`pin_inbound` / `verify_outbound` operate on a process-wide
    singleton so production call sites that don't hold an explicit
    guard reference still see the handler's pinned chat_id."""
    guard = get_global_guard()
    guard.clear_violations()
    token = pin_inbound(C_WORLDARCH)
    try:
        # verify_outbound reads the same pin as guard.active_chat_id.
        assert guard.active_chat_id == C_WORLDARCH
        assert verify_outbound(C_WORLDARCH) is True
        assert verify_outbound(C_HOME) is False
        # The violation was recorded on the same singleton instance.
        assert len(guard.violations) >= 1
    finally:
        unpin_inbound(token)
    # After unpinning, verify_outbound is unrestricted again.
    assert verify_outbound(C_HOME) is True


def test_pin_none_marks_handler_active_and_refuses_sends():
    """Regression for the second CodeRabbit Major on this PR.

    When a handler calls `enter(None)` (or `pin_inbound(None)`), it
    signals: "this handler is active but its inbound chat_id is
    unknown/unparseable". The guard must:
      * `is_active` becomes True
      * `active_chat_id` returns None
      * ANY subsequent verify_send — regardless of chat_id, including
        None — records a violation. Sends through a handler that has
        no idea what channel to target are unverifiable and must be
        refused.

    Without the `_NO_ACTIVE_INBOUND` sentinel the guard would
    conflate this state with "no handler is running" and let the send
    through silently — which is exactly the misroute incident class.
    """
    guard = OutboundGuard()
    # Before any pin: inactive.
    assert guard.is_active is False
    assert guard.active_chat_id is None

    token = guard.enter(None)
    try:
        # Handler is active, but inbound chat_id is missing.
        assert guard.is_active is True
        assert guard.active_chat_id is None

        # ANY verify_send under this pin must refuse — there is no
        # destination that could be verified against the missing inbound.
        assert guard.verify_send(C_HOME) is False
        assert guard.verify_send(C_WORLDARCH) is False
        assert guard.verify_send(None) is False

        # All three were recorded as violations.
        assert len(guard.violations) == 3
        reasons = {v["reason"] for v in guard.violations}
        assert "handler is active but inbound chat_id is missing" in reasons
        for v in guard.violations:
            assert v["active_inbound_chat_id"] is None
    finally:
        guard.reset(token)

    # After reset, inactive again — verify_send is unrestricted.
    assert guard.is_active is False
    assert guard.active_chat_id is None
    assert guard.verify_send(C_HOME) is True


def test_pin_none_also_blocks_via_module_singleton():
    """Same sentinel semantics must hold for the module-level
    singleton used by production call sites (SlackAdapter, delivery,
    stream_consumer)."""
    guard = get_global_guard()
    guard.clear_violations()
    token = pin_inbound(None)
    try:
        assert guard.is_active is True
        assert guard.active_chat_id is None
        # Misroute attempt via singleton: refused.
        assert verify_outbound(C_HOME) is False
        assert verify_outbound(C_WORLDARCH) is False
        assert verify_outbound(None) is False
        assert len(guard.violations) == 3
    finally:
        unpin_inbound(token)
    # After unpin: no handler active, all sends allowed.
    assert guard.is_active is False
    assert verify_outbound(C_HOME) is True


def test_active_chat_id_distinguishes_inactive_from_pin_none():
    """`active_chat_id` returns None in BOTH "no handler" and "handler
    pinned None". The two states are only distinguishable via
    `is_active`. This is the public API contract — callers that care
    about the difference must use `is_active`."""
    guard = OutboundGuard()
    assert guard.is_active is False
    assert guard.active_chat_id is None

    token = guard.enter(None)
    try:
        assert guard.is_active is True
        assert guard.active_chat_id is None  # same value, different state
    finally:
        guard.reset(token)

    assert guard.is_active is False
    assert guard.active_chat_id is None


def test_per_instance_contextvar_is_not_shared_between_guards():
    """Each OutboundGuard instance creates its own ContextVar in
    __post_init__ (regression for coderabbit Major on the dataclass
    mutable-default smell). Pinning one guard must not leak into
    another guard's `active_chat_id`.
    """
    g1 = OutboundGuard()
    g2 = OutboundGuard()
    token = g1.enter(C_WORLDARCH)
    try:
        # g2 was never entered, so its active_chat_id is None even
        # though g1 has a pin in the current task's context.
        assert g2.active_chat_id is None
    finally:
        g1.reset(token)


def _make_slack_adapter_for_guard_test():
    """Construct a SlackAdapter instance with the minimum attributes
    the `send` method touches, so the OutboundGuard wiring can be
    exercised end-to-end without spinning up a real Slack client."""
    from gateway.platforms.slack import SlackAdapter

    adapter = SlackAdapter.__new__(SlackAdapter)
    adapter._app = object()  # truthy, so the early-return path is skipped
    adapter._get_client = MagicMock(return_value=MagicMock(
        chat_postMessage=AsyncMock(return_value={"ts": "12345.6789"}),
    ))
    adapter._resolve_thread_ts = MagicMock(return_value=None)
    adapter._pop_slash_context = MagicMock(return_value=None)
    adapter.format_message = MagicMock(side_effect=lambda x: x)
    adapter.truncate_message = MagicMock(side_effect=lambda x, _limit: [x])
    adapter.MAX_MESSAGE_LENGTH = 40000
    adapter._bot_message_ts = set()
    adapter._BOT_TS_MAX = 5000
    adapter.config = SimpleNamespace(extra={})
    adapter.stop_typing = AsyncMock(return_value=None)
    return adapter


def test_real_slack_send_with_aligned_chat_id_succeeds():
    """End-to-end: SlackAdapter.send with the SAME chat_id as the
    pinned inbound posts through verify_outbound without recording a
    violation, and the underlying chat_postMessage is invoked."""
    import asyncio

    async def _run():
        adapter = _make_slack_adapter_for_guard_test()
        mock_client = adapter._get_client.return_value

        token = pin_inbound(C_WORLDARCH)
        try:
            result = await adapter.send(
                chat_id=C_WORLDARCH, content="hello world"
            )
        finally:
            unpin_inbound(token)

        assert result.success is True
        assert result.message_id == "12345.6789"
        assert mock_client.chat_postMessage.await_count == 1
        kwargs = mock_client.chat_postMessage.await_args.kwargs
        assert kwargs["channel"] == C_WORLDARCH
        assert kwargs["text"] == "hello world"

    asyncio.run(_run())


def test_real_slack_send_with_misaligned_chat_id_is_refused():
    """End-to-end: SlackAdapter.send with a DIFFERENT chat_id than
    the pinned inbound is REFUSED before reaching chat_postMessage,
    and the returned SendResult signals the failure."""
    import asyncio

    async def _run():
        adapter = _make_slack_adapter_for_guard_test()
        mock_client = adapter._get_client.return_value

        token = pin_inbound(C_WORLDARCH)
        try:
            result = await adapter.send(
                chat_id=C_HOME, content="wrong channel"
            )
        finally:
            unpin_inbound(token)

        assert result.success is False
        assert "OutboundGuard" in (result.error or "")
        # The underlying Slack client was never invoked.
        assert mock_client.chat_postMessage.await_count == 0

    asyncio.run(_run())


def test_violations_list_is_bounded(monkeypatch):
    """Regression: the violations retention must be bounded so a
    misbehaving handler cannot grow the deque without limit and exhaust
    process memory. Verify that after >MAX_VIOLATION_HISTORY violations,
    only the most recent MAX_VIOLATION_HISTORY entries are retained.
    """
    from gateway.outbound_guard import MAX_VIOLATION_HISTORY, OutboundGuard

    # Use a smaller cap so the test stays fast and is deterministic.
    monkeypatch.setattr(
        "gateway.outbound_guard.MAX_VIOLATION_HISTORY", 16
    )

    guard = OutboundGuard()
    # Pin a known inbound chat_id so every verify_send mismatches.
    token = guard.enter(C_WORLDARCH)
    try:
        # Generate MAX_VIOLATION_HISTORY + 50 violations.
        total = 16 + 50
        for _ in range(total):
            assert guard.verify_send(C_HOME) is False
        # Bounded retention: only the most recent MAX entries survive.
        assert guard.violation_count == 16
        # Every retained violation is a misalignment (the most recent class).
        assert all(v["outbound_chat_id"] == C_HOME for v in guard.violations)
    finally:
        guard.reset(token)


def test_violations_clear_works_with_bounded_deque():
    """Regression: clear_violations() must still work when violations is
    a bounded deque (the historical API used list.clear()).
    """
    from gateway.outbound_guard import OutboundGuard

    guard = OutboundGuard()
    token = guard.enter(C_WORLDARCH)
    try:
        for _ in range(5):
            assert guard.verify_send(C_HOME) is False
        assert guard.violation_count == 5
        guard.clear_violations()
        assert guard.violation_count == 0
    finally:
        guard.reset(token)
