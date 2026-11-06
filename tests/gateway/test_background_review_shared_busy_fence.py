"""``dispatch_shared_busy`` (gateway/session_ingress.py) publishes an accepted durable follower to
review admission through the same tolerant path every runner-side slot write takes
(``publish_followup_queue_mutation``): an adapter that binds ``FollowupAdmissionMixin`` (every
production adapter) sees the follow-up epoch bump and its registered review cancel run exactly
once, at acceptance, never on a refusal; a minimal adapter without that contract (upstream's own
busy-dispatch doubles) keeps the exact queue behaviour instead of failing on a missing attribute.
The end-to-end probe over a real adapter, authority and drain is
``test_background_review_followup_shared_fifo.py``.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

import gateway.session_ingress as ingress
from gateway.config import Platform
from gateway.platforms.base_followup_admission import FollowupAdmissionMixin
from gateway.platforms.event import MessageEvent, SessionSource
from gateway.session_ingress import admit_message, dispatch_shared_busy

ROUTE = "route"


class _FencedAdapter(FollowupAdmissionMixin):
    """The fence contract and the busy-dispatch surface, nothing else of a platform adapter."""

    name = "fenced"
    gateway_runner = None

    def __init__(self, handler):
        self._message_handler = handler
        self._background_tasks = set()
        self._pending_messages = {}
        self._release_turn_marker = AsyncMock()
        self._notify_turn_error = AsyncMock()


def _queued_authority(admit_gate: asyncio.Event | None = None):
    async def admit(event):
        if admit_gate is not None:
            await admit_gate.wait()
        return SimpleNamespace(status="queued", admission_id="queued")

    return SimpleNamespace(
        admit_native=admit, native_waiters=set(), waiters={}, db=None
    )


def _event(text="next"):
    return MessageEvent(
        text=text, source=SessionSource(platform=Platform.TELEGRAM, chat_id="chat")
    )


@pytest.fixture
def queued_row(monkeypatch):
    delivery = AsyncMock(return_value=True)
    monkeypatch.setattr(ingress, "deliver_response", delivery)
    # admit_message re-reads the committed row; it is still queued behind the running turn.
    monkeypatch.setattr(
        "hermes_state_runtime.get_session_admission",
        lambda db, admission_id: {"status": "queued"},
    )
    return delivery


@pytest.mark.asyncio
async def test_an_adapter_without_the_fence_contract_still_queues_and_delivers(
    queued_row,
):
    """Upstream's busy-dispatch doubles: no ``apply_followup_queue_mutation`` at all."""
    authority = _queued_authority()
    adapter = SimpleNamespace(
        _background_tasks=set(),
        _message_handler=lambda event: admit_message(authority, event),
        _release_turn_marker=AsyncMock(),
    )

    await asyncio.wait_for(dispatch_shared_busy(adapter, _event(), ROUTE), 5)

    assert not authority.waiters["queued"].done()
    queued_row.assert_not_awaited()
    authority.waiters["queued"].set_result("completed answer")
    await asyncio.gather(*adapter._background_tasks)
    queued_row.assert_awaited_once()
    assert queued_row.await_args.args[-1] == "completed answer"


@pytest.mark.asyncio
async def test_an_accepted_follower_fences_the_live_turns_review_exactly_once(
    queued_row,
):
    authority = _queued_authority()
    adapter = _FencedAdapter(lambda event: admit_message(authority, event))
    cancels = []
    adapter.register_followup_review_cancel(ROUTE, lambda: cancels.append(ROUTE))
    assert adapter.followup_admission_state(ROUTE).epoch == 0

    await asyncio.wait_for(dispatch_shared_busy(adapter, _event(), ROUTE), 5)

    # Published at acceptance, while the follower is still queued behind the running turn.
    assert not authority.waiters["queued"].done()
    assert cancels == [ROUTE]
    assert adapter.followup_admission_state(ROUTE).epoch == 1

    authority.waiters["queued"].set_result("completed answer")
    await asyncio.gather(*adapter._background_tasks)
    queued_row.assert_awaited_once()
    # Its own run and send publish nothing more.
    assert cancels == [ROUTE]
    assert adapter.followup_admission_state(ROUTE).epoch == 1


@pytest.mark.asyncio
async def test_a_follower_refused_before_acceptance_is_never_published(queued_row):
    from hermes_state_runtime import RuntimeStoreError

    async def refused(event):
        raise RuntimeStoreError("permission_denied")

    adapter = _FencedAdapter(refused)
    cancels = []
    adapter.register_followup_review_cancel(ROUTE, lambda: cancels.append(ROUTE))

    with pytest.raises(RuntimeStoreError, match="permission_denied"):
        await dispatch_shared_busy(adapter, _event("refused"), ROUTE)

    assert not adapter._background_tasks
    assert cancels == []
    assert adapter.followup_admission_state(ROUTE).epoch == 0
    queued_row.assert_not_awaited()
