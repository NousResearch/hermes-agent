"""``dispatch_shared_busy``'s observer (gateway/session_ingress.py) completes the follower's
review ownership with its own send, then settles the arrival's busy telemetry, each guaranteed:
the telemetry never stands between the turn and its review release, a completion that raises
never skips the telemetry, and neither skips the marker releases that wake the drain's bounded
wait. The observer is driven directly with a stand-in adapter; the handler parks the completion
on the event the way ``_handle_message_with_agent`` does.
"""

from __future__ import annotations

import types

import pytest

import gateway.session_ingress as ingress_module
from gateway.config import Platform
from gateway.platforms.event import MessageEvent
from gateway.session import SessionSource
from gateway.session_ingress import dispatch_shared_busy

SESSION_KEY = "session-key"


def _event():
    return MessageEvent(
        text="follow up",
        source=SessionSource(
            platform=Platform.TELEGRAM, chat_id="chat", chat_type="dm", user_id="user"
        ),
    )


def _adapter(order, handler):
    async def release(event):
        order.append("marker_released")

    async def notify(event, exc):
        order.append(("notified", type(exc).__name__))

    return types.SimpleNamespace(
        name="fixture",
        _message_handler=handler,
        _release_turn_marker=release,
        _notify_turn_error=notify,
        _background_tasks=set(),
        gateway_runner=None,
        apply_followup_queue_mutation=lambda session_key, mutation: mutation(),
    )


def _completion(order, *, raising=False):
    def complete(*, delivery_succeeded, cause=None):
        order.append(("review_completed", delivery_succeeded, cause))
        if raising:
            raise RuntimeError("completion exploded")

    return complete


@pytest.fixture
def order(monkeypatch):
    order = []

    async def deliver(adapter, event, session_key, response):
        order.append(("delivered", response))
        return True

    monkeypatch.setattr(ingress_module, "deliver_response", deliver)
    monkeypatch.setattr(
        ingress_module,
        "settle_busy_arrival",
        lambda event, **_kw: order.append("telemetry_settled"),
    )
    return order


@pytest.mark.asyncio
async def test_the_observer_completes_the_review_with_its_send_then_settles_telemetry(
    order,
):
    async def handler(event):
        event._gateway_review_delivery_complete = _completion(order)
        return "reply"

    await dispatch_shared_busy(_adapter(order, handler), _event(), SESSION_KEY)

    assert order == [
        ("delivered", "reply"),
        ("review_completed", True, None),
        "telemetry_settled",
        "marker_released",
        "marker_released",
    ]


@pytest.mark.asyncio
async def test_a_handler_failure_completes_the_review_unconfirmed_then_settles_telemetry(
    order,
):
    async def handler(event):
        event._gateway_review_delivery_complete = _completion(order)
        raise RuntimeError("handler exploded")

    with pytest.raises(RuntimeError, match="handler exploded"):
        await dispatch_shared_busy(_adapter(order, handler), _event(), SESSION_KEY)

    assert order == [
        ("review_completed", False, None),
        "telemetry_settled",
        "marker_released",
        "marker_released",
    ]


@pytest.mark.asyncio
async def test_a_completion_that_raises_skips_neither_the_telemetry_nor_the_marker_releases(
    order,
):
    async def handler(event):
        event._gateway_review_delivery_complete = _completion(order, raising=True)
        return "reply"

    await dispatch_shared_busy(_adapter(order, handler), _event(), SESSION_KEY)

    assert order == [
        ("delivered", "reply"),
        ("review_completed", True, None),
        "telemetry_settled",
        "marker_released",
        "marker_released",
    ]
