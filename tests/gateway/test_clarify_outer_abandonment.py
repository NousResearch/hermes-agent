"""An outer timeout keeps gateway cleanup, but cannot become a plugin answer."""
from concurrent.futures import Future
from unittest.mock import patch

import pytest

from tools import clarify_gateway as cm
from tests.gateway.test_clarify_card_retire_on_timeout import (
    _CardAdapter, _THREE_QUESTIONS, _run_clarify,
)


def test_abandoned_nested_clarify_retires_card_without_rearming_or_next_question():
    scope = cm.ClarifyWaitScope()

    class Adapter(_CardAdapter):
        async def send_clarify(self, **kwargs):
            response = await super().send_clarify(**kwargs)
            scope.cancel()  # Outer deadline, after registration and send, before waiting.
            return response

    adapter = Adapter()
    try:
        with cm.bind_wait_scope(scope), pytest.raises(cm.ClarifyWaitAbandoned):
            _run_clarify(adapter, questions=_THREE_QUESTIONS, via_tool=True)
        assert len(adapter.asked) == len(adapter.retired) == 1
        assert adapter.resumed == 0
        assert not cm.has_pending("sk1")
    finally:
        cm.clear_session("sk1")


def test_abandoned_wait_disarms_late_delivery_fallback(monkeypatch):
    import gateway.run as gateway_run
    from gateway.run_turn_runner_clarify_delivery import _clarify_send_then_wait

    scope = cm.ClarifyWaitScope()
    future = Future()
    monkeypatch.setattr(gateway_run, "_approval_send_outcome", lambda *a, **kw: "ambiguous")
    with cm.bind_wait_scope(scope):
        entry = cm.register("outer-cancel", "outer-cancel-session", "Continue?", None)
        original_wait = entry.event.wait

        def wait(timeout):
            scope.cancel()
            return original_wait(timeout)

        monkeypatch.setattr(entry.event, "wait", wait)
        response, answered = _clarify_send_then_wait(
            future, clarify_id=entry.clarify_id, session_key=entry.session_key, clarify_mod=cm,
            fallback=lambda: pytest.fail("abandoned prompt tried a late fallback"),
        )
    assert response == cm.CANCELLED and not answered
    assert not cm.has_pending(entry.session_key)
    monkeypatch.setattr(gateway_run, "_approval_send_outcome", lambda *a, **kw: "failed")
    future.set_result(None)


@pytest.mark.parametrize("outcome", ["failed", "declined", "ambiguous", "sent"])
def test_abandonment_during_send_ack_cannot_fallback_or_clear_successor(monkeypatch, outcome):
    import gateway.run as gateway_run
    from gateway.run_turn_runner_clarify_delivery import _clarify_send_then_wait

    scope = cm.ClarifyWaitScope()
    def disposition(*args, **kwargs):
        scope.cancel()
        cm.register("successor", "ack-session", "New turn?", None)
        return outcome

    monkeypatch.setattr(gateway_run, "_approval_send_outcome", disposition)
    try:
        with cm.bind_wait_scope(scope):
            cm.register("abandoned", "ack-session", "Old turn?", None)
            # The gateway callback is on another context, as in the real gateway.
            def external_disposition(*args, **kwargs):
                with cm.bind_wait_scope(None):
                    return disposition(*args, **kwargs)
            monkeypatch.setattr(gateway_run, "_approval_send_outcome", external_disposition)
            reply = _clarify_send_then_wait(
                None, clarify_id="abandoned", session_key="ack-session", clarify_mod=cm,
                fallback=lambda: pytest.fail("abandoned send retried as text"),
            )
        assert reply == (cm.CANCELLED, False)
        assert cm.resolve_gateway_clarify("successor", "still active")
    finally:
        cm.clear_session("ack-session")


@pytest.mark.parametrize("outcome", ["failed", "declined"])
def test_late_watch_rechecks_ownership_after_disposition(monkeypatch, outcome):
    from gateway.run_turn_runner_clarify_delivery import _LateFailureWatch
    scope = cm.ClarifyWaitScope()
    with cm.bind_wait_scope(scope):
        cm.register("late-owned", "late-session", "Old?", None)
    cm.register("late-other", "late-session", "Other?", None)
    future = Future()
    watch = _LateFailureWatch(
        future, clarify_id="late-owned", session_key="late-session", clarify_mod=cm,
        fallback=lambda: pytest.fail("cancelled watch retried as text"),
    )
    def disposition(_future):
        scope.cancel()  # after _armed was read, before fallback/release
        return outcome
    monkeypatch.setattr(watch, "_outcome", disposition)
    try:
        # Invoke directly: Future logs callback exceptions rather than raising them.
        watch._on_card_done(future)
        assert cm.resolve_gateway_clarify("late-other", "still active")
    finally:
        watch.disarm()
        cm.clear_session("late-session")


def test_queued_text_fallback_does_not_start_after_abandonment():
    import asyncio
    from gateway.run_turn_runner_clarify_delivery import text_fallback_coro
    from tests.gateway.test_clarify_delivery_fallback import _CardAdapter

    async def unused_card():
        pytest.fail("native card should not be sent")
    adapter = _CardAdapter(unused_card)
    scope = cm.ClarifyWaitScope()
    with cm.bind_wait_scope(scope):
        cm.register("queued", "queued-session", "Old?", None)
    queued = text_fallback_coro(adapter, chat_id="42", clarify_id="queued", question="Old?", choices=None)
    scope.cancel()
    assert queued is not None
    result = asyncio.run(queued)
    assert result.success is False
    assert adapter.sent_text == []
    assert not cm.has_pending("queued-session")


@pytest.mark.parametrize("pending", [True, False])
def test_text_fallback_push_marker_and_pending_fence_compose(pending):
    """#132516's human-decision marker rides only on a fallback whose prompt is still pending."""
    import asyncio
    from gateway.run_turn_runner_clarify_delivery import text_fallback_coro
    from tests.gateway.test_clarify_delivery_fallback import _CardAdapter

    class Adapter(_CardAdapter):
        def __init__(self):
            super().__init__(None)
            self.sent_metadata = []

        async def send(self, chat_id, content, reply_to=None, metadata=None):
            self.sent_metadata.append(metadata)
            return await super().send(chat_id, content, reply_to=reply_to, metadata=metadata)

    adapter = Adapter()
    cm.register("marked", "marked-session", "Which env?", None)
    try:
        queued = text_fallback_coro(adapter, chat_id="42", clarify_id="marked", question="Which env?",
                                    choices=None, session_key="marked-session", metadata={"thread_id": "t1"})
        if not pending:
            assert cm.cancel_prompt("marked")  # settled while the fallback was queued
        assert asyncio.run(queued).success is pending
        assert adapter.sent_metadata == ([{"thread_id": "t1", "is_approval_prompt": True}] if pending else [])
    finally:
        cm.clear_session("marked-session")


@pytest.mark.parametrize("cancel_at", ["boundary", "flush", "queued"])
def test_initial_card_never_starts_after_outer_abandonment(monkeypatch, cancel_at):
    import asyncio
    from types import SimpleNamespace
    from gateway.run_turn_runner import TurnRunner
    from tools.clarify_tool import clarify_tool
    scope = cm.ClarifyWaitScope()
    adapter = _CardAdapter()
    runner = object.__new__(TurnRunner)
    def cancel():
        scope.cancel()
        with cm.bind_wait_scope(None):
            cm.register("initial-successor", "initial-session", "New?", None)
    def flush(**kwargs):
        if cancel_at == "flush":
            cancel()
    runner._ctx = SimpleNamespace(
        _status_adapter=adapter, _status_chat_id="42", _status_thread_metadata={},
        session_key="initial-session", stream_consumer_holder=[SimpleNamespace(flush_pending_sync=flush)])
    runner._close_native_stream_boundary = lambda *a, **kw: cancel() if cancel_at == "boundary" else None
    def schedule(coro, label):
        if cancel_at == "queued" and label == "Clarify send failed to schedule":
            cancel()
        result = asyncio.run(coro)
        return SimpleNamespace(result=lambda timeout=None: result)
    runner._schedule = schedule
    events = []

    def fake_invoke_hook(name, **kwargs):
        if name.startswith("on_human_input_"):
            events.append((name, kwargs.get("outcome")))
        return []
    try:
        with cm.bind_wait_scope(scope), pytest.raises(cm.ClarifyWaitAbandoned), \
                patch("hermes_cli.plugins.invoke_hook", side_effect=fake_invoke_hook):
            clarify_tool([{"question": "Old?"}], callback=runner._clarify_callback_sync)
        assert adapter.asked == []
        # The withdrawn prompt resolves its observers once, as the gateway's cancellation.
        assert events == [("on_human_input_request", None), ("on_human_input_resolved", "cancelled")]
        assert cm.resolve_gateway_clarify("initial-successor", "still active")
    finally:
        cm.clear_session("initial-session")
