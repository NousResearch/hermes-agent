"""Delivery receipts must not stall the parent or suppress later receipts."""
from contextvars import copy_context
from threading import Event, Thread


def test_delivery_receipt_obeys_hook_timeout(monkeypatch):
    from agent.tool_result_context import acknowledge_tool_result_context
    from hermes_cli import plugins
    from hermes_constants import get_hermes_home

    monkeypatch.setattr(plugins, "_resolve_hook_callback_timeout", lambda: 0.05)
    entered, release, finished = Event(), Event(), Event()
    observed, receipts = [], []
    expected_home = get_hermes_home()

    def blocked(persisted):
        observed.append((persisted, get_hermes_home()))
        entered.set()
        release.wait()

    def deliver():
        try:
            acknowledge_tool_result_context([blocked, receipts.append], True)
        finally:
            finished.set()

    worker = Thread(target=copy_context().run, args=(deliver,), daemon=True)
    worker.start()
    try:
        assert entered.wait(2)
        assert finished.wait(2), "receipt blocked the parent beyond the configured hook timeout"
        assert observed == [(True, expected_home)]
        assert receipts == [True]
    finally:
        release.set()
        worker.join(2)


def test_failed_receipt_does_not_suppress_later_positional_callback():
    from agent.tool_result_context import acknowledge_tool_result_context

    receipts = []

    def broken(persisted):
        raise RuntimeError("receipt storage unavailable")

    acknowledge_tool_result_context([broken, receipts.append], False)
    assert receipts == [False]
