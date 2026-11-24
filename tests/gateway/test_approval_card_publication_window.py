"""A card tap can approve only a request whose card was rendered for it: an id-less tap (a card from an
older build) settles nothing, and a freshly queued request is Deny-only from the moment it is enqueued,
so no tap path can approve it in the window before its renderer has decided what the card may offer."""

import threading

SK = "agent:main:qqbot:dm:u-42"


def _publish_and_hold():
    """A real gateway wait whose notifier is held before any rendering: the request is published."""
    from tools import approval
    from tools.approval_gateway_wait import _await_gateway_decision

    entered, release, out = threading.Event(), threading.Event(), {}

    def notify(data):
        out["request_id"] = data["request_id"]
        entered.set()
        assert release.wait(10)

    def wait():
        out["decision"] = _await_gateway_decision(SK, notify, {"command": "rm -rf /tmp/x"})

    thread = threading.Thread(target=wait)
    thread.start()
    assert entered.wait(10)
    return approval, thread, release, out


def _finish(approval, thread, release, out):
    if approval.has_blocking_approval(SK):
        approval.resolve_gateway_approval(SK, "deny", request_id=out["request_id"])
    release.set()
    thread.join(12)
    assert not thread.is_alive()


def test_a_published_request_is_deny_only_before_its_card_is_rendered():
    approval, thread, release, out = _publish_and_hold()
    try:
        (pending,) = approval._gateway_queues[SK]
        assert pending.card_deny_only is True
        assert approval.resolve_gateway_approval_card(SK, "once", out["request_id"]) == 0
    finally:
        _finish(approval, thread, release, out)
