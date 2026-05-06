from __future__ import annotations

import asyncio

from acp.schema import AllowedOutcome

from acp_adapter.permissions import make_approval_callback


class _Response:
    def __init__(self, option_id: str) -> None:
        self.outcome = AllowedOutcome(optionId=option_id, outcome="selected")


def test_approval_callback_accepts_allow_permanent_false(monkeypatch) -> None:
    seen_options = []

    class _Future:
        def result(self, timeout):
            return _Response("allow_once")

    def request_permission_fn(**kwargs):
        nonlocal seen_options
        seen_options = kwargs["options"]
        return object()

    monkeypatch.setattr(asyncio, "run_coroutine_threadsafe", lambda coro, loop: _Future())

    callback = make_approval_callback(request_permission_fn, object(), "session")

    assert callback("terminal: rm file", "dangerous command", allow_permanent=False) == "once"
    assert [option.option_id for option in seen_options] == ["allow_once", "deny"]
