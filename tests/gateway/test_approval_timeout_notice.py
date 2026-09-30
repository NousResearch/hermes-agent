"""Invariants for the exec-approval timeout notice hook (gateway/run_turn_runner_approval_settle.py)."""

from types import SimpleNamespace

import pytest

from gateway import run_turn_runner_approval_settle as settle_mod


class _Runner:
    def __init__(self, *, current: bool):
        self.scheduled = []
        self._ctx = SimpleNamespace(session_key="telegram:1", _run_still_current=lambda: current)

    def _schedule(self, coro, label):
        coro.close()
        self.scheduled.append(label)
        return None


def _capture_settle(monkeypatch):
    hooks = {}
    monkeypatch.setattr("tools.approval.register_gateway_settle", lambda sk, rid, fn: hooks.__setitem__(rid, fn))
    return hooks


@pytest.mark.parametrize("current, expected", [(True, 1), (False, 0)])
def test_timeout_notice_is_gated_on_the_run_still_being_current(monkeypatch, current, expected):
    hooks = _capture_settle(monkeypatch)
    runner = _Runner(current=current)
    settle_mod.register_timeout_notice(runner, {"request_id": "r1"}, command="rm -rf x", card_message_id="42")
    hooks["r1"]("timeout")
    assert len(runner.scheduled) == expected


def test_non_timeout_reasons_never_post(monkeypatch):
    hooks = _capture_settle(monkeypatch)
    runner = _Runner(current=True)
    settle_mod.register_timeout_notice(runner, {"request_id": "r2"}, command="ls", card_message_id=None)
    for reason in ("answered", "interrupted", "notify_failed"):
        hooks["r2"](reason)
    assert runner.scheduled == []


@pytest.mark.asyncio
async def test_text_prompt_is_never_edited_in_place():
    """card_message_id=None (the text-fallback path) must send a new message, not rewrite the prompt."""
    calls = []

    class _Adapter:
        async def edit_message(self, chat_id, message_id, content):
            calls.append(("edit", message_id))
            return SimpleNamespace(success=True)

        async def send(self, chat_id, content, metadata=None):
            calls.append(("send", content))
            return SimpleNamespace(success=True)

    ctx = SimpleNamespace(_status_adapter=_Adapter(), _status_chat_id="c", _status_thread_metadata=None)
    await settle_mod._post_timeout_notice(ctx, "ls", None, 300)
    assert [kind for kind, _ in calls] == ["send"]
    await settle_mod._post_timeout_notice(ctx, "ls", "card-9", 300)
    assert ("edit", "card-9") in calls
