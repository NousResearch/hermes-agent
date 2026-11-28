"""Guest chats can't answer approval prompts, so their approvals fail at once."""

from types import SimpleNamespace

from gateway.run_turn_runner_approval import unanswerable_approval_reason
from tests.gateway.test_telegram_guest_reply import _make_adapter, _register_guest_chat


def test_guest_chat_reports_approvals_unanswerable():
    adapter = _make_adapter()
    _register_guest_chat(adapter, "42")

    reason = unanswerable_approval_reason(adapter, SimpleNamespace(chat_id="42", chat_type="group", user_id="1"))

    assert reason and "guest chat" in reason


def test_member_chat_keeps_prompting():
    adapter = _make_adapter()
    _register_guest_chat(adapter, "42")

    assert unanswerable_approval_reason(adapter, SimpleNamespace(chat_id="43", chat_type="group", user_id="1")) is None
