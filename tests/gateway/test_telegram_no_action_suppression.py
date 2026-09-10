from types import SimpleNamespace

import pytest

from gateway.config import Platform
from plugins.telegram_no_action_suppression import _on_pre_gateway_dispatch


def _event(text: str, *, chat_type: str = "group", platform=Platform.TELEGRAM):
    return SimpleNamespace(
        text=text,
        source=SimpleNamespace(
            platform=platform,
            chat_id="chat-1",
            chat_name="TCC Social Media",
            chat_type=chat_type,
            user_id="user-1",
            user_name="tester",
            thread_id=None,
            is_bot=False,
        ),
    )


def test_casual_telegram_group_message_is_skipped():
    result = _on_pre_gateway_dispatch(
        _event("Christine - This has been set up but the agents are not built out enough to use yet.")
    )

    assert result == {
        "action": "skip",
        "reason": "casual/no-action Telegram group message",
    }


def test_fgd_123_exact_test_1_message_is_skipped():
    assert _on_pre_gateway_dispatch(_event("FGD-123 Test 1")) == {
        "action": "skip",
        "reason": "casual/no-action Telegram group message",
    }


def test_fgd_123_test_1_casual_no_action_message_is_skipped():
    assert _on_pre_gateway_dispatch(_event("FGD-123 Test 1 - casual/no-action message")) == {
        "action": "skip",
        "reason": "casual/no-action Telegram group message",
    }


def test_fgd_124_zero_output_qa_message_is_skipped():
    assert _on_pre_gateway_dispatch(_event("Just testing whether this casual message stays quiet.")) == {
        "action": "skip",
        "reason": "casual/no-action Telegram group message",
    }


def test_forum_casual_no_action_message_is_skipped():
    assert _on_pre_gateway_dispatch(_event("ok thanks", chat_type="forum")) == {
        "action": "skip",
        "reason": "casual/no-action Telegram group message",
    }


def test_explicit_jimmy_header_is_allowed():
    assert _on_pre_gateway_dispatch(_event("JIMMY: please review this")) is None


@pytest.mark.parametrize(
    "trigger",
    [
        "JIMMY:",
        "REQUEST:",
        "ACTION:",
        "REVIEW:",
        "APPROVAL:",
        "BLOCKER:",
        "ROUTE:",
        "CODEX READ-ONLY:",
        "TIMMY:",
        "BEBE:",
        "DECISION:",
        "GATE:",
        "ESCALATION:",
        "CLOSEOUT:",
        "QUESTION FOR JIMMY:",
    ],
)
def test_explicit_triggers_are_allowed(trigger):
    assert _on_pre_gateway_dispatch(_event(f"{trigger} please review this")) is None


def test_question_for_jimmy_header_is_allowed():
    assert _on_pre_gateway_dispatch(_event("QUESTION FOR JIMMY: what is next?")) is None


def test_direct_natural_language_jimmy_reference_is_allowed():
    assert _on_pre_gateway_dispatch(_event("Jimmy can you check this?")) is None


def test_dm_is_allowed():
    assert _on_pre_gateway_dispatch(_event("hello", chat_type="dm")) is None


def test_dev_approval_message_is_allowed():
    assert _on_pre_gateway_dispatch(_event("Codex read-only approval request for FGD-127")) is None


def test_actionable_qa_gate_message_is_allowed():
    assert _on_pre_gateway_dispatch(_event("QA gate: run FGD-123 live test")) is None


def test_blocker_message_is_allowed():
    assert _on_pre_gateway_dispatch(_event("BLOCKER: branch state is unclear")) is None


def test_correction_message_is_allowed():
    assert _on_pre_gateway_dispatch(_event("Correction required before closeout")) is None


def test_non_telegram_group_message_is_allowed():
    assert _on_pre_gateway_dispatch(_event("casual message", platform=Platform.DISCORD)) is None
