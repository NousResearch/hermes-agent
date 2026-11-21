"""Invariants for the exec-approval template method on BasePlatformAdapter.

The choice set (once / session / always / deny, minus the persistent tiers under a smart deny)
used to be re-derived in every button adapter and needed three separate "same fix × N adapters"
sweeps. It now comes from one place; adapters only render it.
"""

from typing import Any, Dict

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, ExecApprovalPrompt, SendResult
from gateway.run_turn_runner import _renders_exec_approval_buttons


class _Plain(BasePlatformAdapter):
    def __init__(self):
        super().__init__(PlatformConfig(enabled=True), Platform.TELEGRAM)

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        return True

    async def disconnect(self) -> None:
        pass

    async def send(self, *a: Any, **k: Any) -> SendResult:
        return SendResult(success=True)

    async def get_chat_info(self, chat_id: str) -> dict[str, Any]:
        return {}


class _Buttons(_Plain):
    def __init__(self):
        super().__init__()
        self.prompts: list = []

    async def _send_exec_approval_prompt(self, prompt: ExecApprovalPrompt) -> SendResult:
        self.prompts.append(prompt)
        return SendResult(success=True, message_id="m1")


@pytest.mark.parametrize(
    "allow_permanent, allow_session, smart_denied, expected",
    [
        (True, True, False, ["once", "session", "always", "deny"]),
        (False, True, False, ["once", "session", "deny"]),
        (True, False, False, ["once", "deny"]),          # no session tier → no permanent tier either
        (True, True, True, ["once", "deny"]),            # smart deny: owner override is one-shot
        (False, False, True, ["once", "deny"]),
    ],
)
@pytest.mark.asyncio
async def test_choice_set_follows_the_shared_rule(allow_permanent, allow_session, smart_denied, expected):
    adapter = _Buttons()
    result = await adapter.send_exec_approval(
        "chat", "rm -rf /tmp/x", "sess", description="cleanup",
        allow_permanent=allow_permanent, allow_session=allow_session, smart_denied=smart_denied)
    assert result.success
    (prompt,) = adapter.prompts
    assert prompt.choices == expected
    assert "rm -rf /tmp/x" in prompt.text and "cleanup" in prompt.text


@pytest.mark.asyncio
async def test_command_is_truncated_to_the_platform_budget():
    class _Small(_Buttons):
        _EA_CMD_BUDGET = 20

    adapter = _Small()
    await adapter.send_exec_approval("chat", "x" * 100, "sess")
    (prompt,) = adapter.prompts
    assert "x" * 10 + "..." in prompt.text and "x" * 11 not in prompt.text  # half: the rest is the notice
    assert "Preview shortened" in prompt.text
    assert prompt.command == "x" * 100  # the raw command stays available for embeds/cards


def test_runner_only_offers_buttons_to_adapters_that_render_them():
    """Plain adapters get the text ``/approve`` prompt; the base send_exec_approval must not make
    the runner believe every adapter has buttons (its default reports failure and the runner would
    log a spurious fallback on every approval)."""
    assert _renders_exec_approval_buttons(_Buttons) is True
    assert _renders_exec_approval_buttons(_Plain) is False

    class _DuckTyped:
        async def send_exec_approval(self, *a, **k):
            return SendResult(success=True)

    assert _renders_exec_approval_buttons(_DuckTyped) is True
    assert _renders_exec_approval_buttons(object) is False


_SCHEDULE = {"kind": "schedule", "command": "<cronjob_manage> (schedule a new job)",
             "description": 'Schedule "check-in" (every day at 09:00): remind me to check in', "request_id": "r1"}


def _notify(approval: dict, adapter) -> None:
    """Drive the real TurnRunner approval notify (card path, then text fallback) with ``adapter``."""
    import asyncio
    from types import SimpleNamespace
    from gateway.run_turn_runner import TurnRunner

    async def run():
        ctx = SimpleNamespace(_status_adapter=adapter, _status_chat_id="chat", _status_thread_metadata=None,
                              session_key="sess", _loop_for_step=asyncio.get_running_loop(),
                              stream_consumer_holder=[None])
        await asyncio.to_thread(TurnRunner(SimpleNamespace(), ctx)._approval_notify_sync, approval)
    asyncio.run(run())


def test_a_schedule_approval_shows_the_job_not_a_command(monkeypatch):
    """kind=schedule: the card and the /approve text ask about the job, with the job description as the
    body; the synthetic ``<cronjob_manage>`` label is never shown and nothing is "flagged"."""
    monkeypatch.setattr("gateway.platforms.base_exec_approval.approval_timeout_seconds", lambda: 300)
    card = _Buttons()
    _notify(dict(_SCHEDULE), card)
    (prompt,) = card.prompts
    sent = []

    class _Failing(_Buttons):  # the card send fails, so the runner sends the text prompt instead
        async def _send_exec_approval_prompt(self, prompt):
            return SendResult(success=False, error="down")

        async def send(self, chat_id, content, **k):
            sent.append(content)
            return SendResult(success=True)
    _notify(dict(_SCHEDULE), _Failing())
    for text in (prompt.text, sent[0]):
        assert "Hermes wants to schedule a job" in text and _SCHEDULE["description"] in text
        assert "<cronjob_manage>" not in text and "Why it was flagged" not in text
        assert f"```\n{_SCHEDULE['description']}\n```" in text  # model-written: fenced like a command
        assert "the job will NOT be created" in text
    assert [(label, choice) for label, choice, _ in prompt.actions] == [
        ("Allow", "once"), ("Allow for this chat", "session"), ("Always allow new jobs", "always"), ("Deny", "deny")]
    assert "Reply `/approve` to create it" in sent[0]


def test_an_edit_approval_asks_to_change_the_job_with_allow_and_deny_only(monkeypatch):
    """kind=schedule_edit: the card asks to change a job, shows the change list as the body, and an
    edit (each change its own consent) offers only Allow / Deny."""
    monkeypatch.setattr("gateway.platforms.base_exec_approval.approval_timeout_seconds", lambda: 300)
    card = _Buttons()
    edit = {**_SCHEDULE, "kind": "schedule_edit", "command": '<cronjob_manage> (change "check-in")',
            "description": 'Change "check-in": schedule: every day at 09:00 -> every day at 10:00',
            "allow_session": False, "allow_permanent": False}
    _notify(edit, card)
    (prompt,) = card.prompts
    assert "Hermes wants to change a scheduled job" in prompt.text and edit["description"] in prompt.text
    assert "the job will NOT be changed" in prompt.text and "<cronjob_manage>" not in prompt.text
    assert "Why it was flagged" not in prompt.text
    assert [choice for _label, choice, _ in prompt.actions] == ["once", "deny"]


def test_a_command_approval_renders_the_same_with_an_unknown_kind():
    """An approval without ``kind`` (every command) or with a kind this build does not know renders the
    command card; the schedule wording never leaks into it."""
    cards = []
    for kind in (None, "telepathy"):
        adapter = _Buttons()
        approval = {"command": "rm -rf /tmp/x", "description": "recursive delete", "request_id": "r1"}
        if kind:
            approval["kind"] = kind
        _notify(approval, adapter)
        (prompt,) = adapter.prompts
        cards.append((prompt.text, prompt.actions))
    assert cards[0] == cards[1]
    assert "```\nrm -rf /tmp/x\n```" in cards[0][0] and "schedule" not in cards[0][0]
    assert [label for label, _, _ in cards[0][1]] == ["Allow Once", "Allow Session", "Always Allow", "Deny"]


def test_a_schedule_cards_model_written_markup_stays_inside_the_fence(monkeypatch):
    """A job name/prompt is model-written; spoiler, link or small-print markup in it must render as text
    inside the fence, and a backtick run in it cannot close the fence early."""
    monkeypatch.setattr("gateway.platforms.base_exec_approval.approval_timeout_seconds", lambda: 300)
    plan = 'Schedule "x" (daily): ||hidden|| [ok](https://e.example) ``` \n-# small print'
    card = _Buttons()
    _notify({**_SCHEDULE, "description": plan}, card)
    (prompt,) = card.prompts
    fence = "````"
    assert f"{fence}\n{plan}\n{fence}" in prompt.text, prompt.text
    assert prompt.text.index(fence) < prompt.text.index("||hidden||")

