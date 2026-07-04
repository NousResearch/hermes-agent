from __future__ import annotations

from types import SimpleNamespace

import pytest

from gateway.config import GatewayConfig, SessionResetPolicy, Platform
from gateway.run import GatewayRunner
from gateway.session import SessionSource


class FakeTelegramAdapter:
    def __init__(self):
        self.registered = None
        self.sent = []

    def register_post_delivery_callback(self, session_key, callback, *, generation=None):
        self.registered = (session_key, callback, generation)

    async def send_new_session_button(self, chat_id, *, text, button_label, metadata=None):
        self.sent.append(
            {
                "chat_id": chat_id,
                "text": text,
                "button_label": button_label,
                "metadata": metadata,
            }
        )


def _runner_with(config: GatewayConfig, adapter: FakeTelegramAdapter | None = None):
    runner = GatewayRunner.__new__(GatewayRunner)
    setattr(runner, "session_store", SimpleNamespace(config=config))
    setattr(runner, "adapters", {Platform.TELEGRAM: adapter or FakeTelegramAdapter()})
    return runner


@pytest.mark.asyncio
async def test_completed_telegram_turn_registers_post_delivery_new_button():
    adapter = FakeTelegramAdapter()
    runner = _runner_with(
        GatewayConfig(
            default_reset_policy=SessionResetPolicy(
                post_task_new_button=True,
                post_task_new_button_text="Готово. Новая тема?",
                post_task_new_button_label="New",
            )
        ),
        adapter,
    )

    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="-1001",
        chat_type="forum",
        thread_id="2",
        user_id="42",
    )

    runner._register_post_task_new_button(
        source=source,
        session_key="telegram:-1001:2",
        run_generation=7,
        agent_result={"final_response": "Done", "completed": True},
    )

    assert adapter.registered is not None
    session_key, callback, generation = adapter.registered
    assert session_key == "telegram:-1001:2"
    assert generation == 7

    await callback()

    assert adapter.sent == [
        {
            "chat_id": "-1001",
            "text": "Готово. Новая тема?",
            "button_label": "New",
            "metadata": {"thread_id": "2"},
        }
    ]


@pytest.mark.asyncio
async def test_streamed_turn_sends_post_task_new_button_immediately():
    adapter = FakeTelegramAdapter()
    runner = _runner_with(
        GatewayConfig(
            default_reset_policy=SessionResetPolicy(post_task_new_button=True)
        ),
        adapter,
    )

    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="-1001",
        chat_type="forum",
        thread_id="2",
        user_id="42",
    )

    sent = await runner._send_post_task_new_button_now(
        source=source,
        agent_result={"final_response": "Done", "completed": True, "already_sent": True},
        reply_to_message_id="99",
    )

    assert sent is True
    assert adapter.sent == [
        {
            "chat_id": "-1001",
            "text": "Готово. Новая тема?",
            "button_label": "New",
            "metadata": {"thread_id": "2"},
        }
    ]


def test_profile_always_fresh_policy_suppresses_post_task_new_button():
    runner = _runner_with(
        GatewayConfig(
            default_reset_policy=SessionResetPolicy(post_task_new_button=True),
            reset_by_profile={
                "admin": SessionResetPolicy(
                    mode="always",
                    notify=False,
                    post_task_new_button=False,
                )
            },
        )
    )

    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="-1001",
        chat_type="forum",
        thread_id="1",
        profile="admin",
    )

    assert runner._post_task_new_button_policy(
        source,
        {"final_response": "Done", "completed": True},
    ) is None
