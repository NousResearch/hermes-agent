import asyncio
import threading
import time
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from gateway.platforms.base import BasePlatformAdapter
from gateway.wisdom_command import WisdomAction, WisdomItem, WisdomView
from tests.gateway.test_slack_wisdom import _adapter as slack_adapter
from tests.gateway.test_telegram_wisdom_command import _adapter as telegram_adapter
from tui_gateway.wisdom_mediation import poll


@pytest.mark.asyncio
async def test_idle_boundary_does_not_overlap_busy_turn_and_releases_guard():
    adapter = SimpleNamespace(
        _active_sessions={"busy": asyncio.Event()}, _session_tasks={}
    )
    adapter._drain_pending_after_session_command = AsyncMock(
        side_effect=lambda key, guard: adapter._active_sessions.pop(key)
    )
    work = AsyncMock()
    assert not await BasePlatformAdapter.run_idle_activity(adapter, "busy", work)
    work.assert_not_called()

    async def inside():
        assert "idle" in adapter._active_sessions
        assert adapter._session_tasks["idle"] is asyncio.current_task()

    assert await BasePlatformAdapter.run_idle_activity(adapter, "idle", inside)
    assert (
        "idle" not in adapter._active_sessions and "idle" not in adapter._session_tasks
    )


@pytest.mark.asyncio
async def test_idle_boundary_failure_releases_guard():
    adapter = SimpleNamespace(_active_sessions={}, _session_tasks={})
    adapter._drain_pending_after_session_command = AsyncMock(
        side_effect=lambda key, guard: adapter._active_sessions.pop(key)
    )
    with pytest.raises(RuntimeError):
        await BasePlatformAdapter.run_idle_activity(
            adapter, "session", AsyncMock(side_effect=RuntimeError)
        )
    assert not adapter._active_sessions and not adapter._session_tasks


def view():
    return WisdomView(
        "Collective Wisdom",
        "A recommendation",
        items=[
            WisdomItem(
                "<untrusted>",
                "Canonical warnings",
                actions=[
                    WisdomAction("Not Now", callback_data="wi:agent:defer:identity"),
                    WisdomAction(
                        "Review first", callback_data="wi:agent:inspect:identity"
                    ),
                    WisdomAction(
                        "Install",
                        callback_data="wi:agent:confirm:identity",
                        primary=True,
                    ),
                ],
            )
        ],
    )


@pytest.mark.asyncio
async def test_slack_proactive_advice_cannot_consume_slash_response():
    adapter = slack_adapter()
    adapter._pop_slash_context = Mock(
        side_effect=AssertionError("must not consume slash response")
    )
    await adapter.send_wisdom_mediation(
        view(), source=SimpleNamespace(chat_id="D1", scope_id="T1", thread_id="123")
    )
    sent = adapter._team_clients["T1"].chat_postMessage.call_args.kwargs
    assert sent["thread_ts"] == "123"
    actions = [block for block in sent["blocks"] if block["type"] == "actions"][0][
        "elements"
    ]
    assert [button["text"]["text"] for button in actions] == [
        "Not Now",
        "Review first",
        "Install",
    ]
    assert actions[-1]["style"] == "primary"


@pytest.mark.asyncio
async def test_telegram_native_rich_controls_escape_publisher_text():
    adapter = telegram_adapter()
    await adapter.send_wisdom_mediation(
        view(), source=SimpleNamespace(chat_id="42", thread_id=None)
    )
    sent = adapter._bot.do_api_request.call_args.kwargs["api_kwargs"]
    html = sent["rich_message"]["html"]
    assert "&lt;untrusted&gt;" in html
    assert html.index("Not Now") < html.index("Review first") < html.index("Install")


@pytest.mark.asyncio
async def test_telegram_ambiguous_send_never_falls_back_to_duplicate_message():
    adapter = telegram_adapter()
    adapter._bot.do_api_request.side_effect = TimeoutError
    adapter._send_message_with_thread_fallback = AsyncMock()
    with pytest.raises(TimeoutError):
        await adapter.send_wisdom_mediation(
            view(), source=SimpleNamespace(chat_id="42", thread_id=None)
        )
    adapter._send_message_with_thread_fallback.assert_not_called()


@pytest.mark.parametrize(
    "busy,connected,approval",
    [(True, True, False), (False, False, False), (False, True, True)],
)
def test_tui_defers_without_model_or_history_changes(
    monkeypatch, busy, connected, approval
):
    monkeypatch.setattr(
        "tools.approval.get_pending_gateway_approval", lambda _: approval
    )
    monkeypatch.setattr("tools.clarify_gateway.has_pending", lambda _: False)
    monkeypatch.setattr(
        "hermes_wisdom.service.WisdomService",
        Mock(side_effect=AssertionError("must not poll")),
    )
    session = {
        "session_key": "s",
        "_wisdom_user_activity": time.time(),
        "agent": object(),
        "history_lock": threading.Lock(),
        "running": busy,
        "history": [{"role": "user", "content": "hello"}],
    }
    emit = Mock()
    poll(
        session,
        emit=emit,
        profile_scope=lambda _: nullcontext(),
        connected=lambda: connected,
    )
    assert session["running"] is busy
    assert session["history"] == [{"role": "user", "content": "hello"}]
    emit.assert_not_called()


@pytest.mark.parametrize("cancel", [False, True])
def test_tui_assessment_does_not_overwrite_a_new_turn_after_stop(monkeypatch, cancel):
    monkeypatch.setattr("tools.approval.get_pending_gateway_approval", lambda _: False)
    monkeypatch.setattr("tools.clarify_gateway.has_pending", lambda _: False)
    service = Mock()
    service.store.active_org_id.return_value = "org"
    monkeypatch.setattr("hermes_wisdom.service.WisdomService", lambda: service)
    mediation = Mock()
    mediation.queue.claim_refresh.return_value = False
    monkeypatch.setattr(
        "tui_gateway.wisdom_mediation.WisdomMediation", lambda _: mediation
    )
    session = {
        "session_key": "s",
        "_wisdom_user_activity": time.time(),
        "agent": Mock(),
        "history_lock": threading.Lock(),
        "running": False,
        "history": [],
    }

    def prepare(*args, **kwargs):
        assert session["running"]
        if cancel:
            session["_queued_prompt_generation"] = 1
        return [
            {
                "assessment": {"id": "event", "lease_token": "token"},
                "advice": {
                    "title": "Advice",
                    "explanation": "Useful",
                    "relevance": "digest",
                },
                "interaction": None,
            }
        ]

    mediation.prepare.side_effect = prepare
    mediation.begin_delivery.side_effect = lambda org, items: items
    emit = Mock()
    poll(session, emit=emit, profile_scope=lambda _: nullcontext())
    assert session["history"] == []
    assert session["running"] is cancel
    if cancel:
        mediation.begin_delivery.assert_not_called()
        emit.assert_not_called()
    else:
        emit.assert_called_once()
        assert emit.call_args.args[0] == "notification.show"
        mediation.queue.complete_delivery.assert_called_once()
