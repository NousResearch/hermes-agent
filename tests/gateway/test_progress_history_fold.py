"""Long-turn history fold (``display.progress_fold_after``): a turn that posts many progress bubbles
keeps only the newest N on screen, with the older ones collapsed into one summary message.

Drives the real progress mixin's ``send_progress_messages`` loop (and the real Slack adapter for the
Block Kit rendering); only the chat transport is faked.
"""

import asyncio
import queue
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.display_config import resolve_display_setting
from gateway.platforms.base import SendResult


class _Chat:
    """An editable, deletable chat: ``shown`` maps message id to what is on screen now."""

    def __init__(self):
        self.shown: dict[str, str] = {}
        self._next = 0

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        self._next += 1
        self.shown[str(self._next)] = content
        return SendResult(success=True, message_id=str(self._next))

    async def edit_message(self, chat_id, message_id, content, *, finalize=False):
        self.shown[str(message_id)] = content
        return SendResult(success=True, message_id=message_id)

    async def delete_message(self, chat_id, message_id):
        return self.shown.pop(str(message_id), None) is not None

    async def send_typing(self, *a, **kw):
        return None


def _loop(chat, display):
    from gateway.run_turn_progress import GatewayTurnProgressMixin

    loop = GatewayTurnProgressMixin()
    loop._runner = SimpleNamespace(_delivery_adapter_for=lambda _source: chat)
    loop._ctx = SimpleNamespace(
        source=SimpleNamespace(chat_id="C1", platform=Platform.TELEGRAM), _progress_metadata={},
        _progress_reply_to=None, _status_thread_metadata=None, _cleanup_progress=False,
        _run_still_current=lambda: True, progress_grouping="accumulate", _native_slack_task_cards=False,
        agent_holder=[SimpleNamespace(is_interrupted=False)], progress_queue=queue.Queue(),
        last_progress_msg=[None], repeat_count=[0], last_was_terminal_block=[False], event_message_id=None,
        user_config={"display": display}, resolve_display_setting=resolve_display_setting)
    return loop


async def _long_turn(chat, display, bubbles=8):
    """``bubbles`` progress bubbles, each closed by a content message landing below it."""
    loop = _loop(chat, display)
    task = asyncio.create_task(loop.send_progress_messages())
    for i in range(bubbles):
        loop._ctx.progress_queue.put(f"💻 terminal: step {i}a")
        loop._ctx.progress_queue.put(f"💻 terminal: step {i}b")
        loop._ctx.progress_queue.put(("__reset__",))
    for _ in range(400):
        await asyncio.sleep(0.01)
        if loop._ctx.progress_queue.empty() and chat._next >= bubbles:
            break
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)
    for _ in range(20):  # let scheduled fold edits/deletes land
        await asyncio.sleep(0)
    return loop


@pytest.mark.asyncio
async def test_a_long_turn_keeps_only_the_newest_bubbles_and_one_history_summary(monkeypatch):
    _no_throttle(monkeypatch)
    chat = _Chat()
    await _long_turn(chat, {"progress_fold_after": 3})

    assert chat._next == 8
    on_screen = list(chat.shown.values())
    assert len(on_screen) == 4  # one summary + the 3 newest bubbles
    summary = chat.shown[min(chat.shown, key=int)]
    assert "10 steps" in summary  # the 5 folded bubbles held 2 steps each
    assert "step 7b" in on_screen[-1]


@pytest.mark.asyncio
async def test_the_fold_is_off_unless_configured(monkeypatch):
    _no_throttle(monkeypatch)
    chat = _Chat()
    await _long_turn(chat, {})

    assert len(chat.shown) == 8


def _no_throttle(monkeypatch):
    """Every edit-interval check sees the interval elapsed, so each line is delivered before the next."""
    clock = iter(range(0, 10**9, 10))
    monkeypatch.setattr("gateway.run_turn_progress.time", SimpleNamespace(monotonic=lambda: next(clock)))
    real_sleep = asyncio.sleep

    async def fast_sleep(delay, *a, **kw):
        await real_sleep(0)

    monkeypatch.setattr("gateway.run_turn_progress.asyncio", SimpleNamespace(sleep=fast_sleep, CancelledError=asyncio.CancelledError))


@pytest.mark.asyncio
async def test_slack_renders_the_history_as_a_grey_context_block():
    from plugins.platforms.slack.adapter import SlackAdapter

    adapter = SlackAdapter(PlatformConfig(enabled=True, token="***"))
    adapter._app = MagicMock()
    client = AsyncMock()
    client.chat_update = AsyncMock(return_value={"ok": True})
    adapter._get_client = MagicMock(return_value=client)
    adapter._running = True

    result = await adapter.edit_progress_history("C1", "1700.1", ["Reading the logs · 3 steps", "a <b>"])

    assert result.success
    blocks = client.chat_update.call_args.kwargs["blocks"]
    assert blocks == [{"type": "context", "elements": [
        {"type": "mrkdwn", "text": "Reading the logs · 3 steps\na &lt;b&gt;"}]}]
