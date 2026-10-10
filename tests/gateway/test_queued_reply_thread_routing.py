"""Queued terminal replies follow their own thread, not the chain opener's.

Exercise real runner/queue handling and Slack delivery with only the model and
Slack Web API replaced. Session storage and delivery bookkeeping use HERMES_HOME
from the suite's isolated fixture.
"""

import json
import weakref
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import _reply_anchor_for_event, _thread_metadata_for_event
from gateway.platforms.event import MessageEvent
from gateway.run import GatewayRunner
from gateway.session import SessionSource
from gateway.session_identity import RoutingIdentity, identity_of
from gateway.turn_context import TurnContext
from plugins.platforms.slack.adapter import SlackAdapter


class RecordingSlackAdapter(SlackAdapter):
    def __init__(self):
        super().__init__(PlatformConfig(enabled=True, extra={"reply_in_thread": True}))
        self.sent = []
        self.typing_metadata = []
        self.stopped_typing_metadata = []
        self.client = SimpleNamespace(
            chat_postMessage=AsyncMock(return_value={"ok": True, "ts": "900.001"}),
            reactions_add=AsyncMock(return_value={"ok": True}),
            reactions_remove=AsyncMock(return_value={"ok": True}),
            assistant_threads_setStatus=AsyncMock(return_value={"ok": True}),
        )
        self._app = SimpleNamespace(client=self.client)

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        self.sent.append((reply_to, dict(metadata or {})))
        return await super().send(chat_id, content, reply_to, metadata)

    async def _stop_typing_with_metadata(self, chat_id, metadata=None):
        self.stopped_typing_metadata.append(metadata)
        await super()._stop_typing_with_metadata(chat_id, metadata=metadata)

    def _start_typing_refresh(self, event, interrupt_event, metadata):
        self.typing_metadata.append(metadata)
        return None


def _event(message_id, thread_id, *, platform=Platform.SLACK, chat_type="group"):
    return MessageEvent(
        text=f"question {message_id}", message_id=message_id,
        source=SessionSource(
            platform=platform, chat_id="C123", user_id="U123", chat_type=chat_type,
            thread_id=thread_id, scope_id="T123" if platform == Platform.SLACK else None,
        ),
    )


async def _deliver(monkeypatch, tmp_path, opening, queued):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    runner = GatewayRunner(GatewayConfig())
    adapter = RecordingSlackAdapter()
    runner.adapters = {Platform.SLACK: adapter}
    adapter.gateway_runner = runner
    # Pin a receiving transport independently of source.profile, as ingress does.
    source = opening.source
    source._transport_adapter_ref = weakref.ref(adapter)
    source._authorization_profile_home = tmp_path
    source._identity = RoutingIdentity(
        transport_profile="default", runtime_profile="default",
        authorization_home=tmp_path, runtime_home=tmp_path,
        multiplexed=False, transport=weakref.ref(adapter),
    )
    key = runner._session_key_for_source(source)
    runner._session_run_generation[key] = 1
    pending = list(queued)

    async def run_model(**kwargs):
        result = {
            "final_response": f"answer {kwargs['inbound_message_id']}",
            "messages": [], "api_calls": 1, "failed": False,
        }
        if not pending:
            return result
        followup = pending.pop(0)
        pending_event = followup if isinstance(followup, MessageEvent) else None
        ctx = TurnContext(
            source=kwargs["source"], session_id=kwargs["session_id"], session_key=key,
            run_generation=1, history=[], context_prompt=kwargs["context_prompt"],
            event_message_id=kwargs["event_message_id"],
            inbound_message_id=kwargs["inbound_message_id"],
            _interrupt_depth=kwargs.get("_interrupt_depth", 0),
        )
        return await runner._run_agent_queued_followup(
            ctx, adapter=adapter,
            pending=pending_event.text if pending_event is not None else followup,
            pending_event=pending_event,
            response=result["final_response"], result=result, stream_task=None,
        )

    monkeypatch.setattr(runner, "_run_agent", run_model)

    async def handler(event):
        return await runner._handle_message_with_agent(event, event.source, key, 1)

    adapter.set_message_handler(handler)
    try:
        await adapter._process_message_background(opening, key)
        assert not pending
        assert adapter.client.chat_postMessage.await_args_list
        return runner, adapter, source
    finally:
        if runner._executor is not None:
            runner._executor.shutdown(wait=True)
        if runner._housekeeping_executor is not None:
            runner._housekeeping_executor.shutdown(wait=True)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "opener_thread,terminal_thread,override,anchorless",
    [
        (None, None, None, False),
        ("100.001", None, None, False),
        (None, "200.001", None, False),
        ("100.001", "200.001", None, False),
        ("100.001", "100.001", None, False),
        (None, None, "redirect", False),
        ("100.001", "200.001", "redirect", False),
        ("100.001", None, "", False),
        (None, None, None, True),
        ("100.001", None, None, True),
    ],
    ids=["top-level", "thread-to-top", "top-to-thread", "cross-thread", "same-thread",
         "redirect-top", "redirect-thread", "empty-redirect", "channel-only", "thread-to-channel-only"],
)
async def test_queued_terminal_delivery_owns_thread_without_overriding_redirect(
    monkeypatch, tmp_path, opener_thread, terminal_thread, override, anchorless,
):
    opening = _event("101.001", opener_thread)
    opening.reply_anchor_override = override
    # Two recursive follow-ups prove that the innermost routing survives unwind.
    middle = _event("201.001", "150.001")
    terminal = _event("301.001", terminal_thread)
    if anchorless:
        terminal.raw_message = {"_hermes_no_thread_response": True}
    terminal_anchor = _reply_anchor_for_event(terminal) or ""
    original_metadata = _thread_metadata_for_event(opening)
    original_anchor = _reply_anchor_for_event(opening)
    runner, adapter, original_source = await _deliver(
        monkeypatch, tmp_path, opening, [middle, terminal],
    )

    different = (opener_thread != terminal_thread if opener_thread or terminal_thread
                 else original_anchor != terminal_anchor)
    retargeted = different and override is None
    expected_thread = terminal_thread if retargeted else opener_thread
    expected_anchor = terminal_anchor if retargeted else original_anchor
    final_post = adapter.client.chat_postMessage.await_args.kwargs
    assert final_post["text"] == f"answer {terminal.message_id}"
    assert final_post.get("thread_ts") == (expected_thread or expected_anchor or None)
    assert adapter.sent[-1][0] == expected_anchor
    assert opening.source.thread_id == expected_thread
    assert opening.ledger_message_id == terminal.message_id
    assert opening.message_id != terminal.message_id
    assert adapter.typing_metadata == [original_metadata]
    assert adapter.stopped_typing_metadata
    for metadata in adapter.stopped_typing_metadata:
        assert (metadata or {}).get("thread_id") == opener_thread
        assert (metadata or {}).get("slack_team_id") == original_metadata["slack_team_id"]
    assert identity_of(opening.source) is identity_of(original_source)
    assert opening.source._transport_adapter_ref is original_source._transport_adapter_ref
    assert opening.source._authorization_profile_home == original_source._authorization_profile_home
    assert runner._delivery_adapter_for(opening.source) is adapter
    if not retargeted:
        assert opening.source is original_source
        assert opening.reply_anchor_override == override


@pytest.mark.asyncio
@pytest.mark.parametrize("thread_id", [None, "100.001"])
@pytest.mark.parametrize("text_only_followup", [False, True], ids=["nonqueued", "text-only-followup"])
async def test_unchanged_delivery_target_keeps_prehandler_metadata(
    monkeypatch, tmp_path, thread_id, text_only_followup,
):
    opening = _event("101.001", thread_id)
    before = json.dumps({**(_thread_metadata_for_event(opening) or {}), "notify": True})
    queued = ["continue"] if text_only_followup else []
    _, adapter, original_source = await _deliver(monkeypatch, tmp_path, opening, queued)
    assert json.dumps(adapter.sent[-1][1]) == before
    assert opening.source is original_source
    assert opening.reply_anchor_override is None
    assert adapter.sent[-1][0] == opening.message_id
    answered_id = None if text_only_followup else opening.message_id
    assert adapter.client.chat_postMessage.await_args.kwargs["text"] == f"answer {answered_id}"
