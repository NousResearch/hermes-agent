"""Matrix read receipts and processing reactions: configuration, and which input each receipt
acknowledges. The cross-platform lifecycle contracts that these rely on are in
``tests/gateway/test_processing_lifecycle.py``."""

from __future__ import annotations

import asyncio
import os
import threading
from dataclasses import dataclass, field, replace
from threading import Lock
from unittest.mock import AsyncMock, MagicMock, call

import pytest

from agent import secret_scope
from agent.interrupt_control import InterruptControlMixin
from gateway.config import GatewayConfig, Platform, PlatformConfig, load_gateway_config
from gateway.platforms.base import ExecApprovalPrompt, SendResult
from gateway.platforms.event import MessageEvent, ProcessingOutcome
from gateway.run import GatewayRunner, _INTERRUPT_REASON_STOP, _profile_runtime_scope
from gateway.session import SessionSource
from gateway.turn_context import TurnContext
from hermes_cli.config import atomic_config_write
from plugins.platforms.matrix.adapter import MatrixAdapter
from tests.gateway.test_queued_followup_processing_hooks import _install_fake_agent

_ROOM = "!room:example.org"


def _intake_adapter(
    monkeypatch: pytest.MonkeyPatch, mode: str
) -> tuple[MatrixAdapter, MagicMock, AsyncMock]:
    monkeypatch.delenv("MATRIX_REACTIONS", raising=False)
    monkeypatch.setenv("HERMES_GATEWAY_BUSY_ACK_ENABLED", "false")
    adapter = MatrixAdapter(
        PlatformConfig(
            enabled=True,
            typing_indicator=False,
            extra={
                "read_receipts": mode,
                "reactions": False,
                "auto_thread": False,
                "user_id": "@hermes:example.org",
            },
        )
    )
    adapter._dm_rooms["!room:example.org"] = False
    adapter._get_display_name = AsyncMock(return_value="Alice")
    adapter._download_and_cache_media = AsyncMock(
        side_effect=lambda *args: f"/tmp/{args[1]}.ogg"
    )
    receipts = MagicMock()
    sender = AsyncMock(return_value=SendResult(success=True, message_id="$reply"))
    adapter._background_read_receipt = receipts
    adapter.send = sender
    adapter.stop_typing = AsyncMock()
    return adapter, receipts, sender


async def _text_input(adapter: MatrixAdapter, event_id: str, body: str) -> None:
    await adapter._handle_text_message(
        "!room:example.org",
        "@alice:example.org",
        event_id,
        0,
        {"body": body, "m.mentions": {"user_ids": ["@hermes:example.org"]}},
        {},
    )


def _busy_runner(
    monkeypatch: pytest.MonkeyPatch, adapter: MatrixAdapter, mode: str
) -> GatewayRunner:
    runner = GatewayRunner(config=GatewayConfig())
    runner.adapters[Platform.MATRIX] = adapter
    runner._busy_input_mode = mode
    runner._busy_text_mode = "interrupt"
    monkeypatch.setattr(runner, "_is_user_authorized_for_source", lambda source: True)
    adapter.set_busy_session_handler(runner._handle_active_session_busy_message)
    return runner



@pytest.mark.asyncio
@pytest.mark.parametrize("aggregation", ["batch", "queued-photos", "queued-text"])
async def test_aggregated_turn_receipts_cover_the_latest_native_input(monkeypatch, aggregation):
    adapter, receipts, _sender = _intake_adapter(monkeypatch, "after_processing")
    batch_ready, started, release = asyncio.Event(), asyncio.Event(), asyncio.Event()
    sleep = asyncio.sleep

    async def timer(delay):
        if delay == 0:
            await sleep(0)
            return
        await batch_ready.wait()

    monkeypatch.setattr("gateway.platforms.base.asyncio.sleep", timer)
    seen = []

    async def respond(event):
        seen.append((event.message_id, event.text, list(event.media_urls)))
        started.set()
        await release.wait()
        return "done"

    adapter.set_message_handler(respond)
    try:
        if aggregation == "batch":
            await _text_input(adapter, "$first", "first")
            await _text_input(adapter, "$second", "second")
            batch_ready.set()
            await asyncio.gather(*adapter._pending_text_batch_tasks.values())
        else:
            adapter._text_batch_delay_seconds = 0
            runner = _busy_runner(monkeypatch, adapter, "queue")
            await _text_input(adapter, "$opening", "opening")
            await asyncio.wait_for(started.wait(), 30)
            if aggregation == "queued-photos":
                for event_id, body in (("$first", "first"), ("$second", "second")):
                    await adapter._handle_media_message(
                        _ROOM, "@alice:example.org", event_id, 0,
                        {"body": body, "msgtype": "m.image", "url": f"mxc://example.org/{event_id}",
                         "m.mentions": {"user_ids": ["@hermes:example.org"]}},
                        {}, "m.image",
                    )
            else:
                adapter._busy_text_mode = runner._busy_text_mode = "queue"
                await _text_input(adapter, "$first", "first")
                await _text_input(adapter, "$second", "second")
        await asyncio.wait_for(started.wait(), 30)
        assert receipts.call_args_list == []
        release.set()
        while adapter._background_tasks:
            await asyncio.wait_for(asyncio.gather(*adapter._background_tasks), 30)
        photos = ["/tmp/$first.ogg", "/tmp/$second.ogg"] if aggregation == "queued-photos" else []
        merged_text = "first\n\nsecond" if aggregation == "queued-photos" else "first\nsecond"
        opening = [] if aggregation == "batch" else [("$opening", "opening", [])]
        retained_id = "$second" if aggregation == "queued-text" else "$first"
        assert (seen, receipts.call_args_list) == (
            opening + [(retained_id, merged_text, photos)],
            [call(_ROOM, event_id) for event_id in ([] if aggregation == "batch" else ["$opening"]) + ["$second"]],
        )
    finally:
        release.set()
        await adapter.cancel_background_tasks()


@pytest.mark.asyncio
@pytest.mark.parametrize("admission", ["startup", "priority"])
async def test_deferred_admission_receives_no_completion_until_replayed(monkeypatch, admission):
    adapter, receipts, sender = _intake_adapter(monkeypatch, "after_processing")
    adapter._reactions_enabled = True
    adapter._send_reaction = AsyncMock(return_value="$eyes")
    adapter._schedule_reaction_redaction = MagicMock()
    runner = _busy_runner(monkeypatch, adapter, "queue")
    event = MessageEvent(
        text="deferred",
        message_id="$deferred",
        source=SessionSource(
            platform=Platform.MATRIX, chat_id="!room:example.org", chat_type="dm"
        ),
    )
    key = adapter._event_session_key(event)
    adapter._spawn_drain_task = MagicMock()
    adapter._requeue_backoff_delay = MagicMock(return_value=1)
    queued = []
    if admission == "startup":
        runner._startup_restore_in_progress = True
    else:
        runner._session_state(key).turn.agent = MagicMock()

    async def admit(incoming):
        incoming = replace(incoming)
        if admission == "startup":
            result = await runner._handle_message(incoming)
            queued.extend(runner._startup_restore_queue)
            return result
        result = await runner._hm_handle_running_session_message(
            incoming, incoming.source, key
        )
        queued.extend(adapter._pending_messages.values())
        return result

    adapter.set_message_handler(admit)
    await adapter._process_message_background(event, key)
    assert (queued, receipts.call_args_list, sender.await_args_list) == (
        [event],
        [],
        [],
    )
    adapter._pending_messages.clear()
    adapter.set_message_handler(AsyncMock(return_value="processed"))
    await adapter._process_message_background(event, key)
    assert receipts.call_args_list == [call(event.source.chat_id, event.message_id)]
    assert adapter._send_reaction.await_args_list == [
        call(event.source.chat_id, event.message_id, "👀"),
        call(event.source.chat_id, event.message_id, "✅"),
    ]



@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("result", "receipts", "reactions"),
    [
        ({"final_response": "Operation interrupted.", "messages": [], "interrupted": True,
          "completed": False, "interrupt_message": _INTERRUPT_REASON_STOP}, [], ["👀"]),
        ({"final_response": "Provider unavailable", "messages": [], "failed": True, "completed": False},
         ["$opening"], ["👀", "✅"]),
    ],
    ids=["stopped", "failed"],
)
async def test_turn_receipt_follows_the_returned_agent_outcome(
    monkeypatch, tmp_path, result, receipts, reactions
):
    """A stopped turn is not acknowledged. A failed turn delivered its failure notice, so it is."""
    adapter, receipt_calls, _sender = _intake_adapter(monkeypatch, "after_processing")
    adapter._reactions_enabled = True
    adapter._send_reaction = AsyncMock(return_value="$eyes")
    adapter._schedule_reaction_redaction = MagicMock()
    adapter._text_batch_delay_seconds = 0
    runner = _busy_runner(monkeypatch, adapter, "queue")
    monkeypatch.setattr("gateway.run._hermes_home", tmp_path)
    monkeypatch.setattr("gateway.run._resolve_runtime_agent_kwargs", lambda: {"api_key": "test-key"})
    runner._run_agent = AsyncMock(return_value=result)
    adapter.set_message_handler(runner._handle_message)
    try:
        await _text_input(adapter, "$opening", "opening")
        await asyncio.wait_for(asyncio.gather(*adapter._background_tasks), 30)
        assert (receipt_calls.call_args_list, adapter._send_reaction.await_args_list) == (
            [call(_ROOM, event_id) for event_id in receipts],
            [call(_ROOM, "$opening", emoji) for emoji in reactions],
        )
    finally:
        await adapter.cancel_background_tasks()


@pytest.mark.asyncio
@pytest.mark.parametrize("command_reply", ["active", None, RuntimeError("status failed")])
async def test_inline_command_receipt_is_independent_of_the_active_turn(monkeypatch, command_reply):
    adapter, receipts, _sender = _intake_adapter(monkeypatch, "after_processing")
    adapter._reactions_enabled = True
    adapter._send_reaction = AsyncMock(return_value="$eyes")
    adapter._schedule_reaction_redaction = MagicMock()
    adapter._text_batch_delay_seconds = 0
    _busy_runner(monkeypatch, adapter, "queue")
    started, release = asyncio.Event(), asyncio.Event()

    async def respond(event):
        if event.get_command() == "status":
            if isinstance(command_reply, Exception):
                raise command_reply
            return command_reply
        started.set()
        await release.wait()
        return "done"

    adapter.set_message_handler(respond)
    try:
        await _text_input(adapter, "$opening", "opening")
        await asyncio.wait_for(started.wait(), 30)
        await _text_input(adapter, "$status", "/status")
        during = (list(receipts.call_args_list), list(adapter._send_reaction.await_args_list))
        release.set()
        await asyncio.wait_for(asyncio.gather(*adapter._background_tasks), 30)
        assert (during, receipts.call_args_list, adapter._send_reaction.await_args_list) == (
            ([call(_ROOM, "$status")], [call(_ROOM, "$opening", "👀")]),
            [call(_ROOM, "$status"), call(_ROOM, "$opening")],
            [call(_ROOM, "$opening", "👀"), call(_ROOM, "$opening", "✅")],
        )
    finally:
        release.set()
        await adapter.cancel_background_tasks()


@pytest.mark.asyncio
async def test_late_steer_preserves_the_completed_batch_receipt(monkeypatch):
    adapter, receipts, _sender = _intake_adapter(monkeypatch, "after_processing")
    source = SessionSource(
        platform=Platform.MATRIX, chat_id="!room:example.org", chat_type="dm"
    )
    opening = MessageEvent(text="first", message_id="$first", source=source)
    batched = MessageEvent(text="second", message_id="$second", source=source)
    steer = MessageEvent(text="steering", message_id="$steer", source=source)
    opening.absorb_turn_input(batched)
    await adapter._run_processing_hook("on_processing_start", opening)
    opening.absorb_turn_input(steer, input_text=steer.text)
    pending = opening._processing_state.take_pending_input(steer.text)
    await adapter._run_processing_hook(
        "on_processing_complete", opening, ProcessingOutcome.SUCCESS
    )
    assert (pending, opening.message_id, receipts.call_args_list) == (
        steer,
        "$first",
        [call(source.chat_id, batched.message_id)],
    )



@pytest.mark.parametrize(
    ("section", "env_override"),
    [("matrix", False), ("matrix", True), ("matrix", None), ("platforms", False), ("gateway", False)],
    ids=["matrix-yaml", "matrix-override", "matrix-env-only", "platforms-yaml", "gateway-yaml"],
)
def test_feedback_config_resolves_for_each_served_profile(
    tmp_path, monkeypatch, section, env_override
):
    homes = [tmp_path / "a", tmp_path / "b"]
    settings = [
        {"read_receipts": "disabled", "reactions": False},
        {"read_receipts": "after_processing", "reactions": True},
    ]
    for home, feedback in zip(homes, settings):
        home.mkdir()
        matrix = {"enabled": True, **feedback}
        if env_override is None:
            matrix.pop("reactions")
        config = {"matrix": matrix}
        if section == "platforms":
            config = {"platforms": config}
        if section == "gateway":
            config = {"gateway": {"platforms": config}}
        atomic_config_write(home / "config.yaml", config)
        reactions = "true" if home == homes[0] else "false"
        (home / ".env").write_text(
            f"MATRIX_REACTIONS={reactions}\n" if env_override is not False else "",
            encoding="utf-8",
        )

    monkeypatch.setenv("MATRIX_REACTIONS", "false")
    monkeypatch.setenv("MATRIX_READ_RECEIPTS", "immediate")
    original_env = {
        key: os.environ.get(key) for key in ("MATRIX_REACTIONS", "MATRIX_READ_RECEIPTS")
    }
    previous_multiplex = secret_scope.is_multiplex_active()
    secret_scope.set_multiplex_active(True)
    try:
        resolved = []
        for home in (homes[0], homes[1], homes[0]):
            with _profile_runtime_scope(home, hydrate_secrets=False):
                config = load_gateway_config().platforms[Platform.MATRIX]
                adapter = MatrixAdapter(config)
                resolved.append((
                    adapter._read_receipts_mode,
                    adapter._reactions_enabled,
                ))
        expected_reactions = (False, True) if env_override is False else (True, False)
        assert resolved == [
            ("disabled", expected_reactions[0]),
            ("after_processing", expected_reactions[1]),
            ("disabled", expected_reactions[0]),
        ]
        assert {key: os.environ.get(key) for key in original_env} == original_env
    finally:
        secret_scope.set_multiplex_active(previous_multiplex)



@pytest.mark.parametrize(
    ("value", "mode"),
    [
        ("  AFTER_PROCESSING ", "after_processing"),
        ("DISABLED", "disabled"),
        (None, "immediate"),
        ("sometimes", "immediate"),
    ],
)
def test_receipt_configuration_normalises_modes(monkeypatch, value, mode):
    monkeypatch.delenv("MATRIX_REACTIONS", raising=False)
    adapter = MatrixAdapter(
        PlatformConfig(enabled=True, extra={"read_receipts": value})
    )
    assert (adapter._read_receipts_mode, adapter._reactions_enabled) == (mode, True)


@pytest.mark.parametrize(
    ("matrix", "mode"),
    [
        ("send_read_receipts: false\n", "disabled"),
        ("send_read_receipts: false\n    read_receipts: after_processing\n", "disabled"),
        ("send_read_receipts: true\n    read_receipts: after_processing\n", "after_processing"),
        ("send_read_receipts: true\n", "immediate"),
    ],
)
def test_shared_receipt_switch_turns_matrix_receipts_off(tmp_path, monkeypatch, matrix, mode):
    """``send_read_receipts`` is the receipt switch that WhatsApp and BlueBubbles read. On Matrix
    it turns automatic receipts off, and ``read_receipts`` chooses their timing when they are on."""
    monkeypatch.delenv("MATRIX_REACTIONS", raising=False)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text(f"platforms:\n  matrix:\n    enabled: true\n    {matrix}")
    adapter = MatrixAdapter(load_gateway_config().platforms[Platform.MATRIX])
    assert adapter._read_receipts_mode == mode



@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["immediate", "after_processing", "disabled"])
async def test_claimed_voice_receipts_include_the_triggering_mention(monkeypatch, mode):
    monkeypatch.delenv("MATRIX_REACTIONS", raising=False)
    adapter = MatrixAdapter(
        PlatformConfig(
            enabled=True,
            extra={
                "read_receipts": mode,
                "reactions": False,
                "auto_thread": False,
                "user_id": "@hermes:example.org",
            },
        )
    )
    room_id, sender = "!room:example.org", "@alice:example.org"
    adapter._dm_rooms[room_id] = False
    adapter._get_display_name = AsyncMock(return_value="Alice")
    adapter.handle_message = AsyncMock()
    adapter._background_read_receipt = MagicMock()
    voice = {"msgtype": "m.audio", "body": "voice", "org.matrix.msc3245.voice": {}}
    await adapter._handle_media_message(
        room_id, sender, "$voice", 0, voice, {}, "m.audio"
    )
    adapter.handle_message.assert_not_awaited()
    await adapter._handle_text_message(
        room_id,
        sender,
        "$mention",
        0,
        {
            "body": "@hermes:example.org",
            "m.mentions": {"user_ids": ["@hermes:example.org"]},
        },
        {},
    )
    (invocation,) = adapter.handle_message.await_args_list
    event = invocation.args[0]
    assert event.message_id == "$voice"
    arrival = list(adapter._background_read_receipt.call_args_list)
    await adapter.on_processing_complete(event, ProcessingOutcome.SUCCESS)
    assert (arrival, adapter._background_read_receipt.call_args_list) == {
        "immediate": (
            [call(room_id, "$voice"), call(room_id, "$mention")],
            [call(room_id, "$voice"), call(room_id, "$mention")],
        ),
        "after_processing": ([], [call(room_id, "$mention")]),
        "disabled": ([], []),
    }[mode]



@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["immediate", "after_processing", "disabled"])
@pytest.mark.parametrize("outcome", list(ProcessingOutcome))
async def test_receipt_timing_follows_processing_outcome(monkeypatch, mode, outcome):
    monkeypatch.delenv("MATRIX_REACTIONS", raising=False)
    adapter = MatrixAdapter(
        PlatformConfig(enabled=True, extra={"read_receipts": mode, "reactions": False, "user_id": "@hermes:example.org"})
    )
    adapter._text_batch_delay_seconds = 0
    adapter._dm_rooms["!room:example.org"] = True
    adapter._read_room_member_profiles = AsyncMock(
        return_value=({adapter._user_id, "@alice:example.org"}, {})
    )
    adapter._get_display_name = AsyncMock(return_value="Alice")
    adapter.handle_message = AsyncMock()
    adapter._background_read_receipt = MagicMock()
    await adapter._handle_text_message(
        "!room:example.org",
        "@alice:example.org",
        "$message",
        0,
        {"body": "hello"},
        {},
    )
    (invocation,) = adapter.handle_message.await_args_list
    event = invocation.args[0]
    arrival = list(adapter._background_read_receipt.call_args_list)
    await adapter.on_processing_start(event)
    await adapter.on_processing_complete(event, outcome)
    receipt = [call(event.source.chat_id, event.message_id)]
    assert (arrival, adapter._background_read_receipt.call_args_list) == (
        receipt if mode == "immediate" else [],
        receipt
        if mode == "immediate"
        or (mode == "after_processing" and outcome != ProcessingOutcome.CANCELLED)
        else [],
    )



@pytest.mark.asyncio
@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("outcome", list(ProcessingOutcome))
async def test_lifecycle_toggle_preserves_completion_and_cancellation(
    monkeypatch, enabled, outcome
):
    from gateway.platforms.event import MessageEvent
    from gateway.session import SessionSource

    monkeypatch.delenv("MATRIX_REACTIONS", raising=False)
    adapter = MatrixAdapter(
        PlatformConfig(
            enabled=True, extra={"read_receipts": "disabled", "reactions": enabled}
        )
    )
    adapter._send_reaction = AsyncMock(return_value="$eyes")
    adapter._schedule_reaction_redaction = MagicMock()
    event = MessageEvent(
        text="hello",
        message_id="$message",
        source=SessionSource(
            platform=Platform.MATRIX,
            chat_id="!room:example.org",
            chat_type="dm",
        ),
    )
    await adapter.on_processing_start(event)
    await adapter.on_processing_complete(event, outcome)
    expected_reactions = []
    expected_redactions = []
    expected_pending = {}
    if enabled:
        expected_reactions.append(call(event.source.chat_id, event.message_id, "👀"))
        if outcome == ProcessingOutcome.CANCELLED:
            expected_pending[(event.source.chat_id, event.message_id)] = "$eyes"
        else:
            emoji = "✅" if outcome == ProcessingOutcome.SUCCESS else "❌"
            expected_reactions.append(
                call(event.source.chat_id, event.message_id, emoji)
            )
            expected_redactions.append(
                call(event.source.chat_id, "$eyes", "processing complete")
            )
    assert (
        adapter._send_reaction.await_args_list,
        adapter._schedule_reaction_redaction.call_args_list,
        adapter._pending_reactions,
    ) == (expected_reactions, expected_redactions, expected_pending)



@pytest.mark.asyncio
@pytest.mark.parametrize("control", ["approval", "model", "choice"])
async def test_lifecycle_disabled_keeps_reaction_controls(monkeypatch, control):
    monkeypatch.delenv("MATRIX_REACTIONS", raising=False)
    adapter = MatrixAdapter(PlatformConfig(enabled=True, extra={"reactions": False}))
    adapter._client = MagicMock()
    adapter.send = AsyncMock(
        return_value=SendResult(success=True, message_id="$prompt")
    )
    adapter._send_reaction = AsyncMock(return_value="$control")
    room_id = "!room:example.org"
    if control == "approval":
        await adapter._send_exec_approval_prompt(
            ExecApprovalPrompt(
                session_key="session",
                chat_id=room_id,
                text="Approve command",
                actions=[("Approve", "once", "primary"), ("Deny", "deny", "danger")],
                command="echo hello",
                description="Print hello",
                smart_denied=False,
                request_id="fixture-approval",
            )
        )
        choices = ["✅", "❌"]
    elif control == "model":
        await adapter.send_model_picker(
            room_id,
            [{"slug": "custom", "models": ["test-model"]}],
            "test-model",
            "custom",
            "session",
            AsyncMock(),
        )
        choices = ["1️⃣"]
    else:
        await adapter.send_choice_picker(
            room_id,
            "Choose",
            [{"value": "fast", "label": "Fast"}],
            "session",
            AsyncMock(),
        )
        choices = ["1️⃣"]
    assert adapter._send_reaction.await_args_list == [
        call(room_id, "$prompt", emoji) for emoji in choices
    ]



class _SteeringReceiver(InterruptControlMixin):
    def __init__(self) -> None:
        self._pending_steer = None
        self._pending_steer_lock = Lock()
        self._active_children = []



def _receipt_event(text: str) -> MessageEvent:
    return MessageEvent(
        text=text,
        message_id="$" + text,
        source=SessionSource(
            platform=Platform.MATRIX, chat_id="!room:example.org", chat_type="dm"
        ),
    )



def _receipt_turn(
    runner: GatewayRunner,
    adapter: MatrixAdapter,
    event: MessageEvent,
    receiver: _SteeringReceiver,
) -> TurnContext:
    key = adapter._event_session_key(event)
    ctx = TurnContext(
        source=event.source,
        session_key=key,
        session_id="steering-receipts",
        event_message_id=event.message_id,
        inbound_message_id=event.message_id,
        processing_event=event,
    )
    turn = runner._session_state(key).turn
    turn.agent, turn.event, turn.processing_event, turn.ctx = (
        receiver,
        event,
        event,
        ctx,
    )
    return ctx



@dataclass
class _ModelTurns:
    """Gates each model call of ``_GatedAgent`` so the test decides when every turn finishes."""

    loop: asyncio.AbstractEventLoop
    consumed_steers: int
    messages: list = field(default_factory=list)
    started: list = field(default_factory=lambda: [asyncio.Event() for _ in range(4)])
    release: list = field(default_factory=lambda: [threading.Event() for _ in range(4)])


class _GatedAgent:
    """Fake AIAgent behind the real runner. The first ``consumed_steers`` steers of a turn
    count as consumed by the model; later ones come back as ``pending_steer``, as the agent returns
    a steer that arrived after its last tool batch."""

    turns: _ModelTurns
    _supports_active_turn_redirect = True

    def __init__(self, **kwargs):
        self.tools, self._active_children, self._steers = [], [], []

    def steer(self, text):
        self._steers.append(text)
        return True

    redirect = steer

    def run_conversation(self, message, conversation_history=None, task_id=None, **_kwargs):
        turns = type(self).turns
        index = len(turns.messages)
        turns.messages.append(message)
        turns.loop.call_soon_threadsafe(turns.started[index].set)
        assert turns.release[index].wait(30), "The test did not release the model turn"
        leftover, self._steers = self._steers[turns.consumed_steers:], []
        result = {"final_response": f"done-{index + 1}", "messages": [], "api_calls": 1}
        if leftover:
            result["pending_steer"] = "\n".join(leftover)
        return result


@dataclass(frozen=True)
class _Chain:
    busy_mode: str
    inputs: tuple  # (model turn index, Matrix event ID, body) sent while that turn runs
    turns: int
    feedback: tuple
    consumed_steers: int = 99
    cancel_turn: int | None = None


_OPENING_DONE = (("reply", "done-1"),)
_QUEUED = ("reply", "Queued for the next turn.")


def _completed(event_id, receipt_id=None):
    return (("read", receipt_id or event_id), ("react", event_id, "✅"))


_CHAINS = {
    "queue": _Chain(
        "queue", ((0, "$work", "/queue work"),), 2,
        (("react", "$opening", "👀"), _QUEUED, *_OPENING_DONE, *_completed("$opening"),
         ("react", "$work", "👀"), ("reply", "done-2"), *_completed("$work")),
    ),
    "queue-cancelled": _Chain(
        "queue", ((0, "$work", "/queue work"),), 2,
        (("react", "$opening", "👀"), _QUEUED, *_OPENING_DONE, *_completed("$opening"),
         ("react", "$work", "👀")),
        cancel_turn=1,
    ),
    "steer": _Chain(
        "steer", ((0, "$late", "late"),), 1,
        (("react", "$opening", "👀"), *_OPENING_DONE, *_completed("$opening", "$late")),
    ),
    "slash-steer": _Chain(
        "steer", ((0, "$late", "/steer late"),), 1,
        (("react", "$opening", "👀"),
         ("reply", "⏩ Steer queued into current run — arrives after the next tool call: 'late'"),
         *_OPENING_DONE, *_completed("$opening", "$late")),
    ),
    "redirect": _Chain(
        "interrupt", ((0, "$late", "late"),), 1,
        (("react", "$opening", "👀"), *_OPENING_DONE, *_completed("$opening", "$late")),
    ),
    "leftover-steer": _Chain(
        "steer", ((0, "$late", "late"),), 2,
        (("react", "$opening", "👀"), *_OPENING_DONE, *_completed("$opening"),
         ("react", "$late", "👀"), ("reply", "done-2"), *_completed("$late")),
        consumed_steers=0,
    ),
    "consumed-then-leftover": _Chain(
        "steer", ((0, "$consumed", "consumed"), (0, "$late", "late")), 2,
        (("react", "$opening", "👀"), *_OPENING_DONE, *_completed("$opening", "$consumed"),
         ("react", "$late", "👀"), ("reply", "done-2"), *_completed("$late")),
        consumed_steers=1,
    ),
    "correction-in-queued-turn": _Chain(
        "steer", ((0, "$work", "/queue work"), (1, "$fix", "fix")), 2,
        (("react", "$opening", "👀"), _QUEUED, *_OPENING_DONE, *_completed("$opening"),
         ("react", "$work", "👀"), ("reply", "done-2"), *_completed("$work", "$fix")),
    ),
}


async def _running_model_turn(runner: GatewayRunner, turns: _ModelTurns, index: int) -> str:
    """Wait for model turn ``index`` and for the runner to register its agent. Busy input can only
    be steered into a registered agent. Returns the session key."""
    await asyncio.wait_for(turns.started[index].wait(), 30)

    async def registered() -> str:
        while True:
            for key, state in list(runner._sessions_map().items()):
                turn = state.turn
                if isinstance(turn.agent, _GatedAgent) and turn.ctx is not None and turn.ctx.result_holder[0] is None:
                    return key
            await asyncio.sleep(0.01)

    return await asyncio.wait_for(registered(), 30)


@pytest.mark.asyncio
@pytest.mark.parametrize("chain", list(_CHAINS))
async def test_each_turn_acknowledges_the_input_that_it_processed(monkeypatch, tmp_path, chain):
    """In ``after_processing`` mode a turn reads and reacts only after its reply. The receipt goes to
    the latest input that the turn's model consumed, and input that the model did not consume runs,
    and is acknowledged, as its own turn."""
    scenario = _CHAINS[chain]
    turns = _ModelTurns(asyncio.get_running_loop(), scenario.consumed_steers)
    _GatedAgent.turns = turns
    _install_fake_agent(monkeypatch, tmp_path, _GatedAgent)
    monkeypatch.setenv("MATRIX_HOME_ROOM", _ROOM)
    adapter, _receipts, _sender = _intake_adapter(monkeypatch, "after_processing")
    adapter._text_batch_delay_seconds = 0
    adapter._reactions_enabled = True
    adapter._schedule_reaction_redaction = MagicMock()
    feedback = []

    async def send(chat_id, content, reply_to=None, metadata=None):
        feedback.append(("reply", content))
        return SendResult(success=True, message_id="$reply")

    async def react(room_id, event_id, emoji):
        feedback.append(("react", event_id, emoji))
        return "$reaction"

    monkeypatch.setattr(adapter, "send", send)
    monkeypatch.setattr(adapter, "_send_reaction", react)
    monkeypatch.setattr(adapter, "_background_read_receipt",
                        lambda room_id, event_id: feedback.append(("read", event_id)))
    runner = _busy_runner(monkeypatch, adapter, scenario.busy_mode)
    adapter.set_message_handler(runner._handle_message)
    try:
        await _text_input(adapter, "$opening", "opening")
        for index in range(scenario.turns):
            key = await _running_model_turn(runner, turns, index)
            for _turn, event_id, body in (entry for entry in scenario.inputs if entry[0] == index):
                await _text_input(adapter, event_id, body)
            if index == scenario.cancel_turn:
                cancellation = asyncio.create_task(adapter.cancel_session_processing(key))
                await asyncio.sleep(0)
                turns.release[index].set()
                await asyncio.wait_for(cancellation, 30)
                break
            turns.release[index].set()
        while adapter._background_tasks:
            await asyncio.wait_for(asyncio.gather(*adapter._background_tasks), 30)
        assert (len(turns.messages), tuple(feedback)) == (scenario.turns, scenario.feedback)
    finally:
        for release in turns.release:
            release.set()
        await adapter.cancel_background_tasks()


@pytest.mark.asyncio
@pytest.mark.parametrize("confirmation", ["success", "refused", "cancelled"])
async def test_plaintext_approval_receipt_is_independent_of_the_active_turn(monkeypatch, confirmation):
    """A plain-text approval is acknowledged on its own, even when its confirmation is refused, and
    leaves the active turn's receipt and reactions to that turn."""
    from tools.approval import register_gateway_settle, unregister_gateway_notify
    from tools.approval_gateway_wait import _await_gateway_decision

    adapter, receipts, _sender = _intake_adapter(monkeypatch, "after_processing")
    adapter._text_batch_delay_seconds = 0
    adapter._reactions_enabled = True
    adapter._send_reaction = AsyncMock(return_value="$eyes")
    adapter._schedule_reaction_redaction = MagicMock()
    adapter._dm_rooms["!room:example.org"] = True
    adapter._read_room_member_profiles = AsyncMock(
        return_value=({adapter._user_id, "@alice:example.org"}, {})
    )
    runner = _busy_runner(monkeypatch, adapter, "steer")
    opening = _receipt_event("opening")
    key = adapter._event_session_key(opening)
    started, release = asyncio.Event(), asyncio.Event()
    notified = asyncio.Event()
    settled = []
    loop = asyncio.get_running_loop()
    monkeypatch.setattr("tools.approval_context._get_approval_timeout", lambda: 5)

    def notify(data):
        assert register_gateway_settle(key, data["request_id"], settled.append)
        loop.call_soon_threadsafe(notified.set)

    async def respond(event):
        _receipt_turn(runner, adapter, event, _SteeringReceiver())
        started.set()
        await release.wait()
        return "done"

    adapter.set_message_handler(respond)
    waiter = None
    control_task = None
    delivery_started = asyncio.Event()
    delivery_release = asyncio.Event()

    async def confirmation_send(**kwargs):
        if confirmation == "cancelled":
            delivery_started.set()
            await delivery_release.wait()
        return SendResult(
            success=confirmation != "refused",
            message_id="$confirmation",
            error="refused",
        )

    try:
        assert opening.message_id is not None
        await _text_input(adapter, opening.message_id, opening.text)
        await asyncio.wait_for(started.wait(), 30)
        waiter = asyncio.create_task(
            asyncio.to_thread(
                _await_gateway_decision,
                key,
                notify,
                {"command": "receipt-test", "description": "test"},
            )
        )
        await asyncio.wait_for(notified.wait(), 30)
        _sender.side_effect = confirmation_send
        control_task = asyncio.create_task(_text_input(adapter, "$approval", "yes"))
        if confirmation == "cancelled":
            await asyncio.wait_for(delivery_started.wait(), 30)
            control_task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await control_task
        else:
            await asyncio.wait_for(control_task, 30)
        _sender.side_effect = None
        decision = await asyncio.wait_for(waiter, 30)
        assert (
            decision,
            settled,
            receipts.call_args_list,
            adapter._send_reaction.await_args_list,
        ) == (
            {
                "resolved": True,
                "choice": "once",
                "reason": None,
            },
            ["resolved"],
            [] if confirmation == "cancelled" else [call(opening.source.chat_id, "$approval")],
            [call(opening.source.chat_id, "$opening", "👀")],
        )
        release.set()
        await asyncio.wait_for(asyncio.gather(*adapter._background_tasks), 30)
        assert receipts.call_args_list == (
            [] if confirmation == "cancelled" else [call(opening.source.chat_id, "$approval")]
        ) + [call(opening.source.chat_id, "$opening")]
        assert (adapter._send_reaction.await_args_list, adapter._pending_messages) == (
            [call(opening.source.chat_id, "$opening", emoji) for emoji in ("👀", "✅")],
            {},
        )
    finally:
        unregister_gateway_notify(key)
        delivery_release.set()
        if control_task is not None and not control_task.done():
            await asyncio.wait_for(control_task, 30)
        release.set()
        await adapter.cancel_background_tasks()
        if waiter is not None:
            await asyncio.wait_for(waiter, 30)
