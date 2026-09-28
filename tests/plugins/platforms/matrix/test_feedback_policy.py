"""Matrix feedback settings belong to the receiving adapter's profile."""

from __future__ import annotations

import os
from unittest.mock import AsyncMock, MagicMock, call

import pytest

from agent import secret_scope
from gateway.config import Platform, PlatformConfig, load_gateway_config
from gateway.run import _profile_runtime_scope
from gateway.platforms.event import ProcessingOutcome
from gateway.platforms.base import ExecApprovalPrompt, SendResult
from hermes_cli.config import atomic_config_write
from plugins.platforms.matrix.adapter import MatrixAdapter


@pytest.mark.parametrize("section", ["matrix", "platforms", "gateway"])
@pytest.mark.parametrize(
    "env_override", [False, True, None], ids=["yaml", "override", "env-only"]
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


def test_receipt_default_does_not_read_an_unreleased_env_setting(monkeypatch):
    monkeypatch.setenv("MATRIX_READ_RECEIPTS", "disabled")
    monkeypatch.delenv("MATRIX_REACTIONS", raising=False)
    adapter = MatrixAdapter(PlatformConfig(enabled=True))
    assert (adapter._read_receipts_mode, adapter._reactions_enabled) == (
        "immediate",
        True,
    )


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
