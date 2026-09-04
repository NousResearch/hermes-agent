"""Focused contract tests for Telegram Bot API 10.3 rich buttons."""

import logging
from copy import deepcopy
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway import rich_sent_store
from gateway.config import PlatformConfig
from plugins.platforms.telegram.adapter import TelegramAdapter
from telegram.error import BadRequest, NetworkError


def _make_adapter(*, rich_messages=True):
    config = PlatformConfig(
        enabled=True,
        token="fake-token",
        extra={"rich_messages": rich_messages},
    )
    adapter = TelegramAdapter(config)
    bot = MagicMock()
    bot.do_api_request = AsyncMock(return_value=SimpleNamespace(message_id=123))
    bot.send_message = AsyncMock(return_value=SimpleNamespace(message_id=1))
    bot.send_chat_action = AsyncMock()
    bot.edit_message_text = AsyncMock(return_value=True)
    adapter._bot = bot
    return adapter


def _button(text, action, value, *, style=None):
    button = {"text": text, action: value}
    if style is not None:
        button["style"] = style
    return button


VALID_CARD = {
    "blocks": [
        {
            "type": "paragraph",
            "text": [
                {"type": "plain", "text": "Choose "},
                {
                    "type": "button",
                    "button": _button(
                        "Inline", "callback_data", "demo:inline", style="primary"
                    ),
                },
            ],
        },
        {
            "type": "buttons",
            "align": "center",
            "buttons": [
                _button(
                    [
                        {
                            "type": "custom_emoji",
                            "custom_emoji_id": "1",
                            "alternative_text": "⭐",
                        },
                        " URL",
                    ],
                    "url",
                    "https://example.com",
                ),
                _button("Callback", "callback_data", "demo:callback", style="link"),
                _button("Web app", "web_app", {"url": "https://example.com/app"}),
                _button(
                    {
                        "type": "date_time",
                        "text": "Now",
                        "unix_time": 1_800_000_000,
                        "date_time_format": "day_month_year hour_minute",
                    },
                    "login_url",
                    {"url": "https://example.com/login"},
                ),
                _button("Inline", "switch_inline_query", "search"),
                _button("Here", "switch_inline_query_current_chat", "search"),
                _button(
                    "Choose chat",
                    "switch_inline_query_chosen_chat",
                    {"query": "search", "allow_user_chats": True},
                ),
                _button("Copy", "copy_text", {"text": "copied"}, style="success"),
            ],
        },
        {
            "type": "buttons",
            "align": "right",
            "buttons": [_button("Unavailable", "disabled", {}, style="danger")],
        },
    ]
}


def _mock_bot(adapter):
    bot = adapter._bot
    assert bot is not None
    return cast(Any, bot)


def _api_kwargs(adapter, endpoint):
    call = _mock_bot(adapter).do_api_request.call_args
    assert call.args[0] == endpoint
    return call.kwargs["api_kwargs"]


@pytest.mark.asyncio
async def test_send_rich_message_preserves_all_actions_and_routing():
    adapter = _make_adapter()
    card = deepcopy(VALID_CARD)

    result = await adapter.send_rich_message(
        "-100123",
        card,
        reply_to="42",
        metadata={"thread_id": "77"},
        context_text="Approval card",
    )

    assert result.success is True
    assert result.message_id == "123"
    payload = _api_kwargs(adapter, "sendRichMessage")
    assert payload["chat_id"] == -100123
    assert payload["rich_message"] == card
    assert payload["rich_message"] is card
    assert payload["message_thread_id"] == 77
    assert payload["reply_parameters"] == {"message_id": 42}
    assert payload["disable_notification"] is True
    _mock_bot(adapter).send_message.assert_not_called()


@pytest.mark.asyncio
async def test_edit_rich_message_preserves_payload_and_omits_send_routing():
    adapter = _make_adapter()
    card = deepcopy(VALID_CARD)

    result = await adapter.edit_rich_message(
        "-100123", "456", card, context_text="Updated approval card"
    )

    assert result.success is True
    assert result.message_id == "456"
    payload = _api_kwargs(adapter, "editMessageText")
    assert payload == {
        "chat_id": -100123,
        "message_id": 456,
        "rich_message": card,
    }
    _mock_bot(adapter).edit_message_text.assert_not_called()


@pytest.mark.asyncio
async def test_context_text_is_the_only_content_recorded(monkeypatch):
    adapter = _make_adapter()
    record = MagicMock()
    monkeypatch.setattr(rich_sent_store, "record", record)

    await adapter.send_rich_message(
        "123", deepcopy(VALID_CARD), context_text="Safe reply context"
    )
    record.assert_called_once_with("123", "123", "Safe reply context")

    record.reset_mock()
    await adapter.edit_rich_message("123", "456", deepcopy(VALID_CARD))
    record.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "payload",
    [
        None,
        [],
        {},
        {"markdown": "one", "blocks": []},
        {"markdown": ""},
        {"blocks": []},
        {"blocks": [object()]},
    ],
)
async def test_invalid_top_level_payload_fails_before_api(payload):
    adapter = _make_adapter()

    result = await adapter.send_rich_message("123", payload)

    assert result.success is False
    assert result.retryable is False
    assert result.error_kind == "bad_format"
    assert result.error.startswith("invalid_rich_message:")
    _mock_bot(adapter).do_api_request.assert_not_called()
    _mock_bot(adapter).send_message.assert_not_called()


def _card_for(button, *, align="left", row_size=1):
    return {
        "blocks": [
            {
                "type": "buttons",
                "align": align,
                "buttons": [deepcopy(button) for _ in range(row_size)],
            }
        ]
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "payload",
    [
        _card_for({"callback_data": "x"}),
        _card_for({"text": 1, "callback_data": "x"}),
        _card_for({"text": [], "callback_data": "x"}),
        _card_for({"text": {}, "callback_data": "x"}),
        _card_for(_button([["nested"]], "callback_data", "x")),
        _card_for(_button({"unexpected": True}, "callback_data", "x")),
        _card_for(
            _button(
                {"type": "custom_emoji", "custom_emoji_id": "1"},
                "callback_data",
                "x",
            )
        ),
        _card_for(
            _button(
                {
                    "type": "date_time",
                    "text": "Now",
                    "unix_time": True,
                    "date_time_format": "day_month_year",
                },
                "callback_data",
                "x",
            )
        ),
        _card_for(
            {"text": "Unknown", "callback_data": "x", "future_action": {}}
        ),
        _card_for(_button("No action", "callback_data", "x") | {"url": "https://x"}),
        _card_for(_button("Bad style", "callback_data", "x", style="secondary")),
        _card_for(_button("Wrong link", "url", "https://x", style="link")),
        _card_for(_button("Copy", "copy_text", {"text": "x" * 257})),
        _card_for(_button("Web", "web_app", {})),
        _card_for(_button("Disabled", "disabled", None)),
        _card_for(_button("Disabled", "disabled", {"unexpected": True})),
        _card_for(_button("Nine", "callback_data", "x"), row_size=9),
        _card_for(_button("Align", "callback_data", "x"), align="justify"),
    ],
)
async def test_invalid_button_contract_fails_before_api(payload):
    adapter = _make_adapter()

    result = await adapter.send_rich_message("123", payload)

    assert result.success is False
    assert result.retryable is False
    assert result.error_kind == "bad_format"
    _mock_bot(adapter).do_api_request.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("callback_data", "expected_success"),
    [
        ("", False),
        ("x", True),
        ("x" * 64, True),
        ("x" * 65, False),
        ("🙂" * 16, True),
        ("🙂" * 16 + "x", False),
    ],
)
async def test_callback_data_exact_utf8_boundaries(callback_data, expected_success):
    adapter = _make_adapter()
    payload = _card_for(_button("Callback", "callback_data", callback_data))

    result = await adapter.send_rich_message("123", payload)

    assert result.success is expected_success
    if expected_success:
        _mock_bot(adapter).do_api_request.assert_awaited_once()
    else:
        _mock_bot(adapter).do_api_request.assert_not_called()


@pytest.mark.asyncio
async def test_inline_button_uses_the_same_callback_byte_validation():
    adapter = _make_adapter()
    payload = {
        "blocks": [
            {
                "type": "paragraph",
                "text": [
                    {
                        "type": "button",
                        "button": _button(
                            "Too large", "callback_data", "🙂" * 16 + "x"
                        ),
                    }
                ],
            }
        ]
    }

    result = await adapter.send_rich_message("123", payload)

    assert result.success is False
    assert result.error is not None
    assert "1-64 UTF-8 bytes" in result.error
    _mock_bot(adapter).do_api_request.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("length", "expected_success"),
    [(0, False), (1, True), (256, True), (257, False)],
)
async def test_copy_text_exact_character_boundaries(length, expected_success):
    adapter = _make_adapter()
    payload = _card_for(
        _button("Copy", "copy_text", {"text": "x" * length})
    )

    result = await adapter.send_rich_message("123", payload)

    assert result.success is expected_success
    if expected_success:
        _mock_bot(adapter).do_api_request.assert_awaited_once()
    else:
        _mock_bot(adapter).do_api_request.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("row_size", "expected_success"),
    [(0, False), (1, True), (8, True), (9, False)],
)
async def test_button_row_exact_size_boundaries(row_size, expected_success):
    adapter = _make_adapter()
    payload = _card_for(
        _button("Choice", "callback_data", "x"), row_size=row_size
    )

    result = await adapter.send_rich_message("123", payload)

    assert result.success is expected_success
    if expected_success:
        _mock_bot(adapter).do_api_request.assert_awaited_once()
    else:
        _mock_bot(adapter).do_api_request.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("align", ["left", "center", "right"])
async def test_all_button_row_alignments_are_accepted(align):
    adapter = _make_adapter()
    result = await adapter.send_rich_message(
        "123", _card_for(_button("Choice", "callback_data", "x"), align=align)
    )

    assert result.success is True


@pytest.mark.asyncio
@pytest.mark.parametrize("style", [None, "danger", "success", "primary", "link"])
async def test_all_button_styles_are_accepted_for_callbacks(style):
    adapter = _make_adapter()
    result = await adapter.send_rich_message(
        "123", _card_for(_button("Choice", "callback_data", "x", style=style))
    )

    assert result.success is True


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "payload",
    [{"html": "<b>Hello</b>"}, {"markdown": "**Hello**"}],
)
async def test_valid_html_and_markdown_modes_pass_through(payload):
    adapter = _make_adapter()

    result = await adapter.send_rich_message("123", payload)

    assert result.success is True
    assert _api_kwargs(adapter, "sendRichMessage")["rich_message"] == payload


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["html", "markdown"])
@pytest.mark.parametrize(
    ("length", "expected_success"), [(32_768, True), (32_769, False)]
)
async def test_rich_text_mode_exact_character_boundary(
    mode, length, expected_success
):
    adapter = _make_adapter()

    result = await adapter.send_rich_message("123", {mode: "x" * length})

    assert result.success is expected_success
    if expected_success:
        _mock_bot(adapter).do_api_request.assert_awaited_once()
    else:
        _mock_bot(adapter).do_api_request.assert_not_called()


@pytest.mark.asyncio
async def test_non_json_container_and_non_string_key_fail_before_api():
    adapter = _make_adapter()
    invalid_button = _button("Bad", "callback_data", "")
    tuple_payload = {"blocks": [("wrapper", {"type": "button", "button": invalid_button})]}
    keyed_payload = {"blocks": [{1: "not-json-object-key"}]}

    for payload in (tuple_payload, keyed_payload):
        result = await adapter.send_rich_message("123", payload)
        assert result.success is False
        assert result.error_kind == "bad_format"
        _mock_bot(adapter).do_api_request.assert_not_called()


@pytest.mark.asyncio
async def test_button_like_objects_in_unknown_block_fields_pass_through():
    adapter = _make_adapter()
    payload = {
        "blocks": [
            {
                "type": "future_block",
                "metadata": {
                    "type": "button",
                    "button": {"not": "a rich-text button at this position"},
                },
                "opaque": {"type": "buttons", "buttons": []},
            }
        ]
    }

    result = await adapter.send_rich_message("123", payload)

    assert result.success is True
    assert _api_kwargs(adapter, "sendRichMessage")["rich_message"] == payload


@pytest.mark.asyncio
async def test_button_rows_nested_in_list_item_blocks_are_validated():
    valid = {
        "blocks": [
            {
                "type": "list",
                "items": [
                    {
                        "blocks": [
                            _card_for(
                                _button("Nested", "callback_data", "nested")
                            )["blocks"][0]
                        ]
                    }
                ],
            }
        ]
    }
    invalid = {
        "blocks": [
            {
                "type": "list",
                "items": [
                    {"blocks": [{"type": "buttons", "buttons": []}]}
                ],
            }
        ]
    }

    valid_adapter = _make_adapter()
    valid_result = await valid_adapter.send_rich_message("123", valid)
    assert valid_result.success is True

    invalid_adapter = _make_adapter()
    invalid_result = await invalid_adapter.send_rich_message("123", invalid)
    assert invalid_result.success is False
    assert invalid_result.error_kind == "bad_format"
    _mock_bot(invalid_adapter).do_api_request.assert_not_called()


@pytest.mark.asyncio
async def test_shared_container_identity_is_valid_json_and_is_preserved():
    adapter = _make_adapter()
    shared = {"type": "paragraph", "text": "shared"}
    payload = {"blocks": [shared, shared]}

    result = await adapter.send_rich_message("123", payload)

    assert result.success is True
    assert _api_kwargs(adapter, "sendRichMessage")["rich_message"] is payload


@pytest.mark.asyncio
async def test_shared_reference_dag_is_bounded_by_occurrence_count():
    adapter = _make_adapter()
    node = {}
    for _ in range(13):
        node = {"left": node, "right": node}
    payload = {"blocks": [node]}

    result = await adapter.send_rich_message("123", payload)

    assert result.success is False
    assert result.error == "invalid_rich_message: rich message structure is too large"
    _mock_bot(adapter).do_api_request.assert_not_called()


def _nested_blocks_payload(wrapper_count):
    node = {}
    for _ in range(wrapper_count):
        node = {"child": node}
    return {"blocks": [node]}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("wrapper_count", "expected_success"), [(30, True), (31, False)]
)
async def test_traversal_depth_exact_boundary(wrapper_count, expected_success):
    adapter = _make_adapter()

    result = await adapter.send_rich_message(
        "123", _nested_blocks_payload(wrapper_count)
    )

    assert result.success is expected_success
    if expected_success:
        _mock_bot(adapter).do_api_request.assert_awaited_once()
    else:
        _mock_bot(adapter).do_api_request.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("block_count", "expected_success"), [(9_998, True), (9_999, False)]
)
async def test_container_count_exact_boundary(block_count, expected_success):
    adapter = _make_adapter()
    payload = {"blocks": [{} for _ in range(block_count)]}

    result = await adapter.send_rich_message("123", payload)

    assert result.success is expected_success
    if expected_success:
        _mock_bot(adapter).do_api_request.assert_awaited_once()
    else:
        _mock_bot(adapter).do_api_request.assert_not_called()


@pytest.mark.asyncio
async def test_cyclic_payload_fails_closed_without_echoing_payload():
    adapter = _make_adapter()
    payload = {"blocks": []}
    payload["blocks"].append(payload)

    result = await adapter.send_rich_message("123", payload)

    assert result.success is False
    assert result.error == (
        "invalid_rich_message: rich_message must not contain container cycles"
    )
    _mock_bot(adapter).do_api_request.assert_not_called()


@pytest.mark.asyncio
async def test_disabled_and_unavailable_rich_messages_fail_without_api_call():
    disabled = _make_adapter(rich_messages=False)
    disabled_result = await disabled.send_rich_message("123", deepcopy(VALID_CARD))
    assert disabled_result.success is False
    assert disabled_result.error == "rich_messages_disabled"
    assert disabled_result.retryable is False
    disabled._bot.do_api_request.assert_not_called()

    unavailable = _make_adapter()
    unavailable._rich_send_disabled = True
    unavailable_result = await unavailable.send_rich_message(
        "123", deepcopy(VALID_CARD)
    )
    assert unavailable_result.success is False
    assert unavailable_result.error == "rich_messages_unavailable"
    assert unavailable_result.retryable is False
    unavailable._bot.do_api_request.assert_not_called()


@pytest.mark.asyncio
async def test_invalid_routing_metadata_fails_before_api_call():
    adapter = _make_adapter()

    bad_reply = await adapter.send_rich_message(
        "123", deepcopy(VALID_CARD), reply_to="not-an-int"
    )
    assert bad_reply.success is False
    assert bad_reply.error_kind == "bad_format"
    _mock_bot(adapter).do_api_request.assert_not_called()

    bad_metadata = await adapter.send_rich_message(
        "123",
        deepcopy(VALID_CARD),
        metadata=cast(Any, ["not", "an", "object"]),
    )
    assert bad_metadata.success is False
    assert bad_metadata.error_kind == "bad_format"
    _mock_bot(adapter).do_api_request.assert_not_called()


@pytest.mark.asyncio
async def test_permanent_rejection_never_legacy_resends_or_echoes_payload(caplog):
    adapter = _make_adapter()
    marker = "private-callback-marker"
    caplog.set_level(logging.DEBUG)
    _mock_bot(adapter).do_api_request = AsyncMock(
        side_effect=BadRequest(f"can't parse rich message: {marker}")
    )

    result = await adapter.send_rich_message("123", deepcopy(VALID_CARD))

    assert result.success is False
    assert result.retryable is False
    assert result.error == "rich_message_rejected"
    assert marker not in caplog.text
    _mock_bot(adapter).do_api_request.assert_awaited_once()
    _mock_bot(adapter).send_message.assert_not_called()


@pytest.mark.asyncio
async def test_transient_failure_never_duplicates_or_echoes_payload(caplog):
    adapter = _make_adapter()
    marker = "private-callback-marker"
    caplog.set_level(logging.WARNING)
    _mock_bot(adapter).do_api_request = AsyncMock(
        side_effect=NetworkError(f"temporary failure: {marker}")
    )

    result = await adapter.send_rich_message("123", deepcopy(VALID_CARD))

    assert result.success is False
    assert result.retryable is True
    assert result.error == "rich_message_transport_failure"
    assert marker not in caplog.text
    _mock_bot(adapter).do_api_request.assert_awaited_once()
    _mock_bot(adapter).send_message.assert_not_called()


@pytest.mark.asyncio
async def test_capability_failure_latches_and_does_not_legacy_resend():
    endpoint_not_found = type("EndPointNotFound", (Exception,), {})
    adapter = _make_adapter()
    _mock_bot(adapter).do_api_request = AsyncMock(
        side_effect=endpoint_not_found("endpoint unavailable")
    )

    result = await adapter.send_rich_message("123", deepcopy(VALID_CARD))

    assert result.success is False
    assert result.retryable is False
    assert adapter._rich_send_disabled is True
    _mock_bot(adapter).send_message.assert_not_called()


@pytest.mark.asyncio
async def test_edit_rich_message_connection_and_availability_states():
    permanent = _make_adapter()
    permanent._bot = None
    permanent._replacement_telegram_adapter = MagicMock(return_value=None)
    permanent._is_permanent_fatal = MagicMock(return_value=True)
    permanent._wait_for_reconnection = AsyncMock(return_value=True)

    disconnected = await permanent.edit_rich_message(
        "123", "456", deepcopy(VALID_CARD)
    )
    assert disconnected.success is False
    assert disconnected.error == "Not connected"
    assert disconnected.retryable is False
    permanent._wait_for_reconnection.assert_not_awaited()

    degraded = _make_adapter()
    degraded._send_path_degraded = True
    degraded_result = await degraded.edit_rich_message(
        "123", "456", deepcopy(VALID_CARD)
    )
    assert degraded_result.success is False
    assert degraded_result.error == "send_path_degraded"
    assert degraded_result.retryable is True
    _mock_bot(degraded).do_api_request.assert_not_called()

    unavailable = _make_adapter()
    unavailable._rich_send_disabled = True
    unavailable_result = await unavailable.edit_rich_message(
        "123", "456", deepcopy(VALID_CARD)
    )
    assert unavailable_result.success is False
    assert unavailable_result.error == "rich_messages_unavailable"
    assert unavailable_result.retryable is False
    _mock_bot(unavailable).do_api_request.assert_not_called()


@pytest.mark.asyncio
async def test_edit_not_modified_is_success_and_invalid_id_never_calls_api():
    adapter = _make_adapter()
    _mock_bot(adapter).do_api_request = AsyncMock(
        side_effect=BadRequest("Message is not modified")
    )

    unchanged = await adapter.edit_rich_message(
        "123", "456", deepcopy(VALID_CARD)
    )
    assert unchanged.success is True
    assert unchanged.message_id == "456"

    _mock_bot(adapter).do_api_request.reset_mock()
    invalid = await adapter.edit_rich_message(
        "123", "not-an-int", deepcopy(VALID_CARD)
    )
    assert invalid.success is False
    assert invalid.error_kind == "bad_format"
    _mock_bot(adapter).do_api_request.assert_not_called()
