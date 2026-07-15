from __future__ import annotations

import time

import pytest

from gateway.discord_connector_protocol import (
    MAX_HISTORY_MESSAGES,
    PROTOCOL_VERSION,
    DiscordConnectorHistoryAuthority,
    DiscordConnectorHistoryMessage,
    DiscordConnectorHistoryPage,
    DiscordConnectorKind,
    DiscordConnectorProtocolError,
    DiscordConnectorTarget,
    DiscordConnectorTargetType,
    canonical_json_bytes,
    decode_frame,
    parse_request,
    receipt,
    request_message,
    validate_receipt,
)


_HISTORY_AUTHORITY = DiscordConnectorHistoryAuthority.authenticated_user(
    "400"
).to_mapping()


def test_protocol_rejects_duplicate_unknown_and_private_target_shapes() -> None:
    with pytest.raises(DiscordConnectorProtocolError, match="invalid_frame_json"):
        decode_frame(b'{"protocol":"a","protocol":"b"}')

    hello = request_message(DiscordConnectorKind.HELLO, {"consumer": "gateway"})
    hello["extra"] = True
    with pytest.raises(DiscordConnectorProtocolError, match="invalid_request_shape"):
        parse_request(hello)

    with pytest.raises(
        DiscordConnectorProtocolError, match="forbidden_or_invalid_target"
    ):
        DiscordConnectorTarget.from_mapping(
            {
                "target_type": "dm",
                "guild_id": "100",
                "channel_id": "200",
            }
        )


def test_send_schema_only_accepts_public_bound_target_and_short_deadline() -> None:
    target = DiscordConnectorTarget(
        DiscordConnectorTargetType.PUBLIC_GUILD_CHANNEL,
        "100",
        "200",
    )
    message = request_message(
        DiscordConnectorKind.MESSAGE_SEND,
        {
            "idempotency_key": "case:1",
            "target": target.to_mapping(),
            "content": "arbitrary words are not classified",
            "reply_to_message_id": None,
            "deadline_unix_ms": int(time.time() * 1_000) + 5_000,
        },
    )
    assert parse_request(message).payload["content"] == "arbitrary words are not classified"

    message["payload"]["deadline_unix_ms"] = int(time.time() * 1_000) + 31_000
    with pytest.raises(DiscordConnectorProtocolError, match="invalid_send_deadline"):
        parse_request(message)


def test_receipt_is_digest_bound_to_exact_request() -> None:
    request = parse_request(
        request_message(DiscordConnectorKind.TARGET_GET, {"channel_id": "200"})
    )
    value = receipt(request=request, status="blocked", result={})
    parsed = validate_receipt(
        value,
        expected_kind=request.kind,
        expected_request_id=request.request_id,
    )
    assert parsed["status"] == "blocked"

    value["status"] = "ok"
    with pytest.raises(DiscordConnectorProtocolError, match="invalid_connector_receipt"):
        validate_receipt(
            value,
            expected_kind=request.kind,
            expected_request_id=request.request_id,
        )


def test_history_query_is_bounded_and_cursors_are_mutually_exclusive() -> None:
    value = request_message(
        DiscordConnectorKind.HISTORY_FETCH,
        {
            "channel_id": "200",
            "limit": MAX_HISTORY_MESSAGES,
            "before_message_id": None,
            "after_message_id": "300",
            "authority": _HISTORY_AUTHORITY,
        },
    )
    assert parse_request(value).payload == {
        "channel_id": "200",
        "limit": MAX_HISTORY_MESSAGES,
        "before_message_id": None,
        "after_message_id": "300",
        "authority": _HISTORY_AUTHORITY,
    }

    value["payload"]["limit"] = MAX_HISTORY_MESSAGES + 1
    with pytest.raises(DiscordConnectorProtocolError, match="invalid_history_query"):
        parse_request(value)

    value["payload"].update(
        {"limit": 1, "before_message_id": "299", "after_message_id": "300"}
    )
    with pytest.raises(DiscordConnectorProtocolError, match="invalid_history_query"):
        parse_request(value)


def test_history_page_cannot_encode_dm_or_private_thread_target() -> None:
    with pytest.raises(
        DiscordConnectorProtocolError,
        match="forbidden_or_invalid_target",
    ):
        DiscordConnectorHistoryPage.from_mapping(
            {
                "target": {
                    "target_type": "dm",
                    "guild_id": "100",
                    "channel_id": "200",
                },
                "messages": [],
                "query": {
                    "limit": 1,
                    "before_message_id": None,
                    "after_message_id": None,
                },
                "has_more": False,
                "order": "oldest_to_newest",
            }
        )


def test_maximum_history_page_stays_inside_connector_response_frame() -> None:
    target = DiscordConnectorTarget(
        DiscordConnectorTargetType.PUBLIC_GUILD_CHANNEL,
        "100",
        "200",
    )
    page = DiscordConnectorHistoryPage(
        target=target,
        messages=tuple(
            DiscordConnectorHistoryMessage.from_mapping(
                {
                    "message_id": str(1_000 + index),
                    "author_id": "400",
                    "author_name": "😀" * 160,
                    "author_is_bot": False,
                    "content": "\\" * 2_000,
                    "content_truncated": False,
                    "created_at_unix_ms": 1_000 + index,
                    "reply_to_message_id": None,
                }
            )
            for index in range(MAX_HISTORY_MESSAGES)
        ),
        limit=MAX_HISTORY_MESSAGES,
        before_message_id=None,
        after_message_id=None,
        has_more=True,
    )
    request = parse_request(
        request_message(
            DiscordConnectorKind.HISTORY_FETCH,
                {
                    "channel_id": "200",
                    "limit": MAX_HISTORY_MESSAGES,
                    "before_message_id": None,
                    "after_message_id": None,
                    "authority": _HISTORY_AUTHORITY,
                },
        )
    )
    body = canonical_json_bytes(
        receipt(
            request=request,
            status="ok",
            result={"page": page.to_mapping(), "page_sha256": page.sha256},
        )
    )
    assert len(body) < 128 * 1024


def test_history_protocol_requires_internal_authority_and_rejects_forged_shape() -> None:
    payload = {
        "channel_id": "200",
        "limit": 1,
        "before_message_id": None,
        "after_message_id": None,
    }
    with pytest.raises(DiscordConnectorProtocolError, match="invalid_history_query"):
        request_message(DiscordConnectorKind.HISTORY_FETCH, payload)

    payload["authority"] = {
        "kind": "authenticated_discord_user",
        "requester_user_id": "400",
        "cron_job_id": "e62f55ca93ca",
    }
    with pytest.raises(
        DiscordConnectorProtocolError,
        match="invalid_history_authority",
    ):
        request_message(DiscordConnectorKind.HISTORY_FETCH, payload)


def test_stale_v1_history_request_fails_protocol_version_before_dispatch() -> None:
    value = request_message(
        DiscordConnectorKind.HISTORY_FETCH,
        {
            "channel_id": "200",
            "limit": 1,
            "before_message_id": None,
            "after_message_id": None,
            "authority": _HISTORY_AUTHORITY,
        },
    )
    assert value["protocol"] == PROTOCOL_VERSION
    value["protocol"] = "discord-connector.v1"
    with pytest.raises(
        DiscordConnectorProtocolError,
        match="unsupported_protocol",
    ):
        parse_request(value)
