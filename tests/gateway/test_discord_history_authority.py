from __future__ import annotations

import uuid

import pytest

from gateway.discord_connector_protocol import (
    PROTOCOL_VERSION,
    DiscordConnectorHistoryAuthority,
    DiscordConnectorHistoryAuthorityKind,
    DiscordConnectorKind,
    DiscordConnectorProtocolError,
    parse_request,
    request_message,
)
from gateway.discord_history_authority import (
    CONTROL_TOWER_CHANNEL_ID,
    VOICE_DIGEST_THREAD_ID,
    DiscordHistoryAuthorityError,
    bind_cron_history_job,
    reset_cron_history_job,
    resolve_discord_history_authority,
)
from gateway.session_context import (
    clear_session_vars,
    reset_session_vars,
    set_session_vars,
)


@pytest.fixture(autouse=True)
def _fresh_context() -> None:
    reset_session_vars()


def _history_payload(authority: DiscordConnectorHistoryAuthority) -> dict:
    return {
        "channel_id": CONTROL_TOWER_CHANNEL_ID,
        "limit": 5,
        "before_message_id": None,
        "after_message_id": None,
        "authority": authority.to_mapping(),
    }


def test_interactive_authority_comes_only_from_authenticated_discord_context(
    monkeypatch,
) -> None:
    tokens = set_session_vars(platform="discord", user_id="1279454038731264061")
    authority = resolve_discord_history_authority(CONTROL_TOWER_CHANNEL_ID)
    assert authority.kind is DiscordConnectorHistoryAuthorityKind.AUTHENTICATED_USER
    assert authority.requester_user_id == "1279454038731264061"

    # Explicit clearing suppresses a stale process-global identity.
    clear_session_vars(tokens)
    monkeypatch.setenv("HERMES_SESSION_PLATFORM", "discord")
    monkeypatch.setenv("HERMES_SESSION_USER_ID", "1279454038731264061")
    with pytest.raises(
        DiscordHistoryAuthorityError,
        match="discord_history_requester_context_missing",
    ):
        resolve_discord_history_authority(CONTROL_TOWER_CHANNEL_ID)


@pytest.mark.parametrize(
    ("job_id", "channel_id"),
    [
        ("06ef64d72891", CONTROL_TOWER_CHANNEL_ID),
        ("e62f55ca93ca", VOICE_DIGEST_THREAD_ID),
    ],
)
def test_exact_reviewed_cron_job_and_target_succeed(job_id, channel_id) -> None:
    token = bind_cron_history_job(job_id)
    try:
        authority = resolve_discord_history_authority(channel_id)
    finally:
        reset_cron_history_job(token)
    assert authority.kind is DiscordConnectorHistoryAuthorityKind.REVIEWED_CRON
    assert authority.cron_job_id == job_id


@pytest.mark.parametrize(
    ("job_id", "channel_id", "code"),
    [
        ("", CONTROL_TOWER_CHANNEL_ID, "discord_history_cron_context_invalid"),
        (
            "deadbeef0000",
            CONTROL_TOWER_CHANNEL_ID,
            "discord_history_cron_not_reviewed",
        ),
        (
            "06ef64d72891",
            VOICE_DIGEST_THREAD_ID,
            "discord_history_cron_target_not_reviewed",
        ),
        (
            "e62f55ca93ca",
            CONTROL_TOWER_CHANNEL_ID,
            "discord_history_cron_target_not_reviewed",
        ),
    ],
)
def test_blank_unknown_and_wrong_target_cron_authority_fail_closed(
    job_id, channel_id, code
) -> None:
    token = bind_cron_history_job(job_id)
    try:
        with pytest.raises(DiscordHistoryAuthorityError, match=code):
            resolve_discord_history_authority(channel_id)
    finally:
        reset_cron_history_job(token)


def test_cron_binding_never_repurposes_origin_sender_context() -> None:
    session_tokens = set_session_vars(
        platform="discord",
        user_id="1279454038731264061",
        chat_id=CONTROL_TOWER_CHANNEL_ID,
    )
    cron_token = bind_cron_history_job("deadbeef0000")
    try:
        with pytest.raises(
            DiscordHistoryAuthorityError,
            match="discord_history_cron_not_reviewed",
        ):
            resolve_discord_history_authority(CONTROL_TOWER_CHANNEL_ID)
    finally:
        reset_cron_history_job(cron_token)
        clear_session_vars(session_tokens)


def test_v1_or_authority_free_history_frame_is_rejected() -> None:
    v1 = {
        "protocol": "discord-public-connector.v1",
        "kind": DiscordConnectorKind.HISTORY_FETCH.value,
        "request_id": str(uuid.uuid4()),
        "payload": {
            "channel_id": CONTROL_TOWER_CHANNEL_ID,
            "limit": 1,
            "before_message_id": None,
            "after_message_id": None,
        },
    }
    with pytest.raises(DiscordConnectorProtocolError, match="unsupported_protocol"):
        parse_request(v1)
    v1["protocol"] = PROTOCOL_VERSION
    with pytest.raises(DiscordConnectorProtocolError, match="invalid_history_query"):
        parse_request(v1)


def test_protocol_rejects_forged_or_ambiguous_authority_shape() -> None:
    user = DiscordConnectorHistoryAuthority.authenticated_user(
        "1279454038731264061"
    )
    parsed = parse_request(
        request_message(DiscordConnectorKind.HISTORY_FETCH, _history_payload(user))
    )
    assert parsed.payload["authority"] == user.to_mapping()

    for forged in (
        {**user.to_mapping(), "cron_job_id": "06ef64d72891"},
        {"kind": "reviewed_production_cron", "cron_job_id": "unknown"},
        {"kind": "authenticated_discord_user", "requester_user_id": "0"},
    ):
        with pytest.raises(DiscordConnectorProtocolError):
            request_message(
                DiscordConnectorKind.HISTORY_FETCH,
                {**_history_payload(user), "authority": forged},
            )
