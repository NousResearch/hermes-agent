from __future__ import annotations

import asyncio
import io
import json
import os
import time
import traceback
import urllib.error
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

import gateway.discord_rest_edge as discord_rest_edge
from gateway.discord_edge_protocol import (
    DiscordEdgeOperation,
    DiscordPublicTarget,
    DiscordPublicTargetType,
)
from gateway.discord_rest_edge import (
    DiscordRestEdgeAdapter,
    DiscordRestEdgeError,
    DiscordRestEdgeErrorCode,
)

GUILD_ID = "100000000000000001"
CHANNEL_ID = "200000000000000002"
PARENT_ID = "210000000000000002"
BOT_ID = "300000000000000003"
BOT_ROLE_ID = "400000000000000004"
MESSAGE_ID = "500000000000000005"
THREAD_ID = "600000000000000006"
THREAD_MESSAGE_ID = "700000000000000007"

VIEW_CHANNEL = 1 << 10
SEND_MESSAGES = 1 << 11
READ_MESSAGE_HISTORY = 1 << 16
CREATE_PUBLIC_THREADS = 1 << 35
SEND_MESSAGES_IN_THREADS = 1 << 38
BOT_PERMISSIONS = (
    VIEW_CHANNEL
    | SEND_MESSAGES
    | READ_MESSAGE_HISTORY
    | CREATE_PUBLIC_THREADS
    | SEND_MESSAGES_IN_THREADS
)
TOKEN = "test.token_value-with-safe-chars_123456789"


class FakeResponse:
    def __init__(
        self,
        value: object = None,
        *,
        status: int = 200,
        body: bytes | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        self.status = status
        self._body = (
            body
            if body is not None
            else json.dumps(value, separators=(",", ":")).encode("utf-8")
        )
        self.headers = headers or {"Content-Length": str(len(self._body))}

    def read(self, amount: int = -1) -> bytes:
        if amount < 0:
            amount = len(self._body)
        body, self._body = self._body[:amount], self._body[amount:]
        return body

    def __enter__(self) -> FakeResponse:
        return self

    def __exit__(self, *_args: object) -> None:
        return None


class FakeOpener:
    def __init__(
        self,
        handler: Callable[[str, str, dict[str, Any] | None], FakeResponse],
    ) -> None:
        self.handler = handler
        self.requests: list[tuple[str, str, dict[str, Any] | None, str]] = []

    def open(self, request: Any, timeout: float) -> FakeResponse:
        assert 0 < timeout <= 1.0
        body = json.loads(request.data) if request.data is not None else None
        authorization = request.get_header("Authorization")
        self.requests.append(
            (request.method, request.full_url, body, authorization)
        )
        return self.handler(request.method, request.full_url, body)


def channel(
    *,
    channel_id: str = CHANNEL_ID,
    guild_id: str = GUILD_ID,
    channel_type: int = 0,
    parent_id: str | None = None,
    overwrites: list[dict[str, object]] | None = None,
    name: str | None = None,
) -> dict[str, object]:
    value: dict[str, object] = {
        "guild_id": guild_id,
        "id": channel_id,
        "parent_id": parent_id,
        "permission_overwrites": overwrites or [],
        "type": channel_type,
    }
    if name is not None:
        value["name"] = name
    if channel_type in {10, 11, 12}:
        value["owner_id"] = BOT_ID
        value["thread_metadata"] = {
            "archived": False,
            "auto_archive_duration": 1440,
            "locked": False,
        }
    return value


def guild(
    *,
    everyone_permissions: int = VIEW_CHANNEL,
    bot_permissions: int = BOT_PERMISSIONS,
) -> dict[str, object]:
    return {
        "id": GUILD_ID,
        "owner_id": "900000000000000009",
        "roles": [
            {"id": GUILD_ID, "permissions": str(everyone_permissions)},
            {"id": BOT_ROLE_ID, "permissions": str(bot_permissions)},
        ],
    }


def member() -> dict[str, object]:
    return {
        "roles": [BOT_ROLE_ID],
        "user": {"bot": True, "id": BOT_ID},
    }


def current_user() -> dict[str, object]:
    return {"bot": True, "id": BOT_ID}


def default_identity_response(method: str, url: str) -> FakeResponse | None:
    if method != "GET":
        return None
    if url.endswith("/users/@me"):
        return FakeResponse(current_user())
    if url.endswith(f"/guilds/{GUILD_ID}"):
        return FakeResponse(guild())
    if url.endswith(f"/guilds/{GUILD_ID}/members/{BOT_ID}"):
        return FakeResponse(member())
    return None


def message(
    *,
    message_id: str = MESSAGE_ID,
    channel_id: str = CHANNEL_ID,
    content: str = "hello",
    author_id: str = BOT_ID,
    reply_to_message_id: str | None = None,
    include_reply_guild: bool = True,
) -> dict[str, object]:
    value: dict[str, object] = {
        "author": {"bot": True, "id": author_id},
        "channel_id": channel_id,
        "content": content,
        "guild_id": GUILD_ID,
        "id": message_id,
    }
    if reply_to_message_id is not None:
        value["message_reference"] = {
            "channel_id": channel_id,
            "message_id": reply_to_message_id,
            "type": 0,
        }
        if include_reply_guild:
            value["message_reference"]["guild_id"] = GUILD_ID
    return value


def target() -> DiscordPublicTarget:
    return DiscordPublicTarget(
        DiscordPublicTargetType.PUBLIC_GUILD_CHANNEL,
        GUILD_ID,
        CHANNEL_ID,
    )


def thread_target(*, parent_id: str = PARENT_ID) -> DiscordPublicTarget:
    return DiscordPublicTarget(
        DiscordPublicTargetType.PUBLIC_GUILD_THREAD,
        GUILD_ID,
        CHANNEL_ID,
        parent_id,
    )


def forum_target() -> DiscordPublicTarget:
    return DiscordPublicTarget(
        DiscordPublicTargetType.PUBLIC_GUILD_FORUM,
        GUILD_ID,
        CHANNEL_ID,
    )


def make_adapter(
    tmp_path: Path,
    opener: FakeOpener,
    *,
    token: str = TOKEN,
) -> DiscordRestEdgeAdapter:
    credentials = tmp_path / "credentials"
    credentials.mkdir(mode=0o700, parents=True)
    credentials.chmod(0o700)
    credential = credentials / "discord-token"
    credential.write_text(f"{token}\n", encoding="ascii")
    credential.chmod(0o400)
    return DiscordRestEdgeAdapter.from_credential_file(
        credential,
        credentials_directory=credentials,
        expected_owner_uid=os.getuid(),
        timeout_seconds=1.0,
        _opener=opener,
        _sleeper=lambda _seconds: None,
        _clock_ms=lambda: 1_000,
    )


def proof_handler(
    target_channel: dict[str, object],
    *,
    guild_value: dict[str, object] | None = None,
    parent_channel: dict[str, object] | None = None,
) -> Callable[[str, str, dict[str, Any] | None], FakeResponse]:
    def handler(method: str, url: str, _body: dict[str, Any] | None) -> FakeResponse:
        assert method == "GET"
        if url.endswith(f"/channels/{CHANNEL_ID}"):
            return FakeResponse(target_channel)
        if parent_channel is not None and url.endswith(f"/channels/{PARENT_ID}"):
            return FakeResponse(parent_channel)
        if url.endswith("/users/@me"):
            return FakeResponse(current_user())
        if url.endswith(f"/guilds/{GUILD_ID}"):
            return FakeResponse(guild_value or guild())
        if url.endswith(f"/guilds/{GUILD_ID}/members/{BOT_ID}"):
            return FakeResponse(member())
        raise AssertionError((method, url))

    return handler


def test_live_proof_rejects_dm_and_private_thread_types(tmp_path: Path) -> None:
    dm_adapter = make_adapter(
        tmp_path / "dm",
        FakeOpener(proof_handler(channel(channel_type=1))),
    )
    with pytest.raises(DiscordRestEdgeError) as dm_error:
        dm_adapter.prove_public_message_send(
            target(),
            deadline_unix_ms=30_000,
            now_unix_ms=1_000,
        )
    assert dm_error.value.code is DiscordRestEdgeErrorCode.TARGET_NOT_PUBLIC

    private_adapter = make_adapter(
        tmp_path / "private",
        FakeOpener(
            proof_handler(
                channel(channel_type=12, parent_id=PARENT_ID),
                parent_channel=channel(channel_id=PARENT_ID),
            )
        ),
    )
    with pytest.raises(DiscordRestEdgeError) as private_error:
        private_adapter.prove_public_message_send(
            thread_target(),
            deadline_unix_ms=30_000,
            now_unix_ms=1_000,
        )
    assert private_error.value.code is DiscordRestEdgeErrorCode.TARGET_NOT_PUBLIC


def test_live_proof_rejects_exact_thread_parent_mismatch(tmp_path: Path) -> None:
    adapter = make_adapter(
        tmp_path,
        FakeOpener(
            proof_handler(
                channel(
                    channel_type=11,
                    parent_id="999999999999999999",
                )
            )
        ),
    )
    with pytest.raises(DiscordRestEdgeError) as error:
        adapter.prove_public_message_send(
            thread_target(),
            deadline_unix_ms=30_000,
            now_unix_ms=1_000,
        )
    assert error.value.code is DiscordRestEdgeErrorCode.TARGET_MISMATCH


def test_live_proof_computes_everyone_visibility_and_bot_overwrites(
    tmp_path: Path,
) -> None:
    overwrites = [
        {
            "allow": "0",
            "deny": str(VIEW_CHANNEL),
            "id": GUILD_ID,
            "type": 0,
        },
        {
            "allow": str(BOT_PERMISSIONS),
            "deny": "0",
            "id": BOT_ROLE_ID,
            "type": 0,
        },
    ]
    adapter = make_adapter(
        tmp_path,
        FakeOpener(proof_handler(channel(overwrites=overwrites))),
    )
    proof = adapter.prove_public_message_send(
        target(),
        deadline_unix_ms=30_000,
        now_unix_ms=1_000,
    )
    assert proof.publicly_viewable is False
    assert proof.bot_can_view is True
    assert proof.bot_has_required_permission is True
    assert proof.bot_user_id == BOT_ID


def test_live_proof_detects_bot_permission_revoke(tmp_path: Path) -> None:
    revoked = BOT_PERMISSIONS & ~SEND_MESSAGES
    adapter = make_adapter(
        tmp_path,
        FakeOpener(
            proof_handler(
                channel(),
                guild_value=guild(bot_permissions=revoked),
            )
        ),
    )
    proof = adapter.prove_public_message_send(
        target(),
        deadline_unix_ms=30_000,
        now_unix_ms=1_000,
    )
    assert proof.publicly_viewable is True
    assert proof.bot_can_view is True
    assert proof.bot_has_required_permission is False


@pytest.mark.parametrize("channel_type", [2, 13])
def test_public_voice_surfaces_remain_messageable(
    tmp_path: Path,
    channel_type: int,
) -> None:
    adapter = make_adapter(
        tmp_path,
        FakeOpener(proof_handler(channel(channel_type=channel_type))),
    )
    proof = adapter.prove_public_message_send(
        target(),
        deadline_unix_ms=30_000,
        now_unix_ms=1_000,
    )
    assert proof.publicly_viewable is True
    assert proof.bot_has_required_permission is True


def test_proof_fetches_permission_channel_last_and_stamps_completion_time(
    tmp_path: Path,
) -> None:
    opener = FakeOpener(proof_handler(channel()))
    adapter = make_adapter(tmp_path, opener)
    adapter._clock_ms = lambda: 9_000  # type: ignore[attr-defined]

    proof = adapter.prove_public_message_send(
        target(),
        deadline_unix_ms=30_000,
        now_unix_ms=1_000,
    )

    assert proof.observed_at_unix_ms == 9_000
    paths = [request[1] for request in opener.requests]
    assert paths[-1].endswith(f"/channels/{CHANNEL_ID}")


def test_forum_proof_uses_discord_create_posts_permission(tmp_path: Path) -> None:
    create_posts_permissions = VIEW_CHANNEL | READ_MESSAGE_HISTORY | SEND_MESSAGES
    adapter = make_adapter(
        tmp_path,
        FakeOpener(
            proof_handler(
                channel(channel_type=15),
                guild_value=guild(bot_permissions=create_posts_permissions),
            )
        ),
    )
    proof = adapter.prove_public_thread_create(
        forum_target(),
        has_initial_message=True,
        deadline_unix_ms=30_000,
        now_unix_ms=1_000,
    )
    assert proof.publicly_viewable is True
    assert proof.bot_can_view is True
    assert proof.bot_has_required_permission is True


def test_empty_text_thread_does_not_require_thread_message_permission(
    tmp_path: Path,
) -> None:
    create_only = VIEW_CHANNEL | CREATE_PUBLIC_THREADS | SEND_MESSAGES
    adapter = make_adapter(
        tmp_path,
        FakeOpener(
            proof_handler(
                channel(),
                guild_value=guild(bot_permissions=create_only),
            )
        ),
    )
    empty_proof = adapter.prove_public_thread_create(
        target(),
        has_initial_message=False,
        deadline_unix_ms=30_000,
        now_unix_ms=1_000,
    )
    assert empty_proof.bot_has_required_permission is True

    with_history_adapter = make_adapter(
        tmp_path / "with-message",
        FakeOpener(
            proof_handler(
                channel(),
                guild_value=guild(bot_permissions=create_only),
            )
        ),
    )
    with_message_proof = with_history_adapter.prove_public_thread_create(
        target(),
        has_initial_message=True,
        deadline_unix_ms=30_000,
        now_unix_ms=1_000,
    )
    assert with_message_proof.bot_has_required_permission is False


@pytest.mark.parametrize(
    ("target_channel", "guild_value", "expected_code"),
    [
        (
            channel(
                overwrites=[
                    {
                        "allow": "0",
                        "deny": str(VIEW_CHANNEL),
                        "id": GUILD_ID,
                        "type": 0,
                    },
                    {
                        "allow": str(BOT_PERMISSIONS),
                        "deny": "0",
                        "id": BOT_ROLE_ID,
                        "type": 0,
                    },
                ]
            ),
            guild(),
            DiscordRestEdgeErrorCode.TARGET_NOT_PUBLIC,
        ),
        (
            channel(),
            guild(bot_permissions=BOT_PERMISSIONS & ~SEND_MESSAGES),
            DiscordRestEdgeErrorCode.BOT_PERMISSION_REVOKED,
        ),
    ],
)
def test_mutation_boundary_reproves_public_visibility_and_bot_permission(
    tmp_path: Path,
    target_channel: dict[str, object],
    guild_value: dict[str, object],
    expected_code: DiscordRestEdgeErrorCode,
) -> None:
    methods: list[str] = []

    def handler(method: str, url: str, _body: dict[str, Any] | None) -> FakeResponse:
        methods.append(method)
        if method == "GET" and url.endswith(f"/channels/{CHANNEL_ID}"):
            return FakeResponse(target_channel)
        if method == "GET" and url.endswith("/users/@me"):
            return FakeResponse(current_user())
        if method == "GET" and url.endswith(f"/guilds/{GUILD_ID}"):
            return FakeResponse(guild_value)
        if method == "GET" and url.endswith(
            f"/guilds/{GUILD_ID}/members/{BOT_ID}"
        ):
            return FakeResponse(member())
        raise AssertionError((method, url))

    adapter = make_adapter(tmp_path, FakeOpener(handler))
    with pytest.raises(DiscordRestEdgeError) as error:
        adapter.send_public_message(
            target(),
            content="must not dispatch",
            reply_to_message_id=None,
            deadline_unix_ms=30_000,
        )
    assert error.value.code is expected_code
    assert "POST" not in methods


def test_expired_mutation_deadline_stops_before_any_discord_request(
    tmp_path: Path,
) -> None:
    opener = FakeOpener(lambda *_args: FakeResponse({}))
    adapter = make_adapter(tmp_path, opener)
    with pytest.raises(DiscordRestEdgeError) as error:
        adapter.send_public_message(
            target(),
            content="must not dispatch",
            reply_to_message_id=None,
            deadline_unix_ms=1_000,
        )
    assert error.value.code is DiscordRestEdgeErrorCode.REQUEST_DEADLINE_EXPIRED
    assert opener.requests == []


def test_deadline_expiry_during_live_proof_stops_remaining_requests(
    tmp_path: Path,
) -> None:
    now = [1_000]

    def handler(method: str, url: str, _body: dict[str, Any] | None) -> FakeResponse:
        assert method == "GET"
        assert url.endswith("/users/@me")
        now[0] = 2_000
        return FakeResponse(current_user())

    opener = FakeOpener(handler)
    adapter = make_adapter(tmp_path, opener)
    adapter._clock_ms = lambda: now[0]  # type: ignore[attr-defined]
    with pytest.raises(DiscordRestEdgeError) as error:
        adapter.prove_public_message_send(
            target(),
            deadline_unix_ms=1_500,
            now_unix_ms=1_000,
        )
    assert error.value.code is DiscordRestEdgeErrorCode.REQUEST_DEADLINE_EXPIRED
    assert len(opener.requests) == 1


def test_exact_message_send_and_live_readback_use_safe_mentions(tmp_path: Path) -> None:
    observed_content = "hello @everyone <@300000000000000003>"

    def handler(method: str, url: str, body: dict[str, Any] | None) -> FakeResponse:
        identity = default_identity_response(method, url)
        if identity is not None:
            return identity
        if method == "GET" and url.endswith(f"/channels/{CHANNEL_ID}"):
            return FakeResponse(channel())
        if method == "POST" and url.endswith(f"/channels/{CHANNEL_ID}/messages"):
            assert body == {
                "allowed_mentions": {
                    "parse": [],
                    "replied_user": False,
                    "roles": [],
                    "users": [],
                },
                "content": observed_content,
                "message_reference": {
                    "channel_id": CHANNEL_ID,
                    "fail_if_not_exists": True,
                    "guild_id": GUILD_ID,
                    "message_id": "800000000000000008",
                },
            }
            return FakeResponse(
                message(
                    content=observed_content,
                    reply_to_message_id="800000000000000008",
                    include_reply_guild=False,
                )
            )
        if method == "GET" and url.endswith(
            f"/channels/{CHANNEL_ID}/messages/{MESSAGE_ID}"
        ):
            return FakeResponse(
                message(
                    content=observed_content,
                    reply_to_message_id="800000000000000008",
                    include_reply_guild=False,
                )
            )
        raise AssertionError((method, url, body))

    opener = FakeOpener(handler)
    adapter = make_adapter(tmp_path, opener)
    accepted = adapter.send_public_message(
        target(),
        content=observed_content,
        reply_to_message_id="800000000000000008",
        deadline_unix_ms=30_000,
    )
    readback = adapter.read_public_message(
        target(),
        operation=DiscordEdgeOperation.PUBLIC_MESSAGE_SEND,
        message_id=MESSAGE_ID,
        expected_reply_to_message_id="800000000000000008",
    )
    assert accepted.discord_object_id == MESSAGE_ID
    assert accepted.bot_user_id == BOT_ID
    assert readback.operation is DiscordEdgeOperation.PUBLIC_MESSAGE_SEND
    assert readback.content == observed_content
    assert readback.author_user_id == BOT_ID
    assert all(request[3] == f"Bot {TOKEN}" for request in opener.requests)


def test_reply_send_rejects_missing_discord_reference(tmp_path: Path) -> None:
    def handler(method: str, url: str, _body: dict[str, Any] | None) -> FakeResponse:
        identity = default_identity_response(method, url)
        if identity is not None:
            return identity
        if method == "GET" and url.endswith(f"/channels/{CHANNEL_ID}"):
            return FakeResponse(channel())
        if method == "POST" and url.endswith(f"/channels/{CHANNEL_ID}/messages"):
            return FakeResponse(message(content="reply"))
        raise AssertionError((method, url))

    adapter = make_adapter(tmp_path, FakeOpener(handler))
    with pytest.raises(DiscordRestEdgeError) as error:
        adapter.send_public_message(
            target(),
            content="reply",
            reply_to_message_id="800000000000000008",
            deadline_unix_ms=30_000,
        )
    assert error.value.code is DiscordRestEdgeErrorCode.MUTATION_BINDING_MISMATCH


def test_readback_needs_read_permissions_not_revoked_send_permission(
    tmp_path: Path,
) -> None:
    sent = False

    def handler(method: str, url: str, _body: dict[str, Any] | None) -> FakeResponse:
        nonlocal sent
        if method == "GET" and url.endswith("/users/@me"):
            return FakeResponse(current_user())
        if method == "GET" and url.endswith(f"/guilds/{GUILD_ID}"):
            permissions = (
                VIEW_CHANNEL | READ_MESSAGE_HISTORY if sent else BOT_PERMISSIONS
            )
            return FakeResponse(guild(bot_permissions=permissions))
        if method == "GET" and url.endswith(f"/guilds/{GUILD_ID}/members/{BOT_ID}"):
            return FakeResponse(member())
        if method == "GET" and url.endswith(f"/channels/{CHANNEL_ID}"):
            return FakeResponse(channel())
        if method == "POST" and url.endswith(f"/channels/{CHANNEL_ID}/messages"):
            sent = True
            return FakeResponse(message(content="exact"))
        if method == "GET" and url.endswith(
            f"/channels/{CHANNEL_ID}/messages/{MESSAGE_ID}"
        ):
            return FakeResponse(message(content="exact"))
        raise AssertionError((method, url))

    adapter = make_adapter(tmp_path, FakeOpener(handler))
    adapter.send_public_message(
        target(),
        content="exact",
        reply_to_message_id=None,
        deadline_unix_ms=30_000,
    )
    readback = adapter.read_public_message(
        target(),
        operation=DiscordEdgeOperation.PUBLIC_MESSAGE_SEND,
        message_id=MESSAGE_ID,
        expected_reply_to_message_id=None,
    )
    assert readback.content == "exact"


def test_message_readback_rejects_untyped_operation_before_http(tmp_path: Path) -> None:
    opener = FakeOpener(lambda *_args: FakeResponse({}))
    adapter = make_adapter(tmp_path, opener)
    with pytest.raises(TypeError):
        adapter.read_public_message(
            target(),
            operation="public.message.send",  # type: ignore[arg-type]
            message_id=MESSAGE_ID,
            expected_reply_to_message_id=None,
        )
    assert opener.requests == []


def test_exact_message_edit_requires_bot_author_and_preserves_operation(
    tmp_path: Path,
) -> None:
    state = {"content": "before"}

    def handler(method: str, url: str, body: dict[str, Any] | None) -> FakeResponse:
        identity = default_identity_response(method, url)
        if identity is not None:
            return identity
        if method == "GET" and url.endswith(f"/channels/{CHANNEL_ID}"):
            return FakeResponse(channel())
        if method == "GET" and url.endswith(
            f"/channels/{CHANNEL_ID}/messages/{MESSAGE_ID}"
        ):
            return FakeResponse(message(content=state["content"]))
        if method == "PATCH" and url.endswith(
            f"/channels/{CHANNEL_ID}/messages/{MESSAGE_ID}"
        ):
            assert body is not None
            state["content"] = str(body["content"])
            return FakeResponse(message(content=state["content"]))
        raise AssertionError((method, url, body))

    adapter = make_adapter(tmp_path, FakeOpener(handler))
    accepted = adapter.edit_public_message(
        target(),
        message_id=MESSAGE_ID,
        content="after",
        deadline_unix_ms=30_000,
    )
    readback = adapter.read_public_message(
        target(),
        operation=DiscordEdgeOperation.PUBLIC_MESSAGE_EDIT,
        message_id=MESSAGE_ID,
        expected_reply_to_message_id=None,
    )
    assert accepted.discord_object_id == MESSAGE_ID
    assert readback.operation is DiscordEdgeOperation.PUBLIC_MESSAGE_EDIT
    assert readback.content == "after"


def test_message_edit_blocks_non_bot_author_before_patch(tmp_path: Path) -> None:
    methods: list[str] = []

    def handler(method: str, url: str, _body: dict[str, Any] | None) -> FakeResponse:
        methods.append(method)
        identity = default_identity_response(method, url)
        if identity is not None:
            return identity
        if url.endswith(f"/channels/{CHANNEL_ID}"):
            return FakeResponse(channel())
        if url.endswith(f"/channels/{CHANNEL_ID}/messages/{MESSAGE_ID}"):
            return FakeResponse(
                message(author_id="999999999999999999", content="before")
            )
        raise AssertionError((method, url))

    adapter = make_adapter(tmp_path, FakeOpener(handler))
    with pytest.raises(DiscordRestEdgeError) as error:
        adapter.edit_public_message(
            target(),
            message_id=MESSAGE_ID,
            content="after",
            deadline_unix_ms=30_000,
        )
    assert error.value.code is DiscordRestEdgeErrorCode.BOT_IDENTITY_MISMATCH
    assert "PATCH" not in methods


def test_text_thread_initial_content_fails_before_any_discord_request(
    tmp_path: Path,
) -> None:
    opener = FakeOpener(lambda *_args: FakeResponse({}))
    adapter = make_adapter(tmp_path, opener)
    with pytest.raises(DiscordRestEdgeError) as error:
        adapter.create_public_thread(
            target(),
            name="Exact thread",
            initial_message="must be separately receipted",
            auto_archive_minutes=1440,
            deadline_unix_ms=30_000,
        )
    assert error.value.code is DiscordRestEdgeErrorCode.MUTATION_BINDING_MISMATCH
    assert opener.requests == []


def test_empty_public_thread_readback_is_restart_safe(tmp_path: Path) -> None:
    parent = channel()
    thread = channel(
        channel_id=THREAD_ID,
        channel_type=11,
        parent_id=CHANNEL_ID,
        name="Empty thread",
    )

    def handler(method: str, url: str, body: dict[str, Any] | None) -> FakeResponse:
        identity = default_identity_response(method, url)
        if identity is not None:
            return identity
        if method == "GET" and url.endswith(f"/channels/{CHANNEL_ID}"):
            return FakeResponse(parent)
        if method == "GET" and url.endswith(f"/channels/{THREAD_ID}"):
            return FakeResponse(thread)
        if method == "POST" and url.endswith(f"/channels/{CHANNEL_ID}/threads"):
            assert body == {"name": "Empty thread", "type": 11}
            return FakeResponse(thread)
        raise AssertionError((method, url, body))

    adapter = make_adapter(tmp_path, FakeOpener(handler))
    adapter.create_public_thread(
        target(),
        name="Empty thread",
        initial_message=None,
        auto_archive_minutes=None,
        deadline_unix_ms=30_000,
    )
    restarted = DiscordRestEdgeAdapter.from_credential_file(
        tmp_path / "credentials" / "discord-token",
        credentials_directory=tmp_path / "credentials",
        expected_owner_uid=os.getuid(),
        timeout_seconds=1.0,
        _opener=adapter._api._opener,  # type: ignore[attr-defined]
        _sleeper=lambda _seconds: None,
        _clock_ms=lambda: 1_000,
    )
    readback = restarted.read_created_public_thread(
        target(),
        thread_id=THREAD_ID,
        expected_content="",
    )
    assert readback.content == ""
    assert readback.author_user_id == BOT_ID


def test_exact_forum_thread_create_embeds_safe_initial_message(tmp_path: Path) -> None:
    forum = channel(channel_type=15)
    thread = channel(
        channel_id=THREAD_ID,
        channel_type=11,
        parent_id=CHANNEL_ID,
        name="Exact forum post",
    )
    thread["message"] = message(
        message_id=THREAD_ID,
        channel_id=THREAD_ID,
        content="opening",
    )

    def handler(method: str, url: str, body: dict[str, Any] | None) -> FakeResponse:
        identity = default_identity_response(method, url)
        if identity is not None:
            return identity
        if method == "GET" and url.endswith(f"/channels/{CHANNEL_ID}"):
            return FakeResponse(forum)
        if method == "GET" and url.endswith(f"/channels/{THREAD_ID}"):
            return FakeResponse(thread)
        if method == "POST" and url.endswith(
            f"/channels/{CHANNEL_ID}/threads?use_nested_fields=1"
        ):
            assert body == {
                "auto_archive_duration": 1440,
                "message": {
                    "allowed_mentions": {
                        "parse": [],
                        "replied_user": False,
                        "roles": [],
                        "users": [],
                    },
                    "content": "opening",
                },
                "name": "Exact forum post",
                "type": 11,
            }
            return FakeResponse(thread)
        if method == "GET" and url.endswith(
            f"/channels/{THREAD_ID}/messages/{THREAD_ID}"
        ):
            return FakeResponse(thread["message"])
        raise AssertionError((method, url, body))

    adapter = make_adapter(tmp_path, FakeOpener(handler))
    accepted = adapter.create_public_thread(
        forum_target(),
        name="Exact forum post",
        initial_message="opening",
        auto_archive_minutes=1440,
        deadline_unix_ms=30_000,
    )
    readback = adapter.read_created_public_thread(
        forum_target(),
        thread_id=THREAD_ID,
        expected_content="opening",
    )
    assert accepted.discord_object_id == THREAD_ID
    assert readback.content == "opening"
    assert readback.author_user_id == BOT_ID


def test_credential_mode_link_and_symlink_contract(tmp_path: Path) -> None:
    credentials = tmp_path / "credentials"
    credentials.mkdir(mode=0o700)
    credential = credentials / "discord-token"
    credential.write_text(TOKEN, encoding="ascii")
    credential.chmod(0o600)
    opener = FakeOpener(lambda *_args: FakeResponse({}))
    with pytest.raises(DiscordRestEdgeError) as mode_error:
        DiscordRestEdgeAdapter.from_credential_file(
            credential,
            credentials_directory=credentials,
            expected_owner_uid=os.getuid(),
            _opener=opener,
        )
    assert mode_error.value.code is DiscordRestEdgeErrorCode.CREDENTIAL_INVALID

    credential.chmod(0o400)
    hardlink = credentials / "second-link"
    os.link(credential, hardlink)
    with pytest.raises(DiscordRestEdgeError) as link_error:
        DiscordRestEdgeAdapter.from_credential_file(
            credential,
            credentials_directory=credentials,
            expected_owner_uid=os.getuid(),
            _opener=opener,
        )
    assert link_error.value.code is DiscordRestEdgeErrorCode.CREDENTIAL_INVALID

    hardlink.unlink()
    symlink = credentials / "symlink-token"
    symlink.symlink_to(credential)
    with pytest.raises(DiscordRestEdgeError) as symlink_error:
        DiscordRestEdgeAdapter.from_credential_file(
            symlink,
            credentials_directory=credentials,
            expected_owner_uid=os.getuid(),
            _opener=opener,
        )
    assert symlink_error.value.code is DiscordRestEdgeErrorCode.CREDENTIAL_INVALID


def test_default_http_client_disables_environment_proxies(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:65534")
    credentials = tmp_path / "credentials"
    credentials.mkdir(mode=0o700)
    credential = credentials / "discord-token"
    credential.write_text(TOKEN, encoding="ascii")
    credential.chmod(0o400)
    adapter = DiscordRestEdgeAdapter.from_credential_file(
        credential,
        credentials_directory=credentials,
        expected_owner_uid=os.getuid(),
    )
    opener = adapter._api._opener  # type: ignore[attr-defined]
    assert opener.uses_environment_proxies is False
    adapter.close()


def test_total_deadline_returns_without_waiting_for_stalled_executor() -> None:
    opener = discord_rest_edge._AiohttpTotalDeadlineOpener()

    async def stalled_exchange(
        _request: urllib.request.Request,
        *,
        timeout: float,
    ) -> object:
        del timeout
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, time.sleep, 0.4)
        raise AssertionError("cancelled exchange must not complete")

    opener._exchange = stalled_exchange  # type: ignore[method-assign]
    request = urllib.request.Request(
        "https://discord.com/api/v10/users/@me",
        method="GET",
    )
    started = time.monotonic()
    with pytest.raises(urllib.error.URLError):
        opener.open(request, timeout=0.05)
    elapsed = time.monotonic() - started
    assert elapsed < 0.2
    time.sleep(0.45)
    opener.close()


def test_token_never_appears_in_errors_or_adapter_repr(tmp_path: Path) -> None:
    def rejected(method: str, url: str, _body: object) -> FakeResponse:
        request = urllib.request.Request(url, method=method)
        error_body = json.dumps({"message": f"reflected {TOKEN}"}).encode()
        raise urllib.error.HTTPError(
            url,
            401,
            f"reflected {TOKEN}",
            {"Content-Length": str(len(error_body))},
            io.BytesIO(error_body),
        )

    adapter = make_adapter(tmp_path, FakeOpener(rejected))
    with pytest.raises(DiscordRestEdgeError) as error:
        adapter.prove_public_message_send(
            target(),
            deadline_unix_ms=30_000,
            now_unix_ms=1_000,
        )
    assert TOKEN not in str(error.value)
    assert TOKEN not in repr(error.value)
    assert TOKEN not in repr(adapter)
    assert error.value.__context__ is None
    assert error.value.__cause__ is None
    assert TOKEN not in "".join(
        traceback.format_exception(
            type(error.value),
            error.value,
            error.value.__traceback__,
        )
    )
    assert error.value.code is DiscordRestEdgeErrorCode.API_REJECTED


@pytest.mark.parametrize(
    ("body", "expected_code"),
    [
        (
            b'{"id":"1","id":"2"}',
            DiscordRestEdgeErrorCode.RESPONSE_INVALID,
        ),
        (
            b"{" + b'"padding":"' + b"x" * (256 * 1024) + b'"}',
            DiscordRestEdgeErrorCode.RESPONSE_TOO_LARGE,
        ),
    ],
)
def test_malformed_and_oversized_discord_responses_are_fail_closed(
    tmp_path: Path,
    body: bytes,
    expected_code: DiscordRestEdgeErrorCode,
) -> None:
    adapter = make_adapter(
        tmp_path,
        FakeOpener(lambda *_args: FakeResponse(body=body)),
    )
    with pytest.raises(DiscordRestEdgeError) as error:
        adapter.prove_public_message_send(
            target(),
            deadline_unix_ms=30_000,
            now_unix_ms=1_000,
        )
    assert error.value.code is expected_code


def test_rate_limit_retry_is_bounded_and_secret_free(tmp_path: Path) -> None:
    attempts = 0
    sleeps: list[float] = []

    class RateLimitOpener:
        def open(self, request: Any, timeout: float) -> FakeResponse:
            nonlocal attempts
            del timeout
            attempts += 1
            body = json.dumps(
                {"message": TOKEN, "retry_after": 0.01}
            ).encode("utf-8")
            raise urllib.error.HTTPError(
                request.full_url,
                429,
                TOKEN,
                {"Content-Length": str(len(body))},
                io.BytesIO(body),
            )

    credentials = tmp_path / "credentials"
    credentials.mkdir(mode=0o700)
    credential = credentials / "discord-token"
    credential.write_text(TOKEN, encoding="ascii")
    credential.chmod(0o400)
    adapter = DiscordRestEdgeAdapter.from_credential_file(
        credential,
        credentials_directory=credentials,
        expected_owner_uid=os.getuid(),
        timeout_seconds=1.0,
        _opener=RateLimitOpener(),
        _sleeper=sleeps.append,
        _clock_ms=lambda: 1_000,
    )
    with pytest.raises(DiscordRestEdgeError) as error:
        adapter.prove_public_message_send(
            target(),
            deadline_unix_ms=30_000,
            now_unix_ms=1_000,
        )
    assert attempts == 3
    assert sleeps == [0.01, 0.01]
    assert error.value.code is DiscordRestEdgeErrorCode.API_RATE_LIMITED
    assert TOKEN not in str(error.value)


@pytest.mark.parametrize(
    "body",
    [
        b"[" * 100 + b"0" + b"]" * 100,
        b'{"retry_after":' + b"9" * 400 + b"}",
    ],
)
def test_adversarial_json_complexity_and_numbers_never_escape_raw_errors(
    tmp_path: Path,
    body: bytes,
) -> None:
    adapter = make_adapter(
        tmp_path,
        FakeOpener(lambda *_args: FakeResponse(body=body)),
    )
    with pytest.raises(DiscordRestEdgeError) as error:
        adapter.prove_public_message_send(
            target(),
            deadline_unix_ms=30_000,
            now_unix_ms=1_000,
        )
    assert error.value.code is DiscordRestEdgeErrorCode.RESPONSE_INVALID


def test_oversized_permission_decimal_is_normalized(tmp_path: Path) -> None:
    adapter = make_adapter(
        tmp_path,
        FakeOpener(
            proof_handler(
                channel(),
                guild_value=guild(bot_permissions=int("9" * 400)),
            )
        ),
    )
    with pytest.raises(DiscordRestEdgeError) as error:
        adapter.prove_public_message_send(
            target(),
            deadline_unix_ms=30_000,
            now_unix_ms=1_000,
        )
    assert error.value.code is DiscordRestEdgeErrorCode.RESPONSE_INVALID
