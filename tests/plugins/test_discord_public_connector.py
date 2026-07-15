from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from types import SimpleNamespace
import threading

import pytest

from gateway.discord_connector_protocol import (
    DiscordConnectorHistoryAuthority,
    DiscordConnectorTargetType,
)
from plugins.platforms.discord import public_connector as public_connector_module
from plugins.platforms.discord.public_connector import (
    DiscordPublicConnectorClient,
    DiscordPublicConnectorError,
    DiscordPublicConnectorPolicy,
)


class _AliveThread:
    @staticmethod
    def is_alive() -> bool:
        return True


class _RunningLoop:
    @staticmethod
    def is_running() -> bool:
        return True


class _Channel:
    def __init__(
        self,
        channel_id: int,
        *,
        guild=None,
        type_value: int = 0,
        parent=None,
        owner_id: int | None = None,
        everyone_view: bool = True,
        everyone_history: bool = True,
        bot_send: bool = True,
        bot_history: bool = True,
        bot_view: bool = True,
        requester_send: bool = True,
        requester_history: bool = True,
        requester_view: bool = True,
    ) -> None:
        self.id = channel_id
        self.guild = guild
        self.type = SimpleNamespace(
            value=type_value,
            name={0: "text", 11: "public_thread", 12: "private_thread"}.get(
                type_value, "other"
            ),
        )
        self.parent = parent
        self.parent_id = getattr(parent, "id", None)
        self.owner_id = owner_id
        self._everyone_view = everyone_view
        self._everyone_history = everyone_history
        self._bot_send = bot_send
        self._bot_history = bot_history
        self._bot_view = bot_view
        self._requester_send = requester_send
        self._requester_history = requester_history
        self._requester_view = requester_view

    def permissions_for(self, role):
        if role is self.guild.default_role:
            return SimpleNamespace(
                view_channel=self._everyone_view,
                read_message_history=self._everyone_history,
            )
        if role is self.guild.me:
            return SimpleNamespace(
                view_channel=self._bot_view,
                send_messages=self._bot_send,
                send_messages_in_threads=self._bot_send,
                read_message_history=self._bot_history,
            )
        return SimpleNamespace(
            view_channel=self._requester_view,
            send_messages=self._requester_send,
            send_messages_in_threads=self._requester_send,
            read_message_history=self._requester_history,
        )


def _policy() -> DiscordPublicConnectorPolicy:
    return DiscordPublicConnectorPolicy.build(
        allowed_guild_ids=["100"],
        allowed_channel_ids=["200"],
        allowed_user_ids=["400"],
        allowed_role_ids=["500"],
        require_mention=False,
    )


def _guild_acl_policy() -> DiscordPublicConnectorPolicy:
    return DiscordPublicConnectorPolicy.build(
        allowed_guild_ids=["100"],
        allowed_channel_ids=["200"],
        allowed_user_ids=[],
        allowed_role_ids=[],
        free_response_channel_ids=["200"],
        public_only=False,
        author_policy="guild_acl",
        require_mention=True,
        thread_require_mention=False,
        reviewed_cron_history_targets={"e62f55ca93ca": ["201"]},
    )


def _guild(*, members=()):
    default_role = object()
    member_map = {int(member.id): member for member in members}
    return SimpleNamespace(
        id=100,
        default_role=default_role,
        me=object(),
        get_member=lambda user_id: member_map.get(int(user_id)),
    )


def test_only_public_allowed_guild_channels_and_threads_are_proven() -> None:
    policy = _policy()
    requester = SimpleNamespace(id=400)
    guild = _guild(members=[requester])
    public = _Channel(200, guild=guild)
    assert policy.prove_target(public, bot_user=object()).target_type is (
        DiscordConnectorTargetType.PUBLIC_GUILD_CHANNEL
    )

    public_thread = _Channel(201, guild=guild, type_value=11, parent=public)
    target = policy.prove_target(public_thread, bot_user=object())
    assert target.target_type is DiscordConnectorTargetType.PUBLIC_GUILD_THREAD
    assert target.parent_channel_id == "200"

    with pytest.raises(DiscordPublicConnectorError, match="target_not_allowed"):
        policy.prove_target(_Channel(200, guild=None), bot_user=object())
    with pytest.raises(DiscordPublicConnectorError, match="private_target_forbidden"):
        policy.prove_target(
            _Channel(201, guild=guild, type_value=12, parent=public),
            bot_user=object(),
        )
    with pytest.raises(DiscordPublicConnectorError, match="target_not_public"):
        policy.prove_target(
            _Channel(200, guild=guild, everyone_view=False),
            bot_user=object(),
        )


def test_event_policy_is_permission_only_and_does_not_classify_text() -> None:
    policy = _policy()
    requester = SimpleNamespace(id=400)
    guild = _guild(members=[requester])
    channel = _Channel(200, guild=guild)
    message = SimpleNamespace(
        id=300,
        channel=channel,
        author=SimpleNamespace(id=400, bot=False, display_name="Emo"),
        type=SimpleNamespace(value=0, name="default"),
        content="deploy delete keyword dispatcher classifier — GPT decides meaning",
        reference=None,
        created_at=datetime.now(timezone.utc),
    )
    event = policy.event_from_message(message, bot_user=object())
    assert event.content == message.content

    message.author = SimpleNamespace(id=401, bot=False, display_name="Other")
    with pytest.raises(DiscordPublicConnectorError, match="author_not_allowed"):
        policy.event_from_message(message, bot_user=object())

    message.author = SimpleNamespace(
        id=402,
        bot=False,
        display_name="Role teammate",
        roles=[SimpleNamespace(id=500)],
    )
    assert policy.event_from_message(message, bot_user=object()).author_id == "402"

    message.author = SimpleNamespace(
        id=403,
        bot=True,
        display_name="Role bot",
        roles=[SimpleNamespace(id=500)],
    )
    with pytest.raises(DiscordPublicConnectorError, match="bot_author_forbidden"):
        policy.event_from_message(message, bot_user=object())


def test_history_proof_blocks_dm_private_and_non_public_history() -> None:
    policy = _policy()
    guild = _guild()
    public = _Channel(200, guild=guild)
    assert policy.prove_history_target(
        public, bot_user=object()
    ).target_type is DiscordConnectorTargetType.PUBLIC_GUILD_CHANNEL

    with pytest.raises(DiscordPublicConnectorError, match="target_not_allowed"):
        policy.prove_history_target(_Channel(200, guild=None), bot_user=object())
    with pytest.raises(DiscordPublicConnectorError, match="private_target_forbidden"):
        policy.prove_history_target(
            _Channel(201, guild=guild, type_value=12, parent=public),
            bot_user=object(),
        )
    with pytest.raises(
        DiscordPublicConnectorError, match="target_history_not_public"
    ):
        policy.prove_history_target(
            _Channel(200, guild=guild, everyone_history=False),
            bot_user=object(),
        )
    with pytest.raises(
        DiscordPublicConnectorError, match="bot_cannot_read_target_history"
    ):
        policy.prove_history_target(
            _Channel(200, guild=guild, bot_history=False),
            bot_user=object(),
        )


def test_history_authority_requires_requester_live_acl_not_only_bot_acl() -> None:
    policy = _guild_acl_policy()
    owner = SimpleNamespace(id=400)
    owner_guild = _guild(members=[owner])
    owner_channel = _Channel(
        200,
        guild=owner_guild,
        everyone_view=False,
        everyone_history=False,
    )
    owner_target = policy.prove_history_target(owner_channel, bot_user=object())
    policy.prove_history_authority(
        owner_channel,
        target=owner_target,
        authority=DiscordConnectorHistoryAuthority.authenticated_user("400"),
        requester_member=owner,
    )

    team_a = SimpleNamespace(id=401)
    team_guild = _guild(members=[team_a])
    bot_only_channel = _Channel(
        200,
        guild=team_guild,
        everyone_view=False,
        everyone_history=False,
        bot_view=True,
        bot_history=True,
        bot_send=True,
        requester_view=False,
        requester_history=False,
    )
    bot_only_target = policy.prove_history_target(
        bot_only_channel,
        bot_user=object(),
    )
    with pytest.raises(
        DiscordPublicConnectorError,
        match="history_requester_cannot_read",
    ):
        policy.prove_history_authority(
            bot_only_channel,
            target=bot_only_target,
            authority=DiscordConnectorHistoryAuthority.authenticated_user("401"),
            requester_member=team_a,
        )


def test_canary_history_is_exact_requester_and_cron_is_exact_job_target() -> None:
    canary = _policy()
    guild = _guild()
    channel = _Channel(200, guild=guild)
    target = canary.prove_history_target(channel, bot_user=object())
    with pytest.raises(
        DiscordPublicConnectorError,
        match="history_requester_not_allowed",
    ):
        canary.prove_history_authority(
            channel,
            target=target,
            authority=DiscordConnectorHistoryAuthority.authenticated_user("401"),
            requester_member=SimpleNamespace(id=401),
        )

    production = _guild_acl_policy()
    parent = _Channel(200, guild=guild, everyone_view=False)
    thread = _Channel(201, guild=guild, type_value=11, parent=parent)
    thread_target = production.prove_history_target(thread, bot_user=object())
    production.prove_history_authority(
        thread,
        target=thread_target,
        authority=DiscordConnectorHistoryAuthority.reviewed_cron("e62f55ca93ca"),
    )
    with pytest.raises(
        DiscordPublicConnectorError,
        match="cron_history_target_not_reviewed",
    ):
        production.prove_history_authority(
            parent,
            target=production.prove_history_target(parent, bot_user=object()),
            authority=DiscordConnectorHistoryAuthority.reviewed_cron(
                "e62f55ca93ca"
            ),
        )
    with pytest.raises(
        DiscordPublicConnectorError,
        match="cron_history_target_not_reviewed",
    ):
        production.prove_history_authority(
            thread,
            target=thread_target,
            authority=DiscordConnectorHistoryAuthority.reviewed_cron(
                "deadbeef0000"
            ),
        )


def test_history_fetch_is_bounded_chronological_and_reports_truncation(
    monkeypatch,
) -> None:
    class _Object:
        def __init__(self, *, id):
            self.id = id

    monkeypatch.setattr(
        public_connector_module,
        "discord",
        SimpleNamespace(Object=_Object),
    )
    requester = SimpleNamespace(id=400)
    guild = _guild(members=[requester])
    channel = _Channel(200, guild=guild)
    messages = [
        SimpleNamespace(
            id=message_id,
            channel=channel,
            author=SimpleNamespace(id=400, bot=False, display_name="Emo"),
            content=content,
            created_at=datetime.fromtimestamp(timestamp, timezone.utc),
            reference=None,
        )
        for message_id, content, timestamp in (
            (303, "latest", 3),
            (302, "x" * 2_100, 2),
            (301, "older", 1),
        )
    ]

    def _history(**kwargs):
        assert kwargs == {"limit": 3}

        async def _items():
            for item in messages:
                yield item

        return _items()

    channel.history = _history
    connector = object.__new__(DiscordPublicConnectorClient)
    connector.policy = _policy()
    connector._client = SimpleNamespace(user=object())

    async def _resolve(_channel_id):
        return channel

    connector._fetch_fresh_history_channel = _resolve
    connector._submit = lambda coroutine, **_kwargs: asyncio.run(coroutine)
    page = connector.fetch_guild_history(
        "200",
        limit=2,
        before_message_id=None,
        after_message_id=None,
        authority=DiscordConnectorHistoryAuthority.authenticated_user("400"),
    )

    assert [item.message_id for item in page.messages] == ["302", "303"]
    assert page.messages[0].content_truncated is True
    assert len(page.messages[0].content) == 2_000
    assert page.has_more is True

    forward_messages = list(reversed(messages))

    def _history_after(**kwargs):
        assert kwargs["limit"] == 3
        assert kwargs["after"].id == 300
        assert kwargs["oldest_first"] is True

        async def _items():
            for item in forward_messages:
                yield item

        return _items()

    channel.history = _history_after
    forward = connector.fetch_guild_history(
        "200",
        limit=2,
        before_message_id=None,
        after_message_id="300",
        authority=DiscordConnectorHistoryAuthority.authenticated_user("400"),
    )
    assert [item.message_id for item in forward.messages] == ["301", "302"]
    assert forward.after_message_id == "300"
    assert forward.has_more is True


def test_history_fetch_rechecks_rest_fetched_acl_before_releasing_page() -> None:
    requester = SimpleNamespace(id=400)
    guild = _guild(members=[requester])
    initially_allowed = _Channel(200, guild=guild)
    revoked = _Channel(
        200,
        guild=guild,
        requester_view=False,
        requester_history=False,
    )

    def _history(**kwargs):
        assert kwargs == {"limit": 2}

        async def _items():
            if False:
                yield None

        return _items()

    initially_allowed.history = _history
    fetched = [initially_allowed, revoked]

    async def _fetch_channel(channel_id):
        assert channel_id == 200
        return fetched.pop(0)

    connector = object.__new__(DiscordPublicConnectorClient)
    connector.policy = _policy()
    connector._client = SimpleNamespace(
        user=object(),
        fetch_channel=_fetch_channel,
    )
    connector._submit = lambda coroutine, **_kwargs: asyncio.run(coroutine)

    with pytest.raises(
        DiscordPublicConnectorError,
        match="history_requester_cannot_read",
    ):
        connector.fetch_guild_history(
            "200",
            limit=1,
            before_message_id=None,
            after_message_id=None,
            authority=DiscordConnectorHistoryAuthority.authenticated_user("400"),
        )
    assert fetched == []


def test_fresh_thread_history_acl_uses_rest_fetched_parent_not_gateway_cache() -> None:
    guild = _guild()
    cached_parent = _Channel(
        200,
        guild=guild,
        everyone_view=False,
        requester_view=False,
    )
    fresh_parent = _Channel(200, guild=guild)
    thread = _Channel(201, guild=guild, type_value=11, parent=cached_parent)
    calls = []

    async def _fetch_channel(channel_id):
        calls.append(channel_id)
        return thread if channel_id == 201 else fresh_parent

    connector = object.__new__(DiscordPublicConnectorClient)
    connector.policy = _policy()
    connector._client = SimpleNamespace(fetch_channel=_fetch_channel)

    view = asyncio.run(connector._fetch_fresh_history_channel("201"))
    assert calls == [201, 200]
    assert view.parent is fresh_parent
    assert view.permissions_for(guild.default_role).view_channel is True


def test_structured_mention_and_connector_thread_followup_policy() -> None:
    policy = DiscordPublicConnectorPolicy.build(
        allowed_guild_ids=["100"],
        allowed_channel_ids=["200"],
        allowed_user_ids=["400"],
        allowed_role_ids=["500"],
        require_mention=True,
        auto_thread=True,
        thread_require_mention=False,
    )
    guild = _guild()
    bot = SimpleNamespace(id=900)
    public = _Channel(200, guild=guild)
    message = SimpleNamespace(
        id=300,
        channel=public,
        author=SimpleNamespace(id=400, bot=False, display_name="Emo"),
        mentions=[],
        raw_mentions=[],
        type=SimpleNamespace(value=0, name="default"),
        content="ordinary public chatter",
        reference=None,
        created_at=datetime.now(timezone.utc),
    )
    with pytest.raises(DiscordPublicConnectorError, match="mention_required"):
        policy.event_from_message(message, bot_user=bot)

    message.mentions = [bot]
    assert policy.event_from_message(message, bot_user=bot).target.channel_id == "200"

    thread = _Channel(201, guild=guild, type_value=11, parent=public)
    message.channel = thread
    message.mentions = []
    assert policy.event_from_message(
        message,
        bot_user=bot,
        connector_thread_ids=frozenset({"201"}),
    ).target.channel_id == "201"

    # Connector-created public threads remain usable after a connector
    # restart: Discord's structured thread owner is the durable proof.
    message.channel = _Channel(
        202,
        guild=guild,
        type_value=11,
        parent=public,
        owner_id=900,
    )
    assert policy.event_from_message(message, bot_user=bot).target.channel_id == "202"

    message.channel = _Channel(
        203,
        guild=guild,
        type_value=11,
        parent=public,
        owner_id=901,
    )
    assert policy.event_from_message(message, bot_user=bot).target.channel_id == "203"


def test_auto_thread_failure_falls_back_to_exact_channel_event_once() -> None:
    guild = _guild()
    channel = _Channel(
        200,
        guild=guild,
        everyone_view=False,
        everyone_history=False,
    )

    async def _forbidden_create_thread(**_kwargs):
        raise RuntimeError("discord_forbidden")

    message = SimpleNamespace(
        id=300,
        channel=channel,
        author=SimpleNamespace(id=400, bot=False, display_name="Emo", roles=[]),
        mentions=[],
        raw_mentions=[],
        type=SimpleNamespace(value=0, name="default"),
        content="please handle the task",
        reference=None,
        created_at=datetime.now(timezone.utc),
        create_thread=_forbidden_create_thread,
    )
    received = []
    connector = object.__new__(DiscordPublicConnectorClient)
    connector.policy = _guild_acl_policy()
    connector._client = SimpleNamespace(user=SimpleNamespace(id=900))
    connector._created_thread_ids = set()
    connector.event_sink = received.append

    asyncio.run(connector._handle_inbound_message(message))

    assert len(received) == 1
    assert received[0].event_id == "300"
    assert received[0].target.target_type is DiscordConnectorTargetType.GUILD_CHANNEL
    assert received[0].target.channel_id == "200"
    assert connector._created_thread_ids == set()


def test_live_readiness_proves_targets_and_disconnect_is_terminal() -> None:
    connector = object.__new__(DiscordPublicConnectorClient)
    connector.policy = _policy()
    connector._thread = _AliveThread()
    connector._loop = _RunningLoop()
    connector._ready = threading.Event()
    connector._ready.set()
    connector._stopped = threading.Event()
    connector._closing = threading.Event()
    connector._health_failed = threading.Event()
    connector._client = SimpleNamespace(
        user=SimpleNamespace(id=500),
        intents=SimpleNamespace(
            guilds=True,
            guild_messages=True,
            message_content=True,
            dm_messages=False,
        ),
        is_ready=lambda: True,
        is_closed=lambda: False,
    )
    connector.prove_public_target = lambda channel_id: connector.policy.prove_target(
        _Channel(int(channel_id), guild=_guild()), bot_user=object()
    )
    connector.prove_public_history_target = (
        lambda channel_id: connector.policy.prove_history_target(
            _Channel(int(channel_id), guild=_guild()), bot_user=object()
        )
    )

    identity = connector.readiness_identity()
    assert identity == {
        "discord_gateway_ready": True,
        "bot_user_id": "500",
        "intents": ["guilds", "guild_messages", "message_content"],
        "dm_messages": False,
        "require_mention": False,
        "auto_thread": True,
        "thread_require_mention": False,
        "public_only": True,
        "author_policy": "exact_ids_or_roles",
        "free_response_channel_ids": [],
        "allowed_user_ids": ["400"],
        "allowed_role_ids": ["500"],
        "allowed_channel_ids": ["200"],
        "reviewed_cron_history_targets_sha256": (
            connector.policy.reviewed_cron_history_targets_sha256
        ),
        "public_target_proofs": [
            {
                "target_type": "public_guild_channel",
                "guild_id": "100",
                "channel_id": "200",
            }
        ],
    }

    connector._mark_disconnected()
    assert connector.wait_for_health_failure(0) is True
    with pytest.raises(DiscordPublicConnectorError, match="discord_not_ready"):
        connector.readiness_identity()


def test_guild_acl_mode_accepts_private_channel_and_public_thread_with_live_permissions() -> None:
    policy = _guild_acl_policy()
    guild = _guild()
    private_channel = _Channel(
        200,
        guild=guild,
        everyone_view=False,
        everyone_history=False,
    )
    target = policy.prove_target(private_channel, bot_user=object())
    assert target.target_type is DiscordConnectorTargetType.GUILD_CHANNEL

    # Registry/baseline IDs are not exhaustive authorization in guild_acl
    # mode; Discord's live guild ACL is the authority.
    future_channel = _Channel(299, guild=guild, everyone_view=False)
    assert policy.prove_target(
        future_channel, bot_user=object()
    ).channel_id == "299"

    thread = _Channel(201, guild=guild, type_value=11, parent=private_channel)
    thread_target = policy.prove_target(thread, bot_user=object())
    assert thread_target.target_type is DiscordConnectorTargetType.GUILD_THREAD
    assert thread_target.parent_channel_id == "200"


def test_guild_acl_mode_rejects_wrong_scope_dm_private_thread_and_missing_bot_permission() -> None:
    policy = _guild_acl_policy()
    guild = _guild()
    parent = _Channel(200, guild=guild, everyone_view=False)
    wrong_guild = SimpleNamespace(id=101, default_role=object(), me=object())
    with pytest.raises(DiscordPublicConnectorError, match="target_not_allowed"):
        policy.prove_target(_Channel(200, guild=wrong_guild), bot_user=object())
    with pytest.raises(DiscordPublicConnectorError, match="target_not_allowed"):
        policy.prove_target(_Channel(200, guild=None, type_value=1), bot_user=object())
    with pytest.raises(DiscordPublicConnectorError, match="target_type_forbidden"):
        policy.prove_target(_Channel(200, guild=guild, type_value=3), bot_user=object())
    with pytest.raises(DiscordPublicConnectorError, match="private_target_forbidden"):
        policy.prove_target(
            _Channel(201, guild=guild, type_value=12, parent=parent),
            bot_user=object(),
        )
    for kwargs, error in (
        ({"bot_view": False}, "bot_cannot_view_target"),
        ({"bot_send": False}, "bot_cannot_send_target"),
        ({"bot_history": False}, "bot_cannot_read_target_history"),
    ):
        with pytest.raises(DiscordPublicConnectorError, match=error):
            policy.prove_target(
                _Channel(200, guild=guild, everyone_view=False, **kwargs),
                bot_user=object(),
            )


def test_guild_acl_author_and_low_noise_mention_policy_follow_channel_acl() -> None:
    policy = _guild_acl_policy()
    guild = _guild()
    bot = SimpleNamespace(id=900)
    channel = _Channel(200, guild=guild, everyone_view=False)
    message = SimpleNamespace(
        id=300,
        channel=channel,
        author=SimpleNamespace(id=777, bot=False, display_name="Teammate"),
        mentions=[],
        raw_mentions=[],
        type=SimpleNamespace(value=0, name="default"),
        content="Exact ACL-authorized team request",
        reference=None,
        created_at=datetime.now(timezone.utc),
    )
    # Control Tower is an exact free-response exception.
    assert policy.event_from_message(message, bot_user=bot).author_id == "777"
    message.channel = _Channel(299, guild=guild, everyone_view=False)
    with pytest.raises(DiscordPublicConnectorError, match="mention_required"):
        policy.event_from_message(message, bot_user=bot)
