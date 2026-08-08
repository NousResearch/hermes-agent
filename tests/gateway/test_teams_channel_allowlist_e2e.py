"""End-to-end proof that the Teams channel allowlist holds at the real
gateway authorization boundary, not only inside the adapter's own unit tests.

This is a security boundary (AGENTS.md: "resolution chains, config
propagation, security boundaries ... real E2E against a temp HERMES_HOME --
green unit mocks are not evidence"). ``tests/gateway/test_teams.py`` proves
``TeamsAdapter._on_message`` builds the right ``SessionSource`` (stubbing
``handle_message``); it never proves the gateway's real authorization gate
actually honors what the adapter stamped. Here nothing about the
authorization *decision* is mocked: a real ``TeamsAdapter._on_message`` builds
a real ``SessionSource``, which is then run through the real
``GatewayAuthorizationMixin._is_user_authorized``
(``gateway/authz_mixin.py``) via a minimal ``GatewayRunner`` built with
``object.__new__`` -- the same construction
``tests/gateway/test_config_driven_access_policy.py`` uses for the
WeCom/WhatsApp own-policy gate -- backed by a real ``PairingStore`` against
the hermetic per-test ``HERMES_HOME`` every test already gets
(``tests/conftest.py::_hermetic_environment``, autouse).
"""

from unittest.mock import MagicMock

import pytest

from gateway.config import GatewayConfig, Platform
from gateway.pairing import PairingStore
from gateway.session import SessionSource
from tests.gateway.test_teams import TeamsAdapter, _make_config
from tests.gateway.test_teams import register as _real_teams_register


def _make_runner(adapter) -> "object":
    """Bare GatewayRunner exposing only what _is_user_authorized touches.

    Mirrors ``tests/gateway/test_config_driven_access_policy.py::_make_runner``.
    ``pairing_store`` is a REAL store (not a mock) so "no pairing entry exists"
    is proven by an empty on-disk store, not by a mock return value.
    """
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig()
    runner.adapters = {Platform("teams"): adapter}
    runner.pairing_store = PairingStore()
    return runner


class _Ctx:
    """Minimal ActivityContext stand-in: only ``.activity`` and
    ``.conversation_ref`` are read by ``TeamsAdapter._on_message``."""

    def __init__(self, activity):
        self.activity = activity
        self.conversation_ref = MagicMock()


def _make_activity(
    *,
    conversation_id: str,
    conversation_type: str,
    mentioned: bool = True,
    bot_id: str = "bot-id",
    user_aad: str = "aad-e2e-user",
):
    activity = MagicMock()
    activity.text = "<at>Hermes</at> hello" if mentioned else "hello"
    activity.id = "activity-e2e-1"
    activity.from_ = MagicMock()
    activity.from_.id = "29:e2e-user"
    activity.from_.aad_object_id = user_aad
    activity.from_.name = "E2E User"
    activity.conversation = MagicMock()
    activity.conversation.id = conversation_id
    activity.conversation.conversation_type = conversation_type
    activity.conversation.name = "E2E Chat"
    activity.conversation.tenant_id = "tenant-e2e"
    activity.attachments = []
    del activity.channel_data
    activity.recipient = MagicMock()
    activity.recipient.id = bot_id
    mention_entity = MagicMock()
    mention_entity.type = "mention"
    mention_entity.mentioned = MagicMock()
    mention_entity.mentioned.id = bot_id if mentioned else "someone-else"
    activity.entities = [mention_entity]
    return activity


async def _dispatch(adapter: TeamsAdapter, activity) -> "SessionSource | None":
    """Run the real _on_message and capture the SessionSource it builds.

    Returns None when the message was dropped before ``handle_message`` --
    i.e. never produced a session source at all.
    """
    captured: dict = {}

    async def _capture(event):
        captured["source"] = event.source

    adapter.handle_message = _capture
    await adapter._on_message(_Ctx(activity))
    return captured.get("source")


def _clear_teams_auth_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in (
        "TEAMS_ALLOWED_USERS",
        "TEAMS_ALLOW_ALL_USERS",
        "TEAMS_ALLOWED_CHANNELS",
        "TEAMS_REQUIRE_MENTION",
        "GATEWAY_ALLOWED_USERS",
        "GATEWAY_ALLOW_ALL_USERS",
    ):
        monkeypatch.delenv(key, raising=False)


class TestTeamsChannelAllowlistE2E:
    @pytest.fixture(autouse=True)
    def _teams_platform_registered(self):
        """Register the real Teams ``PlatformEntry`` in the global
        ``platform_registry`` for the duration of one test, then remove it.

        ``_is_user_authorized`` learns a plugin platform's
        ``allowed_users_env`` / ``allow_all_env`` names from this registry
        (``gateway/authz_mixin.py``), not from a hardcoded map. Driving the
        real ``register()`` this module already loaded through a minimal
        ctx -- rather than a ``MagicMock()`` (the unit-test shortcut
        ``tests/gateway/test_teams.py`` uses) -- proves this suite exercises
        the actual entry the gateway consults, not a stand-in for it.
        """
        from gateway.platform_registry import PlatformEntry, platform_registry

        class _RealCtx:
            def register_platform(
                self, name, label, adapter_factory, check_fn,
                validate_config=None, required_env=None, install_hint="",
                **entry_kwargs,
            ):
                entry_kwargs.setdefault("plugin_name", "teams-platform")
                platform_registry.register(PlatformEntry(
                    name=name, label=label, adapter_factory=adapter_factory,
                    check_fn=check_fn, validate_config=validate_config,
                    required_env=required_env or [], install_hint=install_hint,
                    source="plugin", **entry_kwargs,
                ))

        _real_teams_register(_RealCtx())
        try:
            yield
        finally:
            platform_registry.unregister("teams")

    @pytest.mark.anyio
    async def test_allowed_channel_sender_authorized_without_teams_allowed_users(
        self, monkeypatch
    ):
        """The core requirement: a sender absent from TEAMS_ALLOWED_USERS is
        still authorized end to end when they post, mentioning the bot, in a
        channel on the allowlist."""
        _clear_teams_auth_env(monkeypatch)

        adapter = TeamsAdapter(_make_config(
            client_id="bot-id", client_secret="secret", tenant_id="tenant",
            allowed_channels=["19:allowed@thread.tacv2"],
        ))
        adapter._app = MagicMock()
        adapter._app.id = "bot-id"

        activity = _make_activity(
            conversation_id="19:allowed@thread.tacv2", conversation_type="channel",
        )
        source = await _dispatch(adapter, activity)
        assert source is not None, "message from an allowed, mentioned channel must dispatch"
        assert source.role_authorized is True

        runner = _make_runner(adapter)
        assert runner._is_user_authorized(source) is True

    @pytest.mark.anyio
    async def test_non_listed_channel_never_reaches_the_runner(self, monkeypatch):
        """A message from a channel absent from the allowlist never produces a
        SessionSource -- there is nothing left for the runner to authorize."""
        _clear_teams_auth_env(monkeypatch)

        adapter = TeamsAdapter(_make_config(
            client_id="bot-id", client_secret="secret", tenant_id="tenant",
            allowed_channels=["19:allowed@thread.tacv2"],
        ))
        adapter._app = MagicMock()
        adapter._app.id = "bot-id"

        activity = _make_activity(
            conversation_id="19:not-allowed@thread.tacv2", conversation_type="channel",
        )
        source = await _dispatch(adapter, activity)
        assert source is None

    @pytest.mark.anyio
    async def test_allowed_channel_but_unmentioned_never_reaches_the_runner(
        self, monkeypatch
    ):
        """A listed channel still requires the mention -- the allowlist alone
        is not a bypass for require_mention."""
        _clear_teams_auth_env(monkeypatch)

        adapter = TeamsAdapter(_make_config(
            client_id="bot-id", client_secret="secret", tenant_id="tenant",
            allowed_channels=["19:allowed@thread.tacv2"],
        ))
        adapter._app = MagicMock()
        adapter._app.id = "bot-id"

        activity = _make_activity(
            conversation_id="19:allowed@thread.tacv2",
            conversation_type="channel",
            mentioned=False,
        )
        source = await _dispatch(adapter, activity)
        assert source is None

    @pytest.mark.anyio
    async def test_same_sender_still_denied_over_dm(self, monkeypatch):
        """The channel allowlist authorizes the CHANNEL, not the person: the
        exact same AAD-object-id sender DMing the bot must still be denied,
        because they are not in TEAMS_ALLOWED_USERS and no pairing entry
        exists in the (real, empty) PairingStore."""
        _clear_teams_auth_env(monkeypatch)

        adapter = TeamsAdapter(_make_config(
            client_id="bot-id", client_secret="secret", tenant_id="tenant",
            allowed_channels=["19:allowed@thread.tacv2"],
        ))
        adapter._app = MagicMock()
        adapter._app.id = "bot-id"

        dm_activity = _make_activity(
            conversation_id="19:dm-conversation", conversation_type="personal",
        )
        source = await _dispatch(adapter, dm_activity)
        assert source is not None
        assert source.chat_type == "dm"
        assert source.role_authorized is False

        runner = _make_runner(adapter)
        assert runner._is_user_authorized(source) is False

    @pytest.mark.anyio
    async def test_empty_allowlist_falls_back_to_teams_allowed_users(self, monkeypatch):
        """Backward compatibility, proven end to end: with no channel
        allowlist configured, a channel message still reaches the runner
        (role_authorized False), and the pre-existing TEAMS_ALLOWED_USERS
        allowlist is what decides it -- exactly as before this feature."""
        _clear_teams_auth_env(monkeypatch)
        monkeypatch.setenv("TEAMS_ALLOWED_USERS", "aad-e2e-user")

        adapter = TeamsAdapter(_make_config(
            client_id="bot-id", client_secret="secret", tenant_id="tenant",
        ))
        adapter._app = MagicMock()
        adapter._app.id = "bot-id"

        activity = _make_activity(
            conversation_id="19:any-channel@thread.tacv2", conversation_type="channel",
        )
        source = await _dispatch(adapter, activity)
        assert source is not None
        assert source.role_authorized is False

        runner = _make_runner(adapter)
        assert runner._is_user_authorized(source) is True

    @pytest.mark.anyio
    async def test_unmentioned_channel_message_never_reaches_the_runner_even_without_an_allowlist(
        self, monkeypatch
    ):
        """require_mention is independent of the channel allowlist: an
        unmentioned channel message must never produce a SessionSource at
        all, even with no TEAMS_ALLOWED_CHANNELS configured -- closing the
        RSC-delivered-unmentioned-message gap for every Teams channel."""
        _clear_teams_auth_env(monkeypatch)

        adapter = TeamsAdapter(_make_config(
            client_id="bot-id", client_secret="secret", tenant_id="tenant",
        ))
        adapter._app = MagicMock()
        adapter._app.id = "bot-id"

        activity = _make_activity(
            conversation_id="19:any-channel@thread.tacv2",
            conversation_type="channel",
            mentioned=False,
        )
        source = await _dispatch(adapter, activity)
        assert source is None
