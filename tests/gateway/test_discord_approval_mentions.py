"""Discord blocking prompts can opt into owner mentions."""

import contextlib
import itertools
import os
import time
from types import SimpleNamespace

import hermes_yaml as yaml
import pytest

from gateway.config import PlatformConfig
from plugins.platforms.discord import adapter as discord_adapter
from plugins.platforms.discord.adapter import DiscordAdapter, _apply_yaml_config


class _FakeObject:
    def __init__(self, *, id):
        self.id = id


class _FakeAllowedMentions:
    def __init__(self, *, users, roles, everyone, replied_user):
        self.users = users
        self.roles = roles
        self.everyone = everyone
        self.replied_user = replied_user


class _FakeChannel:
    def __init__(self):
        self.sent_kwargs = None

    async def send(self, **kwargs):
        self.sent_kwargs = kwargs
        return SimpleNamespace(id=12345)


class _FakeClient:
    def __init__(self, channel):
        self.channel = channel

    def get_channel(self, channel_id):
        return self.channel


@pytest.fixture(autouse=True)
def _stable_discord_types(monkeypatch):
    monkeypatch.setattr(discord_adapter.discord, "Object", _FakeObject)
    monkeypatch.setattr(discord_adapter.discord, "AllowedMentions", _FakeAllowedMentions)


def _make_adapter(*, extra=None, allowed_user_ids=None):
    channel = _FakeChannel()
    adapter = object.__new__(DiscordAdapter)
    adapter._client = _FakeClient(channel)
    adapter._allowed_user_ids = allowed_user_ids or {"222", "111", "alice"}
    adapter._allowed_role_ids = set()
    adapter.config = PlatformConfig(enabled=True, extra=extra or {})
    return adapter, channel


# Every prompt that blocks the agent on an owner's answer. A non-owner mention in the body
# must stay inert: only the owners named in the ping line may be notified.
_PROMPTS = {
    "exec_approval": lambda a: a.send_exec_approval(
        chat_id="99", command="make check <@999>", session_key="session-1", description="dangerous command",
    ),
    "slash_confirm": lambda a: a.send_slash_confirm(
        chat_id="99", title="Confirm reset", message="Reset session? <@999>",
        session_key="session-1", confirm_id="confirm-1",
    ),
    "clarify_open": lambda a: a.send_clarify(
        chat_id="99", question="Which environment? <@999>", choices=None,
        clarify_id="clarify-1", session_key="session-1",
    ),
    "clarify_choices": lambda a: a.send_clarify(
        chat_id="99", question="Which environment? <@999>", choices=["staging", "production"],
        clarify_id="clarify-1", session_key="session-1",
    ),
    "update_prompt": lambda a: a.send_update_prompt(
        chat_id="99", prompt="Restore stashed changes? <@999>", session_key="session-1",
    ),
}


async def _send(prompt, adapter, channel):
    result = await _PROMPTS[prompt](adapter)
    assert result.success is True
    assert channel.sent_kwargs is not None
    return channel.sent_kwargs


def _assert_owner_ping(sent):
    assert sent["content"].startswith("<@111> <@222>\n")
    assert "<@alice>" not in sent["content"]
    assert "<@999>" in sent["content"]
    allowed_mentions = sent["allowed_mentions"]
    assert [user.id for user in allowed_mentions.users] == [111, 222]
    assert allowed_mentions.roles is False
    assert allowed_mentions.everyone is False
    assert allowed_mentions.replied_user is False
    assert len(sent["content"]) <= DiscordAdapter.MAX_MESSAGE_LENGTH


@pytest.mark.asyncio
@pytest.mark.parametrize("prompt", sorted(_PROMPTS))
async def test_blocking_prompts_ping_only_numeric_owners_when_enabled(monkeypatch, prompt):
    monkeypatch.setenv("DISCORD_APPROVAL_MENTIONS", "true")
    adapter, channel = _make_adapter(extra={"marker": "bound"})

    sent = await _send(prompt, adapter, channel)

    _assert_owner_ping(sent)
    if prompt == "clarify_open":  # answered by the next typed message, so no buttons
        assert "view" not in sent
    else:  # the pinged card keeps its buttons, still answering only to its requester/opener
        assert sent["view"].live_auth == adapter._component_live_auth
        assert sent["view"].answer_extra == {"marker": "bound"}


@pytest.mark.asyncio
@pytest.mark.parametrize("prompt", sorted(_PROMPTS))
async def test_blocking_prompts_do_not_ping_by_default(monkeypatch, prompt):
    monkeypatch.delenv("DISCORD_APPROVAL_MENTIONS", raising=False)
    adapter, channel = _make_adapter()

    sent = await _send(prompt, adapter, channel)

    assert not sent["content"].startswith("<@")
    assert "allowed_mentions" not in sent


_KEY = "DISCORD_APPROVAL_MENTIONS"


@pytest.fixture
def multiplexed(monkeypatch):
    """Multiplexed gateway with the process env var recorded, so the YAML bridge's writes are undone."""
    from agent import secret_scope

    monkeypatch.setattr(secret_scope, "_MULTIPLEX_ACTIVE", True)
    monkeypatch.setenv(_KEY, "")

    def set_process_env(value):
        if value is None:
            monkeypatch.delenv(_KEY)
        else:
            monkeypatch.setenv(_KEY, value)

    return set_process_env


@contextlib.contextmanager
def _profile(scoped_env):
    """Bind a secondary profile's secret scope; ``None`` is the unscoped default profile."""
    from agent import secret_scope

    if scoped_env is None:
        yield
        return
    token = secret_scope.set_secret_scope(scoped_env)
    try:
        yield
    finally:
        secret_scope.reset_secret_scope(token)


def _profile_adapter(yaml_cfg, scoped_env):
    """Load the profile's ``discord:`` YAML through the real bridge under its own scope."""
    with _profile(scoped_env):
        return _make_adapter(extra=_apply_yaml_config({}, yaml_cfg))


async def _pings(profile_adapter, scoped_env):
    adapter, channel = profile_adapter
    with _profile(scoped_env):
        sent = await _send("clarify_open", adapter, channel)
    if "allowed_mentions" not in sent:
        assert not sent["content"].startswith("<@")
        return False
    _assert_owner_ping(sent)
    return True


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "process_env,scoped_env,yaml_cfg,expected",
    [
        # Default profile (unscoped): its own explicit env beats its YAML, either way.
        ("true", None, {"approval_mentions": False}, True),
        ("false", None, {"approval_mentions": True}, False),
        (None, None, {"approval_mentions": True}, True),
        (None, None, {"approval_mentions": False}, False),
        # Secondary profile: its scoped env beats its YAML; the process env is never consulted.
        ("false", {_KEY: "true"}, {"approval_mentions": False}, True),
        ("true", {_KEY: "false"}, {"approval_mentions": True}, False),
        ("true", {}, {"approval_mentions": False}, False),
        ("false", {}, {"approval_mentions": True}, True),
        ("true", {}, {}, False),
    ],
    ids=[
        "default-env-true-beats-yaml-false", "default-env-false-beats-yaml-true",
        "default-yaml-true", "default-yaml-false",
        "secondary-scoped-true-beats-yaml-false", "secondary-scoped-false-beats-yaml-true",
        "secondary-yaml-false-ignores-process-true", "secondary-yaml-true-ignores-process-false",
        "secondary-unset-ignores-process-true",
    ],
)
async def test_mention_opt_in_resolves_env_then_own_yaml_per_profile(
    multiplexed, process_env, scoped_env, yaml_cfg, expected,
):
    multiplexed(process_env)

    assert await _pings(_profile_adapter(yaml_cfg, scoped_env), scoped_env) is expected
    if scoped_env is not None:
        assert os.environ.get(_KEY) == process_env  # a secondary's YAML never reaches the process env


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "default_yaml,secondary_scope,secondary_yaml",
    [(False, "true", False), (True, "false", True)],
    ids=["secondary-opts-in", "secondary-opts-out"],
)
async def test_mention_opt_in_stays_with_the_owning_profile(
    multiplexed, default_yaml, secondary_scope, secondary_yaml,
):
    """Default → secondary → default → secondary on long-lived adapters with conflicting settings:
    neither profile's opt-in (bridged env, scoped env or YAML) leaks into the other's prompts."""
    multiplexed(None)
    scoped_env = {_KEY: secondary_scope}
    default = _profile_adapter({"approval_mentions": default_yaml}, None)
    secondary = _profile_adapter({"approval_mentions": secondary_yaml}, scoped_env)

    assert os.environ[_KEY] == str(default_yaml).lower()
    for _ in range(2):
        assert await _pings(default, None) is default_yaml
        assert await _pings(secondary, scoped_env) is (secondary_scope == "true")


# Live allowlist: a config.yaml edit reaches the gate without a reconnect (DiscordLiveGatesMixin). A
# prompt raised by a turn that started before the edit must name the owners as of now, even when no
# message, slash command or click has refreshed the gate since.
_BRIDGED_VARS = ("DISCORD_ALLOWED_USERS", "DISCORD_ALLOWED_ROLES", _KEY)
_WRITES = itertools.count()


@pytest.fixture
def live_home(tmp_path, monkeypatch):
    """A real HERMES_HOME whose config.yaml builds the gate; what the loader bridges is undone."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    for var in _BRIDGED_VARS:
        for name in (var, f"HERMES_BRIDGED_{var}"):
            monkeypatch.setenv(name, "")
            monkeypatch.delenv(name)
    return tmp_path


def _write_config(home, allow_from):
    doc = {"platforms": {"discord": {"enabled": True}},
           "discord": {"allow_from": list(allow_from), "approval_mentions": True}}
    path = home / "config.yaml"
    path.write_text(yaml.safe_dump(doc), encoding="utf-8")
    stamp = time.time_ns() - 10_000_000_000 + next(_WRITES) * 1_000_000  # a distinct stat per write
    os.utime(path, ns=(stamp, stamp))


def _connected():
    """A DiscordAdapter whose gate connect() built from the active home's config.yaml."""
    from gateway.config import Platform, load_gateway_config

    adapter = DiscordAdapter(load_gateway_config().platforms[Platform.DISCORD])
    adapter._derive_connect_gates()
    channel = _FakeChannel()
    adapter._client = _FakeClient(channel)
    return adapter, channel


def _pinged(sent):
    """The owner ids a prompt notifies; its ping line and AllowedMentions must name the same ones."""
    if "allowed_mentions" not in sent:
        assert not sent["content"].startswith("<@")
        return []
    ids = [user.id for user in sent["allowed_mentions"].users]
    assert sent["content"].startswith(" ".join(f"<@{uid}>" for uid in ids) + "\n")
    return ids


@pytest.mark.asyncio
@pytest.mark.parametrize("prompt", sorted(_PROMPTS))
async def test_an_allowlist_edit_reaches_the_next_prompt_without_an_inbound_event(live_home, prompt):
    _write_config(live_home, ["111", "222"])
    adapter, channel = _connected()
    _write_config(live_home, ["111", "333"])

    assert _pinged(await _send(prompt, adapter, channel)) == [111, 333]


@pytest.mark.asyncio
async def test_a_username_owner_is_pinged_by_its_resolved_id_until_removed(live_home):
    _write_config(live_home, ["alice", "111"])
    adapter, channel = _connected()
    alice = SimpleNamespace(id=555, name="alice", display_name="alice", global_name="alice", discriminator="0")
    adapter._client.guilds = [SimpleNamespace(members=[alice], member_count=1, name="g")]
    await adapter._resolve_allowed_usernames()  # on_ready

    assert _pinged(await _send("slash_confirm", adapter, channel)) == [111, 555]
    _write_config(live_home, ["alice"])
    assert _pinged(await _send("slash_confirm", adapter, channel)) == [555]
    _write_config(live_home, ["111"])
    assert _pinged(await _send("slash_confirm", adapter, channel)) == [111]


@pytest.mark.asyncio
async def test_each_multiplexed_profile_pings_its_own_current_owners(live_home):
    from agent.secret_scope import reset_multiplex_context, set_multiplex_context
    from gateway.run import _profile_runtime_scope

    homes = {name: live_home / name for name in ("a", "b")}
    token = set_multiplex_context(True)
    try:
        bots = {}
        for name, owners in (("a", ["111", "222"]), ("b", ["444"])):
            homes[name].mkdir()
            _write_config(homes[name], owners)
            with _profile_runtime_scope(homes[name]):
                bots[name] = _connected()
        _write_config(homes["a"], ["111"])
        pings = {}
        for bot, scope in (("a", "b"), ("a", "a"), ("b", "b")):  # a routed task of b's sees a's edit first
            with _profile_runtime_scope(homes[scope]):
                pings[f"{bot}@{scope}"] = _pinged(await _send("clarify_open", *bots[bot]))
    finally:
        reset_multiplex_context(token)
    assert pings == {"a@a": [111], "b@b": [444], "a@b": [111]}


@pytest.mark.asyncio
async def test_an_edit_landing_while_an_approval_is_in_flight_keeps_the_card_within_the_cap(live_home, monkeypatch):
    """The preview budget is fixed when the text renders; an edit and a message refreshing the gate
    before the card is built can lengthen the ping line, which then gets only the room left."""
    _write_config(live_home, ["111"])
    adapter, channel = _connected()
    monkeypatch.setattr(discord_adapter.ExecApprovalView, "remove_item", lambda self, item: None, raising=False)
    get_channel = adapter._client.get_channel

    def edit_then_get_channel(channel_id):
        _write_config(live_home, [str(100_000_000_000_000_000 + index) for index in range(120)])
        adapter._refresh_live_gates()  # a message arriving meanwhile
        return get_channel(channel_id)

    adapter._client.get_channel = edit_then_get_channel
    result = await _OVERSIZED["exec_approval"](adapter)

    assert result.success is True
    assert len(channel.sent_kwargs["content"]) <= adapter.MAX_MESSAGE_LENGTH
    _pinged(channel.sent_kwargs)


_LONG = "Q" * 4_000
_LONG_CHOICES = [f"option {index} " + "x" * 120 for index in range(24)]

# Every prompt kind with a body far past the cap: the bounded ping line and the body share 2,000 chars.
_OVERSIZED = {
    # A ``` in the command cannot be sent verbatim ahead of the card, so the card stays Deny-only.
    "exec_approval": lambda a: a.send_exec_approval(
        chat_id="99", command="echo '```' " + "A" * 4_000, session_key="session-1", description="dangerous command",
    ),
    "slash_confirm": lambda a: a.send_slash_confirm(
        chat_id="99", title="Confirm reset", message=_LONG, session_key="session-1", confirm_id="confirm-1",
    ),
    "clarify_open": lambda a: a.send_clarify(
        chat_id="99", question=_LONG, choices=None, clarify_id="clarify-1", session_key="session-1",
    ),
    "clarify_choices": lambda a: a.send_clarify(
        chat_id="99", question=_LONG, choices=_LONG_CHOICES, clarify_id="clarify-1", session_key="session-1",
    ),
    "update_prompt": lambda a: a.send_update_prompt(chat_id="99", prompt=_LONG, session_key="session-1"),
}


@pytest.mark.asyncio
@pytest.mark.parametrize("prompt", sorted(_OVERSIZED))
async def test_large_allowlist_keeps_prompt_within_discord_limit(monkeypatch, prompt):
    monkeypatch.setenv("DISCORD_APPROVAL_MENTIONS", "true")
    allowed_user_ids = {str(100_000_000_000_000_000 + index) for index in range(120)}
    adapter, channel = _make_adapter(allowed_user_ids=allowed_user_ids)
    removed = []  # the discord.ui.View test double has no remove_item; record what the approval card drops
    monkeypatch.setattr(discord_adapter.ExecApprovalView, "remove_item",
                        lambda self, item: removed.append(item.__name__), raising=False)

    result = await _OVERSIZED[prompt](adapter)

    assert result.success is True
    assert channel.sent_kwargs is not None
    content = channel.sent_kwargs["content"]
    assert content.startswith("<@")
    assert len(content) <= adapter.MAX_MESSAGE_LENGTH
    assert content.count("<@") < len(allowed_user_ids)
    assert len(channel.sent_kwargs["allowed_mentions"].users) == content.count("<@")
    if prompt == "clarify_choices":  # every positional button keeps its numbered line beside the ping
        assert all(f"\n{index}. " in content for index in range(1, len(_LONG_CHOICES) + 1))
    if prompt == "exec_approval":  # the mention never buys back an Approve the preview cannot justify
        assert sorted(removed) == ["allow_always", "allow_once", "allow_session"]


def test_yaml_config_seeds_websocket_health_with_primary_precedence(monkeypatch):
    for key in (
        "HERMES_DISCORD_LIVENESS_INTERVAL_SECONDS",
        "HERMES_DISCORD_LIVENESS_FAILURE_THRESHOLD",
    ):
        monkeypatch.delenv(key, raising=False)

    seeded = _apply_yaml_config(
        {},
        {
            "websocket_liveness_interval_seconds": 11,
            "liveness_interval_seconds": 99,
            "websocket_liveness_failure_threshold": 2,
            "websocket_heartbeat_ack_max_age_seconds": 75,
            "websocket_max_latency_seconds": 30,
        },
    )

    assert os.environ["HERMES_DISCORD_LIVENESS_INTERVAL_SECONDS"] == "11"
    assert os.environ["HERMES_DISCORD_LIVENESS_FAILURE_THRESHOLD"] == "2"
    assert seeded == {
        "websocket_liveness_interval_seconds": 11,
        "websocket_liveness_failure_threshold": 2,
        "websocket_heartbeat_ack_max_age_seconds": 75,
        "websocket_max_latency_seconds": 30,
    }
