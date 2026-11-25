"""A Discord or Telegram allowlist edit in config.yaml reaches the next inbound event, no reconnect.

Each adapter copies its gate at connect time (per-profile isolation, #72348). These tests write a
real config.yaml, build the adapter's gate from the real ``load_gateway_config()``, edit the file,
and judge the next event: the edited list must decide, an operator's env value must still win
(#109032), and a multiplexed profile must read only its own file.
"""

import itertools
import logging
import os
import time
from types import SimpleNamespace

import hermes_yaml as yaml
import pytest

from gateway.config import Platform, load_gateway_config
from plugins.platforms.discord.adapter import DiscordAdapter

GUILD_ID = 222222
USER_ID = 4242
_WRITES = itertools.count()
GATE_VARS = ("DISCORD_ALLOWED_ROLES", "DISCORD_ALLOWED_USERS", "TELEGRAM_ALLOWED_USERS",
             "DISCORD_ALLOW_ALL_USERS", "GATEWAY_ALLOW_ALL_USERS", "GATEWAY_ALLOWED_USERS")


@pytest.fixture(autouse=True)
def _home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    for var in GATE_VARS:  # setenv first so teardown also removes what the loader bridges in
        monkeypatch.setenv(var, "")
        monkeypatch.delenv(var)
        monkeypatch.setenv(f"HERMES_BRIDGED_{var}", "")
        monkeypatch.delenv(f"HERMES_BRIDGED_{var}")
    return tmp_path


def _write(home, discord_cfg=None, telegram_cfg=None):
    doc = {"platforms": {"discord": {"enabled": True}, "telegram": {"enabled": True}}}
    if discord_cfg is not None:
        doc["discord"] = discord_cfg
    if telegram_cfg is not None:
        doc["telegram"] = telegram_cfg
    path = home / "config.yaml"
    path.write_text(yaml.safe_dump(doc), encoding="utf-8")
    # Distinct, settled mtimes: two rewrites in one test never share a stat signature.
    stamp = time.time_ns() - 10_000_000_000 + next(_WRITES) * 1_000_000
    os.utime(path, ns=(stamp, stamp))


def _connected_discord(roles=(2,)):
    """A DiscordAdapter whose gate was built the way connect() builds it, guild member holds *roles*."""
    adapter = DiscordAdapter(load_gateway_config().platforms[Platform.DISCORD])
    adapter._snapshot_gate_env()
    adapter._allowed_user_ids = adapter._get_allowed_users()
    adapter._allowed_role_ids = adapter._get_allowed_roles()
    adapter._mark_live_gate_baseline()
    guild = SimpleNamespace(id=GUILD_ID)
    member = SimpleNamespace(id=USER_ID, roles=[SimpleNamespace(id=r) for r in roles], guild=guild)
    guild.get_member = lambda uid: member if uid == USER_ID else None
    adapter._client = SimpleNamespace(get_guild=lambda gid: guild if gid == GUILD_ID else None)
    return adapter, guild, member


def _allowed(adapter, guild, member):
    return adapter._is_allowed_user(str(USER_ID), member, guild=guild, is_dm=False)


@pytest.mark.parametrize("edit, expected", [
    ({"allowed_roles": ["9"]}, False),
    ({"allowed_roles": ["2", "9"]}, True),
])
def test_role_edit_applies_to_the_next_message(_home, edit, expected):
    _write(_home, {"allowed_roles": ["2"]})
    adapter, guild, member = _connected_discord()
    assert _allowed(adapter, guild, member) is True
    _write(_home, edit)
    assert _allowed(adapter, guild, member) is expected


def test_user_removed_from_allow_from_is_refused_next_message(_home):
    _write(_home, {"allow_from": [str(USER_ID), "777"]})
    adapter, guild, member = _connected_discord(roles=())
    assert _allowed(adapter, guild, member) is True
    _write(_home, {"allow_from": ["777"]})
    assert _allowed(adapter, guild, member) is False


def test_slash_and_component_follow_the_edit(_home):
    from plugins.platforms.discord.adapter_component_auth import _component_check_auth

    _write(_home, {"allowed_roles": ["2"]})
    adapter, guild, member = _connected_discord()
    connect_users, connect_roles = adapter._allowed_user_ids, adapter._allowed_role_ids
    _write(_home, {"allowed_roles": ["9"]})
    users, roles = adapter._current_gate_sets()
    interaction = SimpleNamespace(user=member, guild=guild)
    assert _component_check_auth(interaction, connect_users, connect_roles) is True  # stale copy
    assert _component_check_auth(interaction, users, roles) is False


def test_operator_env_still_beats_the_edited_yaml(_home, monkeypatch):
    from hermes_cli import env_loader
    monkeypatch.setattr(env_loader, "_LOADED_DOTENV_KEYS", {"DISCORD_ALLOWED_ROLES"})  # set in .env
    monkeypatch.setenv("DISCORD_ALLOWED_ROLES", "2")
    _write(_home, {"allowed_roles": ["2"]})
    adapter, guild, member = _connected_discord()
    _write(_home, {"allowed_roles": ["9"]})
    assert _allowed(adapter, guild, member) is True
    assert os.environ["DISCORD_ALLOWED_ROLES"] == "2"


def test_multiplexed_profiles_refresh_only_their_own_lists(tmp_path):
    from agent.secret_scope import reset_multiplex_context, set_multiplex_context
    from gateway.run import _profile_runtime_scope

    homes = {name: tmp_path / name for name in ("a", "b")}
    token = set_multiplex_context(True)
    try:
        adapters = {}
        for name, home in homes.items():
            home.mkdir()
            _write(home, {"allowed_roles": ["2"]})
            with _profile_runtime_scope(home):
                adapters[name] = _connected_discord()
        _write(homes["a"], {"allowed_roles": ["9"]})
        verdicts = {}
        for name, home in homes.items():
            with _profile_runtime_scope(home):
                verdicts[name] = _allowed(*adapters[name])
    finally:
        reset_multiplex_context(token)
    assert verdicts == {"a": False, "b": True}
    assert os.environ.get("DISCORD_ALLOWED_ROLES") is None


def test_a_gate_checked_under_another_profiles_scope_keeps_its_own_lists(tmp_path):
    """A routed ``/voice join`` runs the receiving bot's voice loop in the ROUTED profile's scope;
    the gate check there must not adopt that profile's lists as the receiving bot's own."""
    from agent.secret_scope import reset_multiplex_context, set_multiplex_context
    from gateway.run import _profile_runtime_scope

    homes = {name: tmp_path / name for name in ("a", "b")}
    for home, roles in zip(homes.values(), (["2"], ["9"])):
        home.mkdir()
        _write(home, {"allowed_roles": roles})
    token = set_multiplex_context(True)
    try:
        with _profile_runtime_scope(homes["a"]):
            adapter, guild, member = _connected_discord(roles=(9,))
        with _profile_runtime_scope(homes["b"]):
            verdict = _allowed(adapter, guild, member)
        roles_after = adapter._allowed_role_ids
        _write(homes["a"], {"allowed_roles": ["2", "9"]})
        with _profile_runtime_scope(homes["a"]):
            owner_verdict = _allowed(adapter, guild, member)
    finally:
        reset_multiplex_context(token)
    assert (verdict, roles_after) == (False, {2})
    assert owner_verdict is True  # the owner's own edit still applies live


def test_a_secondary_owners_edit_seen_under_a_routed_scope_is_loaded_once(tmp_path, monkeypatch):
    """A routed voice loop gates the secondary owner's bot under another scope on every utterance; once
    an edit is loaded there, later checks must not reload the unchanged file each time."""
    import gateway.config as gateway_config
    from agent.secret_scope import reset_multiplex_context, set_multiplex_context
    from gateway.run import _profile_runtime_scope

    launch, owner, routed = (tmp_path / n for n in ("default", "a", "b"))
    for home in (launch, owner, routed):
        home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(launch))
    _write(owner, {"allowed_roles": ["2"]})
    _write(routed, {"allowed_roles": ["7"]})
    token = set_multiplex_context(True)
    try:
        with _profile_runtime_scope(owner):
            adapter, guild, member = _connected_discord(roles=(9,))
        _write(owner, {"allowed_roles": ["9"]})
        calls = []
        real = gateway_config.load_gateway_config
        monkeypatch.setattr(gateway_config, "load_gateway_config", lambda **kw: calls.append(1) or real(**kw))
        with _profile_runtime_scope(routed):
            verdicts = [_allowed(adapter, guild, member) for _ in range(5)]
    finally:
        reset_multiplex_context(token)
    assert verdicts == [True] * 5
    assert len(calls) == 1


def test_unreadable_edit_keeps_the_last_good_lists(_home, caplog):
    _write(_home, {"allowed_roles": ["2"]})
    adapter, guild, member = _connected_discord()
    (_home / "config.yaml").write_text("discord: [unclosed\n", encoding="utf-8")
    caplog.set_level(logging.WARNING, logger="gateway.platforms._shared")
    assert [_allowed(adapter, guild, member) for _ in range(3)] == [True, True, True]
    assert len([r for r in caplog.records if "keeping the current allowlists" in r.getMessage()]) == 1


def test_an_edit_that_parses_but_fails_to_load_keeps_the_last_good_lists(tmp_path):
    """load_gateway_config() returns the env-only config when a later YAML stage raises; a secondary
    profile's YAML-only allow_from must not fall back to the channel grant in its own .env."""
    from agent.secret_scope import reset_multiplex_context, set_multiplex_context
    from gateway.run import _profile_runtime_scope

    home = tmp_path / "b"
    home.mkdir()
    (home / ".env").write_text("DISCORD_ALLOWED_CHANNELS=555\n", encoding="utf-8")
    _write(home, {"allow_from": [str(USER_ID)]})
    token = set_multiplex_context(True)
    try:
        with _profile_runtime_scope(home):
            adapter, guild, _member = _connected_discord(roles=())
        stranger = SimpleNamespace(id=9999, roles=[], guild=guild)
        (home / "config.yaml").write_text(
            f"discord:\n  allow_from: ['{USER_ID}']\nplatforms:\n  discord:\n    extra: [1]\n", encoding="utf-8")
        with _profile_runtime_scope(home):
            assert adapter._is_allowed_user("9999", stranger, guild=guild, is_dm=False, channel_ids={"555"}) is False
    finally:
        reset_multiplex_context(token)


def test_transient_read_failure_retries_the_same_allowlist_edit(_home, monkeypatch):
    """A recovered read error must not leave a removed Discord role authorized indefinitely."""
    from gateway import config_loader

    _write(_home, {"allowed_roles": ["2"]})
    adapter, guild, member = _connected_discord()
    assert _allowed(adapter, guild, member) is True
    _write(_home, {"allowed_roles": ["9"]})

    real_read = config_loader.read_yaml_layers
    attempts = 0

    def read_with_one_transient_error(home):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise OSError("config.yaml was temporarily unavailable")
        return real_read(home)

    monkeypatch.setattr(config_loader, "read_yaml_layers", read_with_one_transient_error)
    assert _allowed(adapter, guild, member) is True  # last known-good policy on failed read
    assert _allowed(adapter, guild, member) is False  # same edit, no second write
    assert attempts >= 2


def test_telegram_allow_from_edit_applies_to_the_next_message(_home):
    from gateway.platforms._shared import mark_live_gate_baseline
    from plugins.platforms.telegram.adapter import TelegramAdapter

    _write(_home, telegram_cfg={"allow_from": ["111", "222"]})
    adapter = TelegramAdapter(load_gateway_config().platforms[Platform.TELEGRAM])
    mark_live_gate_baseline(adapter)
    assert adapter._is_callback_user_authorized("222") is True
    _write(_home, telegram_cfg={"allow_from": ["111"]})
    assert adapter._is_callback_user_authorized("222") is False
    assert adapter._is_callback_user_authorized("111") is True
    assert adapter.config.extra.get("allow_from") == ["111"]


def test_a_command_in_a_newly_allowed_telegram_chat_is_processed(_home):
    """Commands meet the allowed_chats gate in _should_process_message before any auth check runs."""
    from gateway.platforms._shared import mark_live_gate_baseline
    from plugins.platforms.telegram.adapter import TelegramAdapter
    from tests.gateway.test_telegram_group_gating import _group_message

    _write(_home, telegram_cfg={"allowed_chats": ["-100"], "require_mention": False})
    adapter = TelegramAdapter(load_gateway_config().platforms[Platform.TELEGRAM])
    adapter._bot = SimpleNamespace(id=999, username="hermes_bot")
    mark_live_gate_baseline(adapter)
    _write(_home, telegram_cfg={"allowed_chats": ["-100", "-200"], "require_mention": False})
    assert adapter._should_process_message(_group_message("/status", chat_id=-200), is_command=True) is True


def test_telegram_topic_and_bot_gates_follow_an_edit(tmp_path):
    """allowed_topics and allow_bots are read from extra too; an edit must reach them without a reconnect. A
    multiplexed profile has no bridged env to fall back on, so only the extra refresh can carry the edit."""
    from agent.secret_scope import reset_multiplex_context, set_multiplex_context
    from gateway.platforms._shared import mark_live_gate_baseline
    from gateway.run import _profile_runtime_scope
    from plugins.platforms.telegram.adapter import TelegramAdapter
    from plugins.platforms.telegram.telegram_live_gates import refresh_telegram_gates

    home = tmp_path / "b"
    home.mkdir()
    _write(home, telegram_cfg={"allowed_topics": ["5"], "allow_bots": "all"})
    token = set_multiplex_context(True)
    try:
        with _profile_runtime_scope(home):
            adapter = TelegramAdapter(load_gateway_config().platforms[Platform.TELEGRAM])
            mark_live_gate_baseline(adapter)
        _write(home, telegram_cfg={"allowed_topics": ["5", "6"], "allow_bots": "none"})
        with _profile_runtime_scope(home):
            refresh_telegram_gates(adapter)
            assert adapter._telegram_allowed_topics() == {"5", "6"}
    finally:
        reset_multiplex_context(token)
    assert adapter.config.extra.get("allow_bots") == "none"


def test_unchanged_config_costs_no_reload(_home, monkeypatch):
    import gateway.config as gateway_config

    _write(_home, {"allowed_roles": ["2"]})
    adapter, guild, member = _connected_discord()
    _allowed(adapter, guild, member)  # settles the baseline after the racy window
    calls = []
    real = gateway_config.load_gateway_config
    monkeypatch.setattr(gateway_config, "load_gateway_config", lambda: calls.append(1) or real())
    for _ in range(50):
        assert _allowed(adapter, guild, member) is True
    assert calls == []


def test_a_bot_message_follows_an_allow_bots_edit(_home, monkeypatch):
    """Bot-authored messages skip _is_allowed_user; the gate must still refresh before allow_bots is read."""
    from plugins.platforms.discord import adapter as discord_adapter

    discord = SimpleNamespace(MessageType=SimpleNamespace(default=0, reply=19), DMChannel=type("DMChannel", (), {}))
    monkeypatch.setattr(discord_adapter, "discord", discord)  # discord.py is an optional extra
    _write(_home, {"allow_bots": "all"})
    adapter, _guild, _member = _connected_discord()
    adapter._client.user = SimpleNamespace(id=99, bot=True)
    adapter._text_batch_delay_seconds = 0
    _write(_home, {"allow_bots": "none"})
    bot = SimpleNamespace(id=77, bot=True)
    message = SimpleNamespace(id=5, author=bot, type=discord.MessageType.default, content="<@99> hi",
                              mentions=[adapter._client.user], channel=SimpleNamespace(id=1), guild=None)
    assert adapter._discord_message_admission(message, claim=False) == (False, False)
    assert adapter._get_allow_bots() == "none"



def test_a_read_failure_after_a_good_read_never_applies_an_env_only_config(tmp_path, monkeypatch):
    """The refresh reads config.yaml once and applies THAT result. Production reads it through
    ``gateway.config_loader.read_yaml_layers`` (from ``_load_yaml_layer``); the patch fails every read
    after the first in the refresh, the transient error a preflight-then-reload swallows into an
    env-only config. A secondary profile's YAML allow_from must not then fall back to its .env channel
    grant (no env bridge masks it there)."""
    from agent.secret_scope import reset_multiplex_context, set_multiplex_context
    from gateway import config_loader
    from gateway.run import _profile_runtime_scope

    home = tmp_path / "b"
    home.mkdir()
    (home / ".env").write_text("DISCORD_ALLOWED_CHANNELS=555\n", encoding="utf-8")
    _write(home, {"allow_from": [str(USER_ID)]})
    token = set_multiplex_context(True)
    try:
        with _profile_runtime_scope(home):
            adapter, guild, _member = _connected_discord(roles=())
        stranger = SimpleNamespace(id=9999, roles=[], guild=guild)
        _write(home, {"allow_from": [str(USER_ID), "777"]})
        real_read = config_loader.read_yaml_layers
        reads = itertools.count()

        def fail_after_first_read(path):
            if next(reads) >= 1:
                raise OSError("config.yaml was temporarily unavailable")
            return real_read(path)

        monkeypatch.setattr(config_loader, "read_yaml_layers", fail_after_first_read)
        with _profile_runtime_scope(home):
            assert adapter._is_allowed_user("9999", stranger, guild=guild, is_dm=False, channel_ids={"555"}) is False
    finally:
        reset_multiplex_context(token)


def test_a_load_that_raises_leaves_the_bridged_env_as_it_was(_home):
    """A YAML stage that raises after a plugin hook bridged its value (here the core top-level
    require_mention fallback on a boolean ``telegram:`` block) must not leave that value in the
    process env: the adapter keeps its last good list, so the runner's env check must too."""
    _write(_home, {"allow_from": ["111"]})
    load_gateway_config()
    assert os.environ["DISCORD_ALLOWED_USERS"] == "111"
    (_home / "config.yaml").write_text(yaml.safe_dump(
        {"discord": {"allow_from": ["222"]}, "require_mention": True, "telegram": True}), encoding="utf-8")
    with pytest.raises(TypeError):
        load_gateway_config(strict=True)
    assert os.environ["DISCORD_ALLOWED_USERS"] == "111"


def test_a_load_that_raises_leaves_the_allow_all_grant_as_it_was(_home, monkeypatch):
    """``allow_all_users: true`` added in the same edit as a later-stage failure must not stay
    published: the adapter keeps its last good lists, so the runner's allow-all check must too."""
    from gateway import config_loader

    monkeypatch.setattr(config_loader, "_BRIDGED_ALLOW_ALL_USERS", False)
    _write(_home, {"allow_from": ["111"]})
    load_gateway_config()
    (_home / "config.yaml").write_text(yaml.safe_dump(
        {"allow_all_users": True, "require_mention": True, "telegram": True}), encoding="utf-8")
    with pytest.raises(TypeError):
        load_gateway_config(strict=True)
    assert os.environ.get("GATEWAY_ALLOW_ALL_USERS") is None
    monkeypatch.setenv("GATEWAY_ALLOW_ALL_USERS", "1")  # the operator's own grant, set afterwards
    _write(_home, {"allow_from": ["111"]})
    load_gateway_config()
    assert (os.environ.get("GATEWAY_ALLOW_ALL_USERS"), config_loader.bridged_allow_all_users()) == ("1", None)


def test_an_edit_made_while_the_adapter_starts_up_still_applies(_home):
    """Startup loads the config, then spends time on dependencies before connect marks the baseline;
    a role removed in that gap must be refused, so the baseline is the version that built the gates."""
    _write(_home, {"allowed_roles": ["2"]})
    adapter = DiscordAdapter(load_gateway_config().platforms[Platform.DISCORD])
    _write(_home, {"allowed_roles": ["9"]})  # edited before connect
    adapter._snapshot_gate_env()
    adapter._allowed_user_ids = adapter._get_allowed_users()
    adapter._allowed_role_ids = adapter._get_allowed_roles()
    adapter._mark_live_gate_baseline()
    guild = SimpleNamespace(id=GUILD_ID)
    member = SimpleNamespace(id=USER_ID, roles=[SimpleNamespace(id=2)], guild=guild)
    guild.get_member = lambda uid: member if uid == USER_ID else None
    adapter._client = SimpleNamespace(get_guild=lambda gid: guild if gid == GUILD_ID else None)
    assert _allowed(adapter, guild, member) is False


def test_a_resolved_username_survives_a_second_on_ready_and_a_later_edit(_home):
    """discord.py fires ``on_ready`` again on a re-IDENTIFY without a new connect(): the resolver
    then sees only numeric IDs, and must still map the configured username for the next refresh."""
    import asyncio

    _write(_home, {"allow_from": ["alice"]})
    adapter, guild, _member = _connected_discord(roles=())
    alice = SimpleNamespace(id=111, name="alice", display_name="alice", global_name="alice", discriminator="0")
    guild.members, guild.member_count, guild.name = [alice], 1, "g"
    adapter._client.guilds = [guild]
    for _ in range(2):  # first on_ready, then the reconnect's
        asyncio.run(adapter._resolve_allowed_usernames())
    _write(_home, {"allow_from": ["alice"], "allowed_channels": ["555"]})  # an unrelated edit
    assert adapter._is_allowed_user("111", alice, guild=guild, is_dm=False) is True
    assert adapter.resolved_allowlist_user_ids() == {"111"}


def test_an_owner_edit_first_seen_under_a_routed_scope_applies_the_owners_lists(tmp_path):
    """The owner's edit can reach the gate first from a routed profile's task (a /voice listen loop).
    The reload and the env re-snapshot must both run as the owner: the routed home's lists and .env
    never become this bot's gate."""
    from agent.secret_scope import reset_multiplex_context, set_multiplex_context
    from gateway.run import _profile_runtime_scope

    owner, routed = tmp_path / "a", tmp_path / "b"
    owner.mkdir()
    routed.mkdir()
    (routed / ".env").write_text("DISCORD_ALLOWED_USERS=9999\n", encoding="utf-8")
    _write(owner, {"allowed_roles": ["2"]})
    _write(routed, {"allowed_roles": ["7"]})
    token = set_multiplex_context(True)
    try:
        with _profile_runtime_scope(owner):
            adapter, guild, member = _connected_discord(roles=(5,))
        _write(owner, {"allowed_roles": ["2", "5"]})
        stranger = SimpleNamespace(id=9999, roles=[SimpleNamespace(id=7)], guild=guild)
        with _profile_runtime_scope(routed):
            assert _allowed(adapter, guild, member) is True
            assert adapter._is_allowed_user("9999", stranger, guild=guild, is_dm=False) is False
        with _profile_runtime_scope(owner):
            assert adapter._is_allowed_user("9999", stranger, guild=guild, is_dm=False) is False
    finally:
        reset_multiplex_context(token)


def test_a_launch_profile_edit_first_seen_under_a_routed_scope_still_applies(tmp_path, monkeypatch):
    """The launch profile's YAML lists live in the process env (the single-profile bridge). Its bound
    scope skips that bridge, so a reload there can only be provisional: the owner's next own event
    must still apply the edit instead of finding it already acknowledged."""
    from agent.secret_scope import reset_multiplex_context, set_multiplex_context
    from gateway.run import _profile_runtime_scope

    default, routed = tmp_path / "default", tmp_path / "b"
    default.mkdir()
    routed.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(default))
    _write(default, {"allowed_roles": ["2"]})
    _write(routed, {"allowed_roles": ["7"]})
    token = set_multiplex_context(True)
    try:
        adapter, guild, member = _connected_discord(roles=(9,))
        assert _allowed(adapter, guild, member) is False
        _write(default, {"allowed_roles": ["9"]})
        with _profile_runtime_scope(routed):
            _allowed(adapter, guild, member)
        assert _allowed(adapter, guild, member) is True
    finally:
        reset_multiplex_context(token)
