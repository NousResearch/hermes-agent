"""A Discord or Telegram allowlist edit in config.yaml reaches the next inbound event, no reconnect.

Each adapter copies its gate at connect time (per-profile isolation, #72348). These tests write a
real config.yaml, build the adapter's gate from the real ``load_gateway_config()``, edit the file,
and judge the next event: the edited list must decide, an operator's env value must still win
(#109032), and a multiplexed profile must read only its own file.
"""

import itertools
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


def test_unreadable_edit_keeps_the_last_good_lists(_home):
    _write(_home, {"allowed_roles": ["2"]})
    adapter, guild, member = _connected_discord()
    (_home / "config.yaml").write_text("discord: [unclosed\n", encoding="utf-8")
    assert _allowed(adapter, guild, member) is True


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
