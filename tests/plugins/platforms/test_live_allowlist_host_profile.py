"""A multiplexed gateway's host (launch) profile follows its own config.yaml allowlist edits.

Once the gateway multiplexes, the host profile's work runs under its bound scope, whose gate env is
``launch_secret_scope``: the env frozen at activation plus ``.env``. These tests freeze that env the way
activation does, edit the host's config.yaml, and judge the next event under that scope: a removed user
or a retracted ``allow_all_users`` grant is refused, while an operator's value (frozen env or ``.env``)
still wins, a failed load keeps the last good lists, and a secondary profile's mapping is never touched.
"""

import os

import pytest

from gateway.config import Platform, load_gateway_config
from tests.plugins.platforms.test_live_allowlist_edits import GATE_VARS, USER_ID, _allowed, _connected_discord, _write


@pytest.fixture(autouse=True)
def _home(tmp_path, monkeypatch):
    home = tmp_path / "default"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    for var in GATE_VARS:  # setenv first so teardown also removes what the loader bridges in
        for name in (var, f"HERMES_BRIDGED_{var}"):
            monkeypatch.setenv(name, "")
            monkeypatch.delenv(name)
    from gateway import config_loader
    monkeypatch.setattr(config_loader, "_BRIDGED_ALLOW_ALL_USERS", False)
    return home


@pytest.fixture
def multiplexed():
    from agent.secret_scope import reset_multiplex_context, set_multiplex_context
    token = set_multiplex_context(True)
    yield
    reset_multiplex_context(token)


def _telegram(home):
    from gateway.platforms._shared import mark_live_gate_baseline
    from plugins.platforms.telegram.adapter import TelegramAdapter
    adapter = TelegramAdapter(load_gateway_config().platforms[Platform.TELEGRAM])
    mark_live_gate_baseline(adapter)
    return adapter


def _freeze():
    """Activation: the launch env is frozen with the bridged lists of that moment."""
    import tui_gateway.launch_profile_policy as lpp
    lpp.capture_launch_env()


def _decision_under_host_scope(home, user_id):
    from gateway.run import _profile_runtime_scope
    from plugins.platforms.telegram.telegram_live_gates import env_allowlist_decision
    with _profile_runtime_scope(home):
        return env_allowlist_decision(user_id)


# Repro tests by BearHuddleston (#154562 review).
def test_a_multiplexed_launch_profiles_removed_user_is_refused_under_its_own_scope(_home):
    from agent.secret_scope import reset_multiplex_context, set_multiplex_context
    from gateway.run import _profile_runtime_scope

    _write(_home, {"allow_from": [str(USER_ID), "777"]})
    adapter, guild, member = _connected_discord(roles=())
    _freeze()
    token = set_multiplex_context(True)
    try:
        _write(_home, {"allow_from": ["777"]})
        with _profile_runtime_scope(_home):  # the scope is bound BEFORE the reload sees the edit
            verdict = _allowed(adapter, guild, member)
    finally:
        reset_multiplex_context(token)
    assert verdict is False


def test_a_multiplexed_launch_profiles_removed_telegram_user_is_refused_by_the_runner_gate(_home, multiplexed):
    from plugins.platforms.telegram.telegram_live_gates import refresh_telegram_gates

    _write(_home, telegram_cfg={"allow_from": ["111", "222"]})
    adapter = _telegram(_home)
    _freeze()
    _write(_home, telegram_cfg={"allow_from": ["111"]})
    refresh_telegram_gates(adapter)
    assert _decision_under_host_scope(_home, "222") is False
    assert _decision_under_host_scope(_home, "111") is True


def test_a_retracted_allow_all_grant_is_gone_under_the_host_scope(_home, multiplexed):
    from gateway.platforms._shared import platform_gate_env
    from gateway.run import _profile_runtime_scope

    (_home / "config.yaml").write_text("allow_all_users: true\nplatforms: {discord: {enabled: true}}\n", encoding="utf-8")
    adapter, _guild, _member = _connected_discord(roles=())
    _freeze()
    with _profile_runtime_scope(_home):
        assert platform_gate_env("GATEWAY_ALLOW_ALL_USERS") == "true"
        _write(_home, {"allow_from": ["777"]})
        adapter._refresh_live_gates()  # seen inside a body that bound the scope before the edit
        in_body = platform_gate_env("GATEWAY_ALLOW_ALL_USERS")
    with _profile_runtime_scope(_home):
        next_body = platform_gate_env("GATEWAY_ALLOW_ALL_USERS")
    assert (in_body, next_body) == ("", "")


def test_the_host_profiles_dotenv_value_still_wins_over_a_yaml_edit(_home, multiplexed):
    from plugins.platforms.telegram.telegram_live_gates import refresh_telegram_gates

    from gateway.run import _profile_runtime_scope
    from plugins.platforms.telegram.telegram_live_gates import env_allowlist_decision

    (_home / ".env").write_text("TELEGRAM_ALLOWED_USERS=111,222\n", encoding="utf-8")
    _write(_home, telegram_cfg={"allow_from": ["111", "222"]})
    adapter = _telegram(_home)
    _freeze()
    _write(_home, telegram_cfg={"allow_from": ["111"]})
    with _profile_runtime_scope(_home):
        refresh_telegram_gates(adapter)
        in_body = env_allowlist_decision("222")
    assert (in_body, _decision_under_host_scope(_home, "222")) == (True, True)


def test_a_frozen_operator_value_is_not_replaced_by_a_later_bridged_one(_home, multiplexed, monkeypatch):
    """An env-only operator value (systemd ``Environment=``, ``op run``) lives only in the frozen env. If
    the live env later loses it and the bridge writes the YAML list there, the host still uses the
    operator's: live process env never overrides a frozen value the bridge did not own."""
    from plugins.platforms.telegram.telegram_live_gates import refresh_telegram_gates

    monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "999")
    _write(_home, telegram_cfg={"allow_from": ["111"]})
    adapter = _telegram(_home)
    assert os.environ["TELEGRAM_ALLOWED_USERS"] == "999"  # the operator's value wins at startup
    _freeze()
    monkeypatch.delenv("TELEGRAM_ALLOWED_USERS")
    _write(_home, telegram_cfg={"allow_from": ["111", "222"]})
    refresh_telegram_gates(adapter)
    assert os.environ["TELEGRAM_ALLOWED_USERS"] == "111,222"  # the bridge now owns the live value
    assert (_decision_under_host_scope(_home, "999"), _decision_under_host_scope(_home, "222")) == (True, False)


@pytest.mark.parametrize("drop_marker", [True, False])
def test_a_frozen_bridged_list_the_bridge_did_not_retract_is_kept(_home, multiplexed, monkeypatch, drop_marker):
    """Only the bridge's own retraction (name and marker both gone) drops a frozen bridged list. A live
    value some other writer left, without a marker or over the bridge's stale one, is not evidence the
    list was removed or replaced: dropping it would leave the host with no list at all, which an
    allowlist-less gate reads as open."""
    _write(_home, telegram_cfg={"allow_from": ["111"]})
    _telegram(_home)
    _freeze()
    monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "333")
    if drop_marker:
        monkeypatch.delenv("HERMES_BRIDGED_TELEGRAM_ALLOWED_USERS")
    assert (_decision_under_host_scope(_home, "111"), _decision_under_host_scope(_home, "333")) == (True, False)


def test_an_operator_allow_all_grant_survives_the_host_scope(_home, multiplexed, monkeypatch):
    """GATEWAY_ALLOW_ALL_USERS from the gateway's own environment (systemd, ``op run``) is not the
    bridge's grant: following the bridge never revokes it."""
    from gateway.platforms._shared import platform_gate_env
    from gateway.run import _profile_runtime_scope

    monkeypatch.setenv("GATEWAY_ALLOW_ALL_USERS", "true")
    _write(_home, telegram_cfg={"allow_from": ["111"]})
    _telegram(_home)
    _freeze()
    with _profile_runtime_scope(_home):
        assert platform_gate_env("GATEWAY_ALLOW_ALL_USERS") == "true"


def test_a_failed_host_reload_keeps_the_last_good_list(_home, multiplexed):
    """An edit whose typed construction fails publishes nothing: the host scope neither gains the added
    user nor loses the listed one; the next good save then applies."""
    from plugins.platforms.telegram.telegram_live_gates import refresh_telegram_gates

    _write(_home, telegram_cfg={"allow_from": ["111"]})
    adapter = _telegram(_home)
    _freeze()
    (_home / "config.yaml").write_text(
        "telegram: {allow_from: ['111', '222']}\nplatforms: {discord: {home_channel: {name: x}}}\n", encoding="utf-8")
    refresh_telegram_gates(adapter)
    assert (_decision_under_host_scope(_home, "111"), _decision_under_host_scope(_home, "222")) == (True, False)
    _write(_home, telegram_cfg={"allow_from": ["222"]})
    refresh_telegram_gates(adapter)
    assert (_decision_under_host_scope(_home, "111"), _decision_under_host_scope(_home, "222")) == (False, True)


def test_a_host_reload_never_touches_a_secondary_profiles_scope(_home, tmp_path, multiplexed):
    """The host's edit can first reach its gate from a secondary profile's body (a routed task). The
    reload runs as the host; the secondary's installed mapping keeps its own value, and a fresh
    secondary scope never gains the host's list."""
    from agent.secret_scope import current_secret_scope
    from gateway.platforms._shared import platform_gate_env
    from gateway.run import _profile_runtime_scope
    from plugins.platforms.telegram.telegram_live_gates import refresh_telegram_gates

    secondary = tmp_path / "b"
    secondary.mkdir()
    (secondary / ".env").write_text("TELEGRAM_ALLOWED_USERS=555\n", encoding="utf-8")
    _write(_home, telegram_cfg={"allow_from": ["111", "222"]})
    adapter = _telegram(_home)
    _freeze()
    _write(_home, telegram_cfg={"allow_from": ["111"]})
    with _profile_runtime_scope(secondary):
        before = dict(current_secret_scope())
        refresh_telegram_gates(adapter)
        assert dict(current_secret_scope()) == before
        assert platform_gate_env("TELEGRAM_ALLOWED_USERS") == "555"
    with _profile_runtime_scope(secondary):
        assert platform_gate_env("TELEGRAM_ALLOWED_USERS") == "555"
    assert _decision_under_host_scope(_home, "222") is False


def test_a_host_reload_under_a_foreign_stamped_scope_waits_for_the_host(_home, tmp_path, multiplexed):
    """A body holding another home's secret scope (no home override) sees the host's edit first. The
    reload runs under the host's own scope, so it never writes the host's lists into the foreign
    mapping, and the host's next own event refuses the removed user."""
    from agent.secret_scope import current_secret_scope, reset_secret_scope, set_secret_scope
    from plugins.platforms.telegram.telegram_live_gates import refresh_telegram_gates

    secondary = tmp_path / "b"
    secondary.mkdir()
    _write(_home, telegram_cfg={"allow_from": ["111", "222"]})
    adapter = _telegram(_home)
    _freeze()
    _write(_home, telegram_cfg={"allow_from": ["111"]})
    token = set_secret_scope({"TELEGRAM_ALLOWED_USERS": "555"}, profile_home=str(secondary))
    try:
        refresh_telegram_gates(adapter)
        assert dict(current_secret_scope()) == {"TELEGRAM_ALLOWED_USERS": "555"}
    finally:
        reset_secret_scope(token)
    refresh_telegram_gates(adapter)
    assert _decision_under_host_scope(_home, "222") is False


class _OrderedScope(dict):
    """A mapping that records every key it loses, in order, and what the gate would see then."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.views: list[tuple[str, bool, bool]] = []

    def pop(self, key, *default):
        value = super().pop(key, *default)
        self.views.append((key, "TELEGRAM_ALLOWED_USERS" in self, "GATEWAY_ALLOW_ALL_USERS" in self))
        return value


def test_a_retracted_grant_goes_before_the_list_it_masked(monkeypatch):
    """One edit removes a list AND the allow_all_users it masked: a reader sharing the mapping
    must never see "no list, grant still on", which admits every sender."""
    from gateway import config_loader
    from gateway.bridged_env import follow_bridged_env
    from gateway.platforms._shared import bridged_env_digest, bridged_env_marker

    name, value = "TELEGRAM_ALLOWED_USERS", "111"
    scope = _OrderedScope({name: value, bridged_env_marker(name): bridged_env_digest(value),
                           "GATEWAY_ALLOW_ALL_USERS": "true"})
    for var in (name, bridged_env_marker(name), "GATEWAY_ALLOW_ALL_USERS"):
        monkeypatch.delenv(var, raising=False)  # the bridge retracted both
    monkeypatch.setattr(config_loader, "_BRIDGED_ALLOW_ALL_USERS", False)

    follow_bridged_env(scope)

    assert name not in scope and "GATEWAY_ALLOW_ALL_USERS" not in scope
    assert not any(not has_list and has_grant for _key, has_list, has_grant in scope.views), scope.views


def test_following_snapshots_the_env_under_the_load_lock(monkeypatch):
    """A load publishes all-or-nothing under YAML_ENV_LOCK; following reads the env only while
    holding it, so it never sees a list rewritten without its marker (or the reverse)."""
    from collections.abc import MutableMapping

    from gateway import bridged_env

    seen: list[bool] = []

    class _Env(MutableMapping):
        def __init__(self, env):
            self._env = dict(env)

        def __getitem__(self, key):
            seen.append(bridged_env.YAML_ENV_LOCK._is_owned())
            return self._env[key]

        def __iter__(self):
            return iter(self._env)

        def __len__(self):
            return len(self._env)

        def __setitem__(self, key, value):
            self._env[key] = value

        def __delitem__(self, key):
            del self._env[key]

        def get(self, key, default=None):
            seen.append(bridged_env.YAML_ENV_LOCK._is_owned())
            return self._env.get(key, default)

    monkeypatch.setattr(bridged_env.os, "environ", _Env(os.environ))

    bridged_env.follow_bridged_env({})

    assert seen and all(seen), seen


def test_a_host_discord_reload_under_a_foreign_stamped_scope_reads_the_hosts_own_lists(_home, tmp_path, multiplexed):
    """A body holding another profile's secret scope (stamped, no home override) sees the host's edit
    first: the host bot's gate must come from the host's own lists, never the secondary's mapping."""
    from agent.secret_scope import reset_secret_scope, set_secret_scope

    secondary = tmp_path / "b"
    secondary.mkdir()
    _write(_home, {"allow_from": ["777"]})
    adapter, guild, member = _connected_discord(roles=())
    _freeze()
    _write(_home, {"allow_from": ["778"]})
    token = set_secret_scope({"DISCORD_ALLOWED_USERS": str(member.id)}, profile_home=str(secondary))
    try:
        verdict = _allowed(adapter, guild, member)
    finally:
        reset_secret_scope(token)
    assert verdict is False


def test_moving_users_between_lists_never_shows_a_reader_no_list(monkeypatch):
    """With a grant on, one edit retracts one bridged list and adds another: the new one must land
    before the old one goes (the retracted name here sorts first)."""
    from gateway import config_loader
    from gateway.bridged_env import follow_bridged_env
    from gateway.platforms._shared import bridged_env_digest, bridged_env_marker

    old, new = "TELEGRAM_ALLOWED_USERS", "TELEGRAM_GROUP_ALLOWED_USERS"
    seen = []

    class _Scope(dict):
        def pop(self, key, *default):
            value = super().pop(key, *default)
            seen.append(old in self or new in self)
            return value

    scope = _Scope({old: "111", bridged_env_marker(old): bridged_env_digest("111"),
                    "GATEWAY_ALLOW_ALL_USERS": "true"})
    for var in (old, bridged_env_marker(old)):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv(new, "111")
    monkeypatch.setenv(bridged_env_marker(new), bridged_env_digest("111"))
    monkeypatch.setenv("GATEWAY_ALLOW_ALL_USERS", "true")
    monkeypatch.setattr(config_loader, "_BRIDGED_ALLOW_ALL_USERS", True)

    follow_bridged_env(scope)

    assert old not in scope and scope[new] == "111" and scope["GATEWAY_ALLOW_ALL_USERS"] == "true"
    assert seen and all(seen), seen
