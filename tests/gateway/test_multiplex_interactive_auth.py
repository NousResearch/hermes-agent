"""Multiplex interactive-auth regressions (#86296, #92840, #72657, #87240 egress).

Real ``GatewayRunner`` methods on an ``object.__new__`` runner, real
``PairingStore`` files under a temp HERMES_HOME, multiplex active.
"""

from pathlib import Path
from types import SimpleNamespace

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.pairing import PairingStore
from gateway.profile_routing import ProfileRoute


@pytest.fixture
def mux_home(tmp_path, monkeypatch):
    from agent import secret_scope

    home = tmp_path / "hh"
    (home / "profiles" / "secondary").mkdir(parents=True)
    (home / ".env").write_text("")
    (home / "profiles" / "secondary" / ".env").write_text("")
    monkeypatch.setenv("HERMES_HOME", str(home))
    for key in (
        "TELEGRAM_ALLOWED_USERS",
        "TELEGRAM_ALLOW_BOTS",
        "GATEWAY_ALLOW_ALL_USERS",
        "GATEWAY_ALLOWED_USERS",
        "SLACK_ALLOW_ALL_USERS",
        "SLACK_ALLOWED_USERS",
    ):
        monkeypatch.delenv(key, raising=False)
    prev = secret_scope.is_multiplex_active()
    secret_scope.set_multiplex_active(True)
    yield home
    secret_scope.set_multiplex_active(prev)


def _runner(home):
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(multiplex_profiles=True)
    runner.config.profile_routes = [
        ProfileRoute(name="r", platform="telegram", chat_id="-100555", profile="secondary")
    ]
    runner.config.platforms = {Platform.TELEGRAM: PlatformConfig(enabled=True, extra={})}
    runner.pairing_store = PairingStore(profile="default")
    runner.pairing_stores = {
        "default": runner.pairing_store,
        "secondary": PairingStore(profile="secondary"),
    }
    runner._primary_profile_name = "default"
    runner._profile_adapters = {"secondary": {}}
    return runner


def _telegram(runner):
    from plugins.platforms.telegram.adapter import TelegramAdapter

    tg = object.__new__(TelegramAdapter)
    tg.config = PlatformConfig(enabled=True, extra={})
    tg._authorization_check = None
    tg._message_handler = runner._primary_message_handler()  # closure, no __self__
    runner.adapters = {Platform.TELEGRAM: tg}
    tg.set_authorization_check(runner._make_adapter_auth_check(Platform.TELEGRAM))
    return tg


def test_routed_primary_callback_uses_routed_pairing_store_and_transport_allowlist(mux_home):
    """#86296: shared primary bot + profile_routes → the inline-button caller
    is authorized by the ROUTED profile's pairing store, while env allowlists
    resolve under the transport (launch) home, exactly like inbound messages."""
    runner = _runner(mux_home)
    store = runner.pairing_stores["secondary"]
    store._save_json(store._approved_path("telegram"), {"777": {}})
    (mux_home / ".env").write_text("TELEGRAM_ALLOWED_USERS=999\n")
    tg = _telegram(runner)

    # Paired only in the routed profile → allowed in the routed chat only.
    assert tg._is_callback_user_authorized("777", chat_id="-100555", chat_type="supergroup") is True
    assert tg._is_callback_user_authorized("777", chat_id="-100999", chat_type="supergroup") is False
    # Transport-home allowlist honored in the routed chat (not the routed profile's empty scope).
    assert tg._is_callback_user_authorized("999", chat_id="-100555", chat_type="supergroup") is True
    assert tg._is_callback_user_authorized("888", chat_id="-100555", chat_type="supergroup") is False


def test_bot_sender_reaches_allow_bots_policy_through_callback(mux_home):
    """#92840: the early prefilter must carry ``is_bot`` so TELEGRAM_ALLOW_BOTS
    admits bot-authored messages under the multiplex closure handler."""
    from gateway.run import _profile_runtime_scope

    runner = _runner(mux_home)
    (mux_home / ".env").write_text("TELEGRAM_ALLOWED_USERS=999\nTELEGRAM_ALLOW_BOTS=all\n")
    tg = _telegram(runner)

    def msg(uid, is_bot):
        return SimpleNamespace(
            from_user=SimpleNamespace(id=uid, is_bot=is_bot, username="x", full_name="X"),
            chat=SimpleNamespace(id=-100777, type="supergroup", is_forum=False),
            sender_chat=None,
            message_thread_id=None,
            is_topic_message=False,
        )

    with _profile_runtime_scope(mux_home):
        assert tg._is_user_authorized_from_message(msg(4242, True)) is True
        assert tg._is_user_authorized_from_message(msg(4343, False)) is False


def test_slack_interactive_auth_prefers_wired_profile_check(mux_home, monkeypatch):
    """#72657: a multiplexed Slack adapter's button gate resolves through the
    wired ``_make_adapter_auth_check`` for its own profile; the DEFAULT
    profile's process-env allow-all never leaks in — not through the
    injected path, and not through the env-only fallback either."""
    from gateway.run import _profile_runtime_scope
    from plugins.platforms.slack.adapter import SlackAdapter

    runner = _runner(mux_home)
    runner.adapters = {}
    sec_home = mux_home / "profiles" / "secondary"
    (sec_home / ".env").write_text("SLACK_ALLOWED_USERS=U_SEC\n")
    monkeypatch.setenv("SLACK_ALLOW_ALL_USERS", "true")

    def slack(with_check):
        sl = object.__new__(SlackAdapter)
        sl.config = PlatformConfig(enabled=True, extra={})
        sl._authorization_check = None
        sl._message_handler = runner._make_profile_message_handler("secondary")
        if with_check:
            runner._profile_adapters = {"secondary": {Platform.SLACK: sl}}
            sl.set_authorization_check(
                runner._make_adapter_auth_check(Platform.SLACK, profile_name="secondary")
            )
        return sl

    with _profile_runtime_scope(sec_home):
        wired = slack(True)
        assert wired._is_interactive_user_authorized("U_SEC", channel_id="C1") is True
        assert wired._is_interactive_user_authorized("U_X", channel_id="C1") is False
        assert slack(False)._is_interactive_user_authorized("U_X", channel_id="C1") is False


def test_authorization_adapter_ignores_per_turn_active_profile(mux_home):
    """#87240 egress half: inside a secondary profile's runtime scope the
    default bot must not be handed to that profile (fail-closed None); the
    launch profile still resolves ``self.adapters``."""
    from gateway.run import _profile_runtime_scope

    runner = _runner(mux_home)
    default_bot = object()
    runner.adapters = {Platform.TELEGRAM: default_bot}

    with _profile_runtime_scope(mux_home / "profiles" / "secondary"):
        assert runner._authorization_adapter(Platform.TELEGRAM, profile="secondary") is None
    assert runner._authorization_adapter(Platform.TELEGRAM, profile="default") is default_bot


def test_channel_directory_path_follows_current_home(mux_home):
    """#87240: the directory file resolves against the CURRENT profile home,
    not the home that happened to import the module."""
    import gateway.channel_directory as cd
    from gateway.run import _profile_runtime_scope

    assert cd.DIRECTORY_PATH is None
    with _profile_runtime_scope(mux_home / "profiles" / "secondary"):
        assert cd._directory_path() == Path(mux_home / "profiles" / "secondary" / "channel_directory.json")
    assert cd._directory_path() == Path(mux_home / "channel_directory.json")
