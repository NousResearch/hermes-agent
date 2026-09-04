"""Bot-to-bot loop guard in ``_is_user_authorized`` (#91481).

The scenarios run with ``TELEGRAM_GROUP_ALLOWED_CHATS`` set: that allowlist admits every sender in
the chat, bots included, before the ``ALLOW_BOTS`` block runs. A guard hooked only into the
``ALLOW_BOTS`` block never sees a bot in an allowlisted group, which is the configuration that
produced the incident.
"""

import logging
import threading
from types import SimpleNamespace

import pytest

from gateway.bot_loop_guard import BotLoopGuard, BotLoopGuardSettings, load_settings, settings_from_config
from gateway.session import Platform, SessionSource

GROUP_CHAT = "-1001234567890"
OTHER_GROUP = "-1009876543210"
BOT_A = "111111111"
BOT_B = "222222222"
HUMAN = "100200300"


@pytest.fixture(autouse=True)
def _isolate_telegram_env(monkeypatch):
    for var in (
        "TELEGRAM_ALLOW_BOTS",
        "TELEGRAM_ALLOWED_USERS",
        "TELEGRAM_ALLOW_ALL_USERS",
        "TELEGRAM_GROUP_ALLOWED_USERS",
        "TELEGRAM_GROUP_ALLOWED_CHATS",
        "GATEWAY_ALLOW_ALL_USERS",
        "GATEWAY_ALLOWED_USERS",
    ):
        monkeypatch.delenv(var, raising=False)


@pytest.fixture
def clock():
    state = {"t": 1_000_000.0}
    return SimpleNamespace(now=lambda: state["t"], advance=lambda secs: state.__setitem__("t", state["t"] + secs))


@pytest.fixture
def settings():
    """Mutable holder so a test can flip settings on a live guard."""
    return {"value": BotLoopGuardSettings(max_events=20, window_seconds=60, cooldown_seconds=60)}


@pytest.fixture
def runner(clock, settings):
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.pairing_store = SimpleNamespace(is_approved=lambda *_a, **_kw: False)
    runner._bot_loop_guard = BotLoopGuard(settings=lambda: settings["value"], clock=clock.now)
    return runner


def _bot(user_id: str, chat_id: str = GROUP_CHAT, chat_type: str = "group") -> SessionSource:
    return SessionSource(
        platform=Platform.TELEGRAM, chat_id=chat_id, chat_type=chat_type,
        user_id=user_id, user_name=f"Bot{user_id}", is_bot=True,
    )


def _human(user_id: str = HUMAN, chat_id: str = GROUP_CHAT, chat_type: str = "group") -> SessionSource:
    return SessionSource(
        platform=Platform.TELEGRAM, chat_id=chat_id, chat_type=chat_type,
        user_id=user_id, user_name="Alice", is_bot=False,
    )


def _incident_config(monkeypatch):
    """ALLOW_BOTS on, a human allowlist, and the group-chat allowlist that short-circuits authz."""
    monkeypatch.setenv("TELEGRAM_ALLOW_BOTS", "mentions")
    monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", HUMAN)
    monkeypatch.setenv("TELEGRAM_GROUP_ALLOWED_CHATS", f"{GROUP_CHAT},{OTHER_GROUP}")


def _ping_pong(runner, turns: int, chat_id: str = GROUP_CHAT) -> list:
    return [runner._is_user_authorized(_bot(BOT_A if t % 2 == 0 else BOT_B, chat_id)) for t in range(turns)]


# --- authz integration -----------------------------------------------------


def test_ping_pong_of_40_turns_is_cut_at_the_budget(monkeypatch, runner):
    _incident_config(monkeypatch)

    verdicts = _ping_pong(runner, 40)

    assert verdicts[:20] == [True] * 20
    assert verdicts[20] is False
    assert not any(verdicts[20:])


def test_human_in_same_group_stays_authorized_during_cooldown(monkeypatch, runner, clock):
    _incident_config(monkeypatch)
    _ping_pong(runner, 25)

    assert runner._is_user_authorized(_bot(BOT_A)) is False
    assert runner._is_user_authorized(_human()) is True
    clock.advance(5)
    assert runner._is_user_authorized(_human()) is True


def test_human_traffic_is_never_metered(monkeypatch, runner):
    monkeypatch.setenv("TELEGRAM_ALLOW_BOTS", "mentions")
    monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", HUMAN)

    for _ in range(50):
        assert runner._is_user_authorized(_human(chat_id="123", chat_type="dm")) is True

    assert runner._bot_loop_guard.tracked_conversations == 0


def test_three_bots_in_one_group_share_one_budget(monkeypatch, runner):
    _incident_config(monkeypatch)
    senders = [BOT_A, BOT_B, "333333333"]

    verdicts = [runner._is_user_authorized(_bot(senders[i % 3])) for i in range(30)]

    assert verdicts[:20] == [True] * 20
    assert not any(verdicts[20:])


def test_other_group_has_its_own_budget(monkeypatch, runner):
    _incident_config(monkeypatch)
    _ping_pong(runner, 25)
    assert runner._is_user_authorized(_bot(BOT_A)) is False

    assert runner._is_user_authorized(_bot(BOT_A, chat_id=OTHER_GROUP)) is True
    assert runner._is_user_authorized(_bot(BOT_B, chat_id=OTHER_GROUP)) is True


def test_slow_traffic_never_trips(monkeypatch, runner, clock):
    _incident_config(monkeypatch)

    verdicts = []
    for turn in range(60):
        verdicts.append(runner._is_user_authorized(_bot(BOT_A if turn % 2 == 0 else BOT_B)))
        clock.advance(10)

    assert all(verdicts)


def test_window_expiry_readmits_without_tripping(monkeypatch, runner, clock):
    _incident_config(monkeypatch)
    assert all(_ping_pong(runner, 20))

    clock.advance(61)
    assert runner._is_user_authorized(_bot(BOT_A)) is True


def test_cooldown_expiry_readmits_with_a_fresh_budget(monkeypatch, runner, clock):
    _incident_config(monkeypatch)
    _ping_pong(runner, 21)
    assert runner._is_user_authorized(_bot(BOT_A)) is False

    clock.advance(61)
    assert all(_ping_pong(runner, 20))
    assert runner._is_user_authorized(_bot(BOT_A)) is False


def test_disabled_via_config_admits_everything(monkeypatch, runner, settings):
    _incident_config(monkeypatch)
    settings["value"] = settings_from_config({"gateway": {"bot_loop_guard": {"enabled": False}}})

    assert all(_ping_pong(runner, 40))


def test_bot_admitted_by_allow_bots_in_a_dm_is_metered(monkeypatch, runner):
    """The plain ALLOW_BOTS path (no chat allowlist) is covered too."""
    monkeypatch.setenv("TELEGRAM_ALLOW_BOTS", "mentions")
    monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", HUMAN)

    verdicts = [runner._is_user_authorized(_bot(BOT_A, chat_id="123", chat_type="dm")) for _ in range(25)]

    assert verdicts[:20] == [True] * 20
    assert not any(verdicts[20:])


def test_rejected_bot_messages_do_not_consume_budget(monkeypatch, runner):
    """Only admitted bot traffic counts, so an unauthorized bot cannot silence an authorized one."""
    monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", HUMAN)

    for _ in range(30):
        assert runner._is_user_authorized(_bot(BOT_A, chat_id="123", chat_type="dm")) is False

    assert runner._bot_loop_guard.tracked_conversations == 0


def test_platform_scopes_the_budget_key(monkeypatch, runner):
    _incident_config(monkeypatch)
    _ping_pong(runner, 21)
    assert runner._is_user_authorized(_bot(BOT_A)) is False

    monkeypatch.setenv("DISCORD_ALLOW_BOTS", "all")
    discord_bot = SessionSource(
        platform=Platform.DISCORD, chat_id=GROUP_CHAT, chat_type="group", user_id=BOT_A, is_bot=True,
    )
    assert runner._is_user_authorized(discord_bot) is True


def test_tripping_logs_one_operator_warning(monkeypatch, runner, caplog):
    _incident_config(monkeypatch)

    with caplog.at_level(logging.WARNING, logger="gateway.authz_mixin"):
        _ping_pong(runner, 25)

    trips = [r for r in caplog.records if r.name == "gateway.authz_mixin" and "Bot loop guard" in r.getMessage()]
    assert len(trips) == 1
    assert GROUP_CHAT in trips[0].getMessage()


def test_guard_is_created_lazily_on_a_bare_runner(monkeypatch):
    from gateway.run import GatewayRunner

    monkeypatch.setenv("TELEGRAM_ALLOW_BOTS", "mentions")
    runner = object.__new__(GatewayRunner)
    runner.pairing_store = SimpleNamespace(is_approved=lambda *_a, **_kw: False)

    assert runner._is_user_authorized(_bot(BOT_A, chat_id="123", chat_type="dm")) is True
    assert isinstance(runner._bot_loop_guard, BotLoopGuard)


# --- BotLoopGuard unit ------------------------------------------------------


def test_admit_states(clock):
    guard = BotLoopGuard(settings=lambda: BotLoopGuardSettings(max_events=2, window_seconds=10, cooldown_seconds=30), clock=clock.now)

    assert guard.admit("c") == (True, "ok")
    assert guard.admit("c") == (True, "ok")
    assert guard.admit("c") == (False, "tripped")
    assert guard.admit("c") == (False, "cooldown")
    clock.advance(31)
    assert guard.admit("c") == (True, "ok")


def test_concurrent_admits_respect_the_budget(clock):
    guard = BotLoopGuard(settings=lambda: BotLoopGuardSettings(max_events=20, window_seconds=60, cooldown_seconds=60), clock=clock.now)
    allowed = []
    lock = threading.Lock()

    def worker():
        for _ in range(10):
            ok, _state = guard.admit("c")
            with lock:
                allowed.append(ok)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert sum(allowed) == 20


def test_sweep_drops_idle_conversations(clock):
    guard = BotLoopGuard(settings=lambda: BotLoopGuardSettings(max_events=5, window_seconds=10, cooldown_seconds=10), clock=clock.now)
    for i in range(50):
        guard.admit(f"chat-{i}")
    assert guard.tracked_conversations == 50

    clock.advance(11)
    guard.admit("fresh")
    assert guard.tracked_conversations == 1


# --- settings ---------------------------------------------------------------


def test_settings_defaults_when_block_missing_or_malformed():
    defaults = BotLoopGuardSettings()
    assert settings_from_config({}) == defaults
    assert settings_from_config(None) == defaults
    assert settings_from_config({"gateway": {"bot_loop_guard": "yes"}}) == defaults
    assert settings_from_config({"gateway": {"bot_loop_guard": {"max_events": -1, "window_seconds": "abc"}}}) == defaults
    assert settings_from_config({"gateway": {"bot_loop_guard": {"enabled": "maybe", "max_events": True}}}) == defaults


def test_settings_parse_configured_values():
    parsed = settings_from_config({"gateway": {"bot_loop_guard": {
        "enabled": "false", "max_events": "7", "window_seconds": 30, "cooldown_seconds": 45.5,
    }}})
    assert parsed == BotLoopGuardSettings(enabled=False, max_events=7, window_seconds=30.0, cooldown_seconds=45.5)


def test_load_settings_reads_config_yaml(monkeypatch):
    import hermes_cli.config as hermes_config

    monkeypatch.setattr(hermes_config, "load_config_readonly", lambda: {"gateway": {"bot_loop_guard": {"max_events": 3}}})
    assert load_settings().max_events == 3

    def boom():
        raise RuntimeError("no config")

    monkeypatch.setattr(hermes_config, "load_config_readonly", boom)
    assert load_settings() == BotLoopGuardSettings()
