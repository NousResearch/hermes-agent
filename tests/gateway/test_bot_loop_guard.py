"""Bot-to-bot pair loop protection (Telegram Bot API 10.0).

Telegram ships no loop guard of its own: core.telegram.org/api/bots/bot-to-bot
requires the BOT to "make bot-message handling terminate predictably" via
dedupe, rate limits and maximum interaction depth per sender/receiver pair.

Every test here was verified FAILING against the tree without the guard
(mutation-checked), which matters more than the count: exercising the real
configuration is what makes them meaningful.  In particular they run with
TELEGRAM_GROUP_ALLOWED_CHATS set, because that env var short-circuits
_is_user_authorized with `return True` ~32 lines before the is_bot block --
a guard placed in the is_bot block would never run for the one configuration
that needs it, and a test without the allowlist would go green through a
different code path.
"""

from types import SimpleNamespace

import pytest

from gateway.session import Platform, SessionSource

GROUP_CHAT = "-1001234567890"
OTHER_GROUP = "-1009876543210"


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch):
    for var in (
        "TELEGRAM_ALLOW_BOTS",
        "TELEGRAM_ALLOWED_USERS",
        "TELEGRAM_ALLOW_ALL_USERS",
        "TELEGRAM_GROUP_ALLOWED_USERS",
        "TELEGRAM_GROUP_ALLOWED_CHATS",
        "GATEWAY_ALLOW_ALL_USERS",
        "GATEWAY_ALLOWED_USERS",
        "HERMES_BOT_LOOP_PROTECTION",
        "HERMES_BOT_LOOP_MAX_EVENTS",
        "HERMES_BOT_LOOP_WINDOW_SEC",
        "HERMES_BOT_LOOP_COOLDOWN_SEC",
    ):
        monkeypatch.delenv(var, raising=False)

    from gateway.bot_loop_guard import reset_loop_guard

    reset_loop_guard()
    yield
    reset_loop_guard()


@pytest.fixture
def fake_clock(monkeypatch):
    """Drive the guard's sliding window deterministically."""
    from gateway import bot_loop_guard

    state = {"t": 1_000_000.0}
    monkeypatch.setattr(bot_loop_guard.time, "monotonic", lambda: state["t"])
    return SimpleNamespace(
        advance=lambda secs: state.__setitem__("t", state["t"] + secs),
        now=lambda: state["t"],
    )


def _runner():
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.pairing_store = SimpleNamespace(is_approved=lambda *_a, **_kw: False)
    return runner


def _bot(user_id: str, chat_id: str = GROUP_CHAT) -> SessionSource:
    """A fresh inbound bot message.

    A new SessionSource per turn, mirroring production (each update builds its
    own).  A hand-rolled stub does not work here: _is_user_authorized touches
    delivered_via_upstream_relay and other real fields and blows up with
    AttributeError.
    """
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id=chat_id,
        chat_type="group",
        user_id=user_id,
        user_name=f"Bot{user_id}",
    )
    source.is_bot = True
    return source


def _human(user_id: str = "100200300", chat_id: str = GROUP_CHAT, chat_type: str = "group") -> SessionSource:
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id=chat_id,
        chat_type=chat_type,
        user_id=user_id,
        user_name="Alice",
    )
    source.is_bot = False
    return source


def _real_config(monkeypatch, *, max_events="20"):
    """The configuration that actually reproduces the incident."""
    monkeypatch.setenv("TELEGRAM_ALLOW_BOTS", "mentions")
    monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "100200300")
    # The short-circuit that makes guard placement matter.
    monkeypatch.setenv("TELEGRAM_GROUP_ALLOWED_CHATS", f"{GROUP_CHAT},{OTHER_GROUP}")
    monkeypatch.setenv("HERMES_BOT_LOOP_MAX_EVENTS", max_events)
    monkeypatch.setenv("HERMES_BOT_LOOP_WINDOW_SEC", "60")
    monkeypatch.setenv("HERMES_BOT_LOOP_COOLDOWN_SEC", "60")


# --------------------------------------------------------------------------- 1


def test_ping_pong_of_40_turns_is_cut_at_turn_21(monkeypatch, fake_clock):
    """The incident, reduced: two bots replying to each other must terminate.

    Observed 2026-08-21 in production: 132 messages between two Hermes profiles
    in one group before a human intervened.  With a budget of 20 the 21st
    inbound bot message must be suppressed.
    """
    _real_config(monkeypatch, max_events="20")
    runner = _runner()

    verdicts = []
    for turn in range(40):
        sender = "111111111" if turn % 2 == 0 else "222222222"
        verdicts.append(runner._is_user_authorized(_bot(sender)))

    assert verdicts[:20] == [True] * 20, "first 20 turns must pass"
    assert verdicts[20] is False, "turn 21 must be suppressed"
    assert not any(verdicts[20:]), "the exchange must not resume inside cooldown"


# --------------------------------------------------------------------------- 2


def test_human_in_same_group_still_authorized_during_cooldown(monkeypatch, fake_clock):
    """The guard must never take the group down for people."""
    _real_config(monkeypatch, max_events="20")
    runner = _runner()

    for turn in range(25):
        runner._is_user_authorized(_bot("111111111" if turn % 2 == 0 else "222222222"))

    assert runner._is_user_authorized(_bot("111111111")) is False, "precondition: bots are cooling down"
    assert runner._is_user_authorized(_human()) is True
    fake_clock.advance(5)
    assert runner._is_user_authorized(_human()) is True


# --------------------------------------------------------------------------- 3


def test_human_dm_unaffected(monkeypatch, fake_clock):
    """Human DMs never enter the guard at all."""
    monkeypatch.setenv("TELEGRAM_ALLOW_BOTS", "mentions")
    monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "100200300")
    monkeypatch.setenv("HERMES_BOT_LOOP_MAX_EVENTS", "20")
    runner = _runner()

    for _ in range(50):
        assert runner._is_user_authorized(_human(chat_id="123", chat_type="dm")) is True

    from gateway.bot_loop_guard import loop_guard_state

    assert loop_guard_state()["tracked_pairs"] == 0, "human traffic must not be metered"


# --------------------------------------------------------------------------- 4


def test_three_bots_in_one_group_share_a_single_budget(monkeypatch, fake_clock):
    """Budget is keyed per CONVERSATION, not per (sender, receiver).

    This process only sees inbound messages, so the receiver is a constant.
    Keying on the pair would hand each sender its own budget, and N bots would
    need N x budget messages to trip a guard meant to cap the whole exchange.
    """
    _real_config(monkeypatch, max_events="20")
    runner = _runner()
    senders = ["111111111", "222222222", "333333333"]

    verdicts = [runner._is_user_authorized(_bot(senders[i % 3])) for i in range(30)]

    assert verdicts[:20] == [True] * 20
    assert not any(verdicts[20:]), "three bots must not get 3 x budget"


# --------------------------------------------------------------------------- 5


def test_other_group_has_its_own_budget(monkeypatch, fake_clock):
    """One noisy group must not silence bots elsewhere."""
    _real_config(monkeypatch, max_events="20")
    runner = _runner()

    for turn in range(25):
        runner._is_user_authorized(_bot("111111111" if turn % 2 == 0 else "222222222"))
    assert runner._is_user_authorized(_bot("111111111")) is False

    assert runner._is_user_authorized(_bot("111111111", chat_id=OTHER_GROUP)) is True
    assert runner._is_user_authorized(_bot("222222222", chat_id=OTHER_GROUP)) is True


# --------------------------------------------------------------------------- 6


def test_slow_traffic_never_blocks(monkeypatch, fake_clock):
    """1 message / 10 s for 10 minutes: legitimate pace, zero false positives."""
    _real_config(monkeypatch, max_events="20")
    runner = _runner()

    verdicts = []
    for turn in range(60):
        verdicts.append(runner._is_user_authorized(_bot("111111111" if turn % 2 == 0 else "222222222")))
        fake_clock.advance(10)

    assert all(verdicts), f"slow traffic blocked at index {verdicts.index(False) if not all(verdicts) else -1}"


# --------------------------------------------------------------------------- 7


def test_kill_switch_disables_the_guard(monkeypatch, fake_clock):
    """HERMES_BOT_LOOP_PROTECTION=off restores the previous behaviour exactly."""
    _real_config(monkeypatch, max_events="20")
    monkeypatch.setenv("HERMES_BOT_LOOP_PROTECTION", "off")
    runner = _runner()

    verdicts = [
        runner._is_user_authorized(_bot("111111111" if turn % 2 == 0 else "222222222"))
        for turn in range(40)
    ]

    assert all(verdicts), "kill switch must suppress the guard entirely"
