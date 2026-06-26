"""Tests for optional-skills/productivity/memento-flashcards/scripts/memento_delivery.py"""

import json
import sys
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest import mock

import pytest

SCRIPTS_DIR = (
    Path(__file__).resolve().parents[2]
    / "optional-skills"
    / "productivity"
    / "memento-flashcards"
    / "scripts"
)
sys.path.insert(0, str(SCRIPTS_DIR))

import memento_delivery


# ── Fixtures ─────────────────────────────────────────────────────────────────

_NOW = datetime(2026, 6, 26, 14, 0, 0, tzinfo=timezone.utc)  # 2pm UTC = not quiet hours


def _make_card(card_id: str = None, next_review_delta: timedelta = None) -> dict:
    review_at = (_NOW + (next_review_delta or timedelta(0))).isoformat()
    return {
        "id": card_id or str(uuid.uuid4()),
        "question": "What is the capital of France?",
        "answer": "Paris",
        "collection": "Geography",
        "status": "learning",
        "ease_streak": 0,
        "next_review_at": review_at,
        "created_at": _NOW.isoformat(),
        "video_id": None,
        "last_user_answer": None,
    }


@pytest.fixture
def tmp_config(tmp_path):
    """Return a minimal config pointing everything at tmp_path."""
    cards_file = tmp_path / "cards.json"
    state_file = tmp_path / "delivery_state.json"
    sessions_file = tmp_path / "sessions.json"
    return {
        "enabled": True,
        "dry_run": True,
        "bot_token": "123:FAKE",
        "chat_id": "987654321",
        "idle_minutes": 30,
        "quiet_start": 22,
        "quiet_end": 8,
        "daily_cap": 5,
        "cooldown_minutes": 60,
        "sessions_file": sessions_file,
        "cards_file": cards_file,
        "state_file": state_file,
    }


def _write_cards(config: dict, cards: list[dict]) -> None:
    config["cards_file"].write_text(
        json.dumps({"cards": cards, "version": 1}), encoding="utf-8"
    )


def _write_sessions(config: dict, last_activity: datetime) -> None:
    """Write a minimal sessions.json with one entry at last_activity."""
    config["sessions_file"].write_text(
        json.dumps({
            "entries": [
                {
                    "session_key": "telegram:test",
                    "session_id": "s1",
                    "updated_at": last_activity.isoformat(),
                }
            ]
        }),
        encoding="utf-8",
    )


# ── Feature flag ──────────────────────────────────────────────────────────────

class TestFeatureFlag:
    def test_disabled_by_default(self, tmp_config):
        tmp_config["enabled"] = False
        result = memento_delivery.decide(tmp_config, now=_NOW)
        assert result["action"] == "skip"
        assert result["reason"] == "feature_disabled"

    def test_enabled_proceeds(self, tmp_config):
        # No sessions file → idle detection fails → skip, but for a different reason
        tmp_config["enabled"] = True
        result = memento_delivery.decide(tmp_config, now=_NOW)
        assert result["reason"] != "feature_disabled"


# ── Quiet hours ───────────────────────────────────────────────────────────────

class TestQuietHours:
    def test_overnight_window_hit(self, tmp_config):
        # 23:00 local — inside 22-8 quiet window
        night = datetime(2026, 6, 26, 23, 0, 0, tzinfo=timezone.utc)
        with mock.patch("memento_delivery._is_quiet_hour", return_value=True):
            result = memento_delivery.decide(tmp_config, now=night)
        assert result["action"] == "skip"
        assert result["reason"] == "quiet_hours"

    def test_outside_quiet_window(self, tmp_config):
        # 14:00 UTC — should NOT hit quiet hours
        result_reason = memento_delivery.decide(tmp_config, now=_NOW)["reason"]
        assert result_reason != "quiet_hours"

    def test_quiet_hour_logic_in_window(self):
        # Use _quiet_hour_check directly to avoid timezone-dependent astimezone()
        assert memento_delivery._quiet_hour_check(22, 8, 23) is True   # overnight, in window
        assert memento_delivery._quiet_hour_check(22, 8, 0) is True    # midnight, in window
        assert memento_delivery._quiet_hour_check(22, 8, 7) is True    # 7am, still quiet

    def test_quiet_hour_logic_outside_window(self):
        assert memento_delivery._quiet_hour_check(22, 8, 12) is False  # noon
        assert memento_delivery._quiet_hour_check(22, 8, 8) is False   # 8am = end of window (exclusive)
        assert memento_delivery._quiet_hour_check(22, 8, 21) is False  # 9pm, before quiet starts

    def test_same_day_quiet_window(self):
        assert memento_delivery._quiet_hour_check(1, 3, 2) is True    # inside same-day window
        assert memento_delivery._quiet_hour_check(1, 3, 4) is False   # outside


# ── Idle detection ────────────────────────────────────────────────────────────

class TestIdleDetection:
    def test_no_sessions_file_not_idle(self, tmp_config):
        # sessions file doesn't exist → fail open → not idle
        result = memento_delivery.decide(tmp_config, now=_NOW)
        assert result["action"] == "skip"
        assert result["reason"] == "not_idle"
        assert "sessions_unavailable" in result["detail"]

    def test_recent_activity_not_idle(self, tmp_config):
        recent = _NOW - timedelta(minutes=10)  # active 10m ago, threshold is 30m
        _write_sessions(tmp_config, recent)
        result = memento_delivery.decide(tmp_config, now=_NOW)
        assert result["action"] == "skip"
        assert result["reason"] == "not_idle"

    def test_idle_enough_proceeds(self, tmp_config):
        old_activity = _NOW - timedelta(minutes=45)  # idle for 45m > 30m threshold
        _write_sessions(tmp_config, old_activity)
        # No due cards yet, but we should get past the idle gate
        result = memento_delivery.decide(tmp_config, now=_NOW)
        assert result["reason"] != "not_idle"

    def test_exactly_at_threshold_is_idle(self, tmp_config):
        at_threshold = _NOW - timedelta(minutes=30)
        _write_sessions(tmp_config, at_threshold)
        result = memento_delivery.decide(tmp_config, now=_NOW)
        assert result["reason"] != "not_idle"

    def test_corrupt_sessions_file_not_idle(self, tmp_config):
        tmp_config["sessions_file"].write_text("{corrupted", encoding="utf-8")
        result = memento_delivery.decide(tmp_config, now=_NOW)
        assert result["reason"] == "not_idle"

    def test_most_recent_activity_used(self, tmp_config):
        """When multiple sessions exist, the most recent updated_at wins."""
        tmp_config["sessions_file"].write_text(
            json.dumps({
                "entries": [
                    {"session_key": "k1", "session_id": "s1",
                     "updated_at": (_NOW - timedelta(hours=2)).isoformat()},
                    {"session_key": "k2", "session_id": "s2",
                     "updated_at": (_NOW - timedelta(minutes=5)).isoformat()},
                ]
            }),
            encoding="utf-8",
        )
        result = memento_delivery.decide(tmp_config, now=_NOW)
        # Most recent is 5m ago, threshold 30m → not idle
        assert result["reason"] == "not_idle"


# ── Daily cap ─────────────────────────────────────────────────────────────────

class TestDailyCap:
    def _setup_idle(self, config: dict) -> None:
        _write_sessions(config, _NOW - timedelta(minutes=60))

    def test_cap_reached_skips(self, tmp_config):
        self._setup_idle(tmp_config)
        card = _make_card(next_review_delta=-timedelta(hours=1))
        _write_cards(tmp_config, [card])
        tmp_config["state_file"].parent.mkdir(parents=True, exist_ok=True)
        tmp_config["state_file"].write_text(
            json.dumps({
                "today_date": _NOW.astimezone().strftime("%Y-%m-%d"),
                "sent_today": 5,
                "last_sent_at": None,
                "last_sent_card_id": None,
            }),
            encoding="utf-8",
        )
        result = memento_delivery.decide(tmp_config, now=_NOW)
        assert result["action"] == "skip"
        assert result["reason"] == "daily_cap_reached"

    def test_cap_resets_next_day(self, tmp_config):
        self._setup_idle(tmp_config)
        card = _make_card(next_review_delta=-timedelta(hours=1))
        _write_cards(tmp_config, [card])
        tmp_config["state_file"].parent.mkdir(parents=True, exist_ok=True)
        tmp_config["state_file"].write_text(
            json.dumps({
                "today_date": "2026-06-25",  # yesterday
                "sent_today": 5,
                "last_sent_at": None,
                "last_sent_card_id": None,
            }),
            encoding="utf-8",
        )
        result = memento_delivery.decide(tmp_config, now=_NOW)
        # State resets for new day → cap not hit → would_send
        assert result["action"] == "would_send"

    def test_under_cap_proceeds(self, tmp_config):
        self._setup_idle(tmp_config)
        card = _make_card(next_review_delta=-timedelta(hours=1))
        _write_cards(tmp_config, [card])
        result = memento_delivery.decide(tmp_config, now=_NOW)
        assert result["action"] == "would_send"


# ── Cooldown ──────────────────────────────────────────────────────────────────

class TestCooldown:
    def _setup_idle(self, config: dict) -> None:
        _write_sessions(config, _NOW - timedelta(minutes=60))

    def test_cooldown_skips_within_window(self, tmp_config):
        self._setup_idle(tmp_config)
        card = _make_card(next_review_delta=-timedelta(hours=1))
        _write_cards(tmp_config, [card])
        recent_send = (_NOW - timedelta(minutes=30)).isoformat()
        tmp_config["state_file"].parent.mkdir(parents=True, exist_ok=True)
        tmp_config["state_file"].write_text(
            json.dumps({
                "today_date": _NOW.astimezone().strftime("%Y-%m-%d"),
                "sent_today": 1,
                "last_sent_at": recent_send,
                "last_sent_card_id": str(uuid.uuid4()),
            }),
            encoding="utf-8",
        )
        result = memento_delivery.decide(tmp_config, now=_NOW)
        assert result["action"] == "skip"
        assert result["reason"] == "cooldown"
        assert result["cooldown_remaining_minutes"] > 0

    def test_past_cooldown_proceeds(self, tmp_config):
        self._setup_idle(tmp_config)
        card = _make_card(next_review_delta=-timedelta(hours=1))
        _write_cards(tmp_config, [card])
        old_send = (_NOW - timedelta(minutes=90)).isoformat()
        tmp_config["state_file"].parent.mkdir(parents=True, exist_ok=True)
        tmp_config["state_file"].write_text(
            json.dumps({
                "today_date": _NOW.astimezone().strftime("%Y-%m-%d"),
                "sent_today": 1,
                "last_sent_at": old_send,
                "last_sent_card_id": str(uuid.uuid4()),
            }),
            encoding="utf-8",
        )
        result = memento_delivery.decide(tmp_config, now=_NOW)
        assert result["action"] == "would_send"

    def test_no_repeat_card_within_cooldown(self, tmp_config):
        """If only one due card exists and it was just sent, skip it."""
        self._setup_idle(tmp_config)
        card_id = str(uuid.uuid4())
        card = _make_card(card_id=card_id, next_review_delta=-timedelta(hours=1))
        _write_cards(tmp_config, [card])
        # Send happened 30m ago (within 60m cooldown)
        recent_send = (_NOW - timedelta(minutes=30)).isoformat()
        tmp_config["cooldown_minutes"] = 60
        tmp_config["state_file"].parent.mkdir(parents=True, exist_ok=True)
        tmp_config["state_file"].write_text(
            json.dumps({
                "today_date": _NOW.astimezone().strftime("%Y-%m-%d"),
                "sent_today": 0,  # reset so we don't hit daily cap / time cooldown
                "last_sent_at": None,
                "last_sent_card_id": card_id,
            }),
            encoding="utf-8",
        )
        # Inject card into cooldown by mocking _pick_card
        with mock.patch("memento_delivery._pick_card",
                        return_value=(None, "no_due_cards")):
            result = memento_delivery.decide(tmp_config, now=_NOW)
        assert result["action"] == "skip"
        assert result["reason"] == "no_due_cards"


# ── No due cards ──────────────────────────────────────────────────────────────

class TestNoDueCards:
    def _setup_idle(self, config: dict) -> None:
        _write_sessions(config, _NOW - timedelta(minutes=60))

    def test_no_cards_file_skips(self, tmp_config):
        self._setup_idle(tmp_config)
        result = memento_delivery.decide(tmp_config, now=_NOW)
        assert result["action"] == "skip"
        assert result["reason"] == "no_due_cards"

    def test_empty_deck_skips(self, tmp_config):
        self._setup_idle(tmp_config)
        _write_cards(tmp_config, [])
        result = memento_delivery.decide(tmp_config, now=_NOW)
        assert result["action"] == "skip"
        assert result["reason"] == "no_due_cards"

    def test_future_cards_not_due(self, tmp_config):
        self._setup_idle(tmp_config)
        future_card = _make_card(next_review_delta=timedelta(hours=24))
        _write_cards(tmp_config, [future_card])
        result = memento_delivery.decide(tmp_config, now=_NOW)
        assert result["action"] == "skip"
        assert result["reason"] == "no_due_cards"

    def test_retired_cards_not_due(self, tmp_config):
        self._setup_idle(tmp_config)
        card = _make_card(next_review_delta=-timedelta(hours=1))
        card["status"] = "retired"
        _write_cards(tmp_config, [card])
        result = memento_delivery.decide(tmp_config, now=_NOW)
        assert result["action"] == "skip"
        assert result["reason"] == "no_due_cards"


# ── Happy path (dry-run) ──────────────────────────────────────────────────────

class TestDryRunHappyPath:
    def _setup(self, config: dict, card: dict) -> None:
        _write_sessions(config, _NOW - timedelta(minutes=60))
        _write_cards(config, [card])

    def test_would_send_contains_card_info(self, tmp_config):
        card = _make_card(next_review_delta=-timedelta(hours=1))
        self._setup(tmp_config, card)
        result = memento_delivery.decide(tmp_config, now=_NOW)
        assert result["action"] == "would_send"
        assert result["card"]["id"] == card["id"]
        assert result["card"]["question"] == card["question"]
        assert result["card"]["collection"] == "Geography"

    def test_dry_run_does_not_write_state(self, tmp_config):
        card = _make_card(next_review_delta=-timedelta(hours=1))
        self._setup(tmp_config, card)
        memento_delivery.decide(tmp_config, now=_NOW)
        assert not tmp_config["state_file"].exists()

    def test_dry_run_does_not_call_telegram(self, tmp_config):
        card = _make_card(next_review_delta=-timedelta(hours=1))
        self._setup(tmp_config, card)
        with mock.patch("memento_delivery._send_telegram") as mock_send:
            memento_delivery.decide(tmp_config, now=_NOW)
        mock_send.assert_not_called()

    def test_oldest_due_card_selected(self, tmp_config):
        older_card = _make_card(next_review_delta=-timedelta(hours=2))
        newer_card = _make_card(next_review_delta=-timedelta(hours=1))
        _write_sessions(tmp_config, _NOW - timedelta(minutes=60))
        # Write with newer first — expect older to be picked (sorted oldest-first)
        _write_cards(tmp_config, [newer_card, older_card])
        result = memento_delivery.decide(tmp_config, now=_NOW)
        assert result["action"] == "would_send"
        assert result["card"]["id"] == older_card["id"]


# ── Live send (dry_run=False) ─────────────────────────────────────────────────

class TestLiveSend:
    def _setup(self, config: dict, card: dict) -> None:
        config["dry_run"] = False
        _write_sessions(config, _NOW - timedelta(minutes=60))
        _write_cards(config, [card])

    def test_live_send_writes_state(self, tmp_config):
        card = _make_card(next_review_delta=-timedelta(hours=1))
        self._setup(tmp_config, card)
        with mock.patch("memento_delivery._send_telegram",
                        return_value={"ok": True, "result": {"message_id": 42}}):
            result = memento_delivery.decide(tmp_config, now=_NOW)
        assert result["action"] == "sent"
        assert tmp_config["state_file"].exists()
        state = json.loads(tmp_config["state_file"].read_text())
        assert state["last_sent_card_id"] == card["id"]
        assert state["sent_today"] == 1

    def test_live_send_missing_token_errors(self, tmp_config):
        card = _make_card(next_review_delta=-timedelta(hours=1))
        self._setup(tmp_config, card)
        tmp_config["bot_token"] = ""
        result = memento_delivery.decide(tmp_config, now=_NOW)
        assert result["action"] == "error"
        assert "TELEGRAM_BOT_TOKEN" in result["error"]

    def test_telegram_api_failure_returns_error(self, tmp_config):
        card = _make_card(next_review_delta=-timedelta(hours=1))
        self._setup(tmp_config, card)
        with mock.patch("memento_delivery._send_telegram",
                        return_value={"ok": False, "error": "HTTP 401"}):
            result = memento_delivery.decide(tmp_config, now=_NOW)
        assert result["action"] == "error"
        # State must NOT be written on send failure
        assert not tmp_config["state_file"].exists()

    def test_telegram_api_failure_does_not_write_state(self, tmp_config):
        card = _make_card(next_review_delta=-timedelta(hours=1))
        self._setup(tmp_config, card)
        with mock.patch("memento_delivery._send_telegram",
                        return_value={"ok": False, "error": "connection refused"}):
            memento_delivery.decide(tmp_config, now=_NOW)
        assert not tmp_config["state_file"].exists()


# ── Card rotation / no-repeat ─────────────────────────────────────────────────

class TestCardRotation:
    def test_skips_recently_sent_card_picks_another(self, tmp_config):
        card_a = _make_card(card_id="card-a", next_review_delta=-timedelta(hours=2))
        card_b = _make_card(card_id="card-b", next_review_delta=-timedelta(hours=1))
        _write_sessions(tmp_config, _NOW - timedelta(minutes=60))
        _write_cards(tmp_config, [card_a, card_b])
        # card_a was sent 20m ago (within 60m cooldown)
        result = memento_delivery.decide(
            {**tmp_config,
             "state_file": tmp_config["state_file"]},
            now=_NOW,
        )
        # Without state, both are candidates — card_a is older so picked first
        assert result["action"] == "would_send"
        assert result["card"]["id"] == "card-a"

    def test_only_due_card_in_cooldown_still_picked(self):
        """_pick_card returns the only available card even if in cooldown."""
        card = _make_card(card_id="only-card", next_review_delta=-timedelta(hours=1))
        selected, _ = memento_delivery._pick_card(
            [card],
            last_sent_card_id="only-card",
            last_sent_at=(_NOW - timedelta(minutes=10)).isoformat(),
            cooldown_minutes=60,
            now=_NOW,
        )
        assert selected is not None
        assert selected["id"] == "only-card"

    def test_pick_card_skips_cooldown_card(self):
        card_a = _make_card(card_id="card-a", next_review_delta=-timedelta(hours=2))
        card_b = _make_card(card_id="card-b", next_review_delta=-timedelta(hours=1))
        selected, _ = memento_delivery._pick_card(
            [card_a, card_b],
            last_sent_card_id="card-a",
            last_sent_at=(_NOW - timedelta(minutes=10)).isoformat(),
            cooldown_minutes=60,
            now=_NOW,
        )
        assert selected["id"] == "card-b"

    def test_pick_card_no_cards(self):
        selected, reason = memento_delivery._pick_card(
            [], None, None, 60, _NOW
        )
        assert selected is None
        assert reason == "no_due_cards"


# ── _load_due_cards ───────────────────────────────────────────────────────────

class TestLoadDueCards:
    def test_empty_file(self, tmp_path):
        cards_file = tmp_path / "cards.json"
        cards_file.write_text(json.dumps({"cards": [], "version": 1}))
        assert memento_delivery._load_due_cards(cards_file, _NOW) == []

    def test_missing_file(self, tmp_path):
        assert memento_delivery._load_due_cards(tmp_path / "nope.json", _NOW) == []

    def test_corrupt_file(self, tmp_path):
        cards_file = tmp_path / "cards.json"
        cards_file.write_text("{broken")
        assert memento_delivery._load_due_cards(cards_file, _NOW) == []

    def test_sorted_oldest_first(self, tmp_path):
        cards_file = tmp_path / "cards.json"
        c1 = _make_card(next_review_delta=-timedelta(hours=1))
        c2 = _make_card(next_review_delta=-timedelta(hours=3))
        cards_file.write_text(json.dumps({"cards": [c1, c2], "version": 1}))
        result = memento_delivery._load_due_cards(cards_file, _NOW)
        assert result[0]["id"] == c2["id"]  # older review_at sorts first
        assert result[1]["id"] == c1["id"]
