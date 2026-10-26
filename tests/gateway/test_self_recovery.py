"""Offline tests for gateway.self_recovery (stdlib-only module).

No gateway imports besides the module under test, so this file runs under the
repo venv pytest and under plain python3 with pytest installed. A fake clock
drives every time-dependent assertion; no sleeps, no network, no state.db.
"""

import json

import pytest

from gateway.self_recovery import (
    ERROR_NOTICE_DIAGNOSIS_SUPPRESS_SECONDS,
    ERROR_NOTICE_POST_LIMIT,
    ERROR_NOTICE_WINDOW_SECONDS,
    SERVICE_RESTART_WAIT_CAP_SECONDS,
    PauseNoticeRegistry,
    ErrorNoticeThrottle,
    effective_restart_wait_timeout,
    error_diagnosis_text,
    get_error_notice_throttle,
    pause_sidecar_path,
)


class FakeClock:
    def __init__(self, start=1_000_000.0):
        self.now = start

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


# --- ErrorNoticeThrottle ---


def test_throttle_posts_first_three_then_diagnosis_then_suppresses():
    clock = FakeClock()
    throttle = ErrorNoticeThrottle(clock=clock)
    key = ("runtime_draining", "slack:general")
    assert throttle.observe(key) == ("post", 1)
    assert throttle.observe(key) == ("post", 2)
    assert throttle.observe(key) == ("post", 3)
    verdict, count = throttle.observe(key)
    assert verdict == "diagnosis" and count == 4
    assert throttle.observe(key) == ("suppress", 5)
    assert throttle.observe(key) == ("suppress", 6)


def test_throttle_window_rolls_and_diagnosis_expires():
    clock = FakeClock()
    throttle = ErrorNoticeThrottle(clock=clock)
    key = ("admission_conflict", "slack:agents")
    for _ in range(4):
        throttle.observe(key)  # post, post, post, diagnosis
    assert throttle.observe(key)[0] == "suppress"
    # Past the 1h diagnosis suppression AND past the 15m occurrence window:
    # a fresh episode posts again instead of staying muted forever.
    clock.advance(ERROR_NOTICE_DIAGNOSIS_SUPPRESS_SECONDS + 1)
    assert throttle.observe(key) == ("post", 1)


def test_throttle_scopes_are_independent():
    clock = FakeClock()
    throttle = ErrorNoticeThrottle(clock=clock)
    draining = ("runtime_draining", "chat-a")
    conflict = ("admission_conflict", "chat-a")
    other_chat = ("runtime_draining", "chat-b")
    for _ in range(4):
        throttle.observe(draining)
    assert throttle.observe(draining)[0] == "suppress"
    assert throttle.observe(conflict) == ("post", 1)
    assert throttle.observe(other_chat) == ("post", 1)


def test_throttle_old_occurrences_leave_the_window():
    clock = FakeClock()
    throttle = ErrorNoticeThrottle(clock=clock)
    key = ("runtime_draining", "chat-a")
    assert throttle.observe(key) == ("post", 1)
    clock.advance(ERROR_NOTICE_WINDOW_SECONDS + 1)
    assert throttle.observe(key) == ("post", 1)


def test_diagnosis_text_names_reason_and_count_not_identifiers():
    text = error_diagnosis_text("runtime_draining", 4)
    assert "runtime_draining" in text
    assert "4 identical" in text
    assert "1 hour" in text
    assert "this conversation" in text
    assert "U09" not in text  # never a user id, never a session id


def test_get_error_notice_throttle_attaches_once():
    class Holder:
        pass

    holder = Holder()
    first = get_error_notice_throttle(holder)
    second = get_error_notice_throttle(holder)
    assert first is second
    assert isinstance(first, ErrorNoticeThrottle)


# --- effective_restart_wait_timeout ---


def test_service_restart_wait_is_capped():
    assert (
        effective_restart_wait_timeout(1800.0, True)
        == SERVICE_RESTART_WAIT_CAP_SECONDS
    )
    assert SERVICE_RESTART_WAIT_CAP_SECONDS == 300.0


def test_explicit_restart_keeps_configured_timeout():
    assert effective_restart_wait_timeout(1800.0, False) == 1800.0
    assert effective_restart_wait_timeout(120.0, True) == 120.0
    assert effective_restart_wait_timeout(300.0, True) == 300.0


def test_zero_timeout_still_means_stop_immediately():
    assert effective_restart_wait_timeout(0.0, True) == 0.0
    assert effective_restart_wait_timeout(0.0, False) == 0.0


# --- PauseNoticeRegistry ---


def test_pause_registry_notifies_once_per_day_per_key(tmp_path):
    clock = FakeClock()
    registry = PauseNoticeRegistry(tmp_path / ".pause_notices.json", clock=clock)
    assert registry.should_notify("profile:session-1", "unknown_execution") is True
    assert registry.should_notify("profile:session-1", "unknown_execution") is False
    # A different session, or a different reason, still notifies.
    assert registry.should_notify("profile:session-2", "unknown_execution") is True
    assert registry.should_notify("profile:session-1", "runtime_draining") is True
    # After 24h the same key notifies again (one fresh notice, not silence).
    clock.advance(86400.0 + 1)
    assert registry.should_notify("profile:session-1", "unknown_execution") is True


def test_pause_registry_survives_restart_via_sidecar(tmp_path):
    clock = FakeClock()
    path = tmp_path / ".pause_notices.json"
    first = PauseNoticeRegistry(path, clock=clock)
    assert first.should_notify("profile:session-9", "unknown_execution") is True
    # A new process (new registry object, same file) must not re-notice.
    second = PauseNoticeRegistry(path, clock=clock)
    assert second.should_notify("profile:session-9", "unknown_execution") is False
    assert path.exists()
    stored = json.loads(path.read_text())
    assert len(stored) == 1


def test_pause_registry_fails_open_on_corrupt_sidecar(tmp_path):
    clock = FakeClock()
    path = tmp_path / ".pause_notices.json"
    path.write_text("{not json")
    registry = PauseNoticeRegistry(path, clock=clock)
    assert registry.should_notify("profile:session-1", "unknown_execution") is True


def test_pause_sidecar_path_is_under_home(tmp_path):
    assert pause_sidecar_path(tmp_path) == tmp_path / ".pause_notices.json"


def test_pause_notice_platforms_are_distinct_keys(tmp_path, monkeypatch):
    """Regression: Slack and Telegram conversations sharing a session id must
    each get their own unknown_execution notice; a second Slack notice for
    the same session is suppressed by the sidecar. (An existing suite test
    exercises the Slack/Telegram pair and failed before the platform key.)"""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from types import SimpleNamespace

    from gateway.config import Platform
    from gateway.session import SessionSource
    from gateway.session_contract import SessionRef
    from gateway.session_ingress import pause_notice

    notices = []
    for platform in (Platform.SLACK, Platform.TELEGRAM):
        live = SimpleNamespace(
            source=SessionSource(platform=platform, chat_id="D42"),
            pause_notified=False,
        )
        notices.append(
            pause_notice(
                SimpleNamespace(sessions={"s": live}),
                SessionRef("p", "s"),
                "unknown_execution",
            )
        )
    assert notices[0] is not None and "/hermes discard" in notices[0]
    assert notices[1] is not None and "Send /discard" in notices[1]
    live = SimpleNamespace(
        source=SessionSource(platform=Platform.SLACK, chat_id="D42"),
        pause_notified=False,
    )
    assert (
        pause_notice(
            SimpleNamespace(sessions={"s": live}),
            SessionRef("p", "s"),
            "unknown_execution",
        )
        is None
    )


def test_pause_registry_prunes_entries_older_than_a_week(tmp_path):
    clock = FakeClock()
    path = tmp_path / ".pause_notices.json"
    registry = PauseNoticeRegistry(path, clock=clock)
    assert registry.should_notify("profile:old", "unknown_execution") is True
    clock.advance(8 * 86400.0)
    assert registry.should_notify("profile:new", "unknown_execution") is True
    stored = json.loads(path.read_text())
    assert not any("profile:old" in key for key in stored)
    assert any("profile:new" in key for key in stored)
