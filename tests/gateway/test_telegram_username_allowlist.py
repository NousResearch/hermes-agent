"""Telegram @username allowlist fallback for TELEGRAM_ALLOWED_USERS (#13206).

Operators frequently put a @username in TELEGRAM_ALLOWED_USERS because the
Telegram UI never surfaces the numeric user ID. Numeric IDs remain the
reliable identifier, but a non-numeric value is matched case-insensitively
against the sender's real Telegram @username (SessionSource.user_handle) as a
best-effort fallback, and a one-time warning is logged.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest

from gateway.session import Platform, SessionSource


@pytest.fixture(autouse=True)
def _isolate_telegram_env(monkeypatch):
    for var in (
        "TELEGRAM_ALLOWED_USERS",
        "TELEGRAM_GROUP_ALLOWED_USERS",
        "TELEGRAM_GROUP_ALLOWED_CHATS",
        "TELEGRAM_ALLOW_ALL_USERS",
        "GATEWAY_ALLOW_ALL_USERS",
        "GATEWAY_ALLOWED_USERS",
    ):
        monkeypatch.delenv(var, raising=False)


def _make_bare_runner():
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.pairing_store = SimpleNamespace(is_approved=lambda *_a, **_kw: False)
    return runner


def _make_source(user_id="123", user_handle=None, user_name="Display Name"):
    return SessionSource(
        platform=Platform.TELEGRAM,
        chat_id=user_id,
        chat_type="dm",
        user_id=user_id,
        user_name=user_name,
        user_handle=user_handle,
    )


def test_numeric_id_still_authorizes():
    runner = _make_bare_runner()
    import os

    os.environ["TELEGRAM_ALLOWED_USERS"] = "469682876"
    try:
        assert runner._is_user_authorized(_make_source(user_id="469682876")) is True
    finally:
        del os.environ["TELEGRAM_ALLOWED_USERS"]


def test_username_in_allowlist_matches_user_handle(monkeypatch):
    runner = _make_bare_runner()
    monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "@ChipuEatFast")

    source = _make_source(user_id="999", user_handle="ChipuEatFast")
    assert runner._is_user_authorized(source) is True


def test_username_match_is_case_insensitive(monkeypatch):
    runner = _make_bare_runner()
    monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "chipueatfast")

    source = _make_source(user_id="999", user_handle="ChipuEatFast")
    assert runner._is_user_authorized(source) is True


def test_username_does_not_match_display_name(monkeypatch):
    # The fallback matches the real @username (user_handle), NOT the display
    # name (user_name). A username allowlist must never authorize a user whose
    # only matching field is their freely-chosen display name.
    runner = _make_bare_runner()
    monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "@SomeName")

    source = _make_source(user_id="999", user_handle="other_handle", user_name="SomeName")
    assert runner._is_user_authorized(source) is False


def test_username_no_handle_is_rejected(monkeypatch):
    # User without a public @username (user_handle is None) cannot be matched
    # by a username allowlist entry.
    runner = _make_bare_runner()
    monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "@someone")

    source = _make_source(user_id="999", user_handle=None)
    assert runner._is_user_authorized(source) is False


def _nonnumeric_warnings(caplog):
    return [
        r for r in caplog.records
        if "Non-numeric Telegram allowlist" in r.getMessage()
    ]


def test_non_numeric_allowlist_warns_once_and_names_env_var(monkeypatch, caplog):
    runner = _make_bare_runner()
    monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "@notnumeric")

    with caplog.at_level(logging.WARNING):
        runner._is_user_authorized(_make_source(user_id="1", user_handle="someone"))
        runner._is_user_authorized(_make_source(user_id="2", user_handle="someone"))

    warnings = _nonnumeric_warnings(caplog)
    assert len(warnings) == 1, "non-numeric allowlist warning should fire once"
    msg = warnings[0].getMessage()
    assert "@notnumeric" in msg
    # Operator must be told which env var holds the bad value.
    assert "TELEGRAM_ALLOWED_USERS=@notnumeric" in msg


def test_warning_attributes_value_to_originating_env_var(monkeypatch, caplog):
    # A non-numeric value in GATEWAY_ALLOWED_USERS must be reported as
    # GATEWAY_ALLOWED_USERS, not the platform allowlist (review feedback).
    runner = _make_bare_runner()
    monkeypatch.setenv("GATEWAY_ALLOWED_USERS", "@globaluser")

    with caplog.at_level(logging.WARNING):
        runner._is_user_authorized(_make_source(user_id="1", user_handle="globaluser"))

    warnings = _nonnumeric_warnings(caplog)
    assert len(warnings) == 1
    assert "GATEWAY_ALLOWED_USERS=@globaluser" in warnings[0].getMessage()


def test_numeric_allowlist_does_not_warn(monkeypatch, caplog):
    runner = _make_bare_runner()
    monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "469682876")

    with caplog.at_level(logging.WARNING):
        runner._is_user_authorized(_make_source(user_id="469682876"))

    assert not _nonnumeric_warnings(caplog)


def test_username_match_logs_numeric_id_once_per_user(monkeypatch, caplog):
    # Self-healing onboarding: when a user is authorized via the username
    # fallback, log their stable numeric ID once so the operator can pin it.
    runner = _make_bare_runner()
    monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "@alice")

    with caplog.at_level(logging.WARNING):
        # Same user (same numeric id) messages twice — log only once.
        runner._is_user_authorized(_make_source(user_id="469682876", user_handle="alice"))
        runner._is_user_authorized(_make_source(user_id="469682876", user_handle="alice"))
        # A different user matching the same allowlist entry logs separately.
        runner._is_user_authorized(_make_source(user_id="111", user_handle="Alice"))

    heals = [
        r for r in caplog.records
        if "add that to TELEGRAM_ALLOWED_USERS" in r.getMessage()
    ]
    assert len(heals) == 2
    assert "469682876" in heals[0].getMessage()
    assert "111" in heals[1].getMessage()


def test_no_self_heal_log_when_numeric_id_used(monkeypatch, caplog):
    runner = _make_bare_runner()
    monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "469682876")

    with caplog.at_level(logging.WARNING):
        runner._is_user_authorized(_make_source(user_id="469682876", user_handle="alice"))

    assert not [
        r for r in caplog.records
        if "add that to TELEGRAM_ALLOWED_USERS" in r.getMessage()
    ]
