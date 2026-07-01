"""Regression guard: pairing store must not bypass a configured allowlist (#23778).

A user who tapped "Always" on an approval button gets a pairing-store entry.
Before the fix, ``_is_user_authorized()`` returned True from the pairing store
BEFORE the allowlist was ever consulted, so a paired-but-not-allowed user
permanently bypassed ``TELEGRAM_ALLOWED_USERS`` (or equivalent) even after being
removed from the allowlist. The fix records pairing membership but only honors
it when no allowlist is configured; when an allowlist IS configured, the paired
user must still appear in it.
"""

from types import SimpleNamespace

import pytest

from gateway.session import Platform, SessionSource


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch):
    for var in (
        "TELEGRAM_ALLOWED_USERS",
        "TELEGRAM_ALLOW_ALL_USERS",
        "TELEGRAM_GROUP_ALLOWED_USERS",
        "TELEGRAM_GROUP_ALLOWED_CHATS",
        "GATEWAY_ALLOW_ALL_USERS",
        "GATEWAY_ALLOWED_USERS",
    ):
        monkeypatch.delenv(var, raising=False)


def _make_runner(*, paired: bool):
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.pairing_store = SimpleNamespace(is_approved=lambda *_a, **_kw: paired)
    return runner


def _make_source(user_id: str = "attacker42", chat_type: str = "dm"):
    return SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="123",
        chat_type=chat_type,
        user_id=user_id,
        user_name="SomeHuman",
        is_bot=False,
    )


def test_paired_user_denied_when_not_in_platform_allowlist(monkeypatch):
    """The core bypass: paired but absent from TELEGRAM_ALLOWED_USERS → denied."""
    runner = _make_runner(paired=True)
    monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "owner1,owner2")

    assert runner._is_user_authorized(_make_source("attacker42")) is False


def test_paired_user_denied_when_not_in_global_allowlist(monkeypatch):
    runner = _make_runner(paired=True)
    monkeypatch.setenv("GATEWAY_ALLOWED_USERS", "owner1,owner2")

    assert runner._is_user_authorized(_make_source("attacker42")) is False


def test_paired_user_allowed_when_still_in_platform_allowlist(monkeypatch):
    """A paired user who is also in the allowlist stays authorized."""
    runner = _make_runner(paired=True)
    monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "owner1,attacker42")

    assert runner._is_user_authorized(_make_source("attacker42")) is True


def test_paired_user_allowed_when_still_in_global_allowlist(monkeypatch):
    runner = _make_runner(paired=True)
    monkeypatch.setenv("GATEWAY_ALLOWED_USERS", "owner1,attacker42")

    assert runner._is_user_authorized(_make_source("attacker42")) is True


def test_paired_user_allowed_with_wildcard_allowlist(monkeypatch):
    """A "*" allowlist means everyone; a paired user is honored."""
    runner = _make_runner(paired=True)
    monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "*")

    assert runner._is_user_authorized(_make_source("attacker42")) is True


def test_paired_user_allowed_when_no_allowlist_configured(monkeypatch):
    """No allowlist configured → pairing is the intended access path, honored."""
    runner = _make_runner(paired=True)

    assert runner._is_user_authorized(_make_source("attacker42")) is True


def test_unpaired_user_denied_when_no_allowlist_configured(monkeypatch):
    """No allowlist and not paired → default-deny (no fail-open)."""
    runner = _make_runner(paired=False)

    assert runner._is_user_authorized(_make_source("randouser")) is False


def test_unpaired_user_in_allowlist_still_allowed(monkeypatch):
    """Pairing changes did not regress the plain allowlist path."""
    runner = _make_runner(paired=False)
    monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "owner1")

    assert runner._is_user_authorized(_make_source("owner1")) is True
