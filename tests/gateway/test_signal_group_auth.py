"""Signal group-sender authorization (SIGNAL_GROUP_ALLOWED_USERS).

Regression guard: Signal previously authorized senders ONLY via
SIGNAL_ALLOWED_USERS in gateway/run.py, so a bot whose SIGNAL_ALLOWED_USERS
listed just the admin would answer *only* the admin in group chats — every
other member was logged "Unauthorized user" and dropped, even when they
@mentioned the bot.

The Signal adapter already gates which *groups* are active
(SIGNAL_GROUP_ALLOWED_USERS, "*" = all) and marks unmentioned messages
observe_only. So group membership — not per-sender allowlisting — should
authorize a group message at the gateway. DMs stay gated by
SIGNAL_ALLOWED_USERS.
"""

from types import SimpleNamespace

import pytest

from gateway.session import Platform, SessionSource


@pytest.fixture(autouse=True)
def _isolate_signal_env(monkeypatch):
    for var in (
        "SIGNAL_ALLOWED_USERS",
        "SIGNAL_GROUP_ALLOWED_USERS",
        "SIGNAL_ALLOW_ALL_USERS",
        "GATEWAY_ALLOW_ALL_USERS",
        "GATEWAY_ALLOWED_USERS",
    ):
        monkeypatch.delenv(var, raising=False)


def _make_bare_runner():
    from gateway.run import GatewayRunner
    runner = object.__new__(GatewayRunner)
    runner.pairing_store = SimpleNamespace(is_approved=lambda *_a, **_kw: False)
    return runner


def _group_source(user_id="+15550001111", gid="abc123groupid"):
    return SessionSource(
        platform=Platform.SIGNAL,
        chat_id=f"group:{gid}",
        chat_type="group",
        user_id=user_id,
        user_name="Charlotte",
        chat_id_alt=gid,
    )


def _dm_source(user_id="+15550001111"):
    return SessionSource(
        platform=Platform.SIGNAL,
        chat_id=user_id,
        chat_type="dm",
        user_id=user_id,
        user_name="Charlotte",
    )


ADMIN = "+15550112233"


def test_non_admin_group_member_authorized_when_groups_open(monkeypatch):
    """SIGNAL_GROUP_ALLOWED_USERS=* authorizes any sender in a group, even
    when SIGNAL_ALLOWED_USERS only lists the admin."""
    monkeypatch.setenv("SIGNAL_ALLOWED_USERS", ADMIN)
    monkeypatch.setenv("SIGNAL_GROUP_ALLOWED_USERS", "*")
    runner = _make_bare_runner()
    assert runner._is_user_authorized(_group_source(user_id="+15550009999")) is True


def test_group_member_authorized_by_explicit_group_id(monkeypatch):
    """An explicit group id in SIGNAL_GROUP_ALLOWED_USERS matches the
    'group:<id>' chat_id / chat_id_alt forms."""
    monkeypatch.setenv("SIGNAL_ALLOWED_USERS", ADMIN)
    monkeypatch.setenv("SIGNAL_GROUP_ALLOWED_USERS", "abc123groupid,otherid")
    runner = _make_bare_runner()
    assert runner._is_user_authorized(_group_source(user_id="+15550009999")) is True


def test_group_member_denied_when_group_not_listed(monkeypatch):
    """A group id not in the allowlist is not authorized (no wildcard)."""
    monkeypatch.setenv("SIGNAL_ALLOWED_USERS", ADMIN)
    monkeypatch.setenv("SIGNAL_GROUP_ALLOWED_USERS", "someothergroup")
    runner = _make_bare_runner()
    assert runner._is_user_authorized(_group_source(user_id="+15550009999")) is False


def test_dm_from_non_admin_still_denied(monkeypatch):
    """Opening groups must NOT open DMs — SIGNAL_ALLOWED_USERS still gates DMs."""
    monkeypatch.setenv("SIGNAL_ALLOWED_USERS", ADMIN)
    monkeypatch.setenv("SIGNAL_GROUP_ALLOWED_USERS", "*")
    runner = _make_bare_runner()
    assert runner._is_user_authorized(_dm_source(user_id="+15550009999")) is False


def test_dm_from_admin_authorized(monkeypatch):
    monkeypatch.setenv("SIGNAL_ALLOWED_USERS", ADMIN)
    monkeypatch.setenv("SIGNAL_GROUP_ALLOWED_USERS", "*")
    runner = _make_bare_runner()
    assert runner._is_user_authorized(_dm_source(user_id=ADMIN)) is True


def test_group_sender_denied_when_no_group_allowlist(monkeypatch):
    """With no SIGNAL_GROUP_ALLOWED_USERS and a DM allowlist set, a group
    sender not on the DM allowlist is denied (defense in depth; the adapter
    also drops these upstream)."""
    monkeypatch.setenv("SIGNAL_ALLOWED_USERS", ADMIN)
    runner = _make_bare_runner()
    assert runner._is_user_authorized(_group_source(user_id="+15550009999")) is False
