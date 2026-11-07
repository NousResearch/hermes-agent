"""Tests for the Telegram Mini App dashboard tier resolution."""

from types import SimpleNamespace

import pytest

from plugins.dashboard_auth.telegram_miniapp.tiers import (
    dashboard_admin_user_ids,
    is_dashboard_admin,
    is_paired_or_allowlisted,
    resolve_tier,
)


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch):
    for var in (
        "TELEGRAM_DASHBOARD_ADMIN_USERS",
        "TELEGRAM_ALLOWED_USERS",
        "TELEGRAM_ALLOW_ALL_USERS",
        "GATEWAY_ALLOW_ALL_USERS",
        "GATEWAY_ALLOWED_USERS",
        "TELEGRAM_GROUP_ALLOWED_USERS",
    ):
        monkeypatch.delenv(var, raising=False)


def _pairing_store(*, paired: bool):
    return SimpleNamespace(is_approved=lambda *_a, **_kw: paired)


def _write_machine_env(text: str) -> None:
    """Write ``.env`` under the current test's ``HERMES_HOME`` (set by the
    autouse ``_hermetic_environment`` fixture via the ``HERMES_HOME`` env
    var, which ``_machine_hermes_home()`` reads directly).

    Calls ``invalidate_env_cache()`` afterward, same as the real
    ``save_env_value()`` write path (``PUT /api/env``) already does
    (``hermes_cli/config.py:7117``) — a raw file write, unlike a real
    operator edit, doesn't go through that path, so skipping this call would
    make a test rely on load_env()'s mtime+size cache key ``happening`` to
    change (it does not always, e.g. two same-length values written within
    the same mtime tick), which is not what a real edit ever depends on.
    """
    import os as _os
    from pathlib import Path

    from hermes_cli.config import invalidate_env_cache

    home = Path(_os.environ["HERMES_HOME"])
    home.mkdir(parents=True, exist_ok=True)
    (home / ".env").write_text(text)
    invalidate_env_cache()


# --------------------------------------------------------------------------
# dashboard_admin_user_ids / is_dashboard_admin — fail-closed by default
# --------------------------------------------------------------------------

def test_admin_allowlist_empty_by_default():
    assert dashboard_admin_user_ids() == frozenset()


def test_is_dashboard_admin_false_when_env_unset():
    assert is_dashboard_admin("42") is False


def test_is_dashboard_admin_false_for_empty_user_id():
    assert is_dashboard_admin("") is False


def test_is_dashboard_admin_true_for_listed_user():
    # TELEGRAM_DASHBOARD_ADMIN_USERS is read fresh from the machine's .env
    # file per call (same staleness rule as TELEGRAM_ALLOWED_USERS), not
    # os.environ, so monkeypatch.setenv wouldn't reach it -- this test
    # writes the file instead.
    _write_machine_env("TELEGRAM_DASHBOARD_ADMIN_USERS=42,99\n")
    assert is_dashboard_admin("42") is True
    assert is_dashboard_admin("99") is True


def test_is_dashboard_admin_false_for_unlisted_user():
    _write_machine_env("TELEGRAM_DASHBOARD_ADMIN_USERS=42\n")
    assert is_dashboard_admin("7") is False


def test_setting_only_os_environ_does_not_grant_admin():
    """The inverse of the fix, mirroring
    test_setting_only_os_environ_no_longer_authorizes below: mutating
    os.environ alone (simulating a stale dashboard process that never
    re-read .env) must NOT grant admin -- only the .env file on disk
    is consulted.
    """
    import os as _os

    _os.environ["TELEGRAM_DASHBOARD_ADMIN_USERS"] = "42"
    try:
        assert is_dashboard_admin("42") is False
    finally:
        del _os.environ["TELEGRAM_DASHBOARD_ADMIN_USERS"]


def test_admin_allowlist_ignores_whitespace_and_blanks():
    _write_machine_env("TELEGRAM_DASHBOARD_ADMIN_USERS= 42 , , 99 \n")
    assert dashboard_admin_user_ids() == frozenset({"42", "99"})


def test_admin_allowlist_has_no_wildcard_support():
    """Unlike inbound-message env allowlists, "*" is not special here."""
    _write_machine_env("TELEGRAM_DASHBOARD_ADMIN_USERS=*\n")
    assert is_dashboard_admin("literally-anyone") is False
    assert dashboard_admin_user_ids() == frozenset({"*"})
    assert dashboard_admin_user_ids() == frozenset({"*"})


# --------------------------------------------------------------------------
# is_paired_or_allowlisted — the gateway's own _is_user_authorized, no live adapter
# --------------------------------------------------------------------------

def test_paired_user_is_authorized():
    assert is_paired_or_allowlisted("42", pairing_store=_pairing_store(paired=True)) is True


def test_unpaired_unlisted_user_is_not_authorized():
    assert is_paired_or_allowlisted("42", pairing_store=_pairing_store(paired=False)) is False


def test_empty_user_id_is_never_authorized():
    assert is_paired_or_allowlisted("", pairing_store=_pairing_store(paired=True)) is False


def test_env_allowlisted_user_is_authorized_without_pairing(monkeypatch):
    monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "42")
    assert is_paired_or_allowlisted("42", pairing_store=_pairing_store(paired=False)) is True
    assert is_paired_or_allowlisted("43", pairing_store=_pairing_store(paired=False)) is False


def test_global_allowlist_authorizes(monkeypatch):
    monkeypatch.setenv("GATEWAY_ALLOWED_USERS", "555")
    assert is_paired_or_allowlisted("555", pairing_store=_pairing_store(paired=False)) is True


def test_group_scoped_allowlist_does_not_grant_dm_access(monkeypatch):
    """TELEGRAM_GROUP_ALLOWED_USERS gates group traffic only; the bot gives it no DM access."""
    monkeypatch.setenv("TELEGRAM_GROUP_ALLOWED_USERS", "42")
    assert is_paired_or_allowlisted("42", pairing_store=_pairing_store(paired=False)) is False


def test_dashboard_allowlist_edit_takes_effect_without_restart():
    """An edit through save_env_value (the Mini App's allowlist endpoints) is published to this
    process, so the next tier check sees it."""
    from hermes_cli.config import remove_env_value, save_env_value

    save_env_value("TELEGRAM_ALLOWED_USERS", "777")
    try:
        assert is_paired_or_allowlisted("777", pairing_store=_pairing_store(paired=False)) is True
    finally:
        remove_env_value("TELEGRAM_ALLOWED_USERS")
    assert is_paired_or_allowlisted("777", pairing_store=_pairing_store(paired=False)) is False


# --------------------------------------------------------------------------
# resolve_tier — the composed decision the Mini App auth layer consumes
# --------------------------------------------------------------------------

def test_resolve_tier_unauthorized_when_not_paired():
    assert resolve_tier("42", pairing_store=_pairing_store(paired=False)) == "unauthorized"


def test_resolve_tier_paired_when_paired_but_not_admin():
    assert resolve_tier("42", pairing_store=_pairing_store(paired=True)) == "paired"


def test_resolve_tier_admin_when_paired_and_admin_listed():
    _write_machine_env("TELEGRAM_DASHBOARD_ADMIN_USERS=42\n")
    assert resolve_tier("42", pairing_store=_pairing_store(paired=True)) == "admin"


def test_resolve_tier_never_admin_without_pairing_even_if_admin_listed():
    """Fail-closed composition: admin-listed but NOT paired/allowlisted stays unauthorized.

    An admin allowlist entry is not itself a pairing/allowlist grant — it
    only upgrades an already-authorized user. This guards against a config
    mistake (adding a user to TELEGRAM_DASHBOARD_ADMIN_USERS without also
    pairing/allowlisting them) silently granting dashboard access.
    """
    _write_machine_env("TELEGRAM_DASHBOARD_ADMIN_USERS=42\n")
    assert resolve_tier("42", pairing_store=_pairing_store(paired=False)) == "unauthorized"
