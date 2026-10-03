"""Tests for the Telegram Mini App dashboard tier resolution (spec §1, §4)."""

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
    # file per call (same staleness fix as TELEGRAM_ALLOWED_USERS) -- an
    # earlier version of dashboard_admin_user_ids() read os.environ
    # directly, which monkeypatch.setenv would have satisfied; it no
    # longer does, so this test writes the file instead.
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
# is_paired_or_allowlisted — delegates to gateway.authz_mixin.is_authorized
# --------------------------------------------------------------------------

def test_paired_user_is_authorized():
    assert is_paired_or_allowlisted("42", pairing_store=_pairing_store(paired=True)) is True


def test_unpaired_unlisted_user_is_not_authorized():
    assert is_paired_or_allowlisted("42", pairing_store=_pairing_store(paired=False)) is False


def test_env_allowlisted_user_is_authorized_without_pairing():
    """TELEGRAM_ALLOWED_USERS is read from the .env FILE, not os.environ —
    this is the targeted per-request read (see the class below for the full
    staleness-closing behavior); write the file, not the process env.
    """
    _write_machine_env("TELEGRAM_ALLOWED_USERS=42\n")
    assert is_paired_or_allowlisted("42", pairing_store=_pairing_store(paired=False)) is True


def test_empty_user_id_is_never_authorized():
    assert is_paired_or_allowlisted("", pairing_store=_pairing_store(paired=True)) is False


def test_setting_only_os_environ_no_longer_authorizes():
    """The inverse of the fix: mutating os.environ (simulating a stale
    dashboard-process env that was never refreshed) must NOT authorize —
    only the .env file on disk is consulted.
    """
    import os as _os

    _os.environ["TELEGRAM_ALLOWED_USERS"] = "42"
    try:
        assert is_paired_or_allowlisted("42", pairing_store=_pairing_store(paired=False)) is False
    finally:
        del _os.environ["TELEGRAM_ALLOWED_USERS"]


# --------------------------------------------------------------------------
# Targeted per-request .env read — closes the dashboard-side staleness
# problem structurally (see gateway/authz_mixin.py's env_get parameter and
# this module's _dashboard_env_get/_read_machine_env_var/_machine_hermes_home).
# --------------------------------------------------------------------------

class TestTargetedEnvReadClosesStaleness:
    def test_env_change_is_picked_up_with_zero_restart_single_profile(self):
        """The load-bearing positive case: edit .env mid-"process" (no
        restart simulated at all — just a file rewrite between two calls in
        the same test, i.e. the same running interpreter) and confirm the
        SECOND call sees the new value with nothing reloaded, no os.environ
        touched, no gateway/dashboard restart of any kind.
        """
        _write_machine_env("TELEGRAM_ALLOWED_USERS=111\n")
        assert is_paired_or_allowlisted("111", pairing_store=_pairing_store(paired=False)) is True
        assert is_paired_or_allowlisted("999", pairing_store=_pairing_store(paired=False)) is False

        # Simulate an operator editing the allowlist via PUT /api/env while
        # this "process" (the test interpreter) keeps running — no restart.
        _write_machine_env("TELEGRAM_ALLOWED_USERS=999\n")

        assert is_paired_or_allowlisted("999", pairing_store=_pairing_store(paired=False)) is True
        assert is_paired_or_allowlisted("111", pairing_store=_pairing_store(paired=False)) is False

    def test_group_allowed_users_var_is_read_fresh_too(self):
        """The seam covers every var is_authorized() reads via env_get, not
        just TELEGRAM_ALLOWED_USERS specifically — GATEWAY_ALLOWED_USERS
        (the global fallback allowlist is_authorized() also consults for a
        DM source) must be equally fresh.
        """
        _write_machine_env("GATEWAY_ALLOWED_USERS=555\n")
        assert is_paired_or_allowlisted("555", pairing_store=_pairing_store(paired=False)) is True

    def test_does_not_leak_a_different_profiles_env_var(self):
        """The specific risk this design is meant to avoid: a concurrent
        dashboard request scoped to a DIFFERENT profile (via
        hermes_constants.set_hermes_home_override, the exact mechanism
        hermes_cli/web_server.py's _profile_scope/_config_profile_scope use)
        must not cause this read to consult that OTHER profile's .env — it
        must always read the machine-default .env regardless of any active
        per-request profile-scope override on the calling task.
        """
        import os as _os
        from pathlib import Path

        from hermes_constants import (
            get_hermes_home,
            reset_hermes_home_override,
            set_hermes_home_override,
        )

        # The machine-default .env (what HERMES_HOME points at for this test).
        _write_machine_env("TELEGRAM_ALLOWED_USERS=machine_user\n")

        # A sibling profile directory with a DIFFERENT allowlist — this must
        # never be consulted by is_paired_or_allowlisted() below.
        other_profile_home = Path(_os.environ["HERMES_HOME"]).parent / "other_profile"
        other_profile_home.mkdir(parents=True, exist_ok=True)
        (other_profile_home / ".env").write_text("TELEGRAM_ALLOWED_USERS=other_profile_user\n")

        token = set_hermes_home_override(str(other_profile_home))
        try:
            # Sanity: the override really is active and would redirect a
            # naive get_hermes_home() call to the other profile.
            assert str(get_hermes_home()) == str(other_profile_home)

            assert (
                is_paired_or_allowlisted("machine_user", pairing_store=_pairing_store(paired=False))
                is True
            )
            assert (
                is_paired_or_allowlisted(
                    "other_profile_user", pairing_store=_pairing_store(paired=False)
                )
                is False
            )

            # The override must still be intact after our read returns —
            # this function must not regress the ambient scope for the rest
            # of the request it was called from.
            assert str(get_hermes_home()) == str(other_profile_home)
        finally:
            reset_hermes_home_override(token)


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
