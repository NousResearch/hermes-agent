"""Regression: #146428 — home-target env lookup must not raise UnscopedSecretError.

On a multiplexed gateway the receipt-recovery path (reconcile_pending ->
_save_compose_deliver -> _resolve_delivery_targets -> _get_home_target_chat_id
-> _env_home_target_chat_id -> _home_env_lookup) runs with NO profile secret
scope installed. get_secret() fails closed there (multiplex active, no scope)
and raises UnscopedSecretError even though the call site passes a "" default —
"absent is fine" is clearly the intent. The raise lands inside run
finalization, so the run never reaches a terminal status: it wedges at
status=running and re-raises on every reconcile tick, forever (a restart
doesn't help — receipts are on disk).

The trigger does not even require the platform to be configured: the mere
presence of a bundled platform plugin with a cron_deliver_env_var is enough
(issue reports: MATRIX_HOME_ROOM raised ~1,250x/day on an install with matrix
never enabled).

Contract: a missing scope answers "unset" ("") — it must NOT fall back to
os.getenv, which on a multiplexed gateway can leak a SIBLING profile's value
(the leak the get_secret read was introduced to fix, #83182). Answering "" lets
the existing config.yaml home_channel fallback work instead of crashing.
"""

import pytest

from agent import secret_scope


@pytest.fixture()
def multiplex_no_scope(monkeypatch):
    """Multiplex ACTIVE, no profile scope installed — the reconcile_pending shape."""
    monkeypatch.delenv("MATRIX_HOME_ROOM", raising=False)
    monkeypatch.delenv("TELEGRAM_HOME_CHANNEL", raising=False)
    secret_scope.set_multiplex_active(True)
    assert secret_scope.current_secret_scope() is None
    try:
        yield
    finally:
        secret_scope.set_multiplex_active(False)


class TestHomeEnvLookupUnscoped:
    def test_unscoped_lookup_answers_unset_instead_of_raising(self, multiplex_no_scope):
        from cron.scheduler_delivery import _home_env_lookup

        # Pre-fix this raised UnscopedSecretError (issue #146428).
        assert _home_env_lookup("MATRIX_HOME_ROOM") == ""

    def test_env_home_target_falls_back_instead_of_raising(self, multiplex_no_scope):
        from cron.scheduler_delivery import _env_home_target_chat_id

        assert _env_home_target_chat_id("matrix") == ""

    def test_unscoped_lookup_does_not_leak_process_env(self, multiplex_no_scope, monkeypatch):
        """The safe shape: unset, NOT the host environ — a sibling profile may own that value."""
        monkeypatch.setenv("MATRIX_HOME_ROOM", "sibling-profile-room")
        from cron.scheduler_delivery import _home_env_lookup

        assert _home_env_lookup("MATRIX_HOME_ROOM") == ""

    def test_thread_id_lookup_also_survives_unscoped(self, multiplex_no_scope):
        from cron.scheduler_delivery import _get_home_target_thread_id

        assert _get_home_target_thread_id("matrix") is None

    def test_scoped_read_still_resolves_normally(self, tmp_path, monkeypatch):
        """A tick that DOES have the scope still resolves the owning profile's value."""
        home = tmp_path / "profiles" / "ops"
        home.mkdir(parents=True)
        (home / ".env").write_text('MATRIX_HOME_ROOM="!room:example.org"\n', encoding="utf-8")
        token = secret_scope.set_secret_scope(
            secret_scope.build_profile_secret_scope(home)
        )
        try:
            from cron.scheduler_delivery import _home_env_lookup

            assert _home_env_lookup("MATRIX_HOME_ROOM") == "!room:example.org"
        finally:
            secret_scope.reset_secret_scope(token)
