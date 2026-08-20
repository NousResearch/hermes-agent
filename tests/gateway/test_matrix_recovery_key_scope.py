"""Regression test for #69090: MATRIX_RECOVERY_KEY must honor the active
profile's secret scope under ``gateway.multiplex_profiles`` so that a
secondary profile resolves its own recovery key (not the default profile's),
otherwise E2EE cross-signing verification fails with "Key MAC does not match".

The fix routes the recovery-key read through ``_scoped_recovery_key()``,
which uses :func:`agent.secret_scope.get_secret` (scope-aware) and only falls
back to ``os.getenv`` for an *unscoped* read under multiplex — mirroring the
established Slack app-token pattern (#59739).
"""
import pytest

from agent import secret_scope as ss
from plugins.platforms.matrix.adapter import _scoped_recovery_key


@pytest.fixture(autouse=True)
def _reset_multiplex():
    """Ensure each test starts and ends with multiplexing off (it's a global)."""
    ss.set_multiplex_active(False)
    yield
    ss.set_multiplex_active(False)


class TestScopedRecoveryKey:
    def test_multiplex_inactive_reads_environ(self, monkeypatch):
        """Default deployment: get_secret transparently reads os.environ."""
        monkeypatch.setenv("MATRIX_RECOVERY_KEY", "default-profile-key")
        assert _scoped_recovery_key() == "default-profile-key"

    def test_multiplex_active_scoped_uses_scope_not_environ(self, monkeypatch):
        """Secondary profile under multiplex must resolve its own key.

        This is the core regression: ``os.getenv`` would have returned the
        default profile's key (from os.environ), failing verification.
        """
        monkeypatch.setenv("MATRIX_RECOVERY_KEY", "default-profile-key")
        ss.set_multiplex_active(True)
        token = ss.set_secret_scope({"MATRIX_RECOVERY_KEY": "secondary-profile-key"})
        try:
            assert _scoped_recovery_key() == "secondary-profile-key"
        finally:
            ss.reset_secret_scope(token)

    def test_multiplex_active_unscoped_falls_back_to_environ(self, monkeypatch):
        """Default-profile startup loop under multiplex: unscoped read is fine.

        An unscoped read raises ``UnscopedSecretError``; in that context
        os.environ holds that profile's own value, so we fall back to it rather
        than crashing startup. This matches the Slack adapter's behavior.
        """
        monkeypatch.setenv("MATRIX_RECOVERY_KEY", "default-profile-key")
        ss.set_multiplex_active(True)
        # No secret scope installed -> get_secret raises UnscopedSecretError.
        assert _scoped_recovery_key() == "default-profile-key"

    def test_multiplex_active_scoped_missing_key_is_empty(self, monkeypatch):
        """A scope without the key must NOT fall through to another profile's env.

        If the secondary profile hasn't configured a recovery key, the scope is
        authoritative: we return empty rather than silently borrowing the
        default profile's key (which would fail verification with a confusing
        "Key MAC does not match").
        """
        monkeypatch.setenv("MATRIX_RECOVERY_KEY", "default-profile-key")
        ss.set_multiplex_active(True)
        token = ss.set_secret_scope({"SOME_OTHER_KEY": "x"})
        try:
            assert _scoped_recovery_key() == ""
        finally:
            ss.reset_secret_scope(token)

    def test_strips_whitespace(self, monkeypatch):
        monkeypatch.setenv("MATRIX_RECOVERY_KEY", "  padded-key  \n")
        assert _scoped_recovery_key() == "padded-key"

    def test_unset_returns_empty(self, monkeypatch):
        monkeypatch.delenv("MATRIX_RECOVERY_KEY", raising=False)
        assert _scoped_recovery_key() == ""
