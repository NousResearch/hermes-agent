"""Tests for the profile-scoped credential primitive (Workstream A / Phase 2)."""
import pytest

from agent import secret_scope as ss


@pytest.fixture(autouse=True)
def _reset_multiplex():
    """Ensure each test starts and ends with multiplexing off (it's a global)."""
    ss.set_multiplex_active(False)
    yield
    ss.set_multiplex_active(False)


class TestMultiplexInactiveBackwardCompat:
    """Default deployment: get_secret transparently reads os.environ."""

    def test_reads_environ(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        assert ss.get_secret("ANTHROPIC_API_KEY") == "sk-test"

    def test_missing_returns_default(self, monkeypatch):
        monkeypatch.delenv("NOPE_KEY", raising=False)
        assert ss.get_secret("NOPE_KEY") is None
        assert ss.get_secret("NOPE_KEY", "fallback") == "fallback"

    def test_no_raise_without_scope(self, monkeypatch):
        monkeypatch.delenv("SOME_KEY", raising=False)
        # multiplex off => unscoped read is fine, returns default
        assert ss.get_secret("SOME_KEY") is None


class TestMultiplexActiveFailClosed:
    """Multiplex on: an unscoped secret read raises instead of leaking."""

    def test_unscoped_read_raises(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-leaky")
        ss.set_multiplex_active(True)
        with pytest.raises(ss.UnscopedSecretError):
            ss.get_secret("ANTHROPIC_API_KEY")


    def test_scoped_missing_key_returns_default_not_environ(self, monkeypatch):
        # Even though the value exists in os.environ, a scope is authoritative:
        # an absent scope key must NOT fall through to the (cross-profile) env.
        monkeypatch.setenv("OPENAI_API_KEY", "sk-other-profile")
        ss.set_multiplex_active(True)
        token = ss.set_secret_scope({"ANTHROPIC_API_KEY": "sk-mine"})
        try:
            assert ss.get_secret("OPENAI_API_KEY") is None
            assert ss.get_secret("OPENAI_API_KEY", "d") == "d"
        finally:
            ss.reset_secret_scope(token)




class TestScopedSingleProfile:
    """Multiplex OFF with a scope installed: the scope is an overlay, not a
    blindfold. The cron scheduler installs a ``<home>/.env`` scope around every
    job unconditionally, and single-profile deployments legitimately supply
    credentials via the process environment only (systemd ``Environment=``,
    ``pass-cli run`` / ``op run`` wrappers) — those must keep resolving."""

    def test_scope_hit_wins_over_environ(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-from-environ")
        token = ss.set_secret_scope({"ANTHROPIC_API_KEY": "sk-from-env-file"})
        try:
            assert ss.get_secret("ANTHROPIC_API_KEY") == "sk-from-env-file"
        finally:
            ss.reset_secret_scope(token)


    def test_scope_miss_absent_everywhere_returns_default(self, monkeypatch):
        monkeypatch.delenv("NOPE_KEY", raising=False)
        token = ss.set_secret_scope({})
        try:
            assert ss.get_secret("NOPE_KEY") is None
            assert ss.get_secret("NOPE_KEY", "d") == "d"
        finally:
            ss.reset_secret_scope(token)

    def test_multiplex_on_still_authoritative(self, monkeypatch):
        # The fallthrough is strictly multiplex-off behavior: turning
        # multiplexing on must restore scope-authoritative semantics.
        monkeypatch.setenv("OPENAI_API_KEY", "sk-other-profile")
        ss.set_multiplex_active(True)
        token = ss.set_secret_scope({})
        try:
            assert ss.get_secret("OPENAI_API_KEY") is None
        finally:
            ss.reset_secret_scope(token)


class TestScopeIsolation:
    """Two scopes never see each other's secrets."""

    def test_nested_scopes_restore(self):
        ss.set_multiplex_active(True)
        t1 = ss.set_secret_scope({"K": "a"})
        try:
            assert ss.get_secret("K") == "a"
            t2 = ss.set_secret_scope({"K": "b"})
            try:
                assert ss.get_secret("K") == "b"
            finally:
                ss.reset_secret_scope(t2)
            assert ss.get_secret("K") == "a"
        finally:
            ss.reset_secret_scope(t1)


class TestEnvFileParsing:
    """load_env_file parses without mutating os.environ."""




    def test_strips_utf8_bom_from_first_key(self, tmp_path):
        """Windows editors often save .env as UTF-8 with BOM (EF BB BF).

        Plain utf-8 keeps U+FEFF on the first key name, so get_secret('NAME')
        misses under an installed scope. utf-8-sig strips the leading BOM.
        """
        env = tmp_path / ".env"
        env.write_bytes(
            b"\xef\xbb\xbfANTHROPIC_API_KEY=sk-x\nOPENAI_API_KEY=sk-y\n"
        )
        out = ss.load_env_file(env)
        assert out == {
            "ANTHROPIC_API_KEY": "sk-x",
            "OPENAI_API_KEY": "sk-y",
        }
        assert "\ufeffANTHROPIC_API_KEY" not in out

        scope = ss.build_profile_secret_scope(tmp_path)
        ss.set_multiplex_active(True)
        token = ss.set_secret_scope(scope)
        try:
            assert ss.get_secret("ANTHROPIC_API_KEY") == "sk-x"
            assert ss.get_secret("OPENAI_API_KEY") == "sk-y"
        finally:
            ss.reset_secret_scope(token)
            ss.set_multiplex_active(False)

    def test_build_profile_secret_scope(self, tmp_path):
        (tmp_path / ".env").write_text("ANTHROPIC_API_KEY=sk-profile\n")
        assert ss.build_profile_secret_scope(tmp_path) == {
            "ANTHROPIC_API_KEY": "sk-profile"
        }

    def test_build_profile_secret_scope_includes_home_external_secrets(
        self, tmp_path, monkeypatch
    ):
        (tmp_path / ".env").write_text("XIAOMI_API_KEY=placeholder\n")
        from hermes_cli import env_loader

        home_key = str(tmp_path.resolve())
        monkeypatch.setitem(
            env_loader._SECRET_SOURCE_VALUES_BY_HOME,
            home_key,
            {"XIAOMI_API_KEY": "sk-from-bitwarden"},
        )

        assert ss.build_profile_secret_scope(tmp_path) == {
            "XIAOMI_API_KEY": "sk-from-bitwarden"
        }

    def test_build_profile_secret_scope_ignores_other_home_external_secrets(
        self, tmp_path, monkeypatch
    ):
        profile = tmp_path / "profile"
        other = tmp_path / "other"
        profile.mkdir()
        other.mkdir()
        from hermes_cli import env_loader

        monkeypatch.setitem(
            env_loader._SECRET_SOURCE_VALUES_BY_HOME,
            str(other.resolve()),
            {"XIAOMI_API_KEY": "sk-other-profile"},
        )

        assert ss.build_profile_secret_scope(profile) == {}
