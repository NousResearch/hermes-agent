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

    def test_scoped_read_uses_scope_not_environ(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-from-environ")
        ss.set_multiplex_active(True)
        token = ss.set_secret_scope({"ANTHROPIC_API_KEY": "sk-from-scope"})
        try:
            assert ss.get_secret("ANTHROPIC_API_KEY") == "sk-from-scope"
        finally:
            ss.reset_secret_scope(token)

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

    def test_global_env_still_reads_environ_under_multiplex(self, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", "/opt/data")
        ss.set_multiplex_active(True)
        # No scope, multiplex on — but HERMES_HOME is global, so no raise.
        assert ss.get_secret("HERMES_HOME") == "/opt/data"

    def test_kanban_prefix_is_global(self, monkeypatch):
        monkeypatch.setenv("HERMES_KANBAN_DB", "/x/kanban.db")
        ss.set_multiplex_active(True)
        assert ss.get_secret("HERMES_KANBAN_DB") == "/x/kanban.db"


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

    def test_parses_basic(self, tmp_path):
        env = tmp_path / ".env"
        env.write_text(
            "# comment\n"
            "ANTHROPIC_API_KEY=sk-abc\n"
            "export OPENAI_API_KEY=sk-def\n"
            'QUOTED="quoted-value"\n'
            "SINGLE='single'\n"
            "\n"
            "BAD_LINE_NO_EQUALS\n"
        )
        out = ss.load_env_file(env)
        assert out == {
            "ANTHROPIC_API_KEY": "sk-abc",
            "OPENAI_API_KEY": "sk-def",
            "QUOTED": "quoted-value",
            "SINGLE": "single",
        }

    def test_does_not_mutate_environ(self, tmp_path, monkeypatch):
        monkeypatch.delenv("ZZZ_KEY", raising=False)
        env = tmp_path / ".env"
        env.write_text("ZZZ_KEY=secret\n")
        ss.load_env_file(env)
        import os
        assert "ZZZ_KEY" not in os.environ

    def test_missing_file_returns_empty(self, tmp_path):
        assert ss.load_env_file(tmp_path / "nope.env") == {}

    def test_build_profile_secret_scope(self, tmp_path):
        (tmp_path / ".env").write_text("ANTHROPIC_API_KEY=sk-profile\n")
        assert ss.build_profile_secret_scope(tmp_path) == {
            "ANTHROPIC_API_KEY": "sk-profile"
        }

    def test_file_backed_secret_descriptor(self, tmp_path):
        secret_path = tmp_path / "TELEGRAM_BOT_TOKEN"
        secret_path.write_text("123456:scoped-token\n", encoding="utf-8")
        (tmp_path / ".env").write_text(
            "TELEGRAM_BOT_TOKEN={'source':'file','path':'"
            + str(secret_path)
            + "'}\n",
            encoding="utf-8",
        )

        assert (
            ss.build_profile_secret_scope(tmp_path)["TELEGRAM_BOT_TOKEN"]
            == "123456:scoped-token"
        )


class TestExternalSecretSourceMerge:
    """BWS / 1Password keys must be visible under an active scope.

    Regression coverage for the cron/multiplex failure where
    ``build_profile_secret_scope`` loaded only ``.env``, hiding
    externally-injected keys that exist in ``os.environ`` with
    provenance tracked in ``env_loader._SECRET_SOURCES``.
    """

    def test_bws_key_visible_under_scope(self, tmp_path, monkeypatch):
        """BWS-proven key resolves via get_secret under an active scope."""
        (tmp_path / ".env").write_text("BWS_ACCESS_TOKEN=bootstrap\n")
        monkeypatch.setenv("OLLAMA_API_KEY", "ollama-bws-key")
        monkeypatch.setenv("BWS_ACCESS_TOKEN", "bootstrap")

        from hermes_cli import env_loader

        env_loader._SECRET_SOURCES["OLLAMA_API_KEY"] = "bitwarden"
        try:
            scope = ss.build_profile_secret_scope(tmp_path)
            assert scope["OLLAMA_API_KEY"] == "ollama-bws-key"

            ss.set_multiplex_active(True)
            token = ss.set_secret_scope(scope)
            try:
                assert ss.get_secret("OLLAMA_API_KEY") == "ollama-bws-key"
            finally:
                ss.reset_secret_scope(token)
        finally:
            env_loader._SECRET_SOURCES.clear()

    def test_env_value_takes_precedence_over_bws(self, tmp_path, monkeypatch):
        """A key in .env must win over the BWS-injected value."""
        (tmp_path / ".env").write_text("OLLAMA_API_KEY=from-dotenv\n")
        monkeypatch.setenv("OLLAMA_API_KEY", "from-bws")

        from hermes_cli import env_loader

        env_loader._SECRET_SOURCES["OLLAMA_API_KEY"] = "bitwarden"
        try:
            scope = ss.build_profile_secret_scope(tmp_path)
            assert scope["OLLAMA_API_KEY"] == "from-dotenv"
        finally:
            env_loader._SECRET_SOURCES.clear()

    def test_disabled_bws_does_not_leak_unrelated_env(self, tmp_path, monkeypatch):
        """When _SECRET_SOURCES is empty, arbitrary os.environ keys are excluded."""
        (tmp_path / ".env").write_text("# empty\n")
        monkeypatch.setenv("SOME_RANDOM_KEY", "should-not-leak")

        from hermes_cli import env_loader

        env_loader._SECRET_SOURCES.clear()
        try:
            scope = ss.build_profile_secret_scope(tmp_path)
            assert "SOME_RANDOM_KEY" not in scope
            assert scope.get("SOME_RANDOM_KEY") is None
        finally:
            env_loader._SECRET_SOURCES.clear()

    def test_multiplex_isolation_preserved(self, tmp_path, monkeypatch):
        """BWS-proven keys appear in scope; untracked env keys do not.

        This is the cross-profile isolation invariant: under multiplex,
        a key from another profile's shell export (no provenance entry)
        must NOT be visible, even though a BWS-proven key must be.
        """
        (tmp_path / ".env").write_text("# empty\n")
        monkeypatch.setenv("OLLAMA_API_KEY", "bws-key")
        monkeypatch.setenv("OTHER_PROFILE_KEY", "leaky")

        from hermes_cli import env_loader

        env_loader._SECRET_SOURCES.clear()
        env_loader._SECRET_SOURCES["OLLAMA_API_KEY"] = "bitwarden"
        try:
            scope = ss.build_profile_secret_scope(tmp_path)
            assert scope["OLLAMA_API_KEY"] == "bws-key"
            assert "OTHER_PROFILE_KEY" not in scope

            ss.set_multiplex_active(True)
            token = ss.set_secret_scope(scope)
            try:
                assert ss.get_secret("OLLAMA_API_KEY") == "bws-key"
                assert ss.get_secret("OTHER_PROFILE_KEY") is None
            finally:
                ss.reset_secret_scope(token)
        finally:
            env_loader._SECRET_SOURCES.clear()
