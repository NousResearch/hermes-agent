"""The default profile keeps its process-environment credentials in multiplex mode.

Turning on ``gateway.multiplex_profiles`` must not strip credentials that the
deployment injected into ``os.environ`` (docker ``environment:``, systemd
``Environment=``, ``op run``, plain exports) from the profile the process was
launched under. Before this was fixed, flipping the flag on made every
process-env provider key invisible to the default profile — provider resolution
failed with "No usable credentials found" while the variable was plainly set —
whereas ``<home>/.env``-based deployments kept working.

The isolation guarantee multiplexing exists to provide is asserted here too:
secondary profiles are still fail-closed and never inherit the process env.
"""
import pytest

from pathlib import Path

from agent import secret_scope as ss


@pytest.fixture(autouse=True)
def _reset():
    ss.set_multiplex_active(False)
    yield
    ss.set_multiplex_active(False)


@pytest.fixture
def homes(tmp_path, monkeypatch):
    """A default profile home plus a secondary profile home underneath it."""
    default_home = tmp_path / "data"
    secondary_home = default_home / "profiles" / "tower-ops"
    secondary_home.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(default_home))
    return default_home, secondary_home


class TestDefaultProfileInheritsProcessEnv:
    def test_default_profile_sees_process_env_credential(self, homes, monkeypatch):
        default_home, _ = homes
        monkeypatch.setenv("OPENCODE_GO_API_KEY", "sk-from-process-env")

        scope = ss.build_profile_secret_scope(default_home)

        assert scope.get("OPENCODE_GO_API_KEY") == "sk-from-process-env"

    def test_resolves_through_get_secret_under_multiplex(self, homes, monkeypatch):
        """The property that actually broke: the real read path, multiplex ON."""
        default_home, _ = homes
        monkeypatch.setenv("OPENCODE_GO_API_KEY", "sk-from-process-env")
        ss.set_multiplex_active(True)

        token = ss.set_secret_scope(ss.build_profile_secret_scope(default_home))
        try:
            assert ss.get_secret("OPENCODE_GO_API_KEY") == "sk-from-process-env"
        finally:
            ss.reset_secret_scope(token)

    def test_env_file_overrides_process_env(self, homes, monkeypatch):
        """``<home>/.env`` is more specific and wins, as it did pre-multiplex."""
        default_home, _ = homes
        monkeypatch.setenv("OPENCODE_GO_API_KEY", "sk-from-process-env")
        (default_home / ".env").write_text(
            "OPENCODE_GO_API_KEY=sk-from-env-file\n", encoding="utf-8"
        )

        scope = ss.build_profile_secret_scope(default_home)

        assert scope.get("OPENCODE_GO_API_KEY") == "sk-from-env-file"

    def test_global_env_vars_still_excluded_from_scope(self, homes, monkeypatch):
        """Globals keep reading os.environ directly; they are not profile secrets."""
        default_home, _ = homes
        monkeypatch.setenv("PATH", "/usr/bin")

        scope = ss.build_profile_secret_scope(default_home)

        assert "PATH" not in scope
        assert "HERMES_HOME" not in scope


class TestSecondaryProfileStaysIsolated:
    def test_secondary_profile_does_not_inherit_process_env(self, homes, monkeypatch):
        """The isolation guarantee: profile B never sees the process env."""
        _, secondary_home = homes
        monkeypatch.setenv("OPENCODE_GO_API_KEY", "sk-from-process-env")

        scope = ss.build_profile_secret_scope(secondary_home)

        assert "OPENCODE_GO_API_KEY" not in scope

    def test_secondary_profile_fails_closed_under_multiplex(self, homes, monkeypatch):
        _, secondary_home = homes
        monkeypatch.setenv("OPENCODE_GO_API_KEY", "sk-from-process-env")
        ss.set_multiplex_active(True)

        token = ss.set_secret_scope(ss.build_profile_secret_scope(secondary_home))
        try:
            assert ss.get_secret("OPENCODE_GO_API_KEY") is None
        finally:
            ss.reset_secret_scope(token)

    def test_secondary_profile_uses_its_own_env_file(self, homes, monkeypatch):
        _, secondary_home = homes
        monkeypatch.setenv("OPENCODE_GO_API_KEY", "sk-from-process-env")
        (secondary_home / ".env").write_text(
            "OPENCODE_GO_API_KEY=sk-secondary\n", encoding="utf-8"
        )

        scope = ss.build_profile_secret_scope(secondary_home)

        assert scope.get("OPENCODE_GO_API_KEY") == "sk-secondary"


class TestPerTaskProfileOverrideDoesNotConfuseDefaultDetection:
    def test_context_local_home_override_is_ignored(self, homes, monkeypatch):
        """A per-task override must not make a secondary profile look default."""
        from hermes_constants import set_hermes_home_override, reset_hermes_home_override

        _, secondary_home = homes
        monkeypatch.setenv("OPENCODE_GO_API_KEY", "sk-from-process-env")

        token = set_hermes_home_override(str(secondary_home))
        try:
            scope = ss.build_profile_secret_scope(secondary_home)
        finally:
            reset_hermes_home_override(token)

        assert "OPENCODE_GO_API_KEY" not in scope
