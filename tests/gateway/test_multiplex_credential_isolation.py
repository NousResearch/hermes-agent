"""End-to-end credential isolation proof for multiplex mode (Workstream A).

These exercise the REAL resolution path (runtime_provider, secret scope, MCP
interpolation) rather than mocking it, proving the property that matters: two
profiles with different keys never see each other's, and an unscoped read in
multiplex mode fails closed instead of leaking.
"""
import pytest

from pathlib import Path

from agent import secret_scope as ss


@pytest.fixture(autouse=True)
def _reset(monkeypatch):
    ss.set_multiplex_active(False)
    yield
    ss.set_multiplex_active(False)


class TestRuntimeProviderUsesScope:
    """hermes_cli.runtime_provider._getenv resolves through the secret scope."""

    def test_getenv_reads_scope_under_multiplex(self, monkeypatch):
        from hermes_cli.runtime_provider import _getenv
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-global-leak")
        ss.set_multiplex_active(True)
        tok = ss.set_secret_scope({"ANTHROPIC_API_KEY": "sk-profileA"})
        try:
            assert _getenv("ANTHROPIC_API_KEY") == "sk-profileA"
        finally:
            ss.reset_secret_scope(tok)

    def test_getenv_two_profiles_isolated(self, monkeypatch):
        from hermes_cli.runtime_provider import _getenv
        ss.set_multiplex_active(True)

        tok_a = ss.set_secret_scope({"OPENAI_API_KEY": "sk-A"})
        try:
            assert _getenv("OPENAI_API_KEY") == "sk-A"
        finally:
            ss.reset_secret_scope(tok_a)

        tok_b = ss.set_secret_scope({"OPENAI_API_KEY": "sk-B"})
        try:
            assert _getenv("OPENAI_API_KEY") == "sk-B"
        finally:
            ss.reset_secret_scope(tok_b)

    def test_getenv_fails_closed_unscoped(self, monkeypatch):
        from hermes_cli.runtime_provider import _getenv
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-leak")
        ss.set_multiplex_active(True)
        with pytest.raises(ss.UnscopedSecretError):
            _getenv("OPENROUTER_API_KEY")

    def test_getenv_global_var_still_reads_environ(self, monkeypatch):
        from hermes_cli.runtime_provider import _getenv
        monkeypatch.setenv("HERMES_MAX_ITERATIONS", "42")
        ss.set_multiplex_active(True)
        # global var: no scope needed, no raise
        assert _getenv("HERMES_MAX_ITERATIONS") == "42"


class TestMcpInterpolationUsesScope:
    """MCP config ${VAR} interpolation resolves through the secret scope."""

    def test_interpolation_reads_scope(self, monkeypatch):
        from tools.mcp_tool import _interpolate_env_vars
        monkeypatch.setenv("MY_MCP_TOKEN", "global-token")
        ss.set_multiplex_active(True)
        tok = ss.set_secret_scope({"MY_MCP_TOKEN": "profile-token"})
        try:
            cfg = {"env": {"TOKEN": "${MY_MCP_TOKEN}"}}
            assert _interpolate_env_vars(cfg) == {"env": {"TOKEN": "profile-token"}}
        finally:
            ss.reset_secret_scope(tok)

    def test_interpolation_unset_keeps_placeholder(self, monkeypatch):
        from tools.mcp_tool import _interpolate_env_vars
        monkeypatch.delenv("UNSET_MCP_VAR", raising=False)
        # multiplex off: unset var keeps literal placeholder (legacy behavior)
        assert _interpolate_env_vars("${UNSET_MCP_VAR}") == "${UNSET_MCP_VAR}"

    def test_interpolation_off_reads_environ(self, monkeypatch):
        from tools.mcp_tool import _interpolate_env_vars
        monkeypatch.setenv("MY_MCP_TOKEN", "env-token")
        # multiplex off: legacy os.environ resolution
        assert _interpolate_env_vars("${MY_MCP_TOKEN}") == "env-token"


class TestProfilePathResolutionUnderMultiplexScope:
    """Profile-scoped paths must follow the per-turn _profile_runtime_scope.

    The multiplexed gateway (gateway.multiplex_profiles) serves every profile
    from ONE process, scoping each inbound turn with _profile_runtime_scope —
    the same in-process-many-profiles topology as the desktop tui_gateway. The
    profile-isolation fixes (per-call path resolution + thread context
    propagation) must therefore hold under THIS scope too, not just desktop.
    This is the regression guard proving reachability is not desktop-only.
    """

    def _profiles(self, tmp_path):
        prof_a = tmp_path / "profA"
        prof_b = tmp_path / "profB"
        for p in (prof_a, prof_b):
            (p / "skills").mkdir(parents=True, exist_ok=True)
            (p / "state").mkdir(parents=True, exist_ok=True)
        return prof_a, prof_b

    def test_skills_dir_follows_multiplex_scope(self, tmp_path):
        from gateway.run import _profile_runtime_scope
        import tools.skills_hub as sh

        prof_a, prof_b = self._profiles(tmp_path)
        with _profile_runtime_scope(prof_a):
            a_seen = Path(sh.SKILLS_DIR)
        with _profile_runtime_scope(prof_b):
            b_seen = Path(sh.SKILLS_DIR)

        assert a_seen == prof_a / "skills"
        assert b_seen == prof_b / "skills"

    def test_cache_dir_follows_multiplex_scope(self, tmp_path):
        from gateway.run import _profile_runtime_scope
        import gateway.platforms.base as gb

        _prof_a, prof_b = self._profiles(tmp_path)
        with _profile_runtime_scope(prof_b):
            seen = gb.get_image_cache_dir()
        assert str(seen).startswith(str(prof_b))

    def test_worker_thread_inherits_multiplex_scope(self, tmp_path):
        """A wrapped worker spawned inside the scope must see the right profile.

        The _profile_runtime_scope docstring relies on copy_context() carrying
        the override into the agent worker thread; this proves the M2 fix
        primitive delivers that under the multiplexer's scope.
        """
        import threading

        from gateway.run import _profile_runtime_scope
        from hermes_constants import get_hermes_home
        from tools.thread_context import propagate_context_to_thread

        _prof_a, prof_b = self._profiles(tmp_path)
        seen = {}

        def worker():
            seen["home"] = str(get_hermes_home())

        with _profile_runtime_scope(prof_b):
            t = threading.Thread(target=propagate_context_to_thread(worker))
            t.start()
            t.join()

        assert seen["home"] == str(prof_b)


class TestApiServerEnvIsScoped:
    """_apply_env_overrides must read API_SERVER_* through the profile scope,
    not the global os.environ, so a secondary profile is not falsely enabled
    by the default profile's key loaded into os.environ (#52307)."""

    def test_secondary_profile_scope_without_key_does_not_enable(self, monkeypatch):
        from gateway.config import _apply_env_overrides, GatewayConfig, Platform

        # Default profile's key is in the global process env (loaded at import).
        monkeypatch.setenv("API_SERVER_KEY", "global-default-key")
        ss.set_multiplex_active(True)

        # Secondary profile scope has NO api_server credential.
        tok = ss.set_secret_scope({"OPENAI_API_KEY": "sk-secondary"})
        try:
            config = GatewayConfig()
            _apply_env_overrides(config)
            api = config.platforms.get(Platform.API_SERVER)
            assert api is None or api.enabled is False
        finally:
            ss.reset_secret_scope(tok)

    def test_profile_scope_with_key_enables(self, monkeypatch):
        from gateway.config import _apply_env_overrides, GatewayConfig, Platform

        monkeypatch.delenv("API_SERVER_KEY", raising=False)
        ss.set_multiplex_active(True)
        tok = ss.set_secret_scope({"API_SERVER_KEY": "profile-scoped-key"})
        try:
            config = GatewayConfig()
            _apply_env_overrides(config)
            api = config.platforms.get(Platform.API_SERVER)
            assert api is not None and api.enabled is True
            assert api.extra.get("key") == "profile-scoped-key"
        finally:
            ss.reset_secret_scope(tok)
