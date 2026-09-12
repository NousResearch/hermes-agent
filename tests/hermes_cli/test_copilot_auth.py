"""Tests for hermes_cli.copilot_auth — Copilot token validation and resolution."""

import pytest
from unittest.mock import patch


class TestTokenValidation:
    """Token type validation."""

    def test_classic_pat_rejected(self):
        from hermes_cli.copilot_auth import validate_copilot_token
        valid, msg = validate_copilot_token("ghp_abcdefghijklmnop1234")
        assert valid is False
        assert "Classic Personal Access Tokens" in msg
        assert "ghp_" in msg


class TestResolveToken:
    """Token resolution with env var priority."""


    def test_gh_token_second_priority(self, monkeypatch):
        from hermes_cli.copilot_auth import resolve_copilot_token
        monkeypatch.delenv("COPILOT_GITHUB_TOKEN", raising=False)
        monkeypatch.setenv("GH_TOKEN", "gho_gh_second")
        monkeypatch.setenv("GITHUB_TOKEN", "gho_github_third")
        token, source = resolve_copilot_token()
        assert token == "gho_gh_second"
        assert source == "GH_TOKEN"




    def test_gh_cli_classic_pat_raises(self, monkeypatch):
        from hermes_cli.copilot_auth import resolve_copilot_token
        monkeypatch.delenv("COPILOT_GITHUB_TOKEN", raising=False)
        monkeypatch.delenv("GH_TOKEN", raising=False)
        monkeypatch.delenv("GITHUB_TOKEN", raising=False)
        with patch("hermes_cli.copilot_auth._try_gh_cli_token", return_value="ghp_classic"):
            with pytest.raises(ValueError, match="classic PAT"):
                resolve_copilot_token()

    def test_invalid_env_var_skips_gh_cli_fallback(self, monkeypatch):
        """When an env var is set but holds an unsupported classic PAT,
        resolve_copilot_token must NOT fall back to ``gh auth token``.

        The user explicitly exported a token; silently substituting one
        from the gh CLI credential store is surprising and the subprocess
        call adds up to 5s of latency on Windows cold starts (#60800).
        Only fall back to the CLI when NO Copilot env var is set at all.
        """
        from hermes_cli.copilot_auth import resolve_copilot_token
        monkeypatch.delenv("COPILOT_GITHUB_TOKEN", raising=False)
        monkeypatch.delenv("GH_TOKEN", raising=False)
        monkeypatch.setenv("GITHUB_TOKEN", "ghp_classic_pat_nope")
        with patch("hermes_cli.copilot_auth._try_gh_cli_token") as mock_cli:
            token, source = resolve_copilot_token()
        assert token == ""
        assert source == ""
        mock_cli.assert_not_called()

    def test_all_env_vars_invalid_skips_gh_cli_fallback(self, monkeypatch):
        """All three env vars set to classic PATs → no gh CLI call."""
        from hermes_cli.copilot_auth import resolve_copilot_token
        monkeypatch.setenv("COPILOT_GITHUB_TOKEN", "ghp_one")
        monkeypatch.setenv("GH_TOKEN", "ghp_two")
        monkeypatch.setenv("GITHUB_TOKEN", "ghp_three")
        with patch("hermes_cli.copilot_auth._try_gh_cli_token") as mock_cli:
            token, source = resolve_copilot_token()
        assert token == ""
        assert source == ""
        mock_cli.assert_not_called()


class TestGhCliTokenCache:
    """The gh-CLI probe result is cached — a miss must not re-spawn gh.

    Regression: /api/model/options ran `gh auth token` four times per build;
    with no gh credential store each probe blocked its full 5s timeout, so the
    Desktop Models/Providers settings pages took 20s per open and exceeded the
    renderer's 15s IPC budget (Aug 2026 desktop audit).
    """

    def _reset(self):
        from hermes_cli.copilot_auth import _invalidate_gh_cli_token_cache
        _invalidate_gh_cli_token_cache()

    def test_miss_is_cached_and_probe_runs_once(self):
        from hermes_cli import copilot_auth
        self._reset()
        with patch.object(copilot_auth, "_probe_gh_cli_token", return_value=None) as probe:
            assert copilot_auth._try_gh_cli_token() is None
            assert copilot_auth._try_gh_cli_token() is None
            assert copilot_auth._try_gh_cli_token() is None
        assert probe.call_count == 1
        self._reset()

    def test_hit_is_cached(self):
        from hermes_cli import copilot_auth
        self._reset()
        with patch.object(copilot_auth, "_probe_gh_cli_token", return_value="gho_cached") as probe:
            assert copilot_auth._try_gh_cli_token() == "gho_cached"
            assert copilot_auth._try_gh_cli_token() == "gho_cached"
        assert probe.call_count == 1
        self._reset()

    def test_ttl_expiry_reprobes(self, monkeypatch):
        from hermes_cli import copilot_auth
        self._reset()
        clock = {"now": 1000.0}
        monkeypatch.setattr(copilot_auth.time, "monotonic", lambda: clock["now"])
        with patch.object(copilot_auth, "_probe_gh_cli_token", return_value=None) as probe:
            copilot_auth._try_gh_cli_token()
            clock["now"] += copilot_auth._GH_CLI_TOKEN_CACHE_TTL_SECONDS + 1
            copilot_auth._try_gh_cli_token()
        assert probe.call_count == 2
        self._reset()

    def test_invalidate_forces_reprobe(self):
        from hermes_cli import copilot_auth
        self._reset()
        with patch.object(copilot_auth, "_probe_gh_cli_token", return_value=None) as probe:
            copilot_auth._try_gh_cli_token()
            copilot_auth._invalidate_gh_cli_token_cache()
            copilot_auth._try_gh_cli_token()
        assert probe.call_count == 2
        self._reset()


class TestGhCliProbeHome:
    """`gh auth token` must be probed against the OS account home too.

    `gh` reads hosts.yml from $HOME. A Hermes process running with HOME={HERMES_HOME}/home
    (container installs, and hosts where is_container() false-positives — #58135) probed the
    empty profile home, got "no oauth token found", and dropped Copilot from every model
    picker even though the user was logged in.
    """

    def _run_result(self, stdout: str, returncode: int = 0):
        from subprocess import CompletedProcess
        return CompletedProcess(args=[], returncode=returncode, stdout=stdout, stderr="")

    def test_retries_with_real_home_when_profile_home_has_no_token(self, tmp_path, monkeypatch):
        from hermes_cli import copilot_auth
        profile_home, real_home = tmp_path / "profile-home", tmp_path / "real-home"
        profile_home.mkdir()
        real_home.mkdir()
        monkeypatch.setenv("HOME", str(profile_home))
        monkeypatch.delenv("GH_CONFIG_DIR", raising=False)
        monkeypatch.setattr(
            "hermes_constants.external_credential_home_candidates",
            lambda env=None: [profile_home, real_home])
        monkeypatch.setattr(copilot_auth, "_gh_cli_candidates", lambda: ["/usr/bin/gh"])

        seen_homes = []

        def _fake_run(cmd, **kwargs):
            home = kwargs["env"]["HOME"]
            seen_homes.append(home)
            return self._run_result("gho_realhome\n" if home == str(real_home) else "")

        monkeypatch.setattr(copilot_auth.subprocess, "run", _fake_run)
        assert copilot_auth._probe_gh_cli_token() == "gho_realhome"
        assert seen_homes == [str(profile_home), str(real_home)]

    def test_stops_at_the_first_home_that_answers(self, tmp_path, monkeypatch):
        from hermes_cli import copilot_auth
        profile_home, real_home = tmp_path / "profile-home", tmp_path / "real-home"
        profile_home.mkdir()
        real_home.mkdir()
        monkeypatch.setenv("HOME", str(profile_home))
        monkeypatch.delenv("GH_CONFIG_DIR", raising=False)
        monkeypatch.setattr(
            "hermes_constants.external_credential_home_candidates",
            lambda env=None: [profile_home, real_home])
        monkeypatch.setattr(copilot_auth, "_gh_cli_candidates", lambda: ["/usr/bin/gh"])

        calls = []
        monkeypatch.setattr(
            copilot_auth.subprocess, "run",
            lambda cmd, **kw: (calls.append(kw["env"]["HOME"]), self._run_result("gho_first\n"))[1])
        assert copilot_auth._probe_gh_cli_token() == "gho_first"
        assert calls == [str(profile_home)]

    def test_explicit_gh_config_dir_is_probed_once_and_home_untouched(self, tmp_path, monkeypatch):
        """GH_CONFIG_DIR already pins the config location — don't override the user's HOME."""
        from hermes_cli import copilot_auth
        monkeypatch.setenv("HOME", str(tmp_path / "profile-home"))
        monkeypatch.setenv("GH_CONFIG_DIR", str(tmp_path / "gh-config"))
        monkeypatch.setattr(copilot_auth, "_gh_cli_candidates", lambda: ["/usr/bin/gh"])

        envs = []
        monkeypatch.setattr(
            copilot_auth.subprocess, "run",
            lambda cmd, **kw: (envs.append(kw["env"]), self._run_result("gho_cfgdir\n"))[1])
        assert copilot_auth._probe_gh_cli_token() == "gho_cfgdir"
        assert len(envs) == 1
        assert envs[0]["HOME"] == str(tmp_path / "profile-home")

    def test_missing_binary_skips_to_the_next_candidate(self, tmp_path, monkeypatch):
        """A FileNotFoundError is about the binary, not the home — don't retry the same path."""
        from hermes_cli import copilot_auth
        home_a, home_b = tmp_path / "a", tmp_path / "b"
        home_a.mkdir()
        home_b.mkdir()
        monkeypatch.setenv("HOME", str(home_a))
        monkeypatch.delenv("GH_CONFIG_DIR", raising=False)
        monkeypatch.setattr(
            "hermes_constants.external_credential_home_candidates", lambda env=None: [home_a, home_b])
        monkeypatch.setattr(copilot_auth, "_gh_cli_candidates", lambda: ["/missing/gh", "/usr/bin/gh"])

        calls = []

        def _fake_run(cmd, **kwargs):
            calls.append((cmd[0], kwargs["env"]["HOME"]))
            if cmd[0] == "/missing/gh":
                raise FileNotFoundError(cmd[0])
            return self._run_result("gho_second\n")

        monkeypatch.setattr(copilot_auth.subprocess, "run", _fake_run)
        assert copilot_auth._probe_gh_cli_token() == "gho_second"
        assert calls == [("/missing/gh", str(home_a)), ("/usr/bin/gh", str(home_a))]


class TestRequestHeaders:
    """Copilot API header generation."""

    def test_default_headers_include_openai_intent(self):
        from hermes_cli.copilot_auth import copilot_request_headers
        headers = copilot_request_headers()
        assert headers["Openai-Intent"] == "conversation-edits"
        assert headers["User-Agent"] == "HermesAgent/1.0"
        assert "Editor-Version" in headers


    def test_no_vision_header_by_default(self):
        from hermes_cli.copilot_auth import copilot_request_headers
        headers = copilot_request_headers()
        assert "Copilot-Vision-Request" not in headers


class TestCopilotDefaultHeaders:
    """The models.py copilot_default_headers uses copilot_auth."""


    def test_agent_turn_explicit(self):
        """Explicitly passing is_agent_turn=True sets x-initiator to 'agent'."""
        from hermes_cli.models import copilot_default_headers
        headers = copilot_default_headers(is_agent_turn=True)
        assert headers["x-initiator"] == "agent"

    def test_param_passthrough_both_values(self):
        """is_agent_turn param correctly maps to x-initiator for both True and False."""
        from hermes_cli.models import copilot_default_headers
        for is_agent, expected in [(True, "agent"), (False, "user")]:
            headers = copilot_default_headers(is_agent_turn=is_agent)
            assert headers["x-initiator"] == expected, (
                f"is_agent_turn={is_agent} should produce x-initiator={expected!r}, "
                f"got {headers['x-initiator']!r}"
            )


class TestApiModeSelection:
    """API mode selection matching opencode's shouldUseCopilotResponsesApi."""

    def test_gpt5_uses_responses(self):
        from hermes_cli.models import _should_use_copilot_responses_api
        assert _should_use_copilot_responses_api("gpt-5.4") is True
        assert _should_use_copilot_responses_api("gpt-5.4-mini") is True
        assert _should_use_copilot_responses_api("gpt-5.3-codex") is True
        assert _should_use_copilot_responses_api("gpt-5.2-codex") is True
        assert _should_use_copilot_responses_api("gpt-5.2") is True
        assert _should_use_copilot_responses_api("gpt-5.1-codex-max") is True

    def test_gpt5_mini_excluded(self):
        from hermes_cli.models import _should_use_copilot_responses_api
        assert _should_use_copilot_responses_api("gpt-5-mini") is False


class TestEnvVarOrder:
    """PROVIDER_REGISTRY has correct env var order."""

    def test_copilot_env_vars_include_copilot_github_token(self):
        from hermes_cli.auth import PROVIDER_REGISTRY
        copilot = PROVIDER_REGISTRY["copilot"]
        assert "COPILOT_GITHUB_TOKEN" in copilot.api_key_env_vars
        # COPILOT_GITHUB_TOKEN should be first
        assert copilot.api_key_env_vars[0] == "COPILOT_GITHUB_TOKEN"

