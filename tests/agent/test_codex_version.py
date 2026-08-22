"""Tests for the authentic Codex CLI version resolver (``agent.codex_version``).

The resolver must present the *installed* codex CLI version (from
``codex --version``) rather than a latest-GitHub-release guess, honor an
operator env override, memoize per executable, and never raise on a hot path
(falling back to a pinned constant when the CLI is absent). The same value
feeds the Cloudflare ``User-Agent`` identity, the ``/models`` probe URL, and
the local app-server ``initialize`` handshake so all three stay consistent.
"""
from __future__ import annotations

import subprocess
from types import SimpleNamespace
from unittest.mock import patch

import pytest

import agent.codex_version as cv


@pytest.fixture(autouse=True)
def _clear_state(monkeypatch):
    """Each test starts with a clean memo + no env override."""
    cv._memo.clear()
    monkeypatch.delenv("HERMES_CODEX_CLI_VERSION", raising=False)
    monkeypatch.delenv("HERMES_CODEX_BIN", raising=False)
    yield
    cv._memo.clear()


def _fake_run(stdout, returncode=0):
    def _runner(*args, **kwargs):
        return SimpleNamespace(stdout=stdout, stderr="", returncode=returncode)
    return _runner


class TestInstalledExecutableResolution:
    def test_parses_installed_codex_version(self, monkeypatch):
        with patch.object(
            cv.subprocess, "run", _fake_run("codex-cli 0.140.2\n")
        ):
            assert cv.get_codex_cli_version() == "0.140.2"

    def test_tolerates_trailing_metadata(self, monkeypatch):
        with patch.object(
            cv.subprocess, "run",
            _fake_run("codex-cli 0.141.0 (rust 1.86, abcdef)\n"),
        ):
            assert cv.get_codex_cli_version() == "0.141.0"

    def test_uses_configured_bin_argument(self, monkeypatch):
        captured = {}

        def _runner(cmd, *args, **kwargs):
            captured["cmd"] = cmd
            return SimpleNamespace(stdout="codex-cli 0.150.0", stderr="", returncode=0)

        with patch.object(cv.subprocess, "run", _runner):
            assert cv.get_codex_cli_version(codex_bin="/opt/codex") == "0.150.0"
        assert captured["cmd"][0] == "/opt/codex"
        assert captured["cmd"][1] == "--version"

    def test_honors_hermes_codex_bin_env(self, monkeypatch):
        monkeypatch.setenv("HERMES_CODEX_BIN", "/custom/codex")
        captured = {}

        def _runner(cmd, *args, **kwargs):
            captured["cmd"] = cmd
            return SimpleNamespace(stdout="codex-cli 0.151.0", stderr="", returncode=0)

        with patch.object(cv.subprocess, "run", _runner):
            assert cv.get_codex_cli_version() == "0.151.0"
        assert captured["cmd"][0] == "/custom/codex"


class TestFallbackBehavior:
    def test_missing_binary_falls_back(self, monkeypatch):
        def _raise(*args, **kwargs):
            raise FileNotFoundError("codex not found")

        with patch.object(cv.subprocess, "run", _raise):
            assert cv.get_codex_cli_version() == cv._FALLBACK_CODEX_CLI_VERSION

    def test_nonzero_exit_falls_back(self, monkeypatch):
        with patch.object(
            cv.subprocess, "run", _fake_run("boom", returncode=1)
        ):
            assert cv.get_codex_cli_version() == cv._FALLBACK_CODEX_CLI_VERSION

    def test_unparseable_output_falls_back(self, monkeypatch):
        with patch.object(
            cv.subprocess, "run", _fake_run("codex-cli unknown\n")
        ):
            assert cv.get_codex_cli_version() == cv._FALLBACK_CODEX_CLI_VERSION

    def test_timeout_falls_back(self, monkeypatch):
        def _timeout(*args, **kwargs):
            raise subprocess.TimeoutExpired(cmd="codex", timeout=10)

        with patch.object(cv.subprocess, "run", _timeout):
            assert cv.get_codex_cli_version() == cv._FALLBACK_CODEX_CLI_VERSION

    def test_fallback_is_semver_shaped(self):
        parts = cv._FALLBACK_CODEX_CLI_VERSION.split(".")
        assert len(parts) == 3 and all(p.isdigit() for p in parts)


class TestOverrideAndMemo:
    def test_env_override_wins_without_subprocess(self, monkeypatch):
        monkeypatch.setenv("HERMES_CODEX_CLI_VERSION", "9.9.9")

        def _boom(*args, **kwargs):
            raise AssertionError("subprocess must not run when override is set")

        with patch.object(cv.subprocess, "run", _boom):
            assert cv.get_codex_cli_version() == "9.9.9"

    def test_env_override_normalized(self, monkeypatch):
        monkeypatch.setenv("HERMES_CODEX_CLI_VERSION", "v1.2.3-beta")
        assert cv.get_codex_cli_version() == "1.2.3"

    def test_result_memoized_per_bin(self, monkeypatch):
        calls = {"n": 0}

        def _runner(*args, **kwargs):
            calls["n"] += 1
            return SimpleNamespace(stdout="codex-cli 0.142.0", stderr="", returncode=0)

        with patch.object(cv.subprocess, "run", _runner):
            first = cv.get_codex_cli_version()
            second = cv.get_codex_cli_version()
        assert first == second == "0.142.0"
        assert calls["n"] == 1  # second call served from memo


class TestIdentityWiring:
    """The resolver must feed all three Codex identity surfaces consistently."""

    def test_cloudflare_user_agent_uses_resolver(self):
        from agent.auxiliary_client import _codex_cloudflare_headers

        with patch(
            "agent.codex_version.get_codex_cli_version", return_value="0.143.0"
        ):
            headers = _codex_cloudflare_headers("not-a-real-jwt")
        assert headers["User-Agent"] == "codex_cli_rs/0.143.0 (Hermes Agent)"
        assert headers["originator"] == "codex_cli_rs"

    def test_models_probe_url_uses_resolver(self):
        import agent.model_metadata as mm

        captured = {}

        def _fake_get(url, *args, **kwargs):
            captured["url"] = url
            raise RuntimeError("stop after URL capture")

        mm._codex_oauth_context_cache.clear()
        with patch("agent.codex_version.get_codex_cli_version", return_value="0.144.0"):
            with patch.object(mm, "_ensure_requests", lambda: None):
                with patch.object(mm, "requests", SimpleNamespace(get=_fake_get)):
                    with patch.object(mm, "_resolve_requests_verify", lambda: True):
                        mm._fetch_codex_oauth_context_lengths_with_source(
                            "dummy-token"
                        )
        assert "client_version=0.144.0" in captured["url"]
