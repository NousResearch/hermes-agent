"""Regression: oneshot (-z) must try fallback providers when the primary
provider raises AuthError at resolution time (issue #81209).

The gateway has ``_try_resolve_fallback_provider`` for this; the oneshot lane
previously called ``resolve_runtime_provider`` bare, so a quota-exhausted (429)
or auth-failed primary killed ``hermes -z`` for the entire quota window even
when a healthy ``fallback_providers`` chain was configured.
"""

from __future__ import annotations

import logging
from unittest.mock import patch

import pytest

from hermes_cli.auth import AuthError


class TestOneshotResolutionTimeFallback:
    """Oneshot must consult the fallback chain when the primary provider's
    resolution fails with AuthError — before AIAgent is constructed."""

    def test_auth_error_tries_fallback_provider(self, tmp_path, monkeypatch):
        """When resolve_runtime_provider raises AuthError on the primary,
        oneshot must iterate the fallback chain and use the first viable entry.
        """
        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir(parents=True, exist_ok=True)
        (hermes_home / "config.yaml").write_text(
            "model:\n"
            "  provider: openai-codex\n"
            "  default: gpt-5.5\n"
            "fallback_providers:\n"
            "  - provider: openrouter\n"
            "    model: anthropic/claude-sonnet-4\n"
        )
        monkeypatch.setattr("hermes_constants.Path.home", lambda: tmp_path)
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))

        call_log: list[dict] = []

        def _mock_resolve(**kwargs):
            call_log.append(kwargs)
            if len(call_log) == 1:
                # Primary resolution fails with quota exhaustion.
                raise AuthError(
                    "Codex provider quota exhausted (429); retry after 39750s."
                )
            # Fallback entry resolves successfully.
            return {
                "api_key": "fallback-key",
                "base_url": "https://openrouter.ai/api/v1",
                "provider": "openrouter",
                "requested_provider": "openrouter",
                "api_mode": "openai_chat",
                "command": None,
                "args": None,
                "credential_pool": None,
            }

        with (
            patch(
                "hermes_cli.runtime_provider.resolve_runtime_provider",
                side_effect=_mock_resolve,
            ),
            patch("run_agent.AIAgent") as mock_agent_cls,
            patch("hermes_cli.oneshot._create_session_db_for_oneshot"),
            patch("hermes_cli.mcp_startup.ensure_mcp_discovery_before_agent_build"),
        ):
            mock_agent = mock_agent_cls.return_value
            mock_agent.run_conversation.return_value = {"final_response": "pong"}

            from hermes_cli.oneshot import _run_agent

            response, _ = _run_agent(prompt="ping", model=None, provider=None)

        # Fallback was attempted.
        assert len(call_log) >= 2, (
            f"Expected primary + fallback resolution attempts, got {len(call_log)}"
        )
        # The fallback entry's provider was requested.
        second_call = call_log[1]
        assert second_call.get("requested") == "openrouter"
        # AIAgent was constructed with the fallback's credentials.
        construct_kwargs = mock_agent_cls.call_args
        assert construct_kwargs.kwargs.get("api_key") == "fallback-key"
        assert construct_kwargs.kwargs.get("provider") == "openrouter"
        # The fallback entry's model was used.
        assert construct_kwargs.kwargs.get("model") == "anthropic/claude-sonnet-4"
        assert response == "pong"

    def test_no_fallback_chain_reraises_auth_error(self, tmp_path, monkeypatch):
        """Without a fallback chain configured, the AuthError must propagate."""
        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir(parents=True, exist_ok=True)
        (hermes_home / "config.yaml").write_text(
            "model:\n"
            "  provider: openai-codex\n"
            "  default: gpt-5.5\n"
        )
        monkeypatch.setattr("hermes_constants.Path.home", lambda: tmp_path)
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))

        def _mock_resolve(**kwargs):
            raise AuthError("Codex provider quota exhausted (429)")

        with (
            patch(
                "hermes_cli.runtime_provider.resolve_runtime_provider",
                side_effect=_mock_resolve,
            ),
            patch("hermes_cli.oneshot._create_session_db_for_oneshot"),
            patch("hermes_cli.mcp_startup.ensure_mcp_discovery_before_agent_build"),
        ):
            from hermes_cli.oneshot import _run_agent

            with pytest.raises(AuthError):
                _run_agent(prompt="ping", model=None, provider=None)

    def test_non_auth_error_reraises_unchanged(self, tmp_path, monkeypatch):
        """Non-AuthError exceptions (e.g. config errors) must propagate
        without consulting the fallback chain."""
        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir(parents=True, exist_ok=True)
        (hermes_home / "config.yaml").write_text(
            "model:\n"
            "  provider: openai-codex\n"
            "  default: gpt-5.5\n"
            "fallback_providers:\n"
            "  - provider: openrouter\n"
            "    model: anthropic/claude-sonnet-4\n"
        )
        monkeypatch.setattr("hermes_constants.Path.home", lambda: tmp_path)
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))

        call_count = {"n": 0}

        def _mock_resolve(**kwargs):
            call_count["n"] += 1
            raise RuntimeError("config file not found")

        with (
            patch(
                "hermes_cli.runtime_provider.resolve_runtime_provider",
                side_effect=_mock_resolve,
            ),
            patch("hermes_cli.oneshot._create_session_db_for_oneshot"),
            patch("hermes_cli.mcp_startup.ensure_mcp_discovery_before_agent_build"),
        ):
            from hermes_cli.oneshot import _run_agent

            with pytest.raises(RuntimeError, match="config file not found"):
                _run_agent(prompt="ping", model=None, provider=None)

        # Only the primary was attempted — no fallback consultation.
        assert call_count["n"] == 1

    def test_all_fallback_entries_fail_reraises_auth_error(
        self, tmp_path, monkeypatch
    ):
        """When every fallback entry also fails, the original AuthError must
        propagate."""
        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir(parents=True, exist_ok=True)
        (hermes_home / "config.yaml").write_text(
            "model:\n"
            "  provider: openai-codex\n"
            "  default: gpt-5.5\n"
            "fallback_providers:\n"
            "  - provider: openrouter\n"
            "    model: anthropic/claude-sonnet-4\n"
            "  - provider: nous\n"
            "    model: hermes-4\n"
        )
        monkeypatch.setattr("hermes_constants.Path.home", lambda: tmp_path)
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))

        def _mock_resolve(**kwargs):
            raise AuthError("all providers exhausted")

        with (
            patch(
                "hermes_cli.runtime_provider.resolve_runtime_provider",
                side_effect=_mock_resolve,
            ),
            patch("hermes_cli.oneshot._create_session_db_for_oneshot"),
            patch("hermes_cli.mcp_startup.ensure_mcp_discovery_before_agent_build"),
        ):
            from hermes_cli.oneshot import _run_agent

            with pytest.raises(AuthError, match="all providers exhausted"):
                _run_agent(prompt="ping", model=None, provider=None)

    def test_second_fallback_entry_used_when_first_fails(
        self, tmp_path, monkeypatch
    ):
        """When the first fallback entry fails, the next one must be tried."""
        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir(parents=True, exist_ok=True)
        (hermes_home / "config.yaml").write_text(
            "model:\n"
            "  provider: openai-codex\n"
            "  default: gpt-5.5\n"
            "fallback_providers:\n"
            "  - provider: openrouter\n"
            "    model: anthropic/claude-sonnet-4\n"
            "  - provider: nous\n"
            "    model: hermes-4\n"
        )
        monkeypatch.setattr("hermes_constants.Path.home", lambda: tmp_path)
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))

        call_log: list[dict] = []

        def _mock_resolve(**kwargs):
            call_log.append(kwargs)
            n = len(call_log)
            if n == 1:
                raise AuthError("primary quota exhausted")
            if n == 2:
                raise AuthError("openrouter also down")
            return {
                "api_key": "nous-key",
                "base_url": "https://inference-api.nousresearch.com/v1",
                "provider": "nous",
                "requested_provider": "nous",
                "api_mode": "openai_chat",
                "command": None,
                "args": None,
                "credential_pool": None,
            }

        with (
            patch(
                "hermes_cli.runtime_provider.resolve_runtime_provider",
                side_effect=_mock_resolve,
            ),
            patch("run_agent.AIAgent") as mock_agent_cls,
            patch("hermes_cli.oneshot._create_session_db_for_oneshot"),
            patch("hermes_cli.mcp_startup.ensure_mcp_discovery_before_agent_build"),
        ):
            mock_agent = mock_agent_cls.return_value
            mock_agent.run_conversation.return_value = {"final_response": "pong"}

            from hermes_cli.oneshot import _run_agent

            response, _ = _run_agent(prompt="ping", model=None, provider=None)

        # Primary + first fallback + second fallback.
        assert len(call_log) == 3
        assert call_log[2].get("requested") == "nous"
        construct_kwargs = mock_agent_cls.call_args
        assert construct_kwargs.kwargs.get("provider") == "nous"
        assert construct_kwargs.kwargs.get("model") == "hermes-4"
        assert response == "pong"


class TestOneshotFallbackHelper:
    """Direct unit tests for the ``_resolve_with_oneshot_fallback`` helper."""

    def test_primary_success_no_fallback_consulted(self):
        """When the primary resolves, the fallback chain is never touched."""
        from hermes_cli.oneshot import _resolve_with_oneshot_fallback

        chain_accessed = {"n": 0}

        def _get_chain(cfg):
            chain_accessed["n"] += 1
            return []

        def _resolve_fn(**kwargs):
            return {"provider": "primary", "api_key": "k"}

        result = _resolve_with_oneshot_fallback(
            _resolve_fn, _get_chain, {}, requested="primary"
        )
        assert result["provider"] == "primary"
        # Primary success must not inject a model key.
        assert "model" not in result
        assert chain_accessed["n"] == 0

    def test_auth_error_with_empty_chain_reraises(self):
        """Empty fallback chain → AuthError propagates immediately."""
        from hermes_cli.oneshot import _resolve_with_oneshot_fallback

        def _get_chain(cfg):
            return []

        def _resolve_fn(**kwargs):
            raise AuthError("nope")

        with pytest.raises(AuthError, match="nope"):
            _resolve_with_oneshot_fallback(_resolve_fn, _get_chain, {})

    def test_non_auth_error_propagates_without_chain(self):
        """A RuntimeError never reaches the fallback chain."""
        from hermes_cli.oneshot import _resolve_with_oneshot_fallback

        chain_accessed = {"n": 0}

        def _get_chain(cfg):
            chain_accessed["n"] += 1
            return [{"provider": "openrouter", "model": "m"}]

        def _resolve_fn(**kwargs):
            raise RuntimeError("config broken")

        with pytest.raises(RuntimeError, match="config broken"):
            _resolve_with_oneshot_fallback(_resolve_fn, _get_chain, {})

        assert chain_accessed["n"] == 0
