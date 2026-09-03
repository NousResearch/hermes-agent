"""Test that AuthError triggers fallback provider resolution (#7230)."""

from unittest.mock import patch

import pytest


class TestResolveRuntimeAgentKwargsAuthFallback:
    """_resolve_runtime_agent_kwargs should try fallback on AuthError."""

    def test_auth_error_tries_fallback(self, tmp_path, monkeypatch):
        """When primary provider raises AuthError, fallback is attempted."""
        from hermes_cli.auth import AuthError

        # Create a config with fallback
        config_path = tmp_path / "config.yaml"
        config_path.write_text(
            "model:\n  provider: openai-codex\n"
            "fallback_model:\n  provider: openrouter\n"
            "  model: meta-llama/llama-4-maverick\n"
        )

        monkeypatch.setattr("gateway.run._hermes_home", tmp_path)

        call_count = {"n": 0}

        def _mock_resolve(**kwargs):
            call_count["n"] += 1
            # First call = primary path (gateway reads model.provider from
            # config.yaml internally; we simulate the auth failure here).
            # Second call = fallback path with explicit_api_key + explicit_base_url
            # supplied by gateway from fallback_model config.
            if call_count["n"] == 1:
                raise AuthError("Codex token refresh failed with status 401")
            return {
                "api_key": "fallback-key",
                "base_url": "https://openrouter.ai/api/v1",
                "provider": "openrouter",
                "api_mode": "openai_chat",
                "command": None,
                "args": None,
                "credential_pool": None,
            }

        with patch(
            "hermes_cli.runtime_provider.resolve_runtime_provider",
            side_effect=_mock_resolve,
        ):
            from gateway.run import _resolve_runtime_agent_kwargs
            result = _resolve_runtime_agent_kwargs()

        assert result["provider"] == "openrouter"
        assert result["api_key"] == "fallback-key"
        # Should have been called at least twice (primary + fallback)
        assert call_count["n"] >= 2




class TestResolveRuntimeAgentKwargsCooldownHandover:
    """A primary whose pooled credentials are all benched by a 429 hands over.

    Resolution succeeds (the key is still returned), but it carries
    ``CREDENTIALS_COOLING_DOWN_KEY``. Sending that request would just 429, so
    the gateway routes to the fallback chain first — and keeps the primary when
    the chain has nothing usable, because a cooldown demotes a provider rather
    than disqualifying it.
    """

    @staticmethod
    def _write_config(tmp_path, monkeypatch, *, with_fallback: bool):
        config = "model:\n  provider: gemini\n"
        if with_fallback:
            config += (
                "fallback_model:\n  provider: openrouter\n"
                "  model: meta-llama/llama-4-maverick\n"
            )
        (tmp_path / "config.yaml").write_text(config)
        monkeypatch.setattr("gateway.run._hermes_home", tmp_path)

    @staticmethod
    def _cooling_primary(calls, *, fallback_result=None):
        from hermes_cli.runtime_provider import CREDENTIALS_COOLING_DOWN_KEY
        import time

        def _mock_resolve(**kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                return {
                    "api_key": "benched-key",
                    "base_url": "https://generativelanguage.googleapis.com/v1beta",
                    "provider": "gemini",
                    "api_mode": "chat_completions",
                    "command": None,
                    "args": None,
                    "credential_pool": None,
                    CREDENTIALS_COOLING_DOWN_KEY: time.time() + 1800,
                }
            if fallback_result is None:
                raise RuntimeError("fallback provider unavailable")
            return fallback_result

        return _mock_resolve

    def test_cooldown_routes_to_fallback(self, tmp_path, monkeypatch):
        self._write_config(tmp_path, monkeypatch, with_fallback=True)
        calls = {"n": 0}
        mock = self._cooling_primary(calls, fallback_result={
            "api_key": "fallback-key",
            "base_url": "https://openrouter.ai/api/v1",
            "provider": "openrouter",
            "api_mode": "openai_chat",
            "command": None,
            "args": None,
            "credential_pool": None,
        })

        with patch(
            "hermes_cli.runtime_provider.resolve_runtime_provider",
            side_effect=mock,
        ):
            from gateway.run import _resolve_runtime_agent_kwargs
            result = _resolve_runtime_agent_kwargs()

        assert result["provider"] == "openrouter"
        assert result["api_key"] == "fallback-key"
        assert calls["n"] >= 2

    def test_cooldown_keeps_primary_when_chain_is_unusable(self, tmp_path, monkeypatch):
        """Last resort: an unusable chain must not strand the turn."""
        self._write_config(tmp_path, monkeypatch, with_fallback=True)
        calls = {"n": 0}

        with patch(
            "hermes_cli.runtime_provider.resolve_runtime_provider",
            side_effect=self._cooling_primary(calls),
        ):
            from gateway.run import _resolve_runtime_agent_kwargs
            result = _resolve_runtime_agent_kwargs()

        assert result["provider"] == "gemini"
        assert result["api_key"] == "benched-key"

    def test_cooling_fallback_is_skipped_for_a_healthy_one(self, tmp_path, monkeypatch):
        """A benched chain entry must not be picked over a healthy later one.

        Otherwise the handover just moves the doomed request one hop down.
        """
        from hermes_cli.runtime_provider import CREDENTIALS_COOLING_DOWN_KEY
        import time

        (tmp_path / "config.yaml").write_text(
            "model:\n  provider: gemini\n"
            "fallback_providers:\n"
            "  - provider: zai\n    model: glm-5.2\n"
            "  - provider: openrouter\n    model: meta-llama/llama-4-maverick\n"
        )
        monkeypatch.setattr("gateway.run._hermes_home", tmp_path)

        seen = []

        def _mock_resolve(**kwargs):
            requested = kwargs.get("requested")
            seen.append(requested)
            base = {
                "api_mode": "openai_chat",
                "command": None,
                "args": None,
                "credential_pool": None,
            }
            if requested is None:
                return {
                    **base, "api_key": "benched-primary",
                    "base_url": "https://generativelanguage.googleapis.com/v1beta",
                    "provider": "gemini",
                    CREDENTIALS_COOLING_DOWN_KEY: time.time() + 1800,
                }
            if requested == "zai":
                return {
                    **base, "api_key": "benched-zai",
                    "base_url": "https://api.z.ai/v1", "provider": "zai",
                    CREDENTIALS_COOLING_DOWN_KEY: time.time() + 1800,
                }
            return {
                **base, "api_key": "healthy-key",
                "base_url": "https://openrouter.ai/api/v1",
                "provider": "openrouter",
            }

        with patch(
            "hermes_cli.runtime_provider.resolve_runtime_provider",
            side_effect=_mock_resolve,
        ):
            from gateway.run import _resolve_runtime_agent_kwargs
            result = _resolve_runtime_agent_kwargs()

        assert result["provider"] == "openrouter"
        assert result["api_key"] == "healthy-key"
        assert result["model"] == "meta-llama/llama-4-maverick"
        assert "zai" in seen  # the benched entry was tried and passed over

    def test_a_fallback_reported_rate_limited_is_passed_over(
        self, tmp_path, monkeypatch
    ):
        """Wiring: whatever the probe reports rate-limited is skipped.

        The entry here pins its own base_url, so it resolves through
        `_resolve_explicit_runtime` and comes back WITHOUT the annotation even
        though its key is drawn from a benched pool — the case the direct pool
        probe exists for. The probe itself is covered by
        `TestFallbackRuntimeIsRateLimited` below; this asserts the chain walk
        acts on its verdict.
        """
        import time

        (tmp_path / "config.yaml").write_text(
            "model:\n  provider: gemini\n"
            "fallback_providers:\n"
            "  - provider: zai\n    model: glm-5.2\n"
            "    base_url: https://api.z.ai/v1\n"
            "  - provider: openrouter\n    model: meta-llama/llama-4-maverick\n"
        )
        monkeypatch.setattr("gateway.run._hermes_home", tmp_path)

        from hermes_cli.runtime_provider import CREDENTIALS_COOLING_DOWN_KEY

        base = {
            "api_mode": "openai_chat",
            "command": None,
            "args": None,
            "credential_pool": None,
        }

        def _mock_resolve(**kwargs):
            requested = kwargs.get("requested")
            if requested is None:
                return {
                    **base, "api_key": "benched-primary", "provider": "gemini",
                    "base_url": "https://generativelanguage.googleapis.com/v1beta",
                    CREDENTIALS_COOLING_DOWN_KEY: time.time() + 1800,
                }
            if requested == "zai":
                # No annotation — the explicit base_url path skipped it.
                return {
                    **base, "api_key": "benched-zai", "provider": "zai",
                    "base_url": "https://api.z.ai/v1",
                }
            return {
                **base, "api_key": "healthy-key", "provider": "openrouter",
                "base_url": "https://openrouter.ai/api/v1",
            }

        with patch(
            "hermes_cli.runtime_provider.resolve_runtime_provider",
            side_effect=_mock_resolve,
        ), patch(
            "gateway.run._fallback_runtime_is_rate_limited",
            side_effect=lambda rt: rt.get("provider") == "zai",
        ):
            from gateway.run import _resolve_runtime_agent_kwargs
            result = _resolve_runtime_agent_kwargs()

        assert result["provider"] == "openrouter"
        assert result["api_key"] == "healthy-key"

    def test_every_fallback_cooling_still_uses_one(self, tmp_path, monkeypatch):
        """All-cooling chain: a benched fallback still beats a benched primary.

        It is a different quota bucket, so it has a chance the primary does not.
        """
        import time

        (tmp_path / "config.yaml").write_text(
            "model:\n  provider: gemini\n"
            "fallback_providers:\n  - provider: zai\n    model: glm-5.2\n"
        )
        monkeypatch.setattr("gateway.run._hermes_home", tmp_path)

        from hermes_cli.runtime_provider import CREDENTIALS_COOLING_DOWN_KEY

        base = {
            "api_mode": "openai_chat", "command": None, "args": None,
            "credential_pool": None,
        }

        def _mock_resolve(**kwargs):
            if kwargs.get("requested") is None:
                return {
                    **base, "api_key": "benched-primary", "provider": "gemini",
                    "base_url": "https://generativelanguage.googleapis.com/v1beta",
                    CREDENTIALS_COOLING_DOWN_KEY: time.time() + 1800,
                }
            return {
                **base, "api_key": "benched-zai", "provider": "zai",
                "base_url": "https://api.z.ai/v1",
                CREDENTIALS_COOLING_DOWN_KEY: time.time() + 1800,
            }

        with patch(
            "hermes_cli.runtime_provider.resolve_runtime_provider",
            side_effect=_mock_resolve,
        ):
            from gateway.run import _resolve_runtime_agent_kwargs
            result = _resolve_runtime_agent_kwargs()

        assert result["provider"] == "zai"
        assert result["model"] == "glm-5.2"


class TestFallbackRuntimeIsRateLimited:
    """The probe that covers fallbacks resolved outside the annotated path."""

    @staticmethod
    def _pool(tmp_path, monkeypatch, entries):
        import json

        home = tmp_path / "hermes"
        home.mkdir(parents=True, exist_ok=True)
        monkeypatch.setenv("HERMES_HOME", str(home))
        for var in ("ZAI_API_KEY", "Z_AI_API_KEY"):
            monkeypatch.delenv(var, raising=False)
        (home / "auth.json").write_text(
            json.dumps({"version": 1, "credential_pool": {"zai": entries}}),
            encoding="utf-8",
        )

    @staticmethod
    def _entry(cred_id, priority, code):
        import time

        return {
            "id": cred_id,
            "label": cred_id,
            "auth_type": "api_key",
            "priority": priority,
            "source": "manual",
            "access_token": f"zai-key-{priority}",
            "base_url": "https://api.z.ai/v1",
            "last_status": "exhausted",
            "last_status_at": time.time() - 60,
            "last_error_code": code,
        }

    def test_benched_pool_behind_an_unannotated_runtime(self, tmp_path, monkeypatch):
        self._pool(
            tmp_path, monkeypatch,
            [self._entry("a", 0, 429), self._entry("b", 1, 429)],
        )
        from gateway.run import _fallback_runtime_is_rate_limited

        # No annotation — this is what _resolve_explicit_runtime returns.
        # source is "explicit", exactly what _resolve_explicit_runtime stamps,
        # so this can only pass through the key-identity match — not the
        # credential_pool: source prefix.
        assert _fallback_runtime_is_rate_limited({
            "provider": "zai",
            "api_key": "zai-key-0",
            "source": "explicit",
        }) is True

    def test_a_foreign_key_is_not_attributed_to_the_pool(self, tmp_path, monkeypatch):
        """A key that is not one of the pool's own must never be flagged."""
        self._pool(
            tmp_path, monkeypatch,
            [self._entry("a", 0, 429), self._entry("b", 1, 429)],
        )
        from gateway.run import _fallback_runtime_is_rate_limited

        assert _fallback_runtime_is_rate_limited({
            "provider": "zai",
            "api_key": "some-other-key",
            "source": "explicit",
        }) is False

    def test_revoked_pool_is_not_reported_as_rate_limited(self, tmp_path, monkeypatch):
        self._pool(
            tmp_path, monkeypatch,
            [self._entry("a", 0, 401), self._entry("b", 1, 401)],
        )
        from gateway.run import _fallback_runtime_is_rate_limited

        assert _fallback_runtime_is_rate_limited({
            "provider": "zai",
            "api_key": "zai-key-0",
            "source": "explicit",
        }) is False
