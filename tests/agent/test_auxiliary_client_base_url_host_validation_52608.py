"""Regression tests for issue #52608.

auxiliary_client `_try_anthropic()` must NOT apply `cfg["model"]["base_url"]`
when the configured base_url host is not an Anthropic-compatible endpoint
(e.g. OpenRouter, OpenAI). Operators routing main traffic through a
non-Anthropic provider's endpoint while keeping `provider: anthropic` would
otherwise have every side-channel call (memory extractors, reflection,
vision, title generation) 401 from the foreign host.
"""
from unittest.mock import MagicMock, patch


def _extract_base_url_passed_to_build(mock_build):
    """Pull the base_url that `_try_anthropic()` actually handed to build_anthropic_client."""
    args, _kwargs = mock_build.call_args
    # build_anthropic_client(token, base_url) per agent/auxiliary_client.py line 2180
    assert len(args) >= 2, f"expected (token, base_url), got args={args}"
    return args[1]


class TestTryAnthropicBaseUrlHostValidation:
    """Issue #52608: side-channel calls must not be sent to a non-Anthropic host."""

    def test_openrouter_base_url_does_not_leak_into_auxiliary(self, tmp_path, monkeypatch):
        """cfg.model.base_url=https://openrouter.ai/api/v1 must NOT override aux base_url."""
        import yaml
        from agent.auxiliary_client import _try_anthropic
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        (tmp_path / "config.yaml").write_text(yaml.safe_dump({
            "model": {
                "provider": "anthropic",
                "model": "claude-haiku-4-5-20251001",
                "base_url": "https://openrouter.ai/api/v1",
            }
        }))

        with (
            patch(
                "agent.auxiliary_client._select_pool_entry", return_value=(False, None)
            ),
            patch(
                "agent.anthropic_adapter.resolve_anthropic_token",
                return_value="***",
            ),
            patch(
                "agent.anthropic_adapter.build_anthropic_client"
            ) as mock_build,
        ):
            mock_build.return_value = MagicMock()
            client, _model = _try_anthropic()

        assert client is not None, "auxiliary client must still be created"
        actual = _extract_base_url_passed_to_build(mock_build)
        assert actual == "https://api.anthropic.com", (
            f"Auxiliary client must use the Anthropic default base_url, "
            f"not the operator's main-session override. Got: {actual!r}"
        )

    def test_anthropic_default_host_is_preserved(self, tmp_path, monkeypatch):
        """The common case (operator sets model.base_url to api.anthropic.com) must still apply."""
        import yaml
        from agent.auxiliary_client import _try_anthropic
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        (tmp_path / "config.yaml").write_text(yaml.safe_dump({
            "model": {
                "provider": "anthropic",
                "model": "claude-haiku-4-5-20251001",
                "base_url": "https://api.anthropic.com",
            }
        }))

        with (
            patch(
                "agent.auxiliary_client._select_pool_entry", return_value=(False, None)
            ),
            patch(
                "agent.anthropic_adapter.resolve_anthropic_token",
                return_value="***",
            ),
            patch(
                "agent.anthropic_adapter.build_anthropic_client"
            ) as mock_build,
        ):
            mock_build.return_value = MagicMock()
            client, _model = _try_anthropic()

        assert client is not None
        actual = _extract_base_url_passed_to_build(mock_build)
        assert actual == "https://api.anthropic.com"

    def test_openai_base_url_does_not_leak(self, tmp_path, monkeypatch):
        """Generic non-Anthropic host must not be applied as auxiliary base_url."""
        import yaml
        from agent.auxiliary_client import _try_anthropic
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        (tmp_path / "config.yaml").write_text(yaml.safe_dump({
            "model": {
                "provider": "anthropic",
                "model": "claude-haiku-4-5-20251001",
                "base_url": "https://api.openai.com/v1",
            }
        }))

        with (
            patch(
                "agent.auxiliary_client._select_pool_entry", return_value=(False, None)
            ),
            patch(
                "agent.anthropic_adapter.resolve_anthropic_token",
                return_value="***",
            ),
            patch(
                "agent.anthropic_adapter.build_anthropic_client"
            ) as mock_build,
        ):
            mock_build.return_value = MagicMock()
            client, _model = _try_anthropic()

        assert client is not None
        actual = _extract_base_url_passed_to_build(mock_build)
        assert actual == "https://api.anthropic.com", (
            f"Non-Anthropic host must not be applied. Got: {actual!r}"
        )

    def test_empty_base_url_falls_back_to_default(self, tmp_path, monkeypatch):
        """Empty model.base_url must not crash and must fall back to default."""
        import yaml
        from agent.auxiliary_client import _try_anthropic
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        (tmp_path / "config.yaml").write_text(yaml.safe_dump({
            "model": {
                "provider": "anthropic",
                "model": "claude-haiku-4-5-20251001",
                "base_url": "",
            }
        }))

        with (
            patch(
                "agent.auxiliary_client._select_pool_entry", return_value=(False, None)
            ),
            patch(
                "agent.anthropic_adapter.resolve_anthropic_token",
                return_value="***",
            ),
            patch(
                "agent.anthropic_adapter.build_anthropic_client"
            ) as mock_build,
        ):
            mock_build.return_value = MagicMock()
            client, _model = _try_anthropic()

        assert client is not None
        actual = _extract_base_url_passed_to_build(mock_build)
        assert actual == "https://api.anthropic.com"

    def test_anthropic_host_with_path_is_preserved(self, tmp_path, monkeypatch):
        """api.anthropic.com with a path suffix must still pass the host check."""
        import yaml
        from agent.auxiliary_client import _try_anthropic
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        (tmp_path / "config.yaml").write_text(yaml.safe_dump({
            "model": {
                "provider": "anthropic",
                "model": "claude-haiku-4-5-20251001",
                "base_url": "https://api.anthropic.com/v1/messages",
            }
        }))

        with (
            patch(
                "agent.auxiliary_client._select_pool_entry", return_value=(False, None)
            ),
            patch(
                "agent.anthropic_adapter.resolve_anthropic_token",
                return_value="***",
            ),
            patch(
                "agent.anthropic_adapter.build_anthropic_client"
            ) as mock_build,
        ):
            mock_build.return_value = MagicMock()
            client, _model = _try_anthropic()

        assert client is not None
        actual = _extract_base_url_passed_to_build(mock_build)
        assert actual == "https://api.anthropic.com/v1/messages", (
            f"Anthropic host with path must be preserved. Got: {actual!r}"
        )
