"""Tests for ``get_max_tokens_from_config`` — model.max_tokens override.

Custom OpenAI-compatible providers that do not advertise a ``max_tokens``
default cause responses to truncate with ``finish_reason='length'``.  The
helper resolves an override from config.yaml, mirroring the existing
``get_custom_provider_context_length`` lookup pattern.
"""
from __future__ import annotations

from hermes_cli.config import get_max_tokens_from_config


class TestTopLevelModelMaxTokens:
    """Top-level ``model.max_tokens`` is the preferred location."""

    def test_returns_value_when_present(self) -> None:
        cfg = {"model": {"max_tokens": 32768}}
        assert (
            get_max_tokens_from_config(
                "qwen3.6-27b-fp8", "http://vllm.invalid/v1", cfg
            )
            == 32768
        )

    def test_returns_value_when_no_model_or_base_url(self) -> None:
        # Top-level lookup does NOT depend on model/base_url.
        cfg = {"model": {"max_tokens": 16000}}
        assert get_max_tokens_from_config("", "", cfg) == 16000

    def test_string_int_is_coerced(self) -> None:
        cfg = {"model": {"max_tokens": "8192"}}
        assert (
            get_max_tokens_from_config("m", "http://h.invalid/v1", cfg)
            == 8192
        )

    def test_invalid_value_falls_through(self) -> None:
        # Bad value at top level must not raise; lookup continues to
        # custom_providers and finds the per-model override.
        cfg = {
            "model": {"max_tokens": "32K"},
            "custom_providers": [
                {
                    "name": "vllm",
                    "base_url": "http://h.invalid/v1",
                    "models": {"m": {"max_tokens": 4096}},
                }
            ],
        }
        assert (
            get_max_tokens_from_config("m", "http://h.invalid/v1", cfg) == 4096
        )

    def test_zero_or_negative_top_level_ignored(self) -> None:
        cfg = {"model": {"max_tokens": 0}}
        assert get_max_tokens_from_config("m", "http://h.invalid/v1", cfg) is None
        cfg = {"model": {"max_tokens": -1}}
        assert get_max_tokens_from_config("m", "http://h.invalid/v1", cfg) is None


class TestPerModelCustomProviderMaxTokens:
    """``custom_providers.<>.models.<>.max_tokens`` is the per-model fallback."""

    def test_returns_per_model_when_no_top_level(self) -> None:
        cfg = {
            "custom_providers": [
                {
                    "name": "vllm",
                    "base_url": "http://h.invalid/v1",
                    "models": {"qwen": {"max_tokens": 16384}},
                }
            ]
        }
        assert (
            get_max_tokens_from_config("qwen", "http://h.invalid/v1", cfg)
            == 16384
        )

    def test_top_level_wins_over_per_model(self) -> None:
        cfg = {
            "model": {"max_tokens": 99999},
            "custom_providers": [
                {
                    "name": "vllm",
                    "base_url": "http://h.invalid/v1",
                    "models": {"qwen": {"max_tokens": 16384}},
                }
            ],
        }
        assert (
            get_max_tokens_from_config("qwen", "http://h.invalid/v1", cfg)
            == 99999
        )

    def test_trailing_slash_insensitive(self) -> None:
        cfg = {
            "custom_providers": [
                {
                    "name": "vllm",
                    "base_url": "http://h.invalid/v1/",
                    "models": {"m": {"max_tokens": 4096}},
                }
            ]
        }
        assert (
            get_max_tokens_from_config("m", "http://h.invalid/v1", cfg) == 4096
        )

    def test_url_mismatch_returns_none(self) -> None:
        cfg = {
            "custom_providers": [
                {
                    "name": "vllm",
                    "base_url": "http://other.invalid/v1",
                    "models": {"m": {"max_tokens": 4096}},
                }
            ]
        }
        assert (
            get_max_tokens_from_config("m", "http://h.invalid/v1", cfg)
            is None
        )

    def test_model_mismatch_returns_none(self) -> None:
        cfg = {
            "custom_providers": [
                {
                    "name": "vllm",
                    "base_url": "http://h.invalid/v1",
                    "models": {"other-model": {"max_tokens": 4096}},
                }
            ]
        }
        assert (
            get_max_tokens_from_config("m", "http://h.invalid/v1", cfg)
            is None
        )


class TestProvidersDictForm:
    """The newer keyed ``providers:`` schema (v12+) is also covered via
    ``get_compatible_custom_providers`` normalization."""

    def test_providers_dict_with_models_subdict(self) -> None:
        cfg = {
            "providers": {
                "vllm": {
                    "name": "vllm",
                    "api": "http://h.invalid/v1",
                    "models": {"qwen": {"max_tokens": 8192}},
                }
            }
        }
        assert (
            get_max_tokens_from_config("qwen", "http://h.invalid/v1", cfg)
            == 8192
        )


class TestEdgeCases:
    def test_empty_config_returns_none(self) -> None:
        assert (
            get_max_tokens_from_config("m", "http://h.invalid/v1", {}) is None
        )

    def test_none_config_arg_does_not_crash(self) -> None:
        # Passing ``config=None`` triggers a load_config(); the call should
        # not raise even when no override is configured.
        result = get_max_tokens_from_config("nonexistent", "http://nowhere.invalid/v1")
        assert result is None or isinstance(result, int)

    def test_non_dict_config_returns_none(self) -> None:
        assert get_max_tokens_from_config("m", "http://h.invalid/v1", "garbage") is None  # type: ignore[arg-type]

    def test_no_model_section(self) -> None:
        cfg = {"other": {}}
        assert (
            get_max_tokens_from_config("m", "http://h.invalid/v1", cfg)
            is None
        )
