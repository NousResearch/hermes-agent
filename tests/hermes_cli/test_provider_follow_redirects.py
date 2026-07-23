"""Tests for ``hermes_cli.timeouts.get_provider_follow_redirects``.

Mirrors the contract of ``get_provider_request_timeout``: per-provider
config.yaml lookup, optional per-model override, default False, never
raises. Default-off means behavior is unchanged for every provider that
does not explicitly opt in.
"""
from unittest.mock import patch

import pytest

from hermes_cli.timeouts import get_provider_follow_redirects


def _config(providers):
    return {"providers": providers}


def test_default_false_when_no_config(monkeypatch):
    """No config.yaml loaded → default is False (httpx native default)."""
    with patch(
        "hermes_cli.config.load_config_readonly",
        return_value={},
    ):
        assert get_provider_follow_redirects("openai-codex", "gpt-5.4") is False


def test_default_false_when_provider_missing(monkeypatch):
    with patch(
        "hermes_cli.config.load_config_readonly",
        return_value=_config({"some_other_provider": {"follow_redirects": True}}),
    ):
        assert get_provider_follow_redirects("openai-codex") is False


def test_provider_level_true(monkeypatch):
    with patch(
        "hermes_cli.config.load_config_readonly",
        return_value=_config({"redirect-provider": {"follow_redirects": True}}),
    ):
        assert get_provider_follow_redirects("redirect-provider", "gpt-5.4") is True


def test_provider_level_explicit_false(monkeypatch):
    with patch(
        "hermes_cli.config.load_config_readonly",
        return_value=_config({"redirect-provider": {"follow_redirects": False}}),
    ):
        assert get_provider_follow_redirects("redirect-provider") is False


def test_model_level_overrides_provider_level(monkeypatch):
    """The model-level key wins when both are set."""
    with patch(
        "hermes_cli.config.load_config_readonly",
        return_value=_config(
            {
                "redirect-provider": {
                    "follow_redirects": True,
                    "models": {"gpt-5.4": {"follow_redirects": False}},
                }
            }
        ),
    ):
        # Model override flips provider True → False
        assert get_provider_follow_redirects("redirect-provider", "gpt-5.4") is False
        # No model override → provider value applies
        assert get_provider_follow_redirects("redirect-provider", "other-model") is True


def test_truthy_string_coercion(monkeypatch):
    """YAML booleans parse to Python booleans, but accept clear truthy strings."""
    for truthy in ("true", "True", "TRUE", "yes", "on", "1"):
        with patch(
            "hermes_cli.config.load_config_readonly",
            return_value=_config({"redirect-provider": {"follow_redirects": truthy}}),
        ):
            assert get_provider_follow_redirects("redirect-provider") is True, truthy


def test_falsy_string_coercion(monkeypatch):
    for falsy in ("false", "False", "FALSE", "no", "off", "0", "", "garbage"):
        with patch(
            "hermes_cli.config.load_config_readonly",
            return_value=_config({"redirect-provider": {"follow_redirects": falsy}}),
        ):
            assert get_provider_follow_redirects("redirect-provider") is False, falsy


def test_garbage_config_does_not_raise(monkeypatch):
    """A broken config loader must not block client construction."""
    with patch(
        "hermes_cli.config.load_config_readonly",
        side_effect=RuntimeError("broken yaml"),
    ):
        assert get_provider_follow_redirects("redirect-provider") is False


def test_garbage_provider_entry_does_not_raise(monkeypatch):
    """A non-dict providers entry (e.g. None, list) must not raise."""
    with patch(
        "hermes_cli.config.load_config_readonly",
        return_value={"providers": None},
    ):
        assert get_provider_follow_redirects("redirect-provider") is False

    with patch(
        "hermes_cli.config.load_config_readonly",
        return_value={"providers": ["not", "a", "dict"]},
    ):
        assert get_provider_follow_redirects("redirect-provider") is False


def test_empty_provider_id_returns_false(monkeypatch):
    with patch(
        "hermes_cli.config.load_config_readonly",
        return_value=_config({"redirect-provider": {"follow_redirects": True}}),
    ):
        assert get_provider_follow_redirects("") is False
        assert get_provider_follow_redirects(None) is False


@pytest.mark.parametrize("model_value", [None, "", "  "])
def test_model_override_skipped_when_model_empty(monkeypatch, model_value):
    """Empty model names must not be used as dict keys (would crash on None)."""
    with patch(
        "hermes_cli.config.load_config_readonly",
        return_value=_config({"redirect-provider": {"follow_redirects": True}}),
    ):
        assert get_provider_follow_redirects("redirect-provider", model_value) is True
