"""Tests for persist-by-default model switching.

Covers:
- ``parse_model_flags`` recognises ``--session`` (and keeps ``--global``).
- ``resolve_persist_behavior`` applies the config-gated default and the
  ``--session`` / ``--global`` overrides.
- The default (no flags) persists, which is the user-facing fix: a plain
  ``/model <name>`` survives across sessions.
"""

from unittest.mock import patch

from hermes_cli.model_switch import (
    ModelSwitchResult,
    apply_model_switch_to_config,
    parse_model_flags,
    resolve_persist_behavior,
)


# ---------------------------------------------------------------------------
# parse_model_flags
# ---------------------------------------------------------------------------


class TestParseModelFlagsSession:
    def test_no_flags(self):
        assert parse_model_flags("sonnet") == ("sonnet", "", False, False, False)

    def test_global_flag(self):
        assert parse_model_flags("sonnet --global") == ("sonnet", "", True, False, False)

    def test_session_flag(self):
        assert parse_model_flags("sonnet --session") == (
            "sonnet",
            "",
            False,
            False,
            True,
        )

    def test_session_with_provider(self):
        assert parse_model_flags("sonnet --provider anthropic --session") == (
            "sonnet",
            "anthropic",
            False,
            False,
            True,
        )

    def test_refresh_flag_still_parsed(self):
        assert parse_model_flags("--refresh") == ("", "", False, True, False)

    def test_unicode_dash_session_normalized(self):
        # Telegram/iOS auto-converts -- to en/em dashes.
        assert parse_model_flags("sonnet \u2013session") == (
            "sonnet",
            "",
            False,
            False,
            True,
        )


# ---------------------------------------------------------------------------
# resolve_persist_behavior
# ---------------------------------------------------------------------------


class TestResolvePersistBehavior:
    def test_session_flag_always_session_only(self):
        # --session opts out even if the config default is True.
        with _config({"model": {"persist_switch_by_default": True}}):
            assert resolve_persist_behavior(False, True) is False

    def test_global_flag_always_persists(self):
        # --global forces persist even if the config default is False.
        with _config({"model": {"persist_switch_by_default": False}}):
            assert resolve_persist_behavior(True, False) is True

    def test_default_persists_when_config_missing(self):
        # No model section at all → built-in default (True).
        with _config({}):
            assert resolve_persist_behavior(False, False) is True

    def test_default_persists_when_key_true(self):
        with _config({"model": {"persist_switch_by_default": True}}):
            assert resolve_persist_behavior(False, False) is True

    def test_default_session_only_when_key_false(self):
        with _config({"model": {"persist_switch_by_default": False}}):
            assert resolve_persist_behavior(False, False) is False

    def test_default_when_model_is_flat_string(self):
        # Fresh install: ``model: ""`` (not a dict) → built-in default True.
        with _config({"model": ""}):
            assert resolve_persist_behavior(False, False) is True

    def test_session_overrides_global_when_both_set(self):
        # --session is the explicit opt-out and wins over --global.
        with _config({"model": {"persist_switch_by_default": True}}):
            assert resolve_persist_behavior(True, True) is False


# ---------------------------------------------------------------------------
# persist_model_switch_config
# ---------------------------------------------------------------------------


class TestPersistModelSwitchConfig:
    def test_provider_change_clears_stale_endpoint_and_context_fields(self):
        """Persisting a provider switch must not leave old endpoint fields
        attached to the new provider (NVIDIA regression vector)."""
        cfg = {
            "model": {
                "default": "old-model",
                "provider": "custom",
                "base_url": "https://previous-provider.example/v1",
                "api_key": "old-inline-key",
                "api_mode": "anthropic_messages",
                "context_length": 12345,
                "extra_body": {"old": True},
            }
        }
        result = ModelSwitchResult(
            success=True,
            new_model="nvidia/llama-3.3-nemotron-super-49b-v1.5",
            target_provider="nvidia",
            provider_changed=True,
            base_url="https://integrate.api.nvidia.com/v1",
            api_key="nv-key",
            api_mode="chat_completions",
        )

        model_cfg = apply_model_switch_to_config(cfg, result)["model"]
        assert model_cfg["default"] == result.new_model
        assert model_cfg["provider"] == "nvidia"
        assert "base_url" not in model_cfg
        assert "api_key" not in model_cfg
        assert "api" not in model_cfg
        assert "api_mode" not in model_cfg
        assert "context_length" not in model_cfg
        assert "extra_body" not in model_cfg

    def test_builtin_provider_switch_clears_stale_endpoint_fields_even_same_provider(self):
        """A dirty built-in-provider config must be cleaned on the next global switch."""
        cfg = {
            "model": {
                "default": "old-nvidia-model",
                "provider": "nvidia",
                "base_url": "https://previous-provider.example/v1",
                "api_key": "old-inline-key",
                "api": "old-alias-key",
                "api_mode": "anthropic_messages",
            }
        }
        result = ModelSwitchResult(
            success=True,
            new_model="nvidia/new-model",
            target_provider="nvidia",
            provider_changed=False,
            base_url="https://integrate.api.nvidia.com/v1",
            api_key="nv-key",
            api_mode="chat_completions",
        )

        model_cfg = apply_model_switch_to_config(cfg, result)["model"]
        assert model_cfg["default"] == "nvidia/new-model"
        assert model_cfg["provider"] == "nvidia"
        assert "base_url" not in model_cfg
        assert "api_key" not in model_cfg
        assert "api" not in model_cfg
        assert "api_mode" not in model_cfg

    def test_custom_provider_switch_persists_endpoint_fields(self):
        """Custom endpoint routing still needs model.base_url persisted."""
        result = ModelSwitchResult(
            success=True,
            new_model="local-model",
            target_provider="custom:lab",
            provider_changed=True,
            base_url="http://localhost:11434/v1",
            api_mode="chat_completions",
        )

        model_cfg = apply_model_switch_to_config({"model": {}}, result)["model"]
        assert model_cfg["provider"] == "custom:lab"
        assert model_cfg["default"] == "local-model"
        assert model_cfg["base_url"] == "http://localhost:11434/v1"
        assert model_cfg["api_mode"] == "chat_completions"

    def test_cli_wrapper_persists_shared_helper_result(self):
        import cli as cli_mod

        saved = []
        result = ModelSwitchResult(
            success=True,
            new_model="local-model",
            target_provider="custom:lab",
            provider_changed=True,
            base_url="http://localhost:11434/v1",
            api_mode="chat_completions",
        )

        with patch("hermes_cli.config.load_config", return_value={"model": {}}), patch(
            "hermes_cli.config.save_config", side_effect=lambda c: saved.append(c)
        ):
            assert cli_mod.persist_model_switch_config(result) is True

        assert saved[-1]["model"]["provider"] == "custom:lab"


# ---------------------------------------------------------------------------
# helper
# ---------------------------------------------------------------------------


class _config:
    """Context manager that patches ``load_config`` to return a fixed dict."""

    def __init__(self, cfg: dict):
        self.cfg = cfg

    def __enter__(self):
        self._patch = patch(
            "hermes_cli.config.load_config",
            return_value=self.cfg,
        )
        # resolve_persist_behavior imports load_config lazily inside the
        # function, so patching the source module is sufficient.
        self._patch.start()
        return self

    def __exit__(self, *exc):
        self._patch.stop()
