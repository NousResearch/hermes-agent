"""Tests that `hermes model` always shows the model selection menu for custom
providers, even when a model is already saved.

Regression test for the bug where _model_flow_named_custom() returned
immediately when provider_info had a saved ``model`` field, making it
impossible to switch models on multi-model endpoints.
"""

import os
from unittest.mock import patch, MagicMock, call

import pytest


@pytest.fixture
def config_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME with a minimal config."""
    home = tmp_path / "hermes"
    home.mkdir()
    config_yaml = home / "config.yaml"
    config_yaml.write_text("model: old-model\ncustom_providers: []\n")
    env_file = home / ".env"
    env_file.write_text("")
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("HERMES_MODEL", raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False)
    monkeypatch.delenv("HERMES_INFERENCE_PROVIDER", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    return home


class TestCustomProviderModelSwitch:
    """Ensure _model_flow_named_custom always probes and shows menu."""

    def test_saved_model_still_probes_endpoint(self, config_home):
        """When a model is already saved, the function must still call
        fetch_api_models to probe the endpoint — not skip with early return."""
        from hermes_cli.main import _model_flow_named_custom

        provider_info = {
            "name": "My vLLM",
            "base_url": "https://vllm.example.com/v1",
            "api_key": "sk-test",
            "model": "model-A",  # already saved
        }

        with patch("hermes_cli.models.fetch_api_models", return_value=["model-A", "model-B"]) as mock_fetch, \
             patch.dict("sys.modules", {"simple_term_menu": None}), \
             patch("builtins.input", return_value="2"), \
             patch("builtins.print"):
            _model_flow_named_custom({}, provider_info)

        # fetch_api_models MUST be called even though model was saved
        mock_fetch.assert_called_once_with(
            "sk-test",
            "https://vllm.example.com/v1",
            timeout=8.0,
            api_mode=None,
        )

    def test_can_switch_to_different_model(self, config_home):
        """User selects a different model than the saved one."""
        import yaml
        from hermes_cli.main import _model_flow_named_custom

        provider_info = {
            "name": "My vLLM",
            "base_url": "https://vllm.example.com/v1",
            "api_key": "sk-test",
            "model": "model-A",
        }

        with patch("hermes_cli.models.fetch_api_models", return_value=["model-A", "model-B"]), \
             patch.dict("sys.modules", {"simple_term_menu": None}), \
             patch("builtins.input", return_value="2"), \
             patch("builtins.print"):
            _model_flow_named_custom({}, provider_info)

        config = yaml.safe_load((config_home / "config.yaml").read_text()) or {}
        model = config.get("model")
        assert isinstance(model, dict)
        assert model["default"] == "model-B"

    def test_probe_failure_falls_back_to_saved(self, config_home):
        """When endpoint probe fails and user presses Enter, saved model is used."""
        import yaml
        from hermes_cli.main import _model_flow_named_custom

        provider_info = {
            "name": "My vLLM",
            "base_url": "https://vllm.example.com/v1",
            "api_key": "sk-test",
            "model": "model-A",
        }

        # fetch returns empty list (probe failed), user presses Enter (empty input)
        with patch("hermes_cli.models.fetch_api_models", return_value=[]), \
             patch("builtins.input", return_value=""), \
             patch("builtins.print"):
            _model_flow_named_custom({}, provider_info)

        config = yaml.safe_load((config_home / "config.yaml").read_text()) or {}
        model = config.get("model")
        assert isinstance(model, dict)
        assert model["default"] == "model-A"

    def test_no_saved_model_still_works(self, config_home):
        """First-time flow (no saved model) still works as before."""
        import yaml
        from hermes_cli.main import _model_flow_named_custom

        provider_info = {
            "name": "My vLLM",
            "base_url": "https://vllm.example.com/v1",
            "api_key": "sk-test",
            # no "model" key
        }

        with patch("hermes_cli.models.fetch_api_models", return_value=["model-X"]), \
             patch.dict("sys.modules", {"simple_term_menu": None}), \
             patch("builtins.input", return_value="1"), \
             patch("builtins.print"):
            _model_flow_named_custom({}, provider_info)

        config = yaml.safe_load((config_home / "config.yaml").read_text()) or {}
        model = config.get("model")
        assert isinstance(model, dict)
        assert model["default"] == "model-X"

    def test_api_mode_set_from_provider_info(self, config_home):
        """When custom_providers entry has api_mode, it should be applied."""
        import yaml
        from hermes_cli.main import _model_flow_named_custom

        provider_info = {
            "name": "Anthropic Proxy",
            "base_url": "https://proxy.example.com/anthropic",
            "api_key": "***",
            "model": "claude-3",
            "api_mode": "anthropic_messages",
        }

        with patch("hermes_cli.models.fetch_api_models", return_value=["claude-3"]), \
             patch.dict("sys.modules", {"simple_term_menu": None}), \
             patch("builtins.input", return_value="1"), \
             patch("builtins.print"):
            _model_flow_named_custom({}, provider_info)

        config = yaml.safe_load((config_home / "config.yaml").read_text()) or {}
        model = config.get("model")
        assert isinstance(model, dict)
        assert model.get("api_mode") == "anthropic_messages"

    def test_api_mode_cleared_when_not_specified(self, config_home):
        """When custom_providers entry has no api_mode, stale api_mode is removed."""
        import yaml
        from hermes_cli.main import _model_flow_named_custom

        # Pre-seed a stale api_mode in config
        config_path = config_home / "config.yaml"
        config_path.write_text(yaml.dump({"model": {"api_mode": "anthropic_messages"}}))

        provider_info = {
            "name": "My vLLM",
            "base_url": "https://vllm.example.com/v1",
            "api_key": "***",
            "model": "llama-3",
        }

        with patch("hermes_cli.models.fetch_api_models", return_value=["llama-3"]), \
             patch.dict("sys.modules", {"simple_term_menu": None}), \
             patch("builtins.input", return_value="1"), \
             patch("builtins.print"):
            _model_flow_named_custom({}, provider_info)

        config = yaml.safe_load((config_home / "config.yaml").read_text()) or {}
        model = config.get("model")
        assert isinstance(model, dict)
        assert "api_mode" not in model, "Stale api_mode should be removed"

    def test_env_template_api_key_is_preserved_in_model_config(self, config_home, monkeypatch):
        """Selecting an env-backed custom provider must not inline the secret."""
        import yaml
        from hermes_cli.main import _model_flow_named_custom

        config_path = config_home / "config.yaml"
        config_path.write_text(
            "model:\n"
            "  default: old-model\n"
            "  provider: openrouter\n"
            "custom_providers:\n"
            "- name: Example Provider\n"
            "  base_url: https://api.example-provider.test/v1\n"
            "  api_key: ${EXAMPLE_PROVIDER_API_KEY}\n"
            "  model: qwen3.6-35b-fast\n"
        )
        monkeypatch.setenv("EXAMPLE_PROVIDER_API_KEY", "sk-live-example-provider")

        provider_info = {
            "name": "Example Provider",
            "base_url": "https://api.example-provider.test/v1",
            "api_key": "sk-live-example-provider",
            "api_key_ref": "${EXAMPLE_PROVIDER_API_KEY}",
            "model": "qwen3.6-35b-fast",
        }

        with patch("hermes_cli.models.fetch_api_models", return_value=["qwen3.6-35b-fast"]) as mock_fetch, \
             patch.dict("sys.modules", {"simple_term_menu": None}), \
             patch("builtins.input", return_value="1"), \
             patch("builtins.print"):
            _model_flow_named_custom({}, provider_info)

        mock_fetch.assert_called_once_with(
            "sk-live-example-provider",
            "https://api.example-provider.test/v1",
            timeout=8.0,
            api_mode=None,
        )
        config = yaml.safe_load(config_path.read_text()) or {}
        assert config["model"]["api_key"] == "${EXAMPLE_PROVIDER_API_KEY}"
        assert config["custom_providers"][0]["api_key"] == "${EXAMPLE_PROVIDER_API_KEY}"
        assert "sk-live-example-provider" not in config_path.read_text()

    def test_key_env_custom_provider_persists_reference_not_secret(self, config_home, monkeypatch):
        """key_env custom providers should also avoid writing plaintext keys."""
        import yaml
        from hermes_cli.main import _model_flow_named_custom

        config_path = config_home / "config.yaml"
        config_path.write_text(
            "model:\n"
            "  default: old-model\n"
            "custom_providers:\n"
            "- name: Example Provider\n"
            "  base_url: https://api.example-provider.test/v1\n"
            "  key_env: EXAMPLE_PROVIDER_API_KEY\n"
            "  model: qwen3.6-35b-fast\n"
        )
        monkeypatch.setenv("EXAMPLE_PROVIDER_API_KEY", "sk-live-example-provider")

        provider_info = {
            "name": "Example Provider",
            "base_url": "https://api.example-provider.test/v1",
            "api_key": "",
            "key_env": "EXAMPLE_PROVIDER_API_KEY",
            "model": "qwen3.6-35b-fast",
        }

        with patch("hermes_cli.models.fetch_api_models", return_value=["qwen3.6-35b-fast"]), \
             patch.dict("sys.modules", {"simple_term_menu": None}), \
             patch("builtins.input", return_value="1"), \
             patch("builtins.print"):
            _model_flow_named_custom({}, provider_info)

        config = yaml.safe_load(config_path.read_text()) or {}
        assert config["model"]["api_key"] == "${EXAMPLE_PROVIDER_API_KEY}"
        assert config["custom_providers"][0]["key_env"] == "EXAMPLE_PROVIDER_API_KEY"
        assert "sk-live-example-provider" not in config_path.read_text()

    def test_env_ref_base_url_preserves_api_key_ref_through_picker(
        self, config_home, monkeypatch
    ):
        """Integration regression: when BOTH ``base_url`` and ``api_key`` use
        ``${VAR}`` templates (the Discord-reported NeuralWatt case), the picker
        must still preserve the env reference in ``model.api_key``.

        The earlier lookup went through ``get_compatible_custom_providers``
        which dropped entries whose ``base_url`` was an env-ref template
        (``urlparse("${NEURALWATT_API_BASE}")`` has no scheme/netloc), causing
        ``api_key_ref`` to stay empty and the resolved secret to be written to
        ``config.yaml``. This test drives the real picker-callsite code path.
        """
        import yaml
        from hermes_cli.main import select_provider_and_model

        config_path = config_home / "config.yaml"
        config_path.write_text(
            "model:\n"
            "  default: old-model\n"
            "  provider: openrouter\n"
            "custom_providers:\n"
            "- name: NeuralWatt\n"
            "  base_url: ${NEURALWATT_API_BASE}\n"
            "  api_key: ${NEURALWATT_API_KEY}\n"
            "  model: qwen3.6-35b-fast\n"
            "  models: []\n"
        )
        monkeypatch.setenv("NEURALWATT_API_BASE", "https://api.neuralwatt.com/v1")
        monkeypatch.setenv("NEURALWATT_API_KEY", "sk-live-neuralwatt-secret")

        # Exercise the real picker: select "custom:neuralwatt" from the
        # provider menu. ``select_provider_and_model`` prompts for a provider
        # choice (returns an index), then hands off to
        # ``_model_flow_named_custom`` with the provider_info built by
        # ``_named_custom_provider_map``.
        def _pick_neuralwatt(labels, default=0):
            for i, label in enumerate(labels):
                if "NeuralWatt" in label:
                    return i
            raise AssertionError(
                f"NeuralWatt entry missing from provider menu: {labels}"
            )

        with patch("hermes_cli.main._prompt_provider_choice",
                   side_effect=_pick_neuralwatt), \
             patch("hermes_cli.models.fetch_api_models",
                   return_value=["qwen3.6-35b-fast"]) as mock_fetch, \
             patch.dict("sys.modules", {"simple_term_menu": None}), \
             patch("builtins.input", return_value="1"), \
             patch("builtins.print"):
            select_provider_and_model()

        # The live probe must still use the resolved secret.
        mock_fetch.assert_called_once()
        probe_args, probe_kwargs = mock_fetch.call_args
        assert probe_args[0] == "sk-live-neuralwatt-secret"

        # But config.yaml must keep the env reference, not the plaintext secret.
        saved = config_path.read_text()
        config = yaml.safe_load(saved) or {}
        assert config["model"]["api_key"] == "${NEURALWATT_API_KEY}"
        assert config["custom_providers"][0]["api_key"] == "${NEURALWATT_API_KEY}"
        assert "sk-live-neuralwatt-secret" not in saved

    def test_key_env_providers_dict_entry_does_not_add_api_key(
        self, config_home, monkeypatch
    ):
        """Regression for #15803: a ``providers:`` (keyed-schema) entry that
        relies on ``key_env`` must not gain an ``api_key`` field after the
        model picker runs.

        Before the fix, ``_model_flow_named_custom`` synthesized
        ``api_key: ${KEY_ENV}`` from the resolved secret and wrote it to the
        ``providers.<key>`` entry, cluttering configs that intentionally keep
        credentials out of ``config.yaml``. The entry already carries
        ``key_env``; the runtime resolves it directly, so no inline
        ``api_key`` belongs on disk.
        """
        import yaml
        from hermes_cli.main import _model_flow_named_custom

        config_path = config_home / "config.yaml"
        config_path.write_text(
            "providers:\n"
            "  crs-henkee:\n"
            "    name: CRS Henkee\n"
            "    base_url: http://127.0.0.1:3000/api/v1\n"
            "    key_env: HERMES_CRS_HENKEE_KEY\n"
            "    transport: anthropic_messages\n"
            "    model: claude-opus-4-7\n"
            "    default_model: claude-opus-4-7\n"
            "custom_providers: []\n"
        )
        monkeypatch.setenv("HERMES_CRS_HENKEE_KEY", "cr_live_secret_xyz")

        # provider_info as built by _named_custom_provider_map for a
        # ``providers:`` entry that has key_env but no inline api_key.
        provider_info = {
            "name": "CRS Henkee",
            "base_url": "http://127.0.0.1:3000/api/v1",
            "api_key": "",
            "key_env": "HERMES_CRS_HENKEE_KEY",
            "model": "claude-opus-4-7",
            "api_mode": "anthropic_messages",
            "provider_key": "crs-henkee",
            "api_key_ref": "",
        }

        with patch(
            "hermes_cli.models.fetch_api_models",
            return_value=["claude-opus-4-7"],
        ) as mock_fetch, \
             patch.dict("sys.modules", {"simple_term_menu": None}), \
             patch("builtins.input", return_value="1"), \
             patch("builtins.print"):
            _model_flow_named_custom({}, provider_info)

        # The /models probe must resolve the secret from the env var.
        mock_fetch.assert_called_once()
        probe_args, _ = mock_fetch.call_args
        assert probe_args[0] == "cr_live_secret_xyz"

        # The providers entry must NOT gain an api_key field — neither the
        # plaintext secret nor a synthesized ${KEY_ENV} template.
        saved_text = config_path.read_text()
        saved = yaml.safe_load(saved_text) or {}
        entry = saved["providers"]["crs-henkee"]
        assert "api_key" not in entry, (
            f"providers.crs-henkee gained an api_key field: {entry.get('api_key')!r}"
        )
        assert entry["key_env"] == "HERMES_CRS_HENKEE_KEY"
        assert entry["default_model"] == "claude-opus-4-7"

        # And the plaintext secret must never appear anywhere on disk.
        assert "cr_live_secret_xyz" not in saved_text
        # The synthesized template is also redundant here — key_env owns it.
        assert "${HERMES_CRS_HENKEE_KEY}" not in saved_text

    def test_key_env_providers_dict_preserves_existing_api_key(
        self, config_home, monkeypatch
    ):
        """A ``providers:`` entry that already has an inline ``api_key``
        template must keep it untouched. Only entries that never declared
        an ``api_key`` should skip the write."""
        import yaml
        from hermes_cli.main import _model_flow_named_custom

        config_path = config_home / "config.yaml"
        config_path.write_text(
            "providers:\n"
            "  crs-henkee:\n"
            "    name: CRS Henkee\n"
            "    base_url: http://127.0.0.1:3000/api/v1\n"
            "    api_key: ${HERMES_CRS_HENKEE_KEY}\n"
            "    key_env: HERMES_CRS_HENKEE_KEY\n"
            "    transport: anthropic_messages\n"
            "    model: claude-opus-4-7\n"
            "    default_model: claude-opus-4-7\n"
            "custom_providers: []\n"
        )
        monkeypatch.setenv("HERMES_CRS_HENKEE_KEY", "cr_live_secret_xyz")

        provider_info = {
            "name": "CRS Henkee",
            "base_url": "http://127.0.0.1:3000/api/v1",
            "api_key": "cr_live_secret_xyz",  # expanded by load_config
            "key_env": "HERMES_CRS_HENKEE_KEY",
            "model": "claude-opus-4-7",
            "api_mode": "anthropic_messages",
            "provider_key": "crs-henkee",
            "api_key_ref": "${HERMES_CRS_HENKEE_KEY}",  # raw template preserved
        }

        with patch(
            "hermes_cli.models.fetch_api_models",
            return_value=["claude-opus-4-7"],
        ), \
             patch.dict("sys.modules", {"simple_term_menu": None}), \
             patch("builtins.input", return_value="1"), \
             patch("builtins.print"):
            _model_flow_named_custom({}, provider_info)

        saved_text = config_path.read_text()
        saved = yaml.safe_load(saved_text) or {}
        entry = saved["providers"]["crs-henkee"]
        # Existing api_key template must survive (the resolved secret must not
        # clobber it via _preserve_env_ref_templates).
        assert entry["api_key"] == "${HERMES_CRS_HENKEE_KEY}"
        assert "cr_live_secret_xyz" not in saved_text
