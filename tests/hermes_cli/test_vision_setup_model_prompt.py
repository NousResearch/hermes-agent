"""Tests for vision setup model prompt in _configure_simple_requirements (#41441)."""

from unittest.mock import patch, call
import pytest


class TestVisionSetupModelPrompt:
    """Verify the vision wizard prompts for model name and saves it correctly."""

    @pytest.fixture(autouse=True)
    def _deps(self, tmp_path, monkeypatch):
        """Isolate config/env for each test."""
        self._cfg_path = tmp_path / "config.yaml"
        self._cfg_path.write_text("{}")
        self._env_path = tmp_path / ".env"
        self._env_path.write_text("")
        monkeypatch.setattr("hermes_cli.tools_config._toolset_has_keys", lambda *_a, **_kw: False)

    # -- helpers ----------------------------------------------------------

    def _run(self, prompt_side_effects, *, is_openai=True):
        """Run the vision branch of _configure_simple_requirements.

        ``prompt_side_effects`` is a list of return values for ``_prompt()``
        in call order: [base_url, api_key, model_name].
        """
        import hermes_cli.tools_config as tc

        with (
            patch.object(tc, "_prompt", side_effect=prompt_side_effects),
            patch.object(tc, "_prompt_choice", return_value=1),  # idx=1 → OpenAI-compatible
            patch.object(tc, "load_config", return_value={}),
            patch.object(tc, "save_config") as mock_save,
            patch.object(tc, "save_env_value") as mock_env,
            patch.object(tc, "_print_success"),
            patch.object(tc, "_print_warning"),
            patch.object(tc, "_print_info"),
        ):
            tc._configure_simple_requirements("vision")
            return mock_save, mock_env

    # -- tests ------------------------------------------------------------

    def test_native_openai_prompts_for_model_with_default(self):
        """Native OpenAI should show 'gpt-4o-mini' as default in the hint."""
        import hermes_cli.tools_config as tc

        prompts = []
        with (
            patch.object(tc, "_prompt", side_effect=lambda msg, **_kw: (prompts.append(msg) or "")),
            patch.object(tc, "_prompt_choice", return_value=1),
            patch.object(tc, "load_config", return_value={}),
            patch.object(tc, "save_config"),
            patch.object(tc, "save_env_value"),
            patch.object(tc, "_print_success"),
            patch.object(tc, "_print_warning"),
            patch.object(tc, "_print_info"),
        ):
            # api.openai.com → is_native_openai=True
            # base_url prompt returns OpenAI URL
            tc._prompt = lambda msg, **_kw: {
                "OPENAI_BASE_URL": "https://api.openai.com/v1",
                "OPENAI_API_KEY": "sk-test",
            }.get(msg.strip().split("\n")[0].strip(), "")
            # Re-patch properly
            with patch.object(tc, "_prompt", side_effect=[
                "https://api.openai.com/v1",  # base_url
                "sk-test",                     # api_key
                "",                            # model_name (blank → use default)
            ]):
                tc._configure_simple_requirements("vision")

    def test_native_openai_saves_default_model(self):
        """When user accepts default, gpt-4o-mini is saved."""
        save_calls = {}
        env_calls = []

        import hermes_cli.tools_config as tc

        with (
            patch.object(tc, "_prompt", side_effect=[
                "https://api.openai.com/v1",  # base_url
                "sk-test",                     # api_key
                "",                            # model_name → default
            ]),
            patch.object(tc, "_prompt_choice", return_value=1),
            patch.object(tc, "load_config", return_value={}),
            patch.object(tc, "save_config") as mock_save,
            patch.object(tc, "save_env_value") as mock_env,
            patch.object(tc, "_print_success"),
            patch.object(tc, "_print_warning"),
            patch.object(tc, "_print_info"),
        ):
            tc._configure_simple_requirements("vision")

        # Check save_env_value was called with the model
        env_calls = [c for c in mock_env.call_args_list if c[0][0] == "AUXILIARY_VISION_MODEL"]
        assert len(env_calls) == 1
        assert env_calls[0][0][1] == "gpt-4o-mini"

    def test_custom_endpoint_prompts_for_model_no_default(self):
        """Custom endpoint should prompt for model without a default."""
        import hermes_cli.tools_config as tc

        prompts_captured = []
        original_prompt = tc._prompt

        def capture_prompt(msg, **kwargs):
            prompts_captured.append(msg)
            if "OPENAI_BASE_URL" in msg or "base_url" in msg.lower():
                return "https://my-vllm.example.com/v1"
            elif "API key" in msg or "OPENAI_API_KEY" in msg:
                return "my-key"
            elif "model" in msg.lower():
                return "llava-v1.6"
            return ""

        with (
            patch.object(tc, "_prompt", side_effect=capture_prompt),
            patch.object(tc, "_prompt_choice", return_value=1),
            patch.object(tc, "load_config", return_value={}),
            patch.object(tc, "save_config"),
            patch.object(tc, "save_env_value") as mock_env,
            patch.object(tc, "_print_success"),
            patch.object(tc, "_print_warning"),
            patch.object(tc, "_print_info"),
        ):
            tc._configure_simple_requirements("vision")

        # Model prompt should NOT contain "gpt-4o-mini" as default for custom endpoints
        model_prompts = [p for p in prompts_captured if "Vision model" in p or "model name" in p.lower()]
        assert len(model_prompts) >= 1, f"Expected model prompt, got: {prompts_captured}"
        assert "gpt-4o-mini" not in model_prompts[0], "Custom endpoint should not default to gpt-4o-mini"

        # Model should be saved
        env_calls = [c for c in mock_env.call_args_list if c[0][0] == "AUXILIARY_VISION_MODEL"]
        assert len(env_calls) == 1
        assert env_calls[0][0][1] == "llava-v1.6"

    def test_custom_endpoint_saves_model_to_config(self):
        """Custom endpoint model should be saved in auxiliary.vision.model config."""
        import hermes_cli.tools_config as tc

        saved_cfg = {}
        def capture_save(cfg):
            saved_cfg.update(cfg)

        with (
            patch.object(tc, "_prompt", side_effect=[
                "https://my-vllm.example.com/v1",  # base_url
                "my-key",                            # api_key
                "llava-v1.6",                        # model_name
            ]),
            patch.object(tc, "_prompt_choice", return_value=1),
            patch.object(tc, "load_config", return_value={}),
            patch.object(tc, "save_config", side_effect=capture_save),
            patch.object(tc, "save_env_value"),
            patch.object(tc, "_print_success"),
            patch.object(tc, "_print_warning"),
            patch.object(tc, "_print_info"),
        ):
            tc._configure_simple_requirements("vision")

        assert saved_cfg.get("auxiliary", {}).get("vision", {}).get("model") == "llava-v1.6"
        assert saved_cfg.get("auxiliary", {}).get("vision", {}).get("base_url") == "https://my-vllm.example.com/v1"

    def test_empty_model_not_saved(self):
        """If user leaves model blank on custom endpoint (no default), skip saving model."""
        import hermes_cli.tools_config as tc

        with (
            patch.object(tc, "_prompt", side_effect=[
                "https://my-vllm.example.com/v1",  # base_url
                "my-key",                            # api_key
                "",                                  # model_name → empty, no default
            ]),
            patch.object(tc, "_prompt_choice", return_value=1),
            patch.object(tc, "load_config", return_value={}),
            patch.object(tc, "save_config"),
            patch.object(tc, "save_env_value") as mock_env,
            patch.object(tc, "_print_success"),
            patch.object(tc, "_print_warning"),
            patch.object(tc, "_print_info"),
        ):
            tc._configure_simple_requirements("vision")

        # No AUXILIARY_VISION_MODEL should be saved when model is blank on custom endpoint
        env_calls = [c for c in mock_env.call_args_list if c[0][0] == "AUXILIARY_VISION_MODEL"]
        assert len(env_calls) == 0, f"Should not save empty model, got: {env_calls}"
