"""Tests for the manual Bedrock application-inference-profile setup flow.

`_model_flow_bedrock_inference_profiles` lets a user register opaque
application-inference-profile ARNs and tag each with the Claude model it
points to. The mapping is persisted to ``bedrock.models.<arn>.name`` so the
adapters can route correctly and send the right thinking config.
"""

from unittest.mock import patch

import pytest


@pytest.fixture
def config_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME with a minimal string-format config."""
    home = tmp_path / "hermes"
    home.mkdir()
    (home / "config.yaml").write_text("model: some-old-model\n")
    (home / ".env").write_text("")
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("HERMES_MODEL", raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False)
    monkeypatch.delenv("HERMES_INFERENCE_PROVIDER", raising=False)
    return home


ARN_A = "arn:aws:bedrock:eu-west-1:123456789012:application-inference-profile/aaa"
ARN_B = "arn:aws:bedrock:eu-west-1:123456789012:application-inference-profile/bbb"


def _load(config_home):
    import yaml

    return yaml.safe_load((config_home / "config.yaml").read_text()) or {}


class TestBedrockInferenceProfilesFlow:
    def test_registers_multiple_arns_and_persists_mapping(self, config_home):
        from hermes_cli.model_setup_flows import (
            _model_flow_bedrock_inference_profiles,
        )

        # ARN_A → family "1" (Opus 4.8), add another;
        # ARN_B → family "5" (Haiku 4.5), stop.
        inputs = [ARN_A, "1", "y", ARN_B, "5", "n"]

        with patch("builtins.input", side_effect=inputs), patch(
            "hermes_cli.auth._prompt_model_selection", return_value=ARN_A
        ), patch("hermes_cli.auth.deactivate_provider"):
            _model_flow_bedrock_inference_profiles({}, "eu-west-1")

        cfg = _load(config_home)
        assert cfg["bedrock"]["models"] == {
            ARN_A: {"name": "Opus 4.8"},
            ARN_B: {"name": "Haiku 4.5"},
        }
        assert cfg["bedrock"]["region"] == "eu-west-1"
        assert cfg["bedrock"]["discovery"]["enabled"] is False
        assert cfg["model"]["provider"] == "bedrock"
        assert cfg["model"]["default"] == ARN_A
        assert (
            cfg["model"]["base_url"]
            == "https://bedrock-runtime.eu-west-1.amazonaws.com"
        )

    def test_single_arn_skips_picker_and_is_default(self, config_home):
        from hermes_cli.model_setup_flows import (
            _model_flow_bedrock_inference_profiles,
        )

        inputs = [ARN_A, "2", "n"]  # one ARN, family "2" (Opus 4.7), stop

        # _prompt_model_selection must NOT be called for a single ARN.
        with patch("builtins.input", side_effect=inputs), patch(
            "hermes_cli.auth._prompt_model_selection"
        ) as picker, patch("hermes_cli.auth.deactivate_provider"):
            _model_flow_bedrock_inference_profiles({}, "us-east-1")

        picker.assert_not_called()
        cfg = _load(config_home)
        assert cfg["bedrock"]["models"] == {ARN_A: {"name": "Opus 4.7"}}
        assert cfg["model"]["default"] == ARN_A

    def test_free_text_family_escape_hatch(self, config_home):
        from hermes_cli.model_setup_flows import (
            _BEDROCK_PROFILE_FAMILIES,
            _model_flow_bedrock_inference_profiles,
        )

        other = str(len(_BEDROCK_PROFILE_FAMILIES) + 1)  # the "Other" option
        inputs = [ARN_A, other, "Opus 5.0", "n"]

        with patch("builtins.input", side_effect=inputs), patch(
            "hermes_cli.auth.deactivate_provider"
        ):
            _model_flow_bedrock_inference_profiles({}, "eu-west-1")

        cfg = _load(config_home)
        assert cfg["bedrock"]["models"][ARN_A] == {"name": "Opus 5.0"}

    def test_no_arns_makes_no_change(self, config_home):
        from hermes_cli.model_setup_flows import (
            _model_flow_bedrock_inference_profiles,
        )

        # Immediately decline to add any ARN (empty ARN, then stop).
        with patch("builtins.input", side_effect=["", "n"]), patch(
            "hermes_cli.auth.deactivate_provider"
        ):
            _model_flow_bedrock_inference_profiles({}, "eu-west-1")

        cfg = _load(config_home)
        # Untouched — model stays the original plain string, no bedrock block.
        assert cfg.get("model") == "some-old-model"
        assert "bedrock" not in cfg

    def test_invalid_family_choice_defaults_to_first(self, config_home):
        from hermes_cli.model_setup_flows import (
            _BEDROCK_PROFILE_FAMILIES,
            _model_flow_bedrock_inference_profiles,
        )

        # Out-of-range numeric and non-numeric both fall back to the first
        # family (with a printed warning).
        inputs = [ARN_A, "99", "n"]
        with patch("builtins.input", side_effect=inputs), patch(
            "hermes_cli.auth.deactivate_provider"
        ):
            _model_flow_bedrock_inference_profiles({}, "eu-west-1")

        cfg = _load(config_home)
        assert cfg["bedrock"]["models"][ARN_A] == {
            "name": _BEDROCK_PROFILE_FAMILIES[0]
        }

    def test_duplicate_arn_last_entry_wins(self, config_home):
        from hermes_cli.model_setup_flows import (
            _model_flow_bedrock_inference_profiles,
        )

        # Same ARN entered twice with different families — the second replaces
        # the first, leaving a single mapping.
        inputs = [ARN_A, "1", "y", ARN_A, "5", "n"]
        with patch("builtins.input", side_effect=inputs), patch(
            "hermes_cli.auth.deactivate_provider"
        ):
            _model_flow_bedrock_inference_profiles({}, "eu-west-1")

        cfg = _load(config_home)
        assert cfg["bedrock"]["models"] == {ARN_A: {"name": "Haiku 4.5"}}

    def test_keyboard_interrupt_on_first_arn_leaves_config_untouched(
        self, config_home
    ):
        from hermes_cli.model_setup_flows import (
            _model_flow_bedrock_inference_profiles,
        )

        with patch("builtins.input", side_effect=KeyboardInterrupt), patch(
            "hermes_cli.auth.deactivate_provider"
        ):
            _model_flow_bedrock_inference_profiles({}, "eu-west-1")

        cfg = _load(config_home)
        assert cfg.get("model") == "some-old-model"
        assert "bedrock" not in cfg

    def test_multi_arn_picker_receives_model_labels(self, config_home):
        """Regression: the multi-ARN path calls _prompt_model_selection with
        model_labels= — that kwarg must exist on the real signature (it would
        otherwise TypeError in production while a loose mock hid it)."""
        import inspect
        from hermes_cli.auth import _prompt_model_selection

        params = inspect.signature(_prompt_model_selection).parameters
        assert "model_labels" in params, (
            "_prompt_model_selection must accept model_labels= for the "
            "Bedrock profile picker"
        )
