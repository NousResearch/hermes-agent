"""Regression coverage for OpenRouter preset references (issue #31739)."""

import json
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from hermes_cli.models import validate_requested_model


@pytest.mark.parametrize(
    "model_name",
    ["@preset/email-copywriter", "@preset/Foo_bar.~9"],
)
def test_direct_openrouter_preset_reference_skips_model_listing(model_name):
    """An account-scoped direct preset has no public model row to probe."""
    with patch(
        "hermes_cli.models.fetch_api_models",
        side_effect=AssertionError("direct preset references must not probe /models"),
    ):
        result = validate_requested_model(
            model_name,
            "openrouter",
            api_key="key",
            base_url="https://openrouter.ai/api/v1",
        )

    assert result == {
        "accepted": True,
        "persist": True,
        "recognized": False,
        "message": None,
    }


def test_combined_openrouter_preset_reference_validates_base_model():
    """Combined references validate the base model, not the preset-decorated ID."""
    with patch(
        "hermes_cli.models.fetch_api_models",
        return_value=["openai/gpt-5.4"],
    ) as mock_fetch:
        result = validate_requested_model(
            "openai/gpt-5.4@preset/email-copywriter",
            "openrouter",
            api_key="key",
            base_url="https://openrouter.ai/api/v1",
        )

    mock_fetch.assert_called_once_with("key", "https://openrouter.ai/api/v1")
    assert result == {
        "accepted": True,
        "persist": True,
        "recognized": True,
        "message": None,
    }


def test_combined_openrouter_preset_reference_rejects_unknown_base_model():
    with patch("hermes_cli.models.fetch_api_models", return_value=["openai/gpt-5.4"]):
        result = validate_requested_model(
            "openai/gpt-5.4-preview@preset/email-copywriter",
            "openrouter",
            api_key="key",
            base_url="https://openrouter.ai/api/v1",
        )

    assert result["accepted"] is False
    assert result["persist"] is False
    assert result["recognized"] is False
    assert "Similar models" in result["message"]
    assert "openai/gpt-5.4" in result["message"]


def test_combined_openrouter_preset_reference_preserves_suffix_on_autocorrect():
    with patch("hermes_cli.models.fetch_api_models", return_value=["openai/gpt-5.4"]):
        result = validate_requested_model(
            "openai/gpt-5.44@preset/email-copywriter",
            "openrouter",
            api_key="key",
            base_url="https://openrouter.ai/api/v1",
        )

    corrected = "openai/gpt-5.4@preset/email-copywriter"
    assert result["accepted"] is True
    assert result["corrected_model"] == corrected
    assert corrected in result["message"]


def test_combined_preset_preserves_suffix_on_catalog_autocorrect():
    with (
        patch("hermes_cli.models.fetch_api_models", return_value=None),
        patch(
            "hermes_cli.models.provider_model_ids",
            return_value=["openai/gpt-5.4"],
        ),
    ):
        result = validate_requested_model(
            "openai/gpt-5.44@preset/email-copywriter",
            "openrouter",
            api_key="key",
            base_url="https://openrouter.ai/api/v1",
        )

    corrected = "openai/gpt-5.4@preset/email-copywriter"
    assert result["accepted"] is True
    assert result["corrected_model"] == corrected
    assert corrected in result["message"]


@pytest.mark.parametrize(
    "model_name",
    [
        "@preset/",
        "openai/gpt-5.4@preset/",
        "@preset/foo/bar",
        "@preset/foo?bar",
        "@preset/☃",
        "@preset/foo@preset/bar",
        "openai/gpt-5.4@preset/foo/bar",
    ],
)
def test_openrouter_preset_reference_requires_a_url_safe_slug(model_name):
    """Malformed preset references must fail before model-list probing."""
    with patch(
        "hermes_cli.models.fetch_api_models",
        side_effect=AssertionError("malformed presets must not probe /models"),
    ):
        result = validate_requested_model(
            model_name,
            "openrouter",
            api_key="key",
            base_url="https://openrouter.ai/api/v1",
        )

    assert result["accepted"] is False
    assert result["persist"] is False
    assert result["recognized"] is False
    assert "URL-safe" in result["message"]


def test_preset_reference_does_not_bypass_other_provider_validation():
    with patch("hermes_cli.models.fetch_api_models", return_value=["gpt-5.4"]):
        result = validate_requested_model(
            "@preset/email-copywriter",
            "openai",
            api_key="key",
            base_url="https://api.openai.com/v1",
        )

    assert result["accepted"] is False
    assert result["persist"] is False


def test_preset_reference_does_not_bypass_custom_endpoint_validation():
    probe = {
        "models": ["local-model"],
        "probed_url": "https://proxy.example/v1/models",
        "resolved_base_url": "https://proxy.example/v1",
        "suggested_base_url": None,
        "used_fallback": False,
    }
    with patch("hermes_cli.models.probe_api_models", return_value=probe) as mock_probe:
        result = validate_requested_model(
            "@preset/email-copywriter",
            "openrouter",
            api_key="key",
            base_url="https://proxy.example/v1",
        )

    mock_probe.assert_called_once()
    assert result["accepted"] is True
    assert result["recognized"] is False
    assert "custom endpoint's model listing" in result["message"]


def test_configured_alias_switches_preset_through_real_resolution_chain(tmp_path):
    """Exercise config loading, alias resolution, runtime resolution, and validation."""
    (tmp_path / "config.yaml").write_text(
        """
model:
  default: openai/gpt-5.4
  provider: openrouter
  base_url: https://openrouter.ai/api/v1
model_aliases:
  email-copywriter:
    model: '@preset/email-copywriter'
    provider: openrouter
""".lstrip(),
        encoding="utf-8",
    )

    script = r"""
import json
import hermes_cli.model_switch as model_switch
import hermes_cli.models as models


def fail_model_probe(*args, **kwargs):
    raise AssertionError("preset references must not probe /models")


models.fetch_api_models = fail_model_probe
model_switch.get_model_capabilities = lambda *args, **kwargs: None
model_switch.get_model_info = lambda *args, **kwargs: None
result = model_switch.switch_model(
    "email-copywriter",
    current_provider="openrouter",
    current_model="openai/gpt-5.4",
    current_base_url="https://openrouter.ai/api/v1",
    current_api_key="key",
)
print(json.dumps(vars(result)))
"""
    env = {
        "HOME": str(tmp_path),
        "HERMES_HOME": str(tmp_path),
        "LANG": "C.UTF-8",
        "OPENROUTER_API_KEY": "key",
        "PATH": os.environ.get("PATH", ""),
        "PYTHONIOENCODING": "utf-8",
    }
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[2],
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, (
        f"subprocess failed with exit {completed.returncode}\n"
        f"stdout:\n{completed.stdout}\n"
        f"stderr:\n{completed.stderr}"
    )
    result = json.loads(completed.stdout.splitlines()[-1])

    assert result["success"] is True
    assert result["new_model"] == "@preset/email-copywriter"
    assert result["target_provider"] == "openrouter"
    assert result["resolved_via_alias"] == "email-copywriter"
    assert result["warning_message"] == ""
