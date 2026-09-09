"""Explicit route identities are validated, never silently repaired."""
import pytest

from hermes_cli.plugin_side_runs import SideRunConfig


@pytest.mark.parametrize("patch", [
    {"provider": " openai"},
    {"provider": "openai "},
    {"provider": "OpenAI"},
    {"model": " model"},
    {"model": "model "},
    {"reasoning": {"effort": []}},
    {"reasoning": {"effort": {}}},
])
def test_invalid_identity_or_reasoning_is_rejected_without_normalization(patch):
    with pytest.raises(ValueError):
        SideRunConfig.from_mapping({"provider": "openai", "model": "model", **patch})
