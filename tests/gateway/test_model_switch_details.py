"""Tests for /model switch confirmation details."""

from hermes_cli.model_switch import ModelSwitchResult

from gateway.run import _append_model_switch_details


class _FakeModelInfo:
    context_window = 128000
    max_output = 8192

    def has_cost_data(self):
        return False

    def format_capabilities(self):
        return "tools"


def test_alias_overrides_beat_model_info_defaults():
    """Alias-specific context/max output should be shown ahead of catalog defaults."""
    lines = []
    result = ModelSwitchResult(
        success=True,
        new_model="custom-model:latest",
        target_provider="custom",
        provider_changed=True,
        base_url="https://example.com/v1",
        api_mode="openai_compat",
        model_info=_FakeModelInfo(),
        context_length=204800,
        max_tokens=131072,
    )

    _append_model_switch_details(lines, result)

    assert "Context: 204,800 tokens" in lines
    assert "Max output: 131,072 tokens" in lines
    assert "Capabilities: tools" in lines
