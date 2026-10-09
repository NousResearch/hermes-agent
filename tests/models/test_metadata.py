"""Canonical raw catalogue interpretation tests."""

from models import ModelRef
from models.metadata import (
    ModelMetadata,
    model_metadata_from_entry,
    model_metadata_patch_from_entry,
)


def test_patch_preserves_unknown_and_explicit_false_capabilities():
    unknown = model_metadata_patch_from_entry({"limit": {"context": 48000}}, unknown_model=True)
    assert unknown.context_window == 48000
    assert unknown.supports_vision is None
    assert unknown.supports_reasoning is None

    explicit_false = model_metadata_patch_from_entry(
        {"attachment": False, "reasoning": False, "tool_call": True},
        unknown_model=True,
    )
    assert explicit_false.supports_vision is False
    assert explicit_false.supports_reasoning is False
    assert explicit_false.supports_tools is True


def test_patch_keeps_historical_context_fallback():
    patch = model_metadata_patch_from_entry({"tool_call": True})
    assert patch.context_window == 200000


def test_entry_interpretation_returns_canonical_metadata():
    metadata = model_metadata_from_entry(
        ModelRef("anthropic", "claude-sonnet-4"),
        {
            "limit": {"context": 200000, "output": 64000},
            "modalities": {"input": ["text", "image"], "output": ["text"]},
            "reasoning": True,
            "tool_call": True,
            "family": "claude",
        },
    )
    assert isinstance(metadata, ModelMetadata)
    assert metadata.ref == ModelRef("anthropic", "claude-sonnet-4")
    assert metadata.context_window == 200000
    assert metadata.max_output_tokens == 64000
    assert metadata.supports_vision is True
    assert metadata.supports_reasoning is True
    assert metadata.input_modalities == ("text", "image")
    assert metadata.model_family == "claude"
