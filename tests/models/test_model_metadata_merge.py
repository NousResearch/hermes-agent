from models import ModelRef
from models.metadata.merge import merge_metadata
from models.metadata.types import ModelMetadataPatch, ReasoningMetadata


def test_merge_metadata_preserves_false_and_tracks_source():
    metadata = merge_metadata(
        ModelRef("openrouter", "vendor/model"),
        (
            ("explicit", ModelMetadataPatch(supports_vision=False)),
            ("catalog", ModelMetadataPatch(supports_vision=True, supports_tools=True)),
        ),
    )

    assert metadata.supports_vision is False
    assert metadata.supports_tools is True
    assert metadata.provenance["supports_vision"] == "explicit"
    assert metadata.provenance["supports_tools"] == "catalog"


def test_merge_metadata_unknown_falls_through():
    metadata = merge_metadata(
        ModelRef("openrouter", "vendor/model"),
        (
            ("live", ModelMetadataPatch(supports_vision=None)),
            ("catalog", ModelMetadataPatch(supports_vision=True)),
        ),
    )

    assert metadata.supports_vision is True
    assert metadata.provenance["supports_vision"] == "catalog"


def test_reasoning_fields_merge_independently():
    metadata = merge_metadata(
        ModelRef("openrouter", "vendor/model"),
        (
            (
                "live",
                ModelMetadataPatch(
                    reasoning=ReasoningMetadata(supported=True, mandatory=True),
                ),
            ),
            (
                "catalog",
                ModelMetadataPatch(
                    reasoning=ReasoningMetadata(
                        supported=True,
                        supported_efforts=("low", "high"),
                        mandatory=False,
                    ),
                ),
            ),
        ),
    )

    assert metadata.supports_reasoning is True
    assert metadata.reasoning.supported is True
    assert metadata.reasoning.supported_efforts == ("low", "high")
    assert metadata.reasoning.mandatory is True
    assert metadata.provenance["reasoning.mandatory"] == "live"
    assert metadata.provenance["reasoning.supported_efforts"] == "catalog"


def test_top_level_reasoning_support_and_nested_support_stay_coherent():
    metadata = merge_metadata(
        ModelRef("openrouter", "vendor/model"),
        (("catalog", ModelMetadataPatch(supports_reasoning=False)),),
    )

    assert metadata.supports_reasoning is False
    assert metadata.reasoning.supported is False
