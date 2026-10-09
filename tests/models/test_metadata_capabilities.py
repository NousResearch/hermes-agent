"""Behavior contracts for the canonical model metadata capability owner."""

from models import ModelRef
from models.metadata import CapabilitySources, ModelMetadataContext, ModelMetadataPatch
from models.metadata.capabilities import resolve_model_metadata, resolve_supports_vision


def test_explicit_and_configured_facts_preserve_false_and_precedence():
    ref = ModelRef("custom", "local-vlm")
    sources = CapabilitySources(
        live=lambda *_: ModelMetadataPatch(supports_vision=True),
    )

    assert resolve_supports_vision(
        ref,
        context=ModelMetadataContext(
            explicit=ModelMetadataPatch(supports_vision=False),
            configured=ModelMetadataPatch(supports_vision=True),
        ),
        sources=sources,
    ) is False


def test_live_source_wins_and_later_sources_are_not_queried_for_vision():
    calls = []

    def live(*_):
        calls.append("live")
        return ModelMetadataPatch(supports_vision=True)

    def catalog(*_):
        calls.append("catalog")
        raise AssertionError("catalog must not be queried after a live answer")

    assert resolve_supports_vision(
        ModelRef("llamacpp", "vision.gguf"),
        sources=CapabilitySources(live=live, catalog=catalog),
    ) is True
    assert calls == ["live"]


def test_metadata_resolution_keeps_unknown_distinct_from_false():
    ref = ModelRef("custom", "unknown")
    metadata = resolve_model_metadata(
        ref,
        context=ModelMetadataContext(configured=ModelMetadataPatch(supports_vision=False)),
    )
    unknown = resolve_model_metadata(ref)

    assert metadata.supports_vision is False
    assert metadata.provenance["supports_vision"] == "configured"
    assert unknown.supports_vision is None
