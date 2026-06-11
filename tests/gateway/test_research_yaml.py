from __future__ import annotations

from pathlib import Path

from gateway.alphahunt.research_yaml import (
    dump_research_yaml,
    load_research_yaml,
    sample_research_envelope,
    validate_research_envelope,
)


def _errors(envelope: dict) -> list[str]:
    return validate_research_envelope(envelope)


def test_protocol_sample_passes():
    envelope = sample_research_envelope("protocol")
    assert envelope["asset_class"] == "DRAFT:crypto_protocol"
    assert "source_references" in envelope["note"]
    assert "source_references" not in envelope
    assert _errors(envelope) == []


def test_stock_and_etf_samples_pass():
    assert _errors(sample_research_envelope("stock")) == []
    assert _errors(sample_research_envelope("etf")) == []


def test_commodity_theme_sample_passes():
    assert _errors(sample_research_envelope("commodity_theme")) == []


def test_macro_event_sample_passes():
    assert _errors(sample_research_envelope("macro_event")) == []


def test_market_sample_passes():
    assert _errors(sample_research_envelope("market")) == []


def test_dump_and_load_round_trip():
    envelope = sample_research_envelope("protocol")
    loaded = load_research_yaml(dump_research_yaml(envelope))
    assert loaded == envelope


def test_samples_on_disk_validate():
    sample_dir = Path("docs/alphahunt/samples")
    for name in [
        "protocol_ethena.yaml",
        "stock_example.yaml",
        "commodity_copper.yaml",
        "macro_fomc.yaml",
        "market_worldcup.yaml",
    ]:
        envelope = load_research_yaml((sample_dir / name).read_text(encoding="utf-8"))
        assert _errors(envelope) == []


def test_missing_thesis_fails():
    envelope = sample_research_envelope("protocol")
    del envelope["note"]["thesis"]

    assert any("note.thesis" in error for error in _errors(envelope))


def test_missing_invalidation_conditions_fails():
    envelope = sample_research_envelope("protocol")
    del envelope["note"]["invalidation_conditions"]

    assert any("note.invalidation_conditions" in error for error in _errors(envelope))


def test_missing_next_check_at_fails():
    envelope = sample_research_envelope("protocol")
    del envelope["note"]["next_check_at"]

    assert any("note.next_check_at" in error for error in _errors(envelope))


def test_missing_note_source_references_fails():
    envelope = sample_research_envelope("protocol")
    del envelope["note"]["source_references"]

    assert any("note.source_references" in error for error in _errors(envelope))


def test_top_level_source_references_fails():
    envelope = sample_research_envelope("protocol")
    envelope["source_references"] = envelope["note"].pop("source_references")

    errors = _errors(envelope)
    assert any("note.source_references" in error for error in errors)
    assert any("source_references must be nested" in error for error in errors)


def test_non_iso_next_check_at_fails():
    envelope = sample_research_envelope("protocol")
    envelope["note"]["next_check_at"] = "next Thursday"

    assert any("ISO8601" in error for error in _errors(envelope))


def test_empty_observables_fails():
    envelope = sample_research_envelope("protocol")
    envelope["note"]["observables"] = []

    assert any("note.observables" in error for error in _errors(envelope))


def test_unknown_kind_fails():
    envelope = sample_research_envelope("protocol")
    envelope["kind"] = "bond"

    assert any("kind must be one of" in error for error in _errors(envelope))


def test_unknown_asset_class_without_draft_prefix_fails():
    envelope = sample_research_envelope("protocol")
    envelope["asset_class"] = "new_vertical"

    assert any("DRAFT:<name>" in error for error in _errors(envelope))


def test_draft_asset_class_passes():
    envelope = sample_research_envelope("protocol")
    envelope["asset_class"] = "DRAFT:new_vertical"

    assert _errors(envelope) == []


def test_market_output_forbidden_terms_fail():
    for term in ["bet", "wager", "execute", "stake", "kelly"]:
        envelope = sample_research_envelope("market")
        envelope["research_markdown"] += f"\nDo not use this term: {term}."

        assert any(term in error for error in _errors(envelope))


def test_market_action_must_be_whitelisted():
    envelope = sample_research_envelope("market")
    envelope["note"]["action_suggestion"] = "participate"

    assert any("market note.action_suggestion" in error for error in _errors(envelope))


def test_market_actions_allowed():
    for action in ["ignore", "observe", "research", "manual_review", "no_participation"]:
        envelope = sample_research_envelope("market")
        envelope["note"]["action_suggestion"] = action

        assert _errors(envelope) == []
