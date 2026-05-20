"""Tests for gateway runtime provenance notes used for self-identification."""

from __future__ import annotations

import logging

from gateway.runtime_provenance import (
    build_model_switch_note,
    resolve_self_identification_note,
)


def test_model_switch_note_is_structured_and_names_runtime_authority():
    note = build_model_switch_note(
        previous_model="claude-opus-4-7",
        requested_model="gpt-5.5",
        requested_provider="openai-codex",
    )

    assert note["previous_model"] == "claude-opus-4-7"
    assert note["requested_model"] == "gpt-5.5"
    assert note["requested_provider"] == "openai-codex"
    assert "runtime metadata" in note["text"]
    assert "self-identification" in note["text"]


def test_resolved_note_prefers_live_runtime_when_pending_note_matches():
    pending = build_model_switch_note(
        previous_model="claude-opus-4-7",
        requested_model="gpt-5.5",
        requested_provider="openai-codex",
    )

    resolved = resolve_self_identification_note(
        pending,
        runtime_model="gpt-5.5",
        runtime_provider="openai-codex",
        session_key="discord:thread",
    )

    assert "Live runtime for this turn: openai-codex/gpt-5.5" in resolved
    assert "stale" not in resolved.lower()


def test_resolved_note_reports_conflict_and_logs_runtime_winner(caplog):
    pending = build_model_switch_note(
        previous_model="gpt-5.5",
        requested_model="claude-opus-4-7",
        requested_provider="anthropic",
    )

    with caplog.at_level(logging.WARNING):
        resolved = resolve_self_identification_note(
            pending,
            runtime_model="gpt-5.5",
            runtime_provider="openai-codex",
            session_key="discord:1506515748535930941",
        )

    assert "Runtime provenance conflict" in resolved
    assert "queued /model switch requested anthropic/claude-opus-4-7" in resolved
    assert "live runtime is openai-codex/gpt-5.5" in resolved
    assert "Prefer the live runtime/footer" in resolved
    assert "runtime provenance conflict" in caplog.text
    assert "discord:1506515748535930941" in caplog.text


def test_legacy_pending_note_gets_conflict_checked(caplog):
    legacy = (
        "[Note: model was just switched from gpt-5.5 to claude-opus-4-7 "
        "via Anthropic. Adjust your self-identification accordingly.]"
    )

    with caplog.at_level(logging.WARNING):
        resolved = resolve_self_identification_note(
            legacy,
            runtime_model="gpt-5.5",
            runtime_provider="openai-codex",
            session_key="discord:thread",
        )

    assert "Runtime provenance conflict" in resolved
    assert "anthropic/claude-opus-4-7" in resolved.lower()
    assert "openai-codex/gpt-5.5" in resolved
    assert "runtime provenance conflict" in caplog.text
