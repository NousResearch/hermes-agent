"""Focused contracts for the VIAJA CON CARLOS general plugin."""

from __future__ import annotations

import json
from pathlib import Path

from plugins.viaja_con_carlos import register
from plugins.viaja_con_carlos.conversation import FIXED_OPENING, conversation_prompt
from plugins.viaja_con_carlos.source_lookup import lookup_sources, normalize_text


class _Context:
    def __init__(self):
        self.tools = []
        self.hooks = []

    def register_tool(self, **kwargs):
        self.tools.append(kwargs)

    def register_hook(self, name, callback):
        self.hooks.append((name, callback))


def test_normalize_text_handles_accents_case_and_whitespace():
    assert normalize_text("  Dúmbar   RÓCK  ") == "dumbar rock"


def test_lookup_returns_relevant_excerpt_and_stable_attribution():
    result = lookup_sources("Palafitos bungalow private pool price")

    assert result["confirmation_required"] is False
    match = next(item for item in result["excerpts"] if item["source_id"] == "PAL-006")
    assert "private pool" in match["excerpt"].lower()
    assert match["document_path"].endswith("palafitos-overwater-maroma.md")


def test_lookup_normalizes_property_aliases():
    result = lookup_sources("Dúmbar   Róck", property_hint="Roatán")

    assert result["confirmation_required"] is False
    assert any(item["source_id"] == "DUN-001" for item in result["excerpts"])


def test_lookup_marks_known_conflicts_for_confirmation():
    result = lookup_sources("Cayo Espanto impuestos resort fee")

    assert result["confirmation_required"] is True
    assert result["confirmation_reason"] == "conflicting_facts"
    assert {"CAY-009", "CAY-010"}.issubset(
        {item["source_id"] for item in result["excerpts"]}
    )


def test_lookup_marks_missing_facts_for_confirmation():
    result = lookup_sources("Cayo Espanto exact price for 2035")

    assert result["excerpts"] == []
    assert result["confirmation_required"] is True
    assert result["confirmation_reason"] == "missing_fact"


def test_conversation_prompt_keeps_opening_adapter_owned_and_policy_explicit():
    assert FIXED_OPENING
    prompt = conversation_prompt()
    assert "exactly once" in prompt
    assert "source lookup" in prompt.lower()
    assert "No inventes" in prompt


def test_plugin_registers_public_tool_and_conversation_hook():
    ctx = _Context()

    register(ctx)

    assert len(ctx.tools) == 1
    tool = ctx.tools[0]
    assert tool["name"] == "viaja_source_lookup"
    assert tool["schema"]["parameters"]["required"] == ["query"]
    assert {name for name, _ in ctx.hooks} == {"pre_llm_call"}

    payload = json.loads(tool["handler"]({"query": "Palafitos"}))
    assert payload["excerpts"]
