"""Tests for the Desktop presentation layer (hermes_cli.memory_provider_surface).

Covers field normalization/derivation (back-compat with legacy schemas),
tier grouping, conditional ``when`` carry-through, and surface assembly with
secret masking.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

import pytest

from agent.memory_provider import MemoryProvider
from hermes_cli.memory_provider_surface import (
    KIND_BOOL,
    KIND_NUMBER,
    KIND_SECRET,
    KIND_SELECT,
    KIND_TEXT,
    TIER_ADVANCED,
    TIER_SAFE,
    build_surface,
    enrich_schema,
    field_visible,
    normalize_field,
)


# --- field derivation / back-compat -----------------------------------------

def test_secret_derives_kind_and_env_key():
    f = normalize_field({"key": "api_key", "secret": True, "env_var": "FOO_KEY"})
    assert f["kind"] == KIND_SECRET
    assert f["env_key"] == "FOO_KEY"
    assert f["tier"] == TIER_SAFE


def test_choices_derive_select_with_options():
    f = normalize_field({"key": "mode", "choices": ["cloud", "local_external"]})
    assert f["kind"] == KIND_SELECT
    assert [o["value"] for o in f["options"]] == ["cloud", "local_external"]


def test_plain_field_is_text():
    assert normalize_field({"key": "base_url"})["kind"] == KIND_TEXT


def test_explicit_kind_respected():
    assert normalize_field({"key": "x", "kind": KIND_BOOL})["kind"] == KIND_BOOL
    assert normalize_field({"key": "y", "kind": KIND_NUMBER})["kind"] == KIND_NUMBER


def test_label_prettifies_snake_and_camel():
    assert normalize_field({"key": "api_key"})["label"] == "Api Key"
    assert normalize_field({"key": "baseUrl"})["label"] == "Base Url"


def test_explicit_label_wins():
    assert normalize_field({"key": "k", "label": "Custom"})["label"] == "Custom"


def test_tier_advanced_passthrough_and_invalid_falls_back_safe():
    assert normalize_field({"key": "k", "tier": "advanced"})["tier"] == TIER_ADVANCED
    assert normalize_field({"key": "k", "tier": "bogus"})["tier"] == TIER_SAFE


def test_enrich_skips_keyless_fields():
    out = enrich_schema([{"key": "a"}, {"description": "no key"}, {"key": "b"}])
    assert [f["key"] for f in out] == ["a", "b"]


# --- conditional (when) carry-through ----------------------------------------

def test_when_clause_carried_through():
    f = normalize_field({"key": "api_url", "when": {"mode": "cloud"}})
    assert f["when"] == {"mode": "cloud"}


def test_duplicate_keys_with_different_when_both_enriched():
    # Hindsight declares api_url 3x, gated by mode — enrich keeps all rows so
    # the renderer can show the matching one.
    schema = [
        {"key": "api_url", "default": "cloud-url", "when": {"mode": "cloud"}},
        {"key": "api_url", "default": "local-url", "when": {"mode": "local_external"}},
    ]
    rows = enrich_schema(schema)
    assert len(rows) == 2
    assert {r["default"] for r in rows} == {"cloud-url", "local-url"}


# --- build_surface assembly --------------------------------------------------

def _write_provider_config(home, name, data):
    d = home / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "config.json").write_text(json.dumps(data), encoding="utf-8")


class _FakeProvider(MemoryProvider):
    @property
    def name(self) -> str:
        return "demo"

    def is_available(self) -> bool:
        return True

    def initialize(self, session_id: str, **kwargs) -> None:
        pass

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return []

    def get_config_schema(self) -> List[Dict[str, Any]]:
        return [
            {"key": "api_key", "secret": True, "env_var": "DEMO_KEY", "description": "key"},
            {"key": "api_url", "default": "https://default"},
            {"key": "mode", "choices": ["cloud", "local"], "tier": "advanced"},
        ]


class _NoConfigProvider(MemoryProvider):
    @property
    def name(self) -> str:
        return "builtin"

    def is_available(self) -> bool:
        return True

    def initialize(self, session_id: str, **kwargs) -> None:
        pass

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return []


def test_build_surface_masks_secret_and_carries_state(tmp_path, monkeypatch):
    monkeypatch.setenv("DEMO_KEY", "abc123")
    _write_provider_config(tmp_path, "demo", {"api_url": "https://custom"})
    surface = build_surface(_FakeProvider(), str(tmp_path))

    assert surface["name"] == "demo"
    fields = {f["key"]: f for f in surface["fields"]}
    assert fields["api_key"]["kind"] == "secret"
    assert fields["api_key"]["value"] == ""
    assert fields["api_key"]["is_set"] is True
    assert fields["api_url"]["value"] == "https://custom"
    assert fields["mode"]["kind"] == "select"
    assert fields["mode"]["tier"] == "advanced"
    assert "abc123" not in json.dumps(surface)


def test_build_surface_empty_for_no_schema(tmp_path):
    surface = build_surface(_NoConfigProvider(), str(tmp_path))
    assert surface["fields"] == []


# --- field_visible (when gating) --------------------------------------------

def test_field_visible_no_when_always_true():
    assert field_visible({"key": "x"}, {}) is True
    assert field_visible({"key": "x"}, {"mode": "cloud"}) is True


def test_field_visible_matches_values():
    f = {"key": "api_url", "when": {"mode": "cloud"}}
    assert field_visible(f, {"mode": "cloud"}) is True
    assert field_visible(f, {"mode": "local_external"}) is False
    assert field_visible(f, {}) is False


def test_field_visible_requires_all_when_keys():
    f = {"key": "u", "when": {"mode": "local_embedded", "llm_provider": "openai_compatible"}}
    assert field_visible(f, {"mode": "local_embedded", "llm_provider": "openai_compatible"}) is True
    assert field_visible(f, {"mode": "local_embedded", "llm_provider": "openai"}) is False


# --- default ABC read_config -------------------------------------------------

def test_default_read_config_values_and_defaults(tmp_path):
    _write_provider_config(tmp_path, "demo", {"api_url": "https://custom"})
    state = _FakeProvider().read_config(str(tmp_path))
    assert state["api_url"] == {"value": "https://custom", "is_set": True}


def test_default_read_config_masks_secret(tmp_path, monkeypatch):
    monkeypatch.setenv("DEMO_KEY", "super-secret")
    state = _FakeProvider().read_config(str(tmp_path))
    assert state["api_key"]["value"] == ""
    assert state["api_key"]["is_set"] is True
    assert "super-secret" not in json.dumps(state)


def test_default_read_config_secret_not_set(tmp_path, monkeypatch):
    monkeypatch.delenv("DEMO_KEY", raising=False)
    state = _FakeProvider().read_config(str(tmp_path))
    assert state["api_key"] == {"value": "", "is_set": False}
