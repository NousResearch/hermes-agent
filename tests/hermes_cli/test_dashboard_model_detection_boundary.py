"""Flat Settings model edits use canonical detection without guessing paid providers."""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

import application_dashboard_model_detection as detection


def test_unknown_model_keeps_current_provider(monkeypatch):
    monkeypatch.setattr(detection, "_authorized", lambda provider: False)
    assert detection.infer_dashboard_model_change(
        "new-private-model", "openrouter", {},
    ) == ("", "new-private-model")


def test_unknown_vendor_slug_does_not_guess_unpaid_openrouter(monkeypatch):
    monkeypatch.setattr(detection, "_authorized", lambda provider: False)
    assert detection.infer_dashboard_model_change(
        "vendor/new-model", "anthropic", {},
    ) == ("", "vendor/new-model")


def test_custom_endpoint_is_not_replaced_by_vendor_heuristic(monkeypatch):
    monkeypatch.setattr(detection, "_authorized", lambda provider: True)
    assert detection.infer_dashboard_model_change(
        "ollama/new-model", "custom:home", {},
    ) == ("", "ollama/new-model")


def test_explicit_vendor_slug_keeps_current_aggregator(monkeypatch):
    monkeypatch.setattr(detection, "_authorized", lambda provider: True)
    assert detection.infer_dashboard_model_change(
        "anthropic/claude-sonnet-5", "openrouter", {},
    ) == ("", "anthropic/claude-sonnet-5")


def test_undecidable_configured_provider_is_not_silently_selected(monkeypatch):
    monkeypatch.setattr(detection, "_authorized", lambda provider: True)
    cfg = {"providers": {
        "a": {"name": "First", "base_url": "https://one.example/v1",
              "models": ["private-model"]},
        "b": {"name": "Second", "base_url": "https://two.example/v1",
              "models": ["private-model"]},
    }}
    assert detection.infer_dashboard_model_change(
        "private-model", "anthropic", cfg,
    ) == ("", "private-model")


def test_flat_model_owner_has_no_legacy_cli_detection_import():
    root = Path(__file__).resolve().parents[2]
    source = (root / "hermes_cli" / "web_server_config.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    function = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_infer_provider_on_model_change"
    )
    modules = {
        node.module for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    }
    assert "hermes_cli.models" not in modules
    assert "application_dashboard_model_detection" in modules
