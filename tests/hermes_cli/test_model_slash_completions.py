"""/model slash-argument completions (flags + provider/id hops)."""

from __future__ import annotations

from unittest.mock import patch

import hermes_cli.commands_completion as commands_mod


def _texts(sub_text: str) -> list[str]:
    return [c.text for c in commands_mod._model_completions(sub_text, sub_text.lower())]


def test_model_space_offers_flags_and_provider_ids():
    catalog = {
        "providers": [("nous", "Nous Portal")],
        "models": [("nous/claude-sonnet-4.6", "Nous Portal")],
    }
    with patch.object(commands_mod, "_model_completion_catalog", return_value=catalog):
        texts = _texts("")
    assert "--provider" in texts
    assert "--reasoning" in texts
    assert "nous/claude-sonnet-4.6" in texts


def test_model_provider_flag_offers_slugs():
    catalog = {
        "providers": [("nous", "Nous Portal"), ("openrouter", "OpenRouter")],
        "models": [],
    }
    with patch.object(commands_mod, "_model_completion_catalog", return_value=catalog):
        texts = _texts("--provider ")
    assert texts == ["nous", "openrouter"]


def test_model_reasoning_flag_offers_efforts():
    catalog = {"providers": [], "models": []}
    with patch.object(commands_mod, "_model_completion_catalog", return_value=catalog):
        texts = _texts("--reasoning ")
    assert "low" in texts
    assert "none" in texts
