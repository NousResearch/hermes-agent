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


def test_model_hops_fuzzy_match_fragments():
    catalog = {
        "providers": [("nous", "Nous Portal")],
        "models": [("nous/claude-sonnet-4.6", "Nous Portal"), ("nous/hermes-4", "Nous Portal")],
    }
    with patch.object(commands_mod, "_model_completion_catalog", return_value=catalog):
        texts = _texts("son4")
    assert texts == ["nous/claude-sonnet-4.6"]
    with patch.object(commands_mod, "_model_completion_catalog", return_value=catalog):
        texts = _texts("hrms")
    assert texts == ["nous/hermes-4"]


def test_refresh_flag_rebuilds_catalog():
    calls: list[bool] = []

    def catalog(*, refresh: bool = False):
        calls.append(refresh)
        return {"providers": [], "models": []}

    with patch.object(commands_mod, "_model_completion_catalog", side_effect=catalog):
        _texts("")
        _texts("--refresh ")
    assert calls == [False, True]


def test_catalog_memo_busts_on_refresh_and_config_sig():
    commands_mod._invalidate_model_completion_catalog()
    payload = {"providers": [{"slug": "nous", "name": "Nous", "models": ["hermes-4"]}]}
    with (
        patch.object(commands_mod, "_model_catalog_sig", return_value=("cfg", 1, 1)),
        patch("hermes_cli.inventory.build_models_payload", return_value=payload) as build,
        patch("hermes_cli.inventory.load_picker_context", return_value=object()),
    ):
        commands_mod._model_completion_catalog()
        commands_mod._model_completion_catalog()
        assert build.call_count == 1
        commands_mod._model_completion_catalog(refresh=True)
        assert build.call_count == 2
    commands_mod._invalidate_model_completion_catalog()
    with (
        patch.object(commands_mod, "_model_catalog_sig", side_effect=[("cfg", 1, 1), ("cfg", 2, 1)]),
        patch("hermes_cli.inventory.build_models_payload", return_value=payload) as build,
        patch("hermes_cli.inventory.load_picker_context", return_value=object()),
    ):
        commands_mod._model_completion_catalog()
        commands_mod._model_completion_catalog()
        assert build.call_count == 2
