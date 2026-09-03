"""Tests for hermes_cli/codex_models.py — Codex model catalog helpers."""


def test_get_codex_model_returns_list():
    from hermes_cli.codex_models import _get_codex_models
    models = _get_codex_models()
    assert isinstance(models, list)
