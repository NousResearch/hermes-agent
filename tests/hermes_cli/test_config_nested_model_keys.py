"""Regression for #153296.

A whole-file save (``save_config(load_config())``) flattens a dict-valued
``model.default`` / ``model.model`` / ``model.name`` to a string id. Unknown
keys nested in that dict must survive, lifted onto ``model`` next to the
promoted id and provider. A key already set on ``model`` is not overwritten.
"""

import os
from pathlib import Path
from unittest.mock import patch

import pytest
import hermes_yaml as yaml

from hermes_cli.config import load_config, save_config

_NESTED = "{model: x/y, provider: openrouter, newer_field: keepme}"


def _save_loaded(home: Path) -> dict:
    with patch.dict(os.environ, {"HERMES_HOME": str(home)}):
        loaded = load_config()
        assert isinstance(loaded["model"]["default"], str)
        assert loaded["model"]["default"] == "x/y"
        save_config(loaded)
        text = (home / "config.yaml").read_text(encoding="utf-8")
    return yaml.safe_load(text)


@pytest.mark.parametrize("source_key", ["default", "model", "name"])
def test_save_config_keeps_unknown_key_nested_in_model_id(tmp_path, source_key):
    """The reported config, saved through ``save_config``, still has ``newer_field``."""
    (tmp_path / "config.yaml").write_text(
        f"_config_version: 53\nmodel:\n  {source_key}: {_NESTED}\n",
        encoding="utf-8",
    )
    raw = _save_loaded(tmp_path)
    model = raw["model"]
    assert model["default"] == "x/y"
    assert model["provider"] == "openrouter"
    assert model["newer_field"] == "keepme"
    if source_key != "default":
        assert source_key not in model


@pytest.mark.parametrize("source_key", ["default", "model", "name"])
def test_save_config_does_not_clobber_existing_model_key(tmp_path, source_key):
    """A key already set on ``model`` wins over the same key nested in the id."""
    nested = (
        "{model: x/y, provider: openrouter, newer_field: keepme, fresh_field: lifted}"
    )
    (tmp_path / "config.yaml").write_text(
        "_config_version: 53\n"
        "model:\n"
        "  provider: anthropic\n"
        "  newer_field: already\n"
        f"  {source_key}: {nested}\n",
        encoding="utf-8",
    )
    raw = _save_loaded(tmp_path)
    model = raw["model"]
    assert model["default"] == "x/y"
    assert model["provider"] == "anthropic"
    assert model["newer_field"] == "already"
    assert model["fresh_field"] == "lifted"
