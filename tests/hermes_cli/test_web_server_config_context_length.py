"""The Settings API must not persist a ``model.context_length`` below the engine's floor.

Regression: the Settings number field accepted any positive value (a stray spinner
click produced ``1``). It was written to ``config.yaml`` unchanged, and because config
is read once per process, every session created afterwards failed its first turn on
``MINIMUM_CONTEXT_LENGTH`` while already-running sessions kept working — the failure
looked unrelated to the setting that caused it.
"""

import pytest
from fastapi import HTTPException

from agent.model_metadata import MINIMUM_CONTEXT_LENGTH
from hermes_cli import web_server_config as wsc


@pytest.fixture
def disk_model(monkeypatch):
    """``load_config()`` as the endpoint sees it: a model dict already on disk."""
    disk = {
        "model": {
            "default": "deepseek-flash",
            "provider": "deepseek",
            "base_url": "https://api.deepseek.com/v1",
            "context_length": 1_000_000,
        },
        "agent": {"max_turns": 80},
    }
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: disk)
    return disk


def test_below_minimum_is_rejected(disk_model):
    """1 is the value a stray wheel tick produces; it must never reach the disk."""
    with pytest.raises(HTTPException) as exc:
        wsc._denormalize_config_from_web({"model_context_length": 1})

    assert exc.value.status_code == 400
    assert "64,000" in exc.value.detail
    assert disk_model["model"]["context_length"] == 1_000_000  # on-disk value untouched


def test_value_just_below_floor_is_rejected(disk_model):
    with pytest.raises(HTTPException):
        wsc._denormalize_config_from_web({"model_context_length": MINIMUM_CONTEXT_LENGTH - 1})


def test_exactly_minimum_is_written(disk_model):
    out = wsc._denormalize_config_from_web({"model_context_length": MINIMUM_CONTEXT_LENGTH})

    assert out["model"]["context_length"] == MINIMUM_CONTEXT_LENGTH


def test_zero_removes_the_override(disk_model):
    """0 is the documented "use the detected window" sentinel — the key is dropped."""
    out = wsc._denormalize_config_from_web({"model_context_length": 0})

    assert "context_length" not in out["model"]


def test_omitted_key_leaves_disk_value_alone(disk_model):
    """A partial autosave that omits the field means "unchanged", never an explicit 0."""
    out = wsc._denormalize_config_from_web({"model": "deepseek-flash"})

    assert out["model"]["context_length"] == 1_000_000
