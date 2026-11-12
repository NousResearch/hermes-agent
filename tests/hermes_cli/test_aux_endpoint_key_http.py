"""Dashboard ``POST /api/model/set``: a write-locked ``.env`` refuses an auxiliary endpoint key
with its own reason, not a generic 500 (#131016)."""

from __future__ import annotations

import asyncio

import pytest
from fastapi import HTTPException

from hermes_cli.config import get_config_path
from hermes_cli.web_models import ModelAssignment
from hermes_cli.web_routers.models import set_model_assignment

SECRET = "sk-aux...CRET"


@pytest.mark.parametrize("task", ["vision", ""], ids=["one task", "all tasks"])
def test_managed_env_refusal_reaches_the_dashboard(monkeypatch, task):
    from hermes_cli import config
    get_config_path().write_text("model:\n  provider: openrouter\n  default: x/y\n")
    monkeypatch.setattr(config, "env_write_refusal", lambda key, action: f"Cannot {action} {key}: managed")

    body = ModelAssignment(scope="auxiliary", task=task, provider="custom", model="qwen3:8b",
                           base_url="http://127.0.0.1:11434/v1", api_key=SECRET,
                           confirm_expensive_model=True)
    with pytest.raises(HTTPException) as refusal:
        asyncio.run(set_model_assignment(body))

    assert refusal.value.status_code == 400
    assert refusal.value.detail.startswith("Cannot set HERMES_CUSTOM_AUX_")
    assert refusal.value.detail.endswith(": managed")
    assert SECRET not in get_config_path().read_text()
    assert "HERMES_CUSTOM_AUX" not in get_config_path().read_text()
