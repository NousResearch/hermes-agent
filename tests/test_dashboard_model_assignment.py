import asyncio
from unittest.mock import patch

from hermes_cli import web_server
from hermes_cli.web_server import ModelAssignment, set_model_assignment


def test_auxiliary_assignment_clears_stale_direct_endpoint_fields():
    cfg = {
        "auxiliary": {
            "vision": {
                "provider": "custom",
                "model": "old-model",
                "base_url": "https://stale.example/v1",
                "api_key": "STALE_SECRET",
                "api_mode": "chat_completions",
                "timeout": 30,
            }
        }
    }
    saved = {}

    with patch("hermes_cli.web_server.load_config", return_value=cfg), patch(
        "hermes_cli.web_server.save_config", side_effect=lambda value: saved.update(value)
    ):
        result = asyncio.run(
            set_model_assignment(
                ModelAssignment(
                    scope="auxiliary",
                    task="vision",
                    provider="openrouter",
                    model="google/gemini-3-flash-preview",
                )
            )
        )

    assert result["ok"] is True
    slot = saved["auxiliary"]["vision"]
    assert slot["provider"] == "openrouter"
    assert slot["model"] == "google/gemini-3-flash-preview"
    assert slot["base_url"] == ""
    assert slot["api_key"] == ""
    assert slot["api_mode"] == ""
    assert slot["timeout"] == 30


def test_auxiliary_reset_clears_stale_direct_endpoint_fields_for_all_slots():
    cfg = {
        "auxiliary": {
            slot: {
                "provider": "custom",
                "model": "old-model",
                "base_url": "https://stale.example/v1",
                "api_key": "STALE_SECRET",
                "api_mode": "chat_completions",
                "extra_body": {"keep": True},
            }
            for slot in web_server._AUX_TASK_SLOTS
        }
    }
    saved = {}

    with patch("hermes_cli.web_server.load_config", return_value=cfg), patch(
        "hermes_cli.web_server.save_config", side_effect=lambda value: saved.update(value)
    ):
        result = asyncio.run(
            set_model_assignment(
                ModelAssignment(scope="auxiliary", task="__reset__", provider="", model="")
            )
        )

    assert result == {"ok": True, "scope": "auxiliary", "reset": True}
    for slot in web_server._AUX_TASK_SLOTS:
        slot_cfg = saved["auxiliary"][slot]
        assert slot_cfg["provider"] == "auto"
        assert slot_cfg["model"] == ""
        assert slot_cfg["base_url"] == ""
        assert slot_cfg["api_key"] == ""
        assert slot_cfg["api_mode"] == ""
        assert slot_cfg["extra_body"] == {"keep": True}
