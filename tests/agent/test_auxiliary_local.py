"""Shared auxiliary.local preset fills empty per-task fields only."""

from unittest.mock import patch

from agent.auxiliary_client import _get_auxiliary_task_config, _resolve_task_provider_model


def test_local_preset_fills_empty_listed_task():
    config = {
        "auxiliary": {
            "local": {
                "base_url": "http://127.0.0.1:11434/v1",
                "model": "qwen3.5:2b",
                "tasks": ["compression", "memory_query_rewrite"],
            },
            "compression": {"provider": "auto", "model": "", "base_url": ""},
        }
    }
    with patch("hermes_cli.config.load_config_readonly", return_value=config):
        task = _get_auxiliary_task_config("compression")
        assert task["base_url"] == "http://127.0.0.1:11434/v1"
        assert task["model"] == "qwen3.5:2b"
        provider, model, base_url, _api_key, _mode = _resolve_task_provider_model(
            task="compression"
        )
        assert provider == "custom"
        assert model == "qwen3.5:2b"
        assert base_url == "http://127.0.0.1:11434/v1"


def test_per_task_values_win_over_local_preset():
    config = {
        "auxiliary": {
            "local": {
                "base_url": "http://127.0.0.1:11434/v1",
                "model": "qwen3.5:2b",
                "tasks": ["compression"],
            },
            "compression": {
                "provider": "auto",
                "model": "kept-model",
                "base_url": "http://127.0.0.1:8080/v1",
            },
        }
    }
    with patch("hermes_cli.config.load_config_readonly", return_value=config):
        provider, model, base_url, _api_key, _mode = _resolve_task_provider_model(
            task="compression"
        )
        assert provider == "custom"
        assert model == "kept-model"
        assert base_url == "http://127.0.0.1:8080/v1"


def test_unlisted_or_empty_tasks_leave_slot_unchanged():
    config = {
        "auxiliary": {
            "local": {
                "base_url": "http://127.0.0.1:11434/v1",
                "model": "qwen3.5:2b",
                "tasks": [],
            },
            "compression": {"provider": "auto", "model": "", "base_url": ""},
            "title_generation": {"provider": "auto", "model": "", "base_url": ""},
        }
    }
    with patch("hermes_cli.config.load_config_readonly", return_value=config):
        compression = _get_auxiliary_task_config("compression")
        assert not str(compression.get("base_url") or "").strip()
        config["auxiliary"]["local"]["tasks"] = ["title_generation"]
        title = _get_auxiliary_task_config("title_generation")
        other = _get_auxiliary_task_config("compression")
        assert title["base_url"] == "http://127.0.0.1:11434/v1"
        assert not str(other.get("base_url") or "").strip()


def test_task_base_url_without_api_key_resolves_custom():
    """Per-task local endpoints must not require a dummy API key."""
    config = {
        "auxiliary": {
            "compression": {
                "provider": "auto",
                "model": "qwen3:8b",
                "base_url": "http://localhost:11434/v1",
            },
        }
    }
    with patch("hermes_cli.config.load_config_readonly", return_value=config):
        provider, model, base_url, api_key, _mode = _resolve_task_provider_model(
            task="compression"
        )
        assert provider == "custom"
        assert model == "qwen3:8b"
        assert base_url == "http://localhost:11434/v1"
        assert not api_key
