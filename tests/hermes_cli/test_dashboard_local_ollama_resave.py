"""Regression tests for #76324 — dashboard re-save of a local Ollama model
must not wipe ``model.base_url`` / ``model.api_mode``.

The CLI (``hermes gateway setup``) writes a local endpoint as::

    model:
      default: gemma4:latest
      provider: custom
      base_url: http://10.0.0.155:11434/v1
      api_mode: chat_completions

The dashboard picker resolves that same endpoint to its canonical
``custom:<name>`` slug and re-saves it via ``POST /api/model/set`` with an
empty ``base_url``. Before this fix the backend treated the bare->named
provider rename as a provider switch and wiped ``base_url`` + ``api_mode``.
"""

from unittest.mock import patch

import yaml

from hermes_cli.web_server import (
    _apply_main_model_assignment,
    _apply_model_assignment_sync,
)


def _load_config(config: dict):
    return patch("hermes_cli.web_server.load_config", return_value=config)


def _save_config_spy(sink: dict):
    def _save(cfg):
        sink.clear()
        sink.update(cfg)

    return patch("hermes_cli.web_server.save_config", side_effect=_save)


# ---------------------------------------------------------------------------
# _apply_main_model_assignment — endpoint preservation on bare->named rename
# ---------------------------------------------------------------------------


def test_bare_custom_to_named_custom_same_endpoint_preserves_base_url_and_api_mode():
    """Re-saving ``custom`` (CLI form) as ``custom:local-ollama`` with the
    same base_url must NOT wipe base_url/api_mode (#76324)."""
    model_cfg = {
        "default": "gemma4:latest",
        "provider": "custom",
        "base_url": "http://10.0.0.155:11434/v1",
        "api_mode": "chat_completions",
    }
    out = _apply_main_model_assignment(
        model_cfg, "custom:local-ollama", "gemma4:latest", "http://10.0.0.155:11434/v1", ""
    )
    assert out["provider"] == "custom"
    assert out["base_url"] == "http://10.0.0.155:11434/v1"
    assert out["api_mode"] == "chat_completions"


def test_bare_custom_to_named_custom_no_base_url_in_request_preserves_endpoint():
    """The dashboard sends no base_url at all; the current endpoint is the
    only durable fact, so it must be preserved (#76324)."""
    model_cfg = {
        "default": "gemma4:latest",
        "provider": "custom",
        "base_url": "http://10.0.0.155:11434/v1",
        "api_mode": "chat_completions",
    }
    out = _apply_main_model_assignment(model_cfg, "custom:local-ollama", "gemma4:latest", "", "")
    assert out["provider"] == "custom"
    assert out["base_url"] == "http://10.0.0.155:11434/v1"
    assert out["api_mode"] == "chat_completions"


def test_named_custom_to_named_custom_same_endpoint_preserves_api_mode():
    """Already-named providers keep their endpoint on a same-endpoint re-pick."""
    model_cfg = {
        "default": "gemma4:latest",
        "provider": "custom:local-ollama",
        "base_url": "http://10.0.0.155:11434/v1",
        "api_mode": "chat_completions",
    }
    out = _apply_main_model_assignment(
        model_cfg, "custom:local-ollama", "gemma4:latest", "http://10.0.0.155:11434/v1", ""
    )
    assert out["provider"] == "custom:local-ollama"
    assert out["base_url"] == "http://10.0.0.155:11434/v1"
    assert out["api_mode"] == "chat_completions"


def test_switch_away_from_custom_still_clears_stale_endpoint():
    """A real provider switch must still drop the old custom endpoint."""
    model_cfg = {
        "default": "gemma4:latest",
        "provider": "custom",
        "base_url": "http://10.0.0.155:11434/v1",
        "api_mode": "chat_completions",
    }
    out = _apply_main_model_assignment(model_cfg, "anthropic", "claude-opus-4-6", "", "")
    assert out["provider"] == "anthropic"
    assert out["base_url"] == ""
    assert out.get("api_mode") is None


def test_switch_between_two_custom_endpoints_clears_old_url():
    """Switching custom endpoints (different URLs) is a real switch."""
    model_cfg = {
        "default": "gemma4:latest",
        "provider": "custom:local-ollama",
        "base_url": "http://10.0.0.155:11434/v1",
    }
    out = _apply_main_model_assignment(
        model_cfg, "custom:other", "other-model", "http://other-host:11434/v1", ""
    )
    assert out["provider"] == "custom:other"
    assert out["base_url"] == "http://other-host:11434/v1"


# ---------------------------------------------------------------------------
# _apply_model_assignment_sync — the full dashboard save path
# ---------------------------------------------------------------------------


def test_dashboard_resave_of_local_ollama_keeps_endpoint(tmp_path):
    """Full regression: config written by the CLI (bare custom + base_url +
    api_mode), then re-saved by the dashboard with the canonical
    ``custom:local-ollama`` slug and no base_url. The on-disk model block
    must keep provider=``custom``, base_url and api_mode (#76324)."""
    initial = {
        "model": {
            "default": "gemma4:latest",
            "provider": "custom",
            "base_url": "http://10.0.0.155:11434/v1",
            "api_mode": "chat_completions",
        },
        "providers": {
            "local-ollama": {
                "name": "Local Ollama",
                "base_url": "http://10.0.0.155:11434/v1",
                "model": "gemma4:latest",
            }
        },
    }
    saved: dict = {}

    with _load_config(dict(initial)), _save_config_spy(saved), patch(
        "hermes_cli.model_cost_guard.expensive_model_warning", lambda *a, **k: None
    ):
        result = _apply_model_assignment_sync(
            "main", "custom:local-ollama", "gemma4:latest", "", ""
        )

    assert result["ok"] is True
    model = saved["model"]
    assert model["provider"] == "custom"
    assert model["base_url"] == "http://10.0.0.155:11434/v1"
    assert model["api_mode"] == "chat_completions"
    assert model["default"] == "gemma4:latest"


def test_dashboard_resave_no_providers_entry_keeps_endpoint(tmp_path):
    """Same as above but without a ``providers:`` entry — the CLI-only form.
    The named slug does not resolve to a configured entry, so the current
    endpoint must be preserved rather than wiped (#76324)."""
    initial = {
        "model": {
            "default": "gemma4:latest",
            "provider": "custom",
            "base_url": "http://10.0.0.155:11434/v1",
            "api_mode": "chat_completions",
        }
    }
    saved: dict = {}

    with _load_config(dict(initial)), _save_config_spy(saved), patch(
        "hermes_cli.model_cost_guard.expensive_model_warning", lambda *a, **k: None
    ):
        result = _apply_model_assignment_sync(
            "main", "custom:local-ollama", "gemma4:latest", "", ""
        )

    assert result["ok"] is True
    model = saved["model"]
    assert model["provider"] == "custom"
    assert model["base_url"] == "http://10.0.0.155:11434/v1"
    assert model["api_mode"] == "chat_completions"


def test_dashboard_resave_reads_base_url_from_providers_entry(tmp_path):
    """When the dashboard sends no base_url but a ``providers:`` entry owns
    the endpoint, the entry's base_url is filled in (also fixes the empty
    ``base_url: ''`` the dashboard used to write)."""
    initial = {
        "model": {"default": "gemma4:latest", "provider": "custom:local-ollama"},
        "providers": {
            "local-ollama": {
                "name": "Local Ollama",
                "base_url": "http://10.0.0.155:11434/v1",
                "model": "gemma4:latest",
            }
        },
    }
    saved: dict = {}

    with _load_config(dict(initial)), _save_config_spy(saved), patch(
        "hermes_cli.model_cost_guard.expensive_model_warning", lambda *a, **k: None
    ):
        result = _apply_model_assignment_sync(
            "main", "custom:local-ollama", "gemma4:latest", "", ""
        )

    assert result["ok"] is True
    model = saved["model"]
    assert model["provider"] == "custom:local-ollama"
    assert model["base_url"] == "http://10.0.0.155:11434/v1"
