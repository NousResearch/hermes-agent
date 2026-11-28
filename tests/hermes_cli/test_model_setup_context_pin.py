"""``hermes model`` drops a ``model.context_length`` pinned for the model/route it replaces.

The pin is scoped to the configured default (``config_context_length_for_runtime``), so a pin the
setup wizard carries over becomes the NEW model's window. ``/model --global`` already drops it
(``model_selection_config_updates``); the docs promise the same for every switch.
"""

from __future__ import annotations

import pytest
import hermes_yaml as yaml

_AZURE = ("model:\n  provider: azure-foundry\n  default: gpt-4.1\n"
          "  base_url: https://fake.openai.azure.com/openai/v1\n  api_mode: chat_completions\n"
          "  context_length: 1047576\n")


def _model_block(home) -> dict:
    return yaml.safe_load((home / "config.yaml").read_text(encoding="utf-8"))["model"]


def _pick_openrouter(model_id):
    from hermes_cli.model_setup_flows_common import _finish_model
    _finish_model(model_id, "openrouter", "done", base_url="https://openrouter.ai/api/v1", api_mode="chat_completions")


def _pick_codex(model_id):
    from hermes_cli.model_setup_flows_common import _activate_provider_model
    _activate_provider_model(model_id, "openai-codex", "https://chatgpt.com/backend-api/codex", "done")


@pytest.mark.parametrize("pick", [_pick_openrouter, _pick_codex], ids=["api-key-flow", "oauth-flow"])
def test_pin_follows_its_route_through_hermes_model(pick, tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text(_AZURE, encoding="utf-8")

    pick("gpt-4.1")  # same model id, another route
    assert "context_length" not in _model_block(tmp_path)

    cfg = yaml.safe_load((tmp_path / "config.yaml").read_text(encoding="utf-8"))
    cfg["model"]["context_length"] = 64000
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
    pick("gpt-4.1")  # same-route re-pick keeps the user's pin
    assert _model_block(tmp_path)["context_length"] == 64000

    pick("gpt-5.4-mini")
    assert "context_length" not in _model_block(tmp_path)


@pytest.mark.parametrize("detected", [128000, None])
def test_azure_pin_is_the_window_detected_for_the_new_route(detected, tmp_path, monkeypatch):
    from hermes_cli import azure_detect
    from hermes_cli import model_setup_flows_azure as az

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text(
        "model:\n  provider: openrouter\n  default: qwen/qwen3-32b\n  base_url: https://openrouter.ai/api/v1\n"
        "  context_length: 40960\n", encoding="utf-8")
    answers = iter(["https://other.openai.azure.com/openai/v1", "1", "sk-azure-fake"])
    monkeypatch.setattr(az, "_ask", lambda *a, **k: next(answers))
    monkeypatch.setattr(az, "_azure_detect_transport", lambda *a: ("chat_completions", ["gpt-4o"]))
    monkeypatch.setattr(az, "_azure_pick_model", lambda *a: "gpt-4o")
    monkeypatch.setattr(azure_detect, "lookup_context_length", lambda *a, **k: detected)

    az._model_flow_azure_foundry({})

    block = _model_block(tmp_path)
    assert (block["provider"], block["default"]) == ("azure-foundry", "gpt-4o")
    assert block.get("context_length") == detected
