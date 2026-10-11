"""Native CLI picker uses the preset keyword and persists mode independently of the model ID."""
import json
from types import SimpleNamespace
from unittest.mock import Mock

from cli import HermesCLI

PAIR = {"lab": {"base_url": "http://lab.invalid/v1", "opusplan": {"plan": "planner", "exec": "worker"}}}


def test_cli_picker_offers_opusplan_and_dispatches_exact_keyword(monkeypatch):
    monkeypatch.setattr("providers.get_provider_profile", lambda p: None)
    monkeypatch.setattr("hermes_cli.models_validate.offered_model_ids", lambda models, *a: models)
    cli = object.__new__(HermesCLI)
    cli._model_picker_state = {"stage": "provider", "selected": 0, "providers": [
        {"slug": "lab", "models": ["planner", "worker"]}], "user_provs": PAIR}
    cli._invalidate = Mock()
    cli._handle_model_picker_selection()
    assert cli._model_picker_state["model_list"] == ["opusplan", "planner", "worker"]
    result = SimpleNamespace(success=True, new_model="planner", opusplan=True)
    pick = Mock(return_value=result)
    monkeypatch.setattr("hermes_cli.cli_model_switch_mixin._switch_model_from", pick)
    monkeypatch.setattr("hermes_cli.cli_model_switch_mixin._picker_offers_reasoning", lambda *a: False)
    cli._commit_picker_result = Mock()
    cli._handle_model_picker_selection()
    assert pick.call_args.args[1] == "opusplan"
    assert pick.call_args.kwargs["explicit_provider"] == "lab"
    cli._commit_picker_result.assert_called_once_with(result, False)


def test_cli_mode_persist_and_resume_on_same_concrete_model():
    cli = object.__new__(HermesCLI)
    cli._session_db, cli.session_id = Mock(), "sid"
    result = SimpleNamespace(target_provider="lab", base_url="", api_mode="chat_completions", new_model="planner", opusplan=True)
    cli._persist_model_switch_to_session(result)
    patch = cli._session_db.patch_session_model_config.call_args.args[1]
    assert patch["opusplan"] is True and "api_key" not in patch
    cli.model, cli.provider = "planner", "lab"
    cli.agent = SimpleNamespace(opusplan_active=False)
    row = {"model": "planner", "model_config": json.dumps(patch)}
    cli._restore_session_model(row)
    assert cli._opusplan_active is True and cli.agent.opusplan_active is True
    result.opusplan = False
    cli._persist_model_switch_to_session(result)
    patch = cli._session_db.patch_session_model_config.call_args.args[1]
    cli._restore_session_model({"model": "planner", "model_config": patch})
    assert cli._opusplan_active is False and cli.agent.opusplan_active is False
