"""Legacy and native selectors expose readable model-first labels with stable route IDs."""
from types import SimpleNamespace

from acp_adapter.config_options import project_options
from acp_adapter.gateway_session_meta import model_state
from acp_adapter.server import HermesACPAgent
from acp_adapter.session import SessionManager
from tests.acp_adapter.test_acp_commands import NoopDb


def test_model_first_labels_keep_provider_identity_in_both_selectors(tmp_path, monkeypatch):
    monkeypatch.setattr("hermes_cli.inventory.build_models_payload", lambda *a, **kw: {"providers": [
        {"slug": "openai-codex", "name": "ChatGPT or Codex Subscription", "models": ["test-model"]},
        {"slug": "openrouter", "name": "OpenRouter", "models": ["test-model"]},
    ]})
    monkeypatch.setattr("acp_adapter.model_catalog._named_custom_provider_catalogs", lambda: [
        ("custom:relay", "Team Relay", [("test-model", "Private endpoint")])])
    info = {"model": "test-model", "provider": "openai-codex"}
    manager = SessionManager(agent_factory=lambda: SimpleNamespace(**info), db=NoopDb())
    agent = HermesACPAgent(session_manager=manager)
    state = manager.create_session(cwd=str(tmp_path))
    for models in (agent._build_model_state(state), model_state(info)):
        labels = {row.model_id: row.name for row in models.available_models}
        assert set(labels) == {"openai-codex:test-model", "openrouter:test-model", "custom:relay:test-model"}
        assert all(label.startswith("test-model · ") for label in labels.values())
        assert len(set(labels.values())) == len(labels)
        assert labels["openai-codex:test-model"] == "test-model · Codex"
        codex = next(row for row in models.available_models if row.model_id == "openai-codex:test-model")
        assert "ChatGPT or Codex Subscription" in codex.description
        assert "OpenRouter" in labels["openrouter:test-model"]
        assert "Team Relay" in labels["custom:relay:test-model"]
        assert models.current_model_id == "openai-codex:test-model"
        native = next(option for option in project_options(info, models) if option.id == "model")
        assert {row.value: row.name for row in native.options} == labels
        assert native.current_value == models.current_model_id

    # A current model absent from inventory still needs the same readable label.
    state.model = info["model"] = "unlisted-model"
    for models in (agent._build_model_state(state), model_state(info)):
        current = next(row for row in models.available_models if row.model_id == models.current_model_id)
        assert current.name.startswith("unlisted-model · ")
