"""Exercise the real initializer's runtime binding publication."""
from types import SimpleNamespace
from agent import agent_init, auxiliary_client
from agent.turn_context import _publish_runtime_main


def test_session_init_retains_pin_for_tools(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    credential = SimpleNamespace(id="entry-a", account_id="account-a")
    agent = SimpleNamespace(max_iterations=4, provider="openai-codex", model="gpt-5",
                            _expiry_aware_session_credential=credential)
    agent_init._init_session_state(agent, "session-a", None, None, None, None,
                                  False, 1, 1, 1)
    binding = {"provider": "openai-codex", "entry_id": "entry-a", "account_id": "account-a"}
    assert agent._session_init_model_config["credential_binding"] == binding
    with auxiliary_client.scoped_runtime_main({}):
        _publish_runtime_main(agent)
        assert auxiliary_client._normalize_main_runtime(None)["credential_binding"] == binding
