"""The legacy external-provider request keeps schema, initialization and session validation."""
import pytest

from tests.fakes.providers.fake_acp_agent import AcpFake, Agent


@pytest.mark.parametrize("initialized,params,expected", [
    (False, {"sessionId": "known", "modelId": "model"}, "request before initialize"),
    (True, {"sessionId": "unknown", "modelId": "model"}, "unknown sessionId"),
    (True, {"sessionId": "known"}, "schema:"),
    (True, {"sessionId": "known", "modelId": 42}, "schema:"),
    (True, {"sessionId": "known", "modelId": "model", "extra": True}, "custom root field"),
    (True, {"sessionId": "known", "modelId": "model", "_meta": {"client": "legacy"}}, None),
])
def test_legacy_model_request_validation(tmp_path, initialized, params, expected):
    scenario = AcpFake(tmp_path, [])
    agent = Agent(scenario.state_dir)
    agent.initialized = initialized
    agent.sessions.add("known")
    errors = agent.validate({"jsonrpc": "2.0", "id": 1, "method": "session/set_model", "params": params})
    if expected is None:
        assert errors == []
    else:
        assert any(expected in error for error in errors), errors
