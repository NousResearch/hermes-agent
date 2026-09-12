"""Recovery uses a real settled account; I/O failure never discards the active result."""
from types import SimpleNamespace
from unittest.mock import Mock


def test_promotion_rehomes_a_live_welcome_agent_before_next_inference(tmp_path, monkeypatch):
    from agent.free_tier import admit_turn
    from hermes_cli import auth, auth_nous
    monkeypatch.setenv("HERMES_SHARED_AUTH_DIR", str(tmp_path / "shared-auth"))
    auth._save_active_provider_state("nous", {"auth_method": "oauth", "access_token": "fixture", "refresh_token": "refresh", "agent_key": "fixture", "inference_base_url": "https://inference-api.nousresearch.com/v1"})
    agent = SimpleNamespace(base_url="https://welcome-api.nousresearch.com/v1", model="nous/welcome", provider="nous")
    def switch(**kwargs):
        agent.base_url = kwargs["base_url"]
    agent.switch_model = Mock(side_effect=switch)
    with admit_turn(agent) as blocked:
        assert blocked is None
        assert agent.base_url == "https://inference-api.nousresearch.com/v1"
    agent.switch_model.assert_called_once()


def test_failed_usage_write_preserves_current_result_and_refuses_next(tmp_path, monkeypatch):
    from agent.free_tier import admit_turn, record_tool_completion, finish_turn
    from hermes_cli import auth, free_tier_usage as usage
    monkeypatch.setenv("HERMES_SHARED_AUTH_DIR", str(tmp_path / "shared-auth"))
    auth._save_active_provider_state("nous", {"auth_method": "anonymous", "anon_token": "anon_io_fixture"})
    agent = SimpleNamespace(base_url="https://welcome-api.nousresearch.com/v1")
    with admit_turn(agent):
        monkeypatch.setattr(auth, "_write_private_file_atomic", Mock(side_effect=OSError("disk full")))
        record_tool_completion(agent)
        result = finish_turn(agent, {"completed": True, "messages": [], "final_response": "actual result"})
        assert result["completed"] and result["final_response"].startswith("actual result")
    # A fresh agent on the same identity must see the failure latch too.
    sibling = SimpleNamespace(base_url=agent.base_url)
    with admit_turn(sibling) as blocked:
        assert blocked["completed"] is False
    assert usage.status()["capped"] is True
