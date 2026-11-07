"""Fallback review contracts: a failed outage probe returns to the backup at once, only the real
primary route re-benches the primary, and effort bookkeeping belongs to the route that owns it."""

from types import SimpleNamespace
from unittest.mock import patch

from agent.error_classifier import FailoverReason
from agent.fallback_cooldown import _leaving_primary


def test_failed_outage_probe_skips_the_healthy_pool_wait():
    from agent.turn_recovery import _pool_blocks_eager_fallback

    agent = SimpleNamespace(_primary_probe_pending=True, _credential_pool=object())
    with patch("run_agent._pool_may_recover_from_rate_limit", return_value=True, create=True), \
            patch("agent.conversation_loop._ra", return_value=SimpleNamespace(
                _pool_may_recover_from_rate_limit=lambda pool: True)):
        assert _pool_blocks_eager_fallback(agent, FailoverReason.overloaded) is False
        agent._primary_probe_pending = False
        assert _pool_blocks_eager_fallback(agent, FailoverReason.rate_limit) is True


def test_a_fallback_on_the_primarys_provider_is_not_the_primary():
    primary = {"provider": "openrouter", "model": "primary/model"}
    on_backup = SimpleNamespace(_fallback_activated=True, provider="openrouter", model="backup/model",
                                _primary_runtime=primary)
    assert _leaving_primary(on_backup) is False
    on_primary = SimpleNamespace(_fallback_activated=True, provider="openrouter", model="primary/model",
                                 _primary_runtime=primary)
    assert _leaving_primary(on_primary) is True


def test_shared_snapshot_installer_leaves_fallback_bookkeeping_alone():
    """A voice route restore must not consume the surrounding fallback's bound effort."""
    from agent.route_binding import reinstall_runtime_snapshot

    agent = SimpleNamespace(reasoning_config={"effort": "low"}, _fallback_bound_reasoning={"effort": "low"})
    rt = {"reasoning_config": {"effort": "low"}, "model": "m", "provider": "p"}
    with patch("agent.agent_runtime_helpers._apply_primary_runtime_fields"), \
            patch("agent.turn_recovery_codex.reset_codex_reasoning_replay"), \
            patch("agent.agent_runtime_helpers._restore_runtime_capabilities"), \
            patch("agent.agent_runtime_helpers._rebuild_primary_client"), \
            patch("agent.chat_completion_helpers._reset_stale_streak"), \
            patch("agent.chat_completion_helpers.rewrite_prompt_model_identity"):
        agent.api_mode, agent.provider = "chat_completions", "p"
        agent.context_compressor = SimpleNamespace(update_model=lambda **k: None)
        rt.update(use_prompt_caching=False, compressor_model="m", compressor_context_length=1,
                  compressor_base_url="", compressor_api_key="", compressor_provider="p")
        reinstall_runtime_snapshot(agent, rt)
    assert agent._fallback_bound_reasoning == {"effort": "low"}



def test_a_backup_rate_limit_keeps_the_primary_probe_owed():
    from agent.fallback_cooldown import arm_primary_cooldown

    agent = SimpleNamespace(_fallback_activated=True, provider="zai", model="glm-5.2",
                            _primary_runtime={"provider": "openrouter", "model": "primary/model"},
                            _primary_outage_cooldown=True, _primary_probe_pending=False,
                            _rate_limited_until=0)
    arm_primary_cooldown(agent, FailoverReason.rate_limit)
    assert agent._primary_outage_cooldown is True


def test_a_codex_response_failed_outage_benches_the_primary():
    """A Codex HTTP-200 ``status=failed`` server error fails over like a raised 5xx: the backup
    holds through the outage cooldown instead of handing the next turn back to the dead primary."""
    from agent.turn_response_check import retry_invalid_response
    from agent.turn_retry_state import TurnRetryState
    from tests.agent.test_fallback_outage_cooldown import _agent, _fallback_client, _restore_at

    agent = _agent()
    agent.api_mode, agent.provider = "codex_responses", "openai-codex"
    failed = SimpleNamespace(status="failed", output=[], output_text="", error=SimpleNamespace(
        code="server_error", message="An error occurred while processing your request."))
    with patch("agent.auxiliary_client.resolve_provider_client", return_value=(_fallback_client(), "glm-5.2")), \
            patch("agent.fallback_cooldown.time.monotonic", return_value=1000.0), \
            patch("hermes_cli.config.load_config", return_value={}):
        verdict = retry_invalid_response(
            agent, response=failed, error_details=["response.status=failed"], _retry=TurnRetryState(),
            thinking_spinner=None, messages=[], api_messages=[], api_kwargs=None, active_system_prompt=None,
            conversation_history=None, retry_count=0, max_retries=3, compression_attempts=0, api_call_count=1,
            api_request_id="r", api_start_time=0.0, api_duration=0.4, effective_task_id="t", turn_id="turn")
    assert verdict.action == "break" and agent.model == "glm-5.2"
    assert _restore_at(agent, 1000.0 + 120)[0] is False
    assert agent.model == "glm-5.2"
