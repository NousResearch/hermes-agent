import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from agent.insights import InsightsEngine
from agent.usage_pricing import CanonicalUsage
from agent.workflow_usage import (
    WorkflowUsageTracker,
    model_decision_reason,
    normalize_reasoning_level,
)
from hermes_state import SessionDB
from run_agent import AIAgent


def test_explicit_model_choice_is_recorded_as_override():
    assert model_decision_reason("gpt-6-sol", explicit_override=True) == (
        "user-selected model gpt-6-sol retained"
    )


def test_model_fallback_reason_is_recorded():
    tracker = WorkflowUsageTracker("turn-fallback")
    tracker.note_model_escalation("fallback gpt-6-luna after provider failure")
    tracker.finish(outcome="completed", decision_reason="primary model selected")
    assert tracker._finished["escalation_reason"] == (
        "fallback gpt-6-luna after provider failure"
    )


def test_reasoning_configuration_is_reported_as_its_level():
    assert normalize_reasoning_level(
        {"reasoning": {"effort": "low", "summary": "auto"}},
        fallback="medium",
    ) == "low"
    assert normalize_reasoning_level(
        {}, fallback="{'effort': 'low', 'summary': 'auto'}"
    ) == "low"


def test_completed_workflow_usage_is_grouped_and_unknowns_stay_unknown(tmp_path):
    db = SessionDB(db_path=tmp_path / "workflow.db")
    try:
        db.create_session("workflow-session", source="cli", model="small-model")
        start = time.time() - 20
        tracker = WorkflowUsageTracker("turn-123", started_at=start)
        tracker.record_attempt(
            request_id="turn-123:api:1", model="small-model", provider="local",
            reasoning_level="low", duration_seconds=0.5,
        )
        tracker.record_attempt(
            request_id="turn-123:api:1", model="small-model", provider="local",
            reasoning_level="low", duration_seconds=1.0,
        )
        tracker.record_response(
            request_id="turn-123:api:1",
            usage=CanonicalUsage(
                input_tokens=100, output_tokens=25, cache_read_tokens=10,
                cache_write_tokens=4, reasoning_tokens=6,
            ),
            estimated_cost_usd=0.002,
            actual_provider_cost_usd=None,
        )
        tracker.record_attempt(
            request_id="turn-123:api:2", model="small-model", provider="local",
            reasoning_level="low", duration_seconds=2.0,
        )
        tracker.record_response(
            request_id="turn-123:api:2",
            usage=CanonicalUsage(input_tokens=50, output_tokens=12),
            estimated_cost_usd=None,
            actual_provider_cost_usd=None,
        )
        tracker.finish(
            outcome="completed", ended_at=start + 10,
            decision_reason="configured active model",
            escalation_reason=None,
        )
        tracker.persist(db, "workflow-session")

        report = InsightsEngine(db).generate(days=30)
        workflow = report["workflows"][0]
        assert workflow["workflow_id"] == "turn-123"
        assert workflow["outcome"] == "completed"
        assert workflow["requests"] == 3
        assert workflow["retries"] == 1
        assert workflow["runtime_seconds"] == 3.5
        assert workflow["wall_runtime_seconds"] == 10
        assert workflow["routes"] == [{
            "model": "small-model",
            "provider": "local",
            "reasoning_level": "low",
            "requests": 3,
            "retries": 1,
            "runtime_seconds": 3.5,
            "tokens": {
                "input": "UNKNOWN",
                "output": "UNKNOWN",
                "cache_read": "UNKNOWN",
                "cache_write": "UNKNOWN",
                "reasoning": "UNKNOWN",
            },
            "estimated_cost_usd": "UNKNOWN",
            "estimated_cost_status": "UNKNOWN",
            "actual_provider_cost_usd": "UNKNOWN",
        }]
        assert "UNKNOWN" in InsightsEngine(db).format_terminal(report)
        assert "Finalized workflows" in InsightsEngine(db).format_terminal(report)
        assert "actual provider cost=UNKNOWN" in InsightsEngine(db).format_gateway(report)
        assert all(row["model"] != "__workflow_ledger__" for row in report["models"])
    finally:
        db.close()


def test_real_turn_persists_request_and_response_into_insights(tmp_path, monkeypatch):
    """The extracted call/usage phases must feed the same turn ledger, not only its unit API."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    db = SessionDB(db_path=tmp_path / "turn.db")
    with (
        patch("model_tools.get_tool_definitions", return_value=[]),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI", return_value=MagicMock()),
    ):
        agent = AIAgent(
            api_key="test-key", base_url="https://api.openai.com/v1", provider="openai-api",
            api_mode="chat_completions", model="gpt-6-luna", session_id="usage-turn", platform="cli", quiet_mode=True,
            skip_context_files=True, skip_memory=True,
        )
    agent._session_db = db
    setattr(agent, "_disable_streaming", True)
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="Done", tool_calls=None), finish_reason="stop")],
        model="gpt-6-luna", usage=SimpleNamespace(prompt_tokens=110, completion_tokens=8),
    )
    try:
        with (
            patch.object(agent, "_interruptible_api_call", return_value=response),
            patch("agent.title_generator.maybe_auto_title", return_value=None),
        ):
            result = agent.run_conversation("Verify this")
        assert result["completed"] is True
        report = InsightsEngine(db).generate()
        workflow = report["workflows"][0]
        assert workflow["outcome"] == "completed"
        assert workflow["requests"] == 1
        assert workflow["routes"][0]["tokens"]["input"] == 110
        assert workflow["routes"][0]["tokens"]["output"] == 8
        assert workflow["routes"][0]["actual_provider_cost_usd"] == "UNKNOWN"
        assert all(model["total_tokens"] <= 118 for model in report["models"])
    finally:
        agent.close()
        db.close()
