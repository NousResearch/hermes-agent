"""Tests that switch_model preserves config_context_length."""

from unittest.mock import MagicMock, patch

from agent.insights import InsightsEngine
from hermes_state import SessionDB
from run_agent import AIAgent
from agent.context_compressor import ContextCompressor


def _make_agent_with_compressor(config_context_length=None) -> AIAgent:
    """Build a minimal AIAgent with a context_compressor, skipping __init__."""
    agent = AIAgent.__new__(AIAgent)

    # Primary model settings
    agent.model = "primary-model"
    agent.provider = "openrouter"
    agent.base_url = "https://openrouter.ai/api/v1"
    agent.api_key = "sk-primary"
    agent.api_mode = "chat_completions"
    agent.client = MagicMock()
    agent.quiet_mode = True

    # Store config_context_length for later use in switch_model
    agent._config_context_length = config_context_length

    # Context compressor with primary model values
    compressor = ContextCompressor(
        model="primary-model",
        threshold_percent=0.50,
        base_url="https://openrouter.ai/api/v1",
        api_key="sk-primary",
        provider="openrouter",
        quiet_mode=True,
        config_context_length=config_context_length,
    )
    agent.context_compressor = compressor

    # For switch_model
    agent._primary_runtime = {}

    return agent


@patch("agent.model_metadata.get_model_context_length", return_value=131_072)
def test_switch_model_preserves_config_context_length(mock_ctx_len):
    """When switching models, config_context_length should be passed to get_model_context_length."""
    agent = _make_agent_with_compressor(config_context_length=32_768)

    assert agent.context_compressor.model == "primary-model"
    assert agent.context_compressor.context_length == 32_768  # From config override

    # Switch model
    agent.switch_model("new-model", "openrouter", api_key="sk-new", base_url="https://openrouter.ai/api/v1")

    # Verify get_model_context_length was called with config_context_length
    mock_ctx_len.assert_called_once()
    call_kwargs = mock_ctx_len.call_args.kwargs
    assert call_kwargs.get("config_context_length") == 32_768

    # Verify compressor was updated
    assert agent.context_compressor.model == "new-model"


def test_switch_model_without_config_context_length():
    """When switching models without config override, config_context_length should be None."""
    agent = _make_agent_with_compressor(config_context_length=None)

    with patch("agent.model_metadata.get_model_context_length", return_value=128_000) as mock_ctx_len:
        # Switch model
        agent.switch_model("new-model", "openrouter", api_key="sk-new", base_url="https://openrouter.ai/api/v1")

        # Verify get_model_context_length was called with None
        mock_ctx_len.assert_called_once()
        call_kwargs = mock_ctx_len.call_args.kwargs
        assert call_kwargs.get("config_context_length") is None


@patch("agent.model_metadata.get_model_context_length", return_value=128_000)
@patch("hermes_cli.timeouts.get_provider_request_timeout", return_value=None)
def test_switch_model_updates_active_session_runtime_metadata(_mock_timeout, _mock_ctx_len, tmp_path):
    """Explicit /model switches should overwrite stale session metadata for /insights."""
    session_db = SessionDB(db_path=tmp_path / "state.db")
    session_db.create_session(
        session_id="sess-1",
        source="telegram",
        model="google/gemini-3.1-flash-lite-pre",
    )
    session_db.update_token_counts(
        "sess-1",
        input_tokens=1000,
        output_tokens=500,
        billing_provider="gmi",
        billing_base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    )

    agent = _make_agent_with_compressor(config_context_length=None)
    agent.session_id = "sess-1"
    agent._session_db = session_db

    agent.switch_model(
        "glm-5.1",
        "zai",
        api_key="sk-zai",
        base_url="https://api.z.ai/api/paas/v4",
    )

    row = session_db.get_session("sess-1")
    assert row is not None
    assert row["model"] == "glm-5.1"
    assert row["billing_provider"] == "zai"
    assert row["billing_base_url"] == "https://api.z.ai/api/paas/v4"

    report = InsightsEngine(session_db).generate(days=30)
    assert report["models"][0]["model"] == "glm-5.1"
