"""Gateway DAG context guard regression tests."""

from collections import OrderedDict
from unittest.mock import patch

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import MessageEvent
from gateway.session import SessionSource, SessionStore
from agent.context_dag_reconcile import reconcile_full_transcript
from agent.context_dag_store import ContextDAGStore
from hermes_state import SessionDB


def _make_gateway(tmp_path):
    from gateway.run import GatewayRunner

    db = SessionDB(db_path=tmp_path / "state.db")
    config = GatewayConfig(
        platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="test-token")},
        sessions_dir=tmp_path / "sessions",
    )
    store = SessionStore(tmp_path / "sessions", config=config)
    store._db = db
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="chat",
        user_id="user",
        chat_type="dm",
    )
    entry = store.get_or_create_session(source)

    runner = GatewayRunner.__new__(GatewayRunner)
    runner.session_store = store
    runner.config = config
    runner._session_db = db
    runner._agent_cache = OrderedDict()
    runner._running_agents = {}
    runner.adapters = {Platform.TELEGRAM: object()}
    runner._session_model_overrides = {}
    runner._resolve_session_agent_runtime = lambda **kwargs: (
        "test-model",
        {
            "api_key": "test-key-1234567890",
            "base_url": "https://openrouter.ai/api/v1",
            "provider": "openrouter",
        },
    )

    event = MessageEvent(text="/compress", source=source, message_id="m1")
    return runner, store, entry, event


@pytest.mark.asyncio
async def test_gateway_compress_uses_legacy_compressor_when_dag_not_gateway_enabled(tmp_path):
    runner, store, entry, event = _make_gateway(tmp_path)
    store.append_to_transcript(entry.session_id, {"role": "user", "content": "one"})
    store.append_to_transcript(entry.session_id, {"role": "assistant", "content": "two"})
    store.append_to_transcript(entry.session_id, {"role": "user", "content": "three"})
    store.append_to_transcript(entry.session_id, {"role": "assistant", "content": "four"})

    cfg = {"context": {"engine": "dag"}, "agent": {}, "compression": {"enabled": True}}

    with (
        patch("hermes_cli.config.load_config", return_value=cfg),
        patch("agent.context_compressor.ContextCompressor.has_content_to_compress", return_value=True),
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
        patch("agent.model_metadata.get_model_context_length", return_value=131_072),
        patch("run_agent.AIAgent._compress_context", return_value=([{"role": "user", "content": "compressed"}], "")) as compress_mock,
        patch.object(store, "rewrite_transcript", wraps=store.rewrite_transcript) as rewrite_mock,
    ):
        result = await runner._handle_compress_command(event)

    assert "compressed" in result.lower()
    compress_mock.assert_called_once()
    rewrite_mock.assert_called_once()


def test_gateway_agent_session_kwargs_make_dag_hygiene_use_gateway_guard(tmp_path):
    runner, store, entry, event = _make_gateway(tmp_path)
    kwargs = runner._gateway_agent_session_kwargs(event.source)

    assert kwargs["platform"] == "telegram"
    assert kwargs["session_db"] is runner._session_db
    assert kwargs["user_id"] == "user"
    assert kwargs["chat_id"] == "chat"
    assert kwargs["chat_type"] == "dm"
    assert kwargs["gateway_session_key"] == runner._session_key_for_source(event.source)

    cfg = {"context": {"engine": "dag"}, "agent": {}, "compression": {"enabled": True}}
    with (
        patch("hermes_cli.config.load_config", return_value=cfg),
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
        patch("agent.model_metadata.get_model_context_length", return_value=131_072),
    ):
        from run_agent import AIAgent

        agent = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            provider="openrouter",
            model="test-model",
            max_iterations=1,
            quiet_mode=True,
            skip_memory=True,
            session_id=entry.session_id,
            **kwargs,
        )

    try:
        assert not getattr(agent.context_compressor, "projection_only_compression", False)
    finally:
        agent.close()


@pytest.mark.asyncio
async def test_gateway_compress_reconciles_projection_only_dag_when_gateway_enabled(tmp_path):
    runner, store, entry, event = _make_gateway(tmp_path)
    for content in ("one", "two", "three", "four"):
        store.append_to_transcript(entry.session_id, {"role": "user", "content": content})

    cfg = {
        "context": {"engine": "dag", "dag": {"gateway_enabled": True}},
        "agent": {},
        "compression": {"enabled": True},
    }

    with (
        patch("hermes_cli.config.load_config", return_value=cfg),
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
        patch("agent.model_metadata.get_model_context_length", return_value=131_072),
        patch("run_agent.AIAgent._compress_context") as compress_mock,
        patch.object(store, "rewrite_transcript", wraps=store.rewrite_transcript) as rewrite_mock,
    ):
        result = await runner._handle_compress_command(event)

    assert "DAG context engine is active" in result
    assert "did not rewrite" in result
    assert "Mirrored missing raw message(s):" in result
    compress_mock.assert_not_called()
    rewrite_mock.assert_not_called()


@pytest.mark.asyncio
async def test_gateway_status_surfaces_dag_projection_reconciliation_state_when_enabled(tmp_path):
    runner, store, entry, event = _make_gateway(tmp_path)
    store.append_to_transcript(entry.session_id, {"role": "user", "content": "one"})
    store.append_to_transcript(entry.session_id, {"role": "assistant", "content": "two"})
    reconcile_full_transcript(
        ContextDAGStore(runner._session_db),
        entry.session_id,
        store.load_transcript(entry.session_id),
        source="test",
    )
    cfg = {
        "context": {"engine": "dag", "dag": {"gateway_enabled": True}},
        "agent": {},
        "compression": {"enabled": True},
    }

    with patch("hermes_cli.config.load_config", return_value=cfg):
        result = await runner._handle_status_command(event)

    assert "DAG context: projection-only" in result
    assert "raw transcript is not rewritten" in result
    assert "DAG reconciliation: checkpointed" in result
    assert "2 transcript message(s)" in result


@pytest.mark.asyncio
async def test_gateway_status_dag_config_without_gateway_flag_reports_inactive_fallback(tmp_path):
    runner, store, entry, event = _make_gateway(tmp_path)
    cfg = {"context": {"engine": "dag"}, "agent": {}, "compression": {"enabled": True}}

    with patch("hermes_cli.config.load_config", return_value=cfg):
        result = await runner._handle_status_command(event)

    assert "DAG context: configured but inactive on gateway" in result
    assert "legacy compressor fallback" in result
    assert "DAG context: projection-only/no transcript rewrite ENABLED" not in result


@pytest.mark.asyncio
async def test_gateway_status_dag_config_with_gateway_env_flag_reports_enabled(tmp_path, monkeypatch):
    runner, store, entry, event = _make_gateway(tmp_path)
    store.append_to_transcript(entry.session_id, {"role": "user", "content": "one"})
    cfg = {"context": {"engine": "dag"}, "agent": {}, "compression": {"enabled": True}}
    monkeypatch.setenv("HERMES_DAG_CONTEXT_GATEWAY_ENABLED", "true")

    with patch("hermes_cli.config.load_config", return_value=cfg):
        result = await runner._handle_status_command(event)

    assert "DAG context: projection-only/no transcript rewrite ENABLED" in result
    assert "configured but inactive on gateway" not in result


@pytest.mark.asyncio
async def test_gateway_status_legacy_output_does_not_include_dag_lines_by_default(tmp_path):
    runner, store, entry, event = _make_gateway(tmp_path)
    cfg = {"context": {"engine": "legacy"}, "agent": {}, "compression": {"enabled": True}}

    with patch("hermes_cli.config.load_config", return_value=cfg):
        result = await runner._handle_status_command(event)

    assert "DAG context:" not in result
    assert "DAG reconciliation:" not in result
