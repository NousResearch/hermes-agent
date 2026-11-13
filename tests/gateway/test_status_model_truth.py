"""/status must report the model the next turn runs once config.yaml moved on (#130572)."""

from datetime import datetime
from unittest.mock import patch

import pytest

from gateway.session import SessionEntry, build_session_key
from hermes_state import AsyncSessionDB, SessionDB
from tests.gateway.test_status_command import _make_event, _make_runner, _make_source

_CONFIG = {"model": {"provider": "openrouter", "default": "thinkingmachines/inkling-small:free"}}


def _runner_with_old_usage(tmp_path):
    entry = SessionEntry(session_key=build_session_key(_make_source()), session_id="sess-1",
                         created_at=datetime.now(), updated_at=datetime.now(),
                         platform=_make_source().platform, chat_type="dm")
    runner = _make_runner(entry)
    runner.session_store.get_model_override.return_value = None
    db = SessionDB(db_path=tmp_path / "state.db")
    runner._session_db = AsyncSessionDB(db)
    db.create_session("sess-1", "telegram", model="dots-studio/dots-3-note-preview:free")
    db.update_token_counts("sess-1", model="dots-studio/dots-3-note-preview:free",
                           billing_provider="openrouter", input_tokens=100, api_call_count=3)
    return runner, db


@pytest.mark.asyncio
async def test_status_reports_config_model_not_stale_usage_route(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    runner, db = _runner_with_old_usage(tmp_path)
    try:
        with patch("gateway.run._load_gateway_config", return_value=_CONFIG):
            status = await runner._handle_message(_make_event("/status"))
            model, _ = runner._resolve_session_agent_runtime(source=_make_source())
        assert model == "thinkingmachines/inkling-small:free"
        assert "**Model:** `thinkingmachines/inkling-small:free` (openrouter)" in status
        assert "dots-studio/dots-3-note-preview:free" not in status
    finally:
        db.close()


@pytest.mark.asyncio
async def test_status_keeps_persisted_route_when_it_is_the_config_model(tmp_path):
    runner, db = _runner_with_old_usage(tmp_path)
    cfg = {"model": {"provider": "openrouter", "default": "dots-studio/dots-3-note-preview:free"}}
    try:
        with patch("gateway.run._load_gateway_config", return_value=cfg):
            status = await runner._handle_message(_make_event("/status"))
        assert "**Model:** `dots-studio/dots-3-note-preview:free` (openrouter)" in status
    finally:
        db.close()


def _fallback_agent():
    from tests.gateway.test_usage_command import _make_mock_agent
    agent = _make_mock_agent(model="minimax/minimax-m3:free", provider="openrouter")
    agent.get_rate_limit_state.return_value = None
    agent._credits_state = None
    agent._fallback_activated = True
    agent._primary_runtime = {"model": "dots-studio/dots-3-note-preview:free", "provider": "openrouter"}
    return agent


@pytest.mark.asyncio
async def test_status_and_usage_label_a_fallback_route_with_the_primary(tmp_path):
    """#133993: /model reports the configured model while the cached agent runs a fallback; /status
    and /usage must say the shown model is a fallback, naming the model /model reports."""
    import threading
    runner, db = _runner_with_old_usage(tmp_path)
    runner._agent_cache_lock = threading.Lock()
    runner._agent_cache[build_session_key(_make_source())] = (_fallback_agent(), "sig")
    cfg = {"model": {"provider": "openrouter", "default": "dots-studio/dots-3-note-preview:free"}}
    try:
        with patch("gateway.run._load_gateway_config", return_value=cfg), \
                patch("gateway.slash_commands_status.fetch_account_usage", return_value=None):
            status = await runner._handle_message(_make_event("/status"))
            usage = await runner._handle_message(_make_event("/usage"))
        for out in (status, usage):
            model_line = next(line for line in out.splitlines() if "minimax/minimax-m3:free" in line)
            assert "dots-studio/dots-3-note-preview:free" in model_line
    finally:
        db.close()
