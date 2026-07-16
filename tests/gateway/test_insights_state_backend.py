"""Focused coverage for /insights durable-state backend selection."""

import asyncio
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from gateway.config import GatewayConfig
from gateway.config import Platform
from gateway.run import GatewayRunner
from gateway.session import SessionSource


def test_runner_uses_configured_state_backend(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    db = MagicMock()

    with patch("hermes_state.SessionDB.for_home", return_value=db) as open_for_home:
        runner = GatewayRunner(
            GatewayConfig(sessions_dir=tmp_path / "sessions")
        )

    assert open_for_home.call_count >= 2
    assert all(call.args[0] == tmp_path for call in open_for_home.call_args_list)
    assert runner.session_store._db is db
    assert runner._session_db._db is db


def test_insights_opens_profile_state_store_off_event_loop(tmp_path, monkeypatch):
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(multiplex_profiles=True)
    runner._resolve_profile_home_for_source = lambda _source: tmp_path
    event = SimpleNamespace(
        source=SessionSource(platform=Platform.TELEGRAM, chat_id="test-chat"),
        get_command_args=lambda: "",
    )
    db = MagicMock()
    opened_on = []
    loop_thread = threading.get_ident()

    def open_for_home(home):
        opened_on.append((home, threading.get_ident()))
        return db

    class FakeInsightsEngine:
        def __init__(self, _db):
            pass

        def generate(self, *, days, source):
            assert days == 30
            assert source is None
            return "report"

        def format_gateway(self, report):
            assert report == "report"
            return "formatted report"

    monkeypatch.setattr("hermes_state.SessionDB.for_home", open_for_home)
    monkeypatch.setattr("agent.insights.InsightsEngine", FakeInsightsEngine)

    result = asyncio.run(runner._handle_insights_command(event))

    assert result == "formatted report"
    assert len(opened_on) == 1
    assert opened_on[0][0] == tmp_path
    assert opened_on[0][1] != loop_thread
    db.close.assert_called_once()
