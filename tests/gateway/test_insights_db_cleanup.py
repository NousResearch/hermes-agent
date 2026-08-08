"""SessionDB lifecycle coverage for the gateway /insights command."""

from unittest.mock import MagicMock

import pytest

from gateway.slash_commands import GatewaySlashCommandsMixin


class _InsightsEvent:
    def get_command_args(self):
        return ""


def _install_insights(monkeypatch, db, engine):
    monkeypatch.setattr("hermes_state.SessionDB", lambda: db)
    monkeypatch.setattr("agent.insights.InsightsEngine", lambda _db: engine)


@pytest.mark.asyncio
async def test_gateway_insights_closes_session_db_after_success(monkeypatch):
    db = MagicMock()
    engine = MagicMock()
    engine.generate.return_value = {"summary": "ok"}
    engine.format_gateway.return_value = "formatted"
    _install_insights(monkeypatch, db, engine)
    handler = object.__new__(GatewaySlashCommandsMixin)

    result = await handler._handle_insights_command(_InsightsEvent())

    assert result == "formatted"
    db.close.assert_called_once_with()


@pytest.mark.asyncio
async def test_gateway_insights_closes_session_db_after_failure(monkeypatch):
    db = MagicMock()
    engine = MagicMock()
    engine.generate.side_effect = RuntimeError("analytics failed")
    _install_insights(monkeypatch, db, engine)
    handler = object.__new__(GatewaySlashCommandsMixin)

    result = await handler._handle_insights_command(_InsightsEvent())

    assert "analytics failed" in result
    db.close.assert_called_once_with()
