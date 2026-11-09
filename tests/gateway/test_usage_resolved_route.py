"""Messaging ``/usage`` before the first turn reads the route that turn resolves.

With no resident agent and no persisted route, ``/usage`` used the raw ``model.provider``. Under
``provider: auto`` with a saved Nous login that is ``auto``: no provider limits were fetched and the
session did not read as a Nous route, although the first turn runs on Nous. It now asks
``_resolve_session_agent_runtime``, the resolver the turn itself calls.
"""
from unittest.mock import MagicMock

import pytest

from hermes_state import AsyncSessionDB
from tests.gateway.test_usage_command import SK, _make_runner

NOUS_URL = "https://inference-api.nousresearch.com/v1"


@pytest.mark.asyncio
async def test_usage_right_after_a_nous_login_shows_the_nous_balance(monkeypatch):
    runner = _make_runner(SK)
    runner._session_db = AsyncSessionDB(MagicMock())
    runner._session_db._db.get_session.return_value = {}
    runner._session_db._db.get_recent_session_model_route.return_value = None
    runner.session_store.get_or_create_session.return_value = MagicMock(session_id="sess-fresh")
    runner.session_store.load_transcript.return_value = []
    monkeypatch.setattr("gateway.run._load_gateway_config", lambda: {"model": {"provider": "auto"}})
    resolved = []

    def resolve(*, source=None, session_key=None, fallback=True, **kw):
        resolved.append(session_key)
        return "m", {"provider": "nous", "base_url": NOUS_URL, "api_key": "k"}
    runner._resolve_session_agent_runtime = resolve
    fetched = []
    monkeypatch.setattr("gateway.slash_commands_status.fetch_account_usage",
                        lambda provider, base_url=None, api_key=None: fetched.append((provider, base_url)))
    nous_calls = []
    monkeypatch.setattr("agent.account_usage_nous.nous_credits_lines",
                        lambda **kw: nous_calls.append(kw["nous_route"]) or ["Nous credits: 7 remaining"])

    result = await runner._handle_usage_command(MagicMock())

    assert resolved == [SK]
    assert fetched == [("nous", NOUS_URL)]
    assert nous_calls == [True]  # a Nous route: the balance with its top-up lines
    assert "Nous credits: 7 remaining" in result
