"""Regression: async completions read their spawning profile's session store.

A multiplexed gateway's default runtime scope points at the root ``state.db``.
An async completion is routed from its persisted SessionSource, so its preflight
must enter that source profile's scope before checking the parent session.
"""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway import run as run_module
from gateway.config import Platform
from gateway.run import GatewayRunner
from gateway.session import SessionSource
from hermes_constants import get_hermes_home


@pytest.mark.asyncio
async def test_completion_classifier_reads_source_profile_db(tmp_path, monkeypatch):
    """A trading completion finds its live parent in trading, not default."""
    default_home = tmp_path / "default"
    trading_home = tmp_path / "profiles" / "trading"
    monkeypatch.setenv("HERMES_HOME", str(default_home))
    default_db = SimpleNamespace(get_session=AsyncMock(return_value=None))
    trading_db = SimpleNamespace(get_session=AsyncMock(return_value={"ended_at": None}))

    runner = object.__new__(GatewayRunner)
    runner._session_db_pinned = run_module._SESSION_DB_UNPINNED
    runner._resolve_profile_home_for_source = lambda source: trading_home

    def open_active_db(raise_on_error=False):
        del raise_on_error
        return trading_db if Path(get_hermes_home()) == trading_home else default_db

    runner._open_session_db_for_active_scope = open_active_db
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="12345",
        chat_type="dm",
        profile="trading",
    )

    assert await runner._classify_completion_target("sess-trading", source=source) == "deliver"
    trading_db.get_session.assert_awaited_once_with("sess-trading")
    default_db.get_session.assert_not_awaited()


@pytest.mark.asyncio
async def test_async_readiness_passes_event_source_to_classifier():
    """The async-delegation gate must scope its parent lookup from the event."""
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="12345",
        chat_type="dm",
        profile="trading",
    )
    runner = object.__new__(GatewayRunner)
    runner._build_process_event_source = lambda evt: source
    runner._classify_completion_target = AsyncMock(return_value="retry")
    event = {"type": "async_delegation", "parent_session_id": "sess-trading"}

    assert await runner._completion_delivery_ready(event) is False
    runner._classify_completion_target.assert_awaited_once_with(
        "sess-trading", source=source,
    )


@pytest.mark.asyncio
async def test_process_preflight_passes_event_source_to_classifier():
    """The process-completion preflight uses the same source-scoped lookup."""
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="12345",
        chat_type="dm",
        profile="trading",
    )
    runner = object.__new__(GatewayRunner)
    runner._build_process_event_source = lambda evt: source
    runner._classify_completion_target = AsyncMock(return_value="deliver")
    event = {"type": "completion", "parent_session_id": "sess-trading"}

    claim = await runner._preflight_completion_delivery(event)
    assert claim.proceed is True
    runner._classify_completion_target.assert_awaited_once_with(
        "sess-trading", source=source,
    )
