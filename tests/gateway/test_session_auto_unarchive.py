from unittest.mock import AsyncMock

import pytest

from gateway.run_turn import _unarchive_session_on_human_activity


@pytest.mark.asyncio
async def test_human_activity_unarchives_session():
    session_db = AsyncMock()
    session_db.unarchive_if_archived.return_value = True

    assert await _unarchive_session_on_human_activity(session_db, "session-1", None) is True
    session_db.unarchive_if_archived.assert_awaited_once_with("session-1")


@pytest.mark.asyncio
async def test_machinery_activity_preserves_archive():
    session_db = AsyncMock()

    assert (
        await _unarchive_session_on_human_activity(
            session_db, "session-1", "internal_notification"
        )
        is False
    )
    session_db.unarchive_if_archived.assert_not_awaited()
