"""The rename route must keep the complete writer lifetime off the event loop."""
import asyncio
import threading
from unittest.mock import patch

import pytest
from fastapi import HTTPException

from hermes_cli.web_models import SessionRename
from hermes_cli.web_routers import sessions


@pytest.mark.parametrize("title", ["renamed", ""])
def test_rename_does_not_open_writer_on_event_loop(title):
    loop_thread = threading.get_ident()
    observed = []

    class DB:
        def resolve_session_id(self, sid):
            observed.append(threading.get_ident())
            return sid

        def set_session_title(self, sid, value):
            observed.append(threading.get_ident())
            assert value == title

        def get_session_title(self, sid):
            observed.append(threading.get_ident())
            return title

        def close(self):
            observed.append(threading.get_ident())

    def open_db(profile, *, read_only):
        assert profile == "selected"
        assert read_only is False
        observed.append(threading.get_ident())
        return DB()

    with patch.object(sessions, "_open_session_db_for_profile", open_db):
        result = asyncio.run(sessions.rename_session_endpoint(
            "session", SessionRename(title=title, profile="selected")))
    assert result == {"ok": True, "title": title}
    assert observed and all(t != loop_thread for t in observed)
    assert len(set(observed)) == 1


@pytest.mark.parametrize("failure,status", [("missing", 404), ("empty", 400), ("invalid", 400)])
def test_rename_preserves_errors_and_closes_writer(failure, status):
    closed = []

    class DB:
        def resolve_session_id(self, sid):
            return None if failure == "missing" else sid

        def set_session_title(self, sid, title):
            raise ValueError("invalid title")

        def close(self):
            closed.append(threading.get_ident())

    body = SessionRename() if failure == "empty" else SessionRename(title="title")
    with patch.object(sessions, "_open_session_db_for_profile", return_value=DB()):
        with pytest.raises(HTTPException) as error:
            asyncio.run(sessions.rename_session_endpoint("session", body))
    assert error.value.status_code == status
    assert len(closed) == 1
    assert closed[0] != threading.get_ident()


def test_rename_preserves_open_failure():
    with patch.object(sessions, "_open_session_db_for_profile", side_effect=RuntimeError("open failed")):
        with pytest.raises(RuntimeError, match="open failed"):
            asyncio.run(sessions.rename_session_endpoint("session", SessionRename(title="title")))


def test_flags_only_update_preserves_existing_title():
    calls = []

    class DB:
        def resolve_session_id(self, sid):
            return sid

        def set_session_pinned(self, sid, value):
            calls.append((sid, value))

        def get_session_title(self, sid):
            return "existing"

        def close(self):
            pass

    with patch.object(sessions, "_open_session_db_for_profile", return_value=DB()):
        result = asyncio.run(sessions.rename_session_endpoint("session", SessionRename(pinned=False)))
    assert calls == [("session", False)]
    assert result == {"ok": True, "title": "existing", "pinned": False}
