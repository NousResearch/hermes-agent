"""/api/status's active-session garnish counts at most once per TTL, and never twice at once (#160281, #155093).

Each count opens a fresh read-only handle, which rebuilds the local lineage index; the 0.75 s
``wait_for`` around it cancels only the await, so a slow count kept running while every later poll
started another one beside it.
"""

import threading
import time

import pytest


@pytest.fixture
def status_with_store(monkeypatch):
    from hermes_cli import web_server_sessions
    from hermes_cli.web_routers import status
    from persistence.conversation.sessions import NewSession

    writer = web_server_sessions._open_session_db_for_profile(None, read_only=False)
    writer.sessions.create(NewSession("s-live", "cli"))
    writer.close()
    monkeypatch.setattr(status, "_status_counts", {}, raising=False)
    monkeypatch.setattr(status, "_status_count_flights", {}, raising=False)

    opens = []
    real_open = web_server_sessions._open_session_db_for_profile

    def counting_open(profile, *, read_only):
        opens.append(threading.get_ident())
        return real_open(profile, read_only=read_only)

    monkeypatch.setattr(web_server_sessions, "_open_session_db_for_profile", counting_open)
    return status, opens


def test_repeated_polls_count_once_within_the_ttl(status_with_store):
    status, opens = status_with_store

    results = [status._count_status_active_sessions() for _ in range(10)]

    assert results == [1] * 10
    assert len(opens) == 1


def test_a_poll_during_a_slow_count_does_not_start_another(status_with_store, monkeypatch):
    status, opens = status_with_store
    from hermes_cli import web_server_sessions

    release, started = threading.Event(), threading.Event()
    counting_open = web_server_sessions._open_session_db_for_profile

    def slow_open(profile, *, read_only):
        store = counting_open(profile, read_only=read_only)
        started.set()
        release.wait(5)
        return store

    monkeypatch.setattr(web_server_sessions, "_open_session_db_for_profile", slow_open)
    first = threading.Thread(target=status._count_status_active_sessions)
    first.start()
    assert started.wait(5)

    t0 = time.monotonic()
    assert status._count_status_active_sessions() == 0  # nothing counted yet: answer without recounting
    assert time.monotonic() - t0 < 0.5
    release.set()
    first.join(5)

    assert len(opens) == 1
