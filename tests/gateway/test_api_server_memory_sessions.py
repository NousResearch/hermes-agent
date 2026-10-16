"""api_server keeps one memory provider per session across requests (#120116)."""

from types import SimpleNamespace

from gateway.platforms.api_server_memory_sessions import ApiServerMemorySessions


class _Manager:
    def __init__(self):
        self.shut_down = False

    def flush_pending(self, timeout=None):
        return True

    def shutdown_all(self):
        self.shut_down = True


def _agent(manager, session_id="sess-1"):
    return SimpleNamespace(_memory_manager=manager, session_id=session_id)


def test_manager_survives_across_requests_and_is_exclusive_per_home(monkeypatch):
    registry = ApiServerMemorySessions(max_size=8, idle_ttl_secs=3600.0)
    home = ["home-a"]
    monkeypatch.setattr(ApiServerMemorySessions, "_owner_home", staticmethod(lambda: (home[0], None)))
    # Shutdowns run inline so the assertion below does not race a daemon thread.
    monkeypatch.setattr(registry, "_shutdown_async", registry._shutdown)

    assert registry.checkout("sess-1") is None  # first request builds its own
    manager = _Manager()
    registry.checkin(_agent(manager))
    assert registry.checkout("sess-1") is manager  # next request on the session gets it back
    assert registry.checkout("sess-1") is None  # ...exclusively: a concurrent request builds a new one
    loser = _Manager()
    registry.checkin(_agent(manager))
    registry.checkin(_agent(loser))  # the concurrent request checks in behind the winner
    assert manager.shut_down and not loser.shut_down
    # Same session id under another profile home is a different key (#120116 x multiplex).
    home[0] = "home-b"
    assert registry.checkout("sess-1") is None
    home[0] = "home-a"
    assert registry.checkout("sess-1") is loser
