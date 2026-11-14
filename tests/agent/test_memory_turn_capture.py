"""``MemoryProvider.capture_turn`` runs inline, before ``sync_turn`` is queued.

``sync_all`` hands ``sync_turn`` to a background worker, so a process that exits right after the reply
could drop the turn. ``capture_turn`` gives a provider one fast, synchronous chance to journal it first.
"""
import threading

from agent.memory_manager import MemoryManager
from agent.memory_provider import MemoryProvider

_SECRET = "sk-proj-abc123def456ghi789jkl012"


class _Provider(MemoryProvider):
    def __init__(self, *, fail_capture: bool = False):
        self.captured: list[tuple] = []
        self.synced: list[tuple] = []
        self.release = threading.Event()
        self._fail_capture = fail_capture

    @property
    def name(self):
        return "journal"

    def initialize(self, session_id="", **kw): pass
    def is_available(self): return True
    def get_tool_schemas(self): return []

    def capture_turn(self, user_content, assistant_content, *, session_id="", messages=None):
        if self._fail_capture:
            raise RuntimeError("journal disk full")
        self.captured.append((user_content, assistant_content, session_id, messages))

    def sync_turn(self, user_content, assistant_content, *, session_id="", messages=None, **kw):
        self.release.wait(10)
        self.synced.append((user_content, assistant_content, session_id))


class _PlainProvider(_Provider):
    capture_turn = MemoryProvider.capture_turn


def test_manager_advertises_durable_turn_capture():
    assert MemoryManager.supports_durable_turn_capture is True


def test_capture_runs_before_sync_all_returns_while_sync_turn_is_still_blocked():
    mgr, p = MemoryManager(), _Provider()
    mgr.add_provider(p)

    mgr.sync_all("hi", "hey", session_id="s1", messages=[{"role": "user", "content": "hi"}])

    assert p.captured == [("hi", "hey", "s1", [{"role": "user", "content": "hi"}])]
    assert p.synced == []
    p.release.set()
    assert mgr.flush_pending(timeout=10) is True
    assert p.synced == [("hi", "hey", "s1")]


def test_capture_gets_scrubbed_content_and_only_the_kwargs_it_accepts():
    mgr, p = MemoryManager(), _Provider()
    p.release.set()
    mgr.add_provider(p)

    # ``turn_author`` is not in this capture_turn signature, so passing it must not raise.
    mgr.sync_all(f"key {_SECRET}", "ok", session_id="s1", turn_author={"id": "1", "name": "Ada", "is_bot": False})

    assert mgr.flush_pending(timeout=10) is True
    (user, _assistant, _sid, _messages), = p.captured
    assert _SECRET not in user


def test_capture_failure_is_logged_and_sync_turn_still_runs(caplog):
    mgr, p = MemoryManager(), _Provider(fail_capture=True)
    p.release.set()
    mgr.add_provider(p)

    mgr.sync_all("hi", "hey", session_id="s1")

    assert mgr.flush_pending(timeout=10) is True
    assert p.synced == [("hi", "hey", "s1")]
    assert "durable turn capture failed" in caplog.text


def test_provider_without_capture_turn_is_unaffected():
    mgr, p = MemoryManager(), _PlainProvider()
    p.release.set()
    mgr.add_provider(p)

    mgr.sync_all("hi", "hey", session_id="s1")

    assert mgr.flush_pending(timeout=10) is True
    assert p.captured == []
    assert p.synced == [("hi", "hey", "s1")]
