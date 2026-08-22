"""Tests for mem0's built-in memory write mirroring (on_memory_write).

When the model uses the built-in ``memory`` tool to add a fact, mem0 should
receive a verbatim copy (infer=False) so curated facts are searchable in the
semantic store. Only ``add`` is mirrored — replace/remove can't be mapped
back to mem0 IDs and would leave stale duplicates.
"""

import threading
import time

import plugins.memory.mem0 as mem0_plugin
from plugins.memory.mem0 import Mem0MemoryProvider


class FakeBackend:
    def __init__(self):
        self.add_calls = []
        self.closed = False

    def add(self, messages, *, user_id, agent_id, infer=False, metadata=None):
        self.add_calls.append({
            "messages": messages,
            "user_id": user_id,
            "agent_id": agent_id,
            "infer": infer,
            "metadata": metadata,
        })

    def close(self):
        self.closed = True


def _provider_with_backend(backend):
    prov = Mem0MemoryProvider()
    prov._mode = "oss"
    prov._user_id = "hermes-user"
    prov._agent_id = "hermes"
    prov._backend = backend
    return prov


def _wait_for(backend, n=1, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if len(backend.add_calls) >= n:
            return
        time.sleep(0.05)
    raise AssertionError(f"mirror write did not land within {timeout}s; calls={backend.add_calls}")


def test_add_is_mirrored_verbatim():
    backend = FakeBackend()
    prov = _provider_with_backend(backend)

    prov.on_memory_write("add", "memory", "user prefers dark mode")

    _wait_for(backend)
    call = backend.add_calls[0]
    assert call["messages"] == [{"role": "user", "content": "user prefers dark mode"}]
    assert call["infer"] is False, "curated fact must be stored verbatim, no extraction"
    assert call["user_id"] == "hermes-user"
    assert call["agent_id"] == "hermes"
    assert call["metadata"].get("write_origin") == "builtin_memory_mirror"
    assert call["metadata"].get("target") == "memory"


def test_user_target_recorded_in_metadata():
    backend = FakeBackend()
    prov = _provider_with_backend(backend)

    prov.on_memory_write("add", "user", "user's name is Echo")

    _wait_for(backend)
    assert backend.add_calls[0]["metadata"]["target"] == "user"


def test_replace_and_remove_not_mirrored():
    backend = FakeBackend()
    prov = _provider_with_backend(backend)

    prov.on_memory_write("replace", "memory", "new text")
    prov.on_memory_write("remove", "memory", "old text")
    time.sleep(0.3)

    assert backend.add_calls == [], "replace/remove must not be mirrored"


def test_empty_content_not_mirrored():
    backend = FakeBackend()
    prov = _provider_with_backend(backend)

    prov.on_memory_write("add", "memory", "")
    time.sleep(0.3)

    assert backend.add_calls == []


def test_no_backend_skips_mirror():
    prov = Mem0MemoryProvider()
    prov._backend = None

    prov.on_memory_write("add", "memory", "anything")

    # No crash, no thread explosion — nothing observable to assert beyond no error.


def test_mirror_failure_is_silent(monkeypatch):
    class ExplodingBackend(FakeBackend):
        def add(self, *args, **kwargs):
            raise RuntimeError("backend down")

    backend = ExplodingBackend()
    prov = _provider_with_backend(backend)

    # Must not raise — mirror failures are logged at debug level only.
    prov.on_memory_write("add", "memory", "fact")
    time.sleep(0.3)


def test_mirror_uses_provider_breaker_state(monkeypatch):
    backend = FakeBackend()
    prov = _provider_with_backend(backend)
    monkeypatch.setattr(prov, "_is_breaker_open", lambda: True)

    prov.on_memory_write("add", "memory", "fact")
    time.sleep(0.3)

    assert backend.add_calls == [], "breaker-open must suppress mirror writes"


# ---------------------------------------------------------------------------
# Lazy backend recovery (_ensure_backend)
# ---------------------------------------------------------------------------


class _FlakyOSSBackend:
    """OSSBackend stand-in that fails until ``allow`` flips to True."""

    allow = False

    def __init__(self, cfg):
        if not _FlakyOSSBackend.allow:
            raise RuntimeError("Storage folder ... already accessed by another instance")
        self.cfg = cfg
        self.closed = False

    def close(self):
        self.closed = True


def test_ensure_backend_recovers_after_transient_failure(monkeypatch):
    """A backend that failed at startup (e.g. Qdrant lock held by another
    process) must be lazily re-initialized on the next call once the
    transient condition clears — not stay wedged until host restart."""
    import plugins.memory.mem0._backend as be_mod

    monkeypatch.setattr(be_mod, "OSSBackend", _FlakyOSSBackend)
    _FlakyOSSBackend.allow = False

    prov = Mem0MemoryProvider()
    prov._mode = "oss"
    prov._config = {"oss": {"vector_store": {"provider": "qdrant", "config": {"path": "~/.hermes/mem0_qdrant"}}}}
    prov._user_id = "u1"
    prov._agent_id = "hermes"

    # First initialize fails (lock held elsewhere at startup).
    backend = prov._create_backend()[0]
    assert backend is None, "startup init must fail while the lock is held"
    assert prov._backend is None

    # The transient condition clears (other process released the lock).
    _FlakyOSSBackend.allow = True

    # Next tool call must recover via _ensure_backend.
    result = prov.handle_tool_call("mem0_search", {"query": "anything", "top_k": 3})
    assert "backend not initialized" not in result.lower(), (
        "tool call must succeed after transient failure clears"
    )
    assert prov._backend is not None and isinstance(prov._backend, _FlakyOSSBackend)
