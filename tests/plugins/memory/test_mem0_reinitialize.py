"""Regression test for releasing the prior Mem0 backend on re-initialize.

The fix (plugins/memory/mem0/__init__.py) calls ``_shutdown_backend()``
before building a new backend inside ``initialize()``. Without it, a prior
session's OSSBackend (and its QdrantLocal file lock on mem0_qdrant/.lock)
is only released by GC, which is non-deterministic and intermittently fails
the next init with "Storage folder ... already accessed by another instance
of Qdrant client".

Reviewer ask: demonstrate repeated ``initialize()`` on the SAME provider
instance and assert the first backend is closed before the second is built.
"""

from plugins.memory.mem0 import Mem0MemoryProvider


class FakeBackend:
    """Track close() so we can assert the old backend was released."""

    def __init__(self, label):
        self.label = label
        self.closed = False

    def close(self):
        self.closed = True

    def __repr__(self):
        return f"<FakeBackend {self.label}>"


def test_repeated_initialize_releases_prior_backend(monkeypatch):
    """initialize() twice on one provider closes backend #1 before #2."""
    provider = Mem0MemoryProvider()

    built = []

    def fake_create_backend():
        b = FakeBackend(len(built))
        built.append(b)
        return b

    monkeypatch.setattr(provider, "_create_backend", fake_create_backend)

    provider.initialize("session-1")
    first = provider._backend
    assert first is built[0]
    assert first.closed is False

    provider.initialize("session-2")
    second = provider._backend

    # The prior backend must have been closed before the new one was built.
    assert first.closed is True
    assert second is built[1]
    assert second.closed is False
    # Only the new (still-live) backend is held.
    assert provider._backend is second


def test_first_initialize_does_not_call_close(monkeypatch):
    """The very first initialize() on a fresh provider never shuts anything down."""
    provider = Mem0MemoryProvider()

    def fake_create_backend():
        return FakeBackend("fresh")

    monkeypatch.setattr(provider, "_create_backend", fake_create_backend)

    # A fresh provider starts with no backend, so initialize() must not touch
    # _shutdown_backend / close(). Guard against the fix regressing into a
    # close-on-first-run.
    provider.initialize("session-1")
    assert provider._backend.label == "fresh"
    assert provider._backend.closed is False
