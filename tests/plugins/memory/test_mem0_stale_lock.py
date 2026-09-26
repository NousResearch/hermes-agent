"""Stale Qdrant local-mode lock auto-recovery in OSSBackend init (issue #58705).

qdrant-client guards its local storage folder with an fcntl.flock (via portalocker) on
``<path>/.lock``; a crashed init can leave the folder un-openable with
"Storage folder ... is already accessed by another instance of Qdrant client" even when
no live process holds the lock. These tests pin the recovery contract in
``plugins/memory/mem0/_backend.py``: probe the flock, delete only provably-stale lock
files, retry construction exactly once, fail closed otherwise, and never touch
live-holder locks, non-qdrant/url configs, or non-POSIX platforms.

The stale-then-recover path injects the empirically-observed qdrant failure around a
REAL qdrant-local storage folder (modern qdrant-client re-opens a leftover-but-unheld
.lock file cleanly in-process, so the crash-residue failure mode is simulated at the
mem0.Memory boundary while every other layer — the flock probe, the unlink, the retry,
and the reopened vector store — is the real thing).
"""

import errno
import fcntl
import os
import sys
import types

import pytest

from plugins.memory.mem0._backend import OSSBackend

pytest.importorskip("qdrant_client", reason="requires the existing mem0 extra")
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

_QDRANT_LOCK_ERROR = (
    "Storage folder {path} is already accessed by another instance of Qdrant client."
    " If you require concurrent access, use Qdrant server instead."
)


def _oss_config(*, provider="qdrant", path=None, url=None):
    vs_cfg = {}
    if path is not None:
        vs_cfg["path"] = str(path)
    if url is not None:
        vs_cfg["url"] = url
    # "custom-embedder" is not in KNOWN_DIMS and carries no embedding_dims, so the
    # dims reconciliation path stays out of these lock-recovery tests entirely.
    return {
        "llm": {"provider": "ollama", "config": {"model": "llama3.1:8b"}},
        "embedder": {"provider": "ollama", "config": {"model": "custom-embedder"}},
        "vector_store": {"provider": provider, "config": vs_cfg},
    }


class _FakeMemory:
    """Stand-in for mem0.Memory returned on successful construction."""

    def __init__(self, qdrant=None):
        self.qdrant = qdrant


def _install_fake_mem0(monkeypatch, construct):
    """Route ``from mem0 import Memory`` in OSSBackend through *construct(config)*."""
    calls = []

    class Memory:
        @classmethod
        def from_config(cls, config):
            calls.append(config)
            return construct(config)

    module = types.ModuleType("mem0")
    module.__path__ = []
    module.Memory = Memory
    monkeypatch.setitem(sys.modules, "mem0", module)
    return calls


def _seed_qdrant_folder(path):
    """Create a functional local-mode storage folder with one collection + points."""
    client = QdrantClient(path=str(path))
    client.create_collection("mem0", vectors_config=VectorParams(size=4, distance=Distance.COSINE))
    client.upsert("mem0", [PointStruct(id=1, vector=[0.1, 0.2, 0.3, 0.4])])
    client.close()


def _flock_hold(path):
    """Take the exclusive flock on <path>/.lock with a raw fd (a live qdrant-equivalent holder)."""
    fd = os.open(os.path.join(str(path), ".lock"), os.O_RDWR)
    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    return fd


class TestStaleLockRecovery:

    def test_stale_lock_no_holder_is_removed_and_backend_constructs(self, tmp_path, monkeypatch):
        """Crash-residue .lock with no live holder -> probe frees it, one retry succeeds on real storage."""
        storage = tmp_path / "qdrant"
        _seed_qdrant_folder(storage)
        lock_path = storage / ".lock"
        assert lock_path.exists()  # qdrant leaves the file behind after a clean close too;
        # the simulated failure below models the empirically-observed crash-residue case
        # where the leftover file keeps blocking init until it is removed.

        def construct(config):
            path = config["vector_store"]["config"]["path"]
            if os.path.exists(os.path.join(path, ".lock")):
                raise RuntimeError(_QDRANT_LOCK_ERROR.format(path=path))
            return _FakeMemory(qdrant=QdrantClient(path=path))

        calls = _install_fake_mem0(monkeypatch, construct)
        backend = OSSBackend(_oss_config(path=storage))

        assert len(calls) == 2  # exactly one retry, not a loop
        # Recovery rebuilt a real, readable vector store over the untouched data.
        assert backend._memory.qdrant.count("mem0").count == 1
        # A fresh .lock now belongs to the live client (the stale one was replaced).
        assert (storage / ".lock").exists()
        backend._memory.qdrant.close()

    def test_live_holder_raises_with_actionable_hint_and_lock_is_preserved(self, tmp_path, monkeypatch):
        """A process genuinely holding the flock -> no deletion, error carries the concurrent-access hint."""
        storage = tmp_path / "qdrant"
        _seed_qdrant_folder(storage)
        holder_fd = _flock_hold(storage)
        try:
            def construct(config):
                # No simulation here: real qdrant-client raises the real single-client
                # error because the raw fd above actively holds the flock.
                return _FakeMemory(qdrant=QdrantClient(path=config["vector_store"]["config"]["path"]))

            calls = _install_fake_mem0(monkeypatch, construct)
            with pytest.raises(RuntimeError) as excinfo:
                OSSBackend(_oss_config(path=storage))

            message = str(excinfo.value)
            assert "already accessed by another instance" in message
            assert "Another Qdrant client is live" in message
            assert "concurrent access" in message
            assert len(calls) == 1  # no retry when the lock is legitimately held
            assert (storage / ".lock").exists()  # nothing deleted
        finally:
            fcntl.flock(holder_fd, fcntl.LOCK_UN)
            os.close(holder_fd)

    def test_second_failure_raises_and_no_further_retries(self, tmp_path, monkeypatch):
        """Fail closed: if the retry also dies, the error propagates without another deletion/retry."""
        storage = tmp_path / "qdrant"
        storage.mkdir()
        (storage / ".lock").write_text("stale", encoding="utf-8")

        def always_fail(config):
            raise RuntimeError(_QDRANT_LOCK_ERROR.format(path=config["vector_store"]["config"]["path"]))

        calls = _install_fake_mem0(monkeypatch, always_fail)
        with pytest.raises(RuntimeError, match="already accessed by another instance"):
            OSSBackend(_oss_config(path=storage))

        assert len(calls) == 2  # one retry only — no loop
        assert not (storage / ".lock").exists()  # the one stale file was cleaned, then we gave up


class TestRecoveryScopeLimits:
    """Recovery is scoped to POSIX + local-path qdrant + the exact qdrant lock error."""

    def test_non_qdrant_provider_is_untouched(self, tmp_path, monkeypatch):
        storage = tmp_path / "vectors"
        storage.mkdir()
        (storage / ".lock").write_text("stale", encoding="utf-8")

        def always_fail(config):
            raise RuntimeError(_QDRANT_LOCK_ERROR.format(path="ignored"))

        calls = _install_fake_mem0(monkeypatch, always_fail)
        with pytest.raises(RuntimeError, match="already accessed by another instance") as excinfo:
            OSSBackend(_oss_config(provider="pgvector", path=storage))

        assert "Another Qdrant client is live" not in str(excinfo.value)  # re-raised unchanged
        assert len(calls) == 1
        assert (storage / ".lock").exists()

    def test_url_based_qdrant_config_is_untouched(self, tmp_path, monkeypatch):
        def always_fail(config):
            raise RuntimeError(_QDRANT_LOCK_ERROR.format(path="ignored"))

        calls = _install_fake_mem0(monkeypatch, always_fail)
        with pytest.raises(RuntimeError, match="already accessed by another instance"):
            OSSBackend(_oss_config(url="http://qdrant.example:6333"))

        assert len(calls) == 1  # server mode has no local .lock to recover

    def test_unrelated_runtime_error_is_not_retried(self, tmp_path, monkeypatch):
        storage = tmp_path / "qdrant"
        storage.mkdir()
        (storage / ".lock").write_text("stale", encoding="utf-8")

        def other_failure(config):
            raise RuntimeError("some unrelated construction failure")

        calls = _install_fake_mem0(monkeypatch, other_failure)
        with pytest.raises(RuntimeError, match="unrelated construction failure"):
            OSSBackend(_oss_config(path=storage))

        assert len(calls) == 1
        assert (storage / ".lock").exists()  # marker must match before anything is deleted

    def test_non_posix_reraises_unchanged(self, tmp_path, monkeypatch):
        storage = tmp_path / "qdrant"
        storage.mkdir()
        (storage / ".lock").write_text("stale", encoding="utf-8")

        def construct(config):
            if os.path.exists(os.path.join(config["vector_store"]["config"]["path"], ".lock")):
                raise RuntimeError(_QDRANT_LOCK_ERROR.format(path=config["vector_store"]["config"]["path"]))
            return _FakeMemory()

        calls = _install_fake_mem0(monkeypatch, construct)
        monkeypatch.setattr(
            "plugins.memory.mem0._backend.os",
            types.SimpleNamespace(name="nt"),
        )
        with pytest.raises(RuntimeError, match="already accessed by another instance") as excinfo:
            OSSBackend(_oss_config(path=storage))

        assert "Another Qdrant client is live" not in str(excinfo.value)
        assert len(calls) == 1  # flock semantics don't hold on Windows: no probe, no delete, no retry
        assert (storage / ".lock").exists()
