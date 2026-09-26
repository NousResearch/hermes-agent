"""Regression tests for resumable ranged model downloads (issue #103416).

Large local-model downloads (16 GB GGUF) run over 8 parallel Range connections
into a preallocated ``.part`` file. Before #103416 a drop on any one connection
raised out of the whole download and the ``.part`` was deleted, discarding every
worker's completed bytes — a 7 GB head start lost to one Wi-Fi/VPN blip, retry
restarting from zero.

These tests pin the fix: each range worker retries its own *remaining* sub-range
from the last byte it wrote, so a transient mid-stream drop costs only a re-fetch
of that worker's tail, never the other workers' gigabytes. A drop that outlasts
the retry budget still fails the download and removes the ``.part`` (a preallocated
zero-filled file is useless without range metadata).
"""

from __future__ import annotations

import threading

import pytest

import hermes_cli.web_routers.local_models as lm


class _FakeResp:
    """Serves a byte slice, optionally raising mid-stream after ``fail_after`` bytes."""

    def __init__(self, data: bytes, fail_after: int | None = None):
        self._data = data
        self._pos = 0
        self._fail_after = fail_after
        self.status = 206

    def read(self, n: int = -1) -> bytes:
        if self._fail_after is not None and self._pos >= self._fail_after:
            raise ConnectionResetError("simulated connection drop")
        end = len(self._data) if n is None or n < 0 else min(self._pos + n, len(self._data))
        if self._fail_after is not None:
            end = min(end, self._fail_after)
        chunk = self._data[self._pos:end]
        self._pos = end
        return chunk

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _install_fake_transport(monkeypatch, master: bytes, *, drop_plan: dict[int, int] | None = None):
    """Patch urlopen so range requests are served from ``master`` in memory.

    ``drop_plan`` maps a range-start offset -> number of bytes to deliver before the
    FIRST request for that start raises (the retry for the advanced start succeeds).
    """
    monkeypatch.setattr(lm, "_probe_range_support", lambda url: len(master))
    monkeypatch.setattr(lm, "_CHUNK", 4)  # small reads so a drop lands mid-range
    monkeypatch.setattr(lm.time, "sleep", lambda *_a, **_k: None)
    seen_starts: set[int] = set()
    seen_lock = threading.Lock()
    drop_plan = dict(drop_plan or {})

    def fake_urlopen(req, timeout=None):
        rng = req.headers["Range"].split("=", 1)[1]
        start_s, end_s = rng.split("-")
        start, end = int(start_s), int(end_s)
        fail_after = None
        with seen_lock:
            if start in drop_plan and start not in seen_starts:
                fail_after = drop_plan[start]
            seen_starts.add(start)
        return _FakeResp(master[start:end + 1], fail_after=fail_after)

    monkeypatch.setattr(lm.urllib.request, "urlopen", fake_urlopen)


def test_range_worker_resumes_after_mid_stream_drop(monkeypatch, tmp_path):
    """A drop on one of the 8 workers costs only that worker's tail — no lost bytes,
    no double counting, the assembled file is byte-exact."""
    total = lm._DOWNLOAD_CONNECTIONS * 8  # 64 bytes, 8 bytes per worker
    master = bytes((i * 7 + 3) & 0xFF for i in range(total))
    # Worker 3 owns bytes [24, 31]; deliver 3 bytes, then drop. Retry resumes at 27.
    _install_fake_transport(monkeypatch, master, drop_plan={24: 3})

    dest = tmp_path / "model.gguf"
    job: dict = {}
    lm.download_file("https://example/model.gguf", dest, job)

    assert dest.read_bytes() == master
    assert not dest.with_suffix(".part").exists()
    # Every byte counted exactly once — the resumed request must not re-pump bytes 24-26.
    assert job["done_bytes"] == total
    assert job["total_bytes"] == total


def test_download_fails_and_clears_part_when_drops_outlast_retries(monkeypatch, tmp_path):
    """A worker that never gets past its first byte exhausts the retry budget; the whole
    download fails and the ``.part`` is removed (no truncated file staged)."""
    monkeypatch.setattr(lm, "_DOWNLOAD_RANGE_RETRIES", 2)
    total = lm._DOWNLOAD_CONNECTIONS * 8
    master = bytes(range(total % 256)) if total <= 256 else bytes(total)
    monkeypatch.setattr(lm, "_probe_range_support", lambda url: total)
    monkeypatch.setattr(lm, "_CHUNK", 4)
    monkeypatch.setattr(lm.time, "sleep", lambda *_a, **_k: None)

    def always_drop(req, timeout=None):
        rng = req.headers["Range"].split("=", 1)[1]
        start = int(rng.split("-", 1)[0])
        # Worker 0 (start 0) always drops immediately; others would succeed.
        return _FakeResp(master[start:], fail_after=0 if start == 0 else None)

    monkeypatch.setattr(lm.urllib.request, "urlopen", always_drop)

    dest = tmp_path / "model.gguf"
    with pytest.raises(Exception):
        lm.download_file("https://example/model.gguf", dest, {})
    assert not dest.with_suffix(".part").exists()
    assert not dest.exists()
