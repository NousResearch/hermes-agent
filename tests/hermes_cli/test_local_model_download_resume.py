"""Resume contract for local-model ranged downloads.

A mid-transfer failure must leave the preallocated .part plus a sidecar so
the next download_file call fetches only missing ranges. Completeness is
always the server-declared total — never catalog size or a preallocated
file's length (zeros look like a full file).
"""

from __future__ import annotations

import io
import json
import threading
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request

import pytest

from hermes_cli.web_routers import local_models


class _FakeResponse(io.BytesIO):
    def __init__(self, body: bytes, *, status: int, headers: dict):
        super().__init__(body)
        self.status = status
        self.headers = headers

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class _RangeServer:
    """Thread-safe urllib.request.urlopen stand-in that honors Range."""

    def __init__(self, body: bytes, *, short_on_first: set[tuple[int, int]] | None = None):
        self.body = body
        self.short_on_first = set(short_on_first or [])
        self._attempts: dict[tuple[int, int], int] = {}
        self.requested: list[tuple[int, int]] = []
        self._lock = threading.Lock()

    def urlopen(self, req, timeout=None):
        if isinstance(req, str):
            return _FakeResponse(
                self.body,
                status=200,
                headers={"Content-Length": str(len(self.body))},
            )
        assert isinstance(req, Request)
        raw = req.headers.get("Range") or req.headers.get("range") or ""
        if not raw:
            return _FakeResponse(
                self.body,
                status=200,
                headers={"Content-Length": str(len(self.body))},
            )
        start_s, end_s = raw.split("=", 1)[1].split("-", 1)
        start, end = int(start_s), int(end_s)
        # Probe is always bytes=0-0; real partitions in these tests are larger.
        if start == 0 and end == 0:
            return _FakeResponse(
                self.body[:1],
                status=206,
                headers={"Content-Range": f"bytes 0-0/{len(self.body)}"},
            )
        with self._lock:
            self.requested.append((start, end))
            n = self._attempts.get((start, end), 0) + 1
            self._attempts[(start, end)] = n
            short = (start, end) in self.short_on_first and n == 1
        if short:
            return _FakeResponse(self.body[start:start + 2], status=206, headers={
                "Content-Range": f"bytes {start}-{start + 1}/{len(self.body)}",
                "Content-Length": "2",
            })
        slice_ = self.body[start:end + 1]
        return _FakeResponse(slice_, status=206, headers={
            "Content-Range": f"bytes {start}-{end}/{len(self.body)}",
            "Content-Length": str(len(slice_)),
        })


def _job() -> dict:
    return {"done_bytes": 0, "total_bytes": 0, "detail": ""}


def test_ranged_incomplete_keeps_part_and_resumes_missing_range(tmp_path, monkeypatch):
    body = bytes(range(32))
    dest = tmp_path / "model.gguf"
    tmp = dest.with_suffix(".part")
    sidecar = dest.with_suffix(".part.json")
    server = _RangeServer(body, short_on_first={(16, 31)})
    monkeypatch.setattr(local_models, "_DOWNLOAD_CONNECTIONS", 2)
    monkeypatch.setattr("urllib.request.urlopen", server.urlopen)

    with pytest.raises(RuntimeError, match=r"download incomplete \(\d+ of 32 bytes\)"):
        local_models.download_file("https://example.test/model.gguf", dest, _job())

    assert not dest.exists()
    assert tmp.exists(), "incomplete ranged download must keep .part for resume"
    assert sidecar.exists(), "range-progress sidecar must remain after a failed ranged fetch"
    saved = json.loads(sidecar.read_text())
    assert saved["url"] == "https://example.test/model.gguf"
    assert saved["total"] == 32
    assert [tuple(p) for p in saved["completed"]] == [(0, 15)]
    first_pass = list(server.requested)
    assert (0, 15) in first_pass and (16, 31) in first_pass

    local_models.download_file("https://example.test/model.gguf", dest, _job())

    assert dest.exists()
    assert dest.read_bytes() == body
    assert not tmp.exists()
    assert not sidecar.exists()
    resumed = server.requested[len(first_pass):]
    assert resumed == [(16, 31)], f"second call must fetch only the missing range, got {resumed}"


def test_no_range_short_body_does_not_stage_dest(tmp_path, monkeypatch):
    dest = tmp_path / "model.gguf"

    class _NoRange:
        def urlopen(self, req, timeout=None):
            if isinstance(req, Request) and (req.headers.get("Range") or req.headers.get("range")):
                return _FakeResponse(b"x", status=200, headers={"Content-Length": "32"})
            return _FakeResponse(b"not the real body", status=200, headers={"Content-Length": "32"})

    monkeypatch.setattr("urllib.request.urlopen", _NoRange().urlopen)

    with pytest.raises(RuntimeError, match="bytes"):
        local_models.download_file("https://example.test/model.gguf", dest, _job())

    assert not dest.exists()


def test_stale_sidecar_url_mismatch_starts_fresh(tmp_path, monkeypatch):
    body = bytes(range(32))
    dest = tmp_path / "model.gguf"
    tmp = dest.with_suffix(".part")
    sidecar = dest.with_suffix(".part.json")
    with open(tmp, "wb") as f:
        f.truncate(32)
        f.write(b"\xff" * 16)
    sidecar.write_text(json.dumps({
        "url": "https://example.test/OTHER.gguf",
        "total": 32,
        "completed": [[0, 15]],
    }))
    server = _RangeServer(body)
    monkeypatch.setattr(local_models, "_DOWNLOAD_CONNECTIONS", 2)
    monkeypatch.setattr("urllib.request.urlopen", server.urlopen)

    local_models.download_file("https://example.test/model.gguf", dest, _job())

    assert dest.read_bytes() == body
    assert set(server.requested) == {(0, 15), (16, 31)}


def test_missing_sidecar_discards_preallocated_part(tmp_path, monkeypatch):
    body = bytes(range(32))
    dest = tmp_path / "model.gguf"
    tmp = dest.with_suffix(".part")
    with open(tmp, "wb") as f:
        f.truncate(32)
        f.write(b"\xee" * 32)
    server = _RangeServer(body)
    monkeypatch.setattr(local_models, "_DOWNLOAD_CONNECTIONS", 2)
    monkeypatch.setattr("urllib.request.urlopen", server.urlopen)

    local_models.download_file("https://example.test/model.gguf", dest, _job())

    assert dest.read_bytes() == body
    assert set(server.requested) == {(0, 15), (16, 31)}


def test_sidecar_total_mismatch_starts_fresh(tmp_path, monkeypatch):
    body = bytes(range(32))
    dest = tmp_path / "model.gguf"
    tmp = dest.with_suffix(".part")
    sidecar = dest.with_suffix(".part.json")
    with open(tmp, "wb") as f:
        f.truncate(16)
    sidecar.write_text(json.dumps({
        "url": "https://example.test/model.gguf",
        "total": 16,
        "completed": [[0, 15]],
    }))
    server = _RangeServer(body)
    monkeypatch.setattr(local_models, "_DOWNLOAD_CONNECTIONS", 2)
    monkeypatch.setattr("urllib.request.urlopen", server.urlopen)

    local_models.download_file("https://example.test/model.gguf", dest, _job())

    assert dest.read_bytes() == body
    assert set(server.requested) == {(0, 15), (16, 31)}


def test_range_fetch_error_keeps_part_for_resume(tmp_path, monkeypatch):
    body = bytes(range(32))
    dest = tmp_path / "model.gguf"
    tmp = dest.with_suffix(".part")
    sidecar = dest.with_suffix(".part.json")

    class _Boom(_RangeServer):
        def urlopen(self, req, timeout=None):
            if isinstance(req, Request):
                raw = req.headers.get("Range") or req.headers.get("range") or ""
                if raw == "bytes=16-31":
                    raise URLError("connection reset")
            return super().urlopen(req, timeout=timeout)

    server = _Boom(body)
    monkeypatch.setattr(local_models, "_DOWNLOAD_CONNECTIONS", 2)
    monkeypatch.setattr("urllib.request.urlopen", server.urlopen)

    with pytest.raises(URLError, match="connection reset"):
        local_models.download_file("https://example.test/model.gguf", dest, _job())

    assert not dest.exists()
    assert tmp.exists()
    assert sidecar.exists()
    assert [tuple(p) for p in json.loads(sidecar.read_text())["completed"]] == [(0, 15)]
