"""Gateway media-deliverable events (D1) — behavior contracts.

Contract: when reply processing extracts media deliverables, a structured
``media.deliverable`` event frame is emitted through the tui-gateway event
transport (seq-stamped, replay-safe). Emission is best-effort and silent when
no desktop client is listening: without a live session transport nothing is
written anywhere (messaging-gateway stdout must stay byte-identical), and the
mature platform-adapter delivery path is untouched.
"""

import threading

import pytest

from tui_gateway import event_replay, server
from tui_gateway.event_replay import events_since, latest_seq, reset_replay_state

from gateway.media_events import (
    describe_media_deliverable,
    emit_media_deliverable,
    extract_media_from_reply,
    MEDIA_EVENT_TYPE,
)


@pytest.fixture(autouse=True)
def _clean_replay():
    reset_replay_state()
    yield
    reset_replay_state()


class _CollectingTransport:
    """Minimal transport capturing write_json frames (test double)."""

    def __init__(self):
        self.frames = []
        self._lock = threading.Lock()

    def write(self, obj):
        with self._lock:
            self.frames.append(obj)
        return True

    def close(self):
        pass


@pytest.fixture()
def listening_session():
    """A live tui_gateway session with a collecting transport."""
    sid = "media-evt-sid"
    transport = _CollectingTransport()
    server._sessions[sid] = {"transport": transport}
    yield sid, transport
    server._sessions.pop(sid, None)


# ── describe_media_deliverable: payload contract ──────────────────────


def test_describe_returns_full_payload_for_real_file(tmp_path):
    f = tmp_path / "chart.png"
    f.write_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * 128)

    payload = describe_media_deliverable(str(f))

    assert payload["path"] == str(f)
    assert payload["kind"] == "image"
    assert payload["mime"] == "image/png"
    assert payload["size"] == f.stat().st_size
    assert set(payload) == {"path", "kind", "mime", "size"}


def test_describe_kind_by_mime_class(tmp_path):
    cases = {
        "clip.mp4": "video",
        "note.ogg": "audio",
        "report.pdf": "document",
        "data.csv": "document",
    }
    for name, kind in cases.items():
        f = tmp_path / name
        f.write_bytes(b"x")
        assert describe_media_deliverable(str(f))["kind"] == kind


def test_describe_none_for_missing_file(tmp_path):
    assert describe_media_deliverable(str(tmp_path / "ghost.png")) is None


def test_describe_none_for_missing_parent_dir(tmp_path):
    assert describe_media_deliverable(str(tmp_path / "no-dir" / "f.png")) is None


# ── extract_media_from_reply: explicit-tag mirror of the delivery gate ─


def test_extract_collects_validated_media_tags(tmp_path):
    f = tmp_path / "plot.png"
    f.write_bytes(b"x")
    reply = f"Here you go:\nMEDIA:{f}\ndone"

    paths = extract_media_from_reply(reply)

    assert paths == [str(f)]


def test_extract_dedupes_repeated_tags(tmp_path):
    f = tmp_path / "plot.png"
    f.write_bytes(b"x")
    reply = f"MEDIA:{f}\ntext\nMEDIA:{f}"

    assert extract_media_from_reply(reply) == [str(f)]


def test_extract_drops_unsafe_or_missing_paths():
    # /etc/passwd is on the denylist; ghost.png does not exist.
    assert extract_media_from_reply("MEDIA:/etc/passwd") == []
    assert extract_media_from_reply("MEDIA:/nonexistent/ghost.png") == []


# ── emit: seq-stamped, replay-safe, silent without a listener ─────────


def test_emit_skips_unstatable_file(listening_session):
    sid, transport = listening_session

    ok = emit_media_deliverable(sid, "/nonexistent/ghost.png", origin="serve")

    # Unstatable file → no event (contract: events describe real files).
    assert ok is False
    assert transport.frames == []


def test_emit_full_frame_for_real_file(listening_session, tmp_path):
    sid, transport = listening_session
    f = tmp_path / "out.png"
    f.write_bytes(b"x" * 42)

    ok = emit_media_deliverable(sid, str(f), origin="serve")

    assert ok is True
    assert len(transport.frames) == 1
    frame = transport.frames[0]
    assert frame["method"] == "event"
    params = frame["params"]
    assert params["type"] == MEDIA_EVENT_TYPE
    assert params["session_id"] == sid
    assert params["seq"] == 1  # stamped by the real replay machinery
    payload = params["payload"]
    assert payload == {
        "path": str(f),
        "kind": "image",
        "mime": "image/png",
        "size": 42,
        "session_id": sid,
        "origin": "serve",
    }


def test_emit_seq_increments_and_replay_returns_media_frames(
    listening_session, tmp_path
):
    sid, _transport = listening_session
    f1 = tmp_path / "a.png"
    f2 = tmp_path / "b.png"
    f1.write_bytes(b"a")
    f2.write_bytes(b"bb")

    emit_media_deliverable(sid, str(f1), origin="serve")
    emit_media_deliverable(sid, str(f2), origin="serve")

    assert latest_seq(sid) == 2
    replayed = events_since(sid, 0)
    media = [p for p in replayed if p.get("type") == MEDIA_EVENT_TYPE]
    assert [p["payload"]["path"] for p in media] == [str(f1), str(f2)]


def test_emit_silent_noop_without_live_session(tmp_path, monkeypatch, capsys):
    """No session transport → nothing written anywhere, no stdout garbage."""
    f = tmp_path / "solo.png"
    f.write_bytes(b"x")

    # Any fallback write would land here; the stdio transport is the only
    # other sink write_json can reach.
    stdio_writes = []
    monkeypatch.setattr(
        server,
        "_stdio_transport",
        type(
            "T",
            (),
            {
                "write": staticmethod(lambda obj: stdio_writes.append(obj) or True),
                "close": staticmethod(lambda: None),
            },
        )(),
    )

    ok = emit_media_deliverable("no-such-session", str(f), origin="gateway")

    assert ok is False
    assert stdio_writes == []
    assert capsys.readouterr().out == ""


def test_emit_never_raises_on_broken_transport(listening_session, tmp_path):
    sid, _transport = listening_session

    def _boom(obj):
        raise RuntimeError("transport exploded")

    server._sessions[sid]["transport"] = type(
        "T", (), {"write": staticmethod(_boom), "close": staticmethod(lambda: None)}
    )()
    f = tmp_path / "x.png"
    f.write_bytes(b"x")

    assert emit_media_deliverable(sid, str(f), origin="serve") is False
