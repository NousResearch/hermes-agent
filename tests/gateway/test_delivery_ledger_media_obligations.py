"""The delivery obligation covers a turn's ATTACHMENTS, not just its text (#106668).

The ledger promised that a crash between finalization and platform ACK "cannot lose a
response silently", but it stored text only. Attachments were sent after the obligation
had already been resolved, so a crash in that window redelivered the text, marked the row
delivered and dropped the media with no trace at all. Media-only turns recorded no
obligation (the text send was the ledger's only producer, gated on non-empty text), and
the TTS-caption path skipped the ledger the same way.

These tests drive the real producer (``_process_message_background``) and the real
recovery (``GatewayStartupMixin._redeliver_claimed_obligations``, via the boot sweep)
against a real ledger DB, so the row states asserted here are the ones production leaves
behind:

* crash in the attachment window   -> row still owed, redelivery resends text AND media
* partial attachment delivery       -> row FAILED (never delivered), full resend with marker
* media-only turn                  -> an obligation exists at all
* released builds' rows/readers    -> still parse and still agree on obligation ids
"""

from __future__ import annotations

import hashlib
import sqlite3
import time
from pathlib import Path
from urllib.parse import quote

import pytest

from gateway import delivery_ledger as dl
from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, SendResult
from gateway.platforms.event import MessageEvent, MessageType
from gateway.run_startup import GatewayStartupMixin
from gateway.session import SessionSource, build_session_key

PLATFORM = Platform.SIGNAL
CHAT_ID = "111"


@pytest.fixture(autouse=True)
def _fresh_db(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(dl, "_db_path", lambda: home / "state.db")
    yield


def _media_file(tmp_path, monkeypatch, name: str, data: bytes) -> Path:
    root = tmp_path / "media-cache"
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    monkeypatch.setattr("gateway.platforms.base.MEDIA_DELIVERY_SAFE_ROOTS", (root,))
    return path.resolve()


def _image_manifest(path: Path) -> list:
    """The replay entry the producer owes for a single batched image path."""
    return [{"op": "send_multiple_images", "images": [[f"file://{quote(str(path))}", ""]]}]


async def _hold_typing(_chat_id, interval=2.0, metadata=None, stop_event=None):
    if stop_event is not None:
        await stop_event.wait()


class _RecordingAdapter(BasePlatformAdapter):
    """Real base-loop adapter: image batches go through the REAL ``send_multiple_images``, so the
    ``file://`` URL round-trip (quote on the way in, unquote on the way out) is exercised; the
    other attachment kinds are recorded directly."""

    def __init__(self, *, documents_ok: bool = True, platform=PLATFORM):
        super().__init__(PlatformConfig(enabled=True, token="fake-token"), platform)
        self.documents_ok = documents_ok
        self.sent: list = []
        self.images: list = []
        self.documents: list = []
        self.voices: list = []
        self.videos: list = []
        self.captions: list = []
        self._keep_typing = _hold_typing

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        return True

    async def disconnect(self) -> None:
        return None

    async def get_chat_info(self, chat_id):
        return {"id": chat_id}

    async def send_typing(self, chat_id, metadata=None) -> None:
        return None

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        self.sent.append(content)
        return SendResult(success=True, message_id=f"text-{len(self.sent)}")

    async def send_image_file(self, chat_id, image_path, caption=None, reply_to=None,
                              metadata=None, **kwargs):
        self.images.append(str(image_path))
        return SendResult(success=True, message_id=f"img-{len(self.images)}")

    async def send_document(self, chat_id, file_path, caption=None, file_name=None, reply_to=None,
                            metadata=None, **kwargs):
        if not self.documents_ok:
            return SendResult(success=False, error="upload refused")
        self.documents.append(str(file_path))
        return SendResult(success=True, message_id=f"doc-{len(self.documents)}")

    async def send_voice(self, chat_id, audio_path, caption=None, reply_to=None, metadata=None,
                         **kwargs):
        self.voices.append(str(audio_path))
        return SendResult(success=True, message_id="voice-1")

    async def send_video(self, chat_id, video_path, caption=None, reply_to=None, metadata=None,
                         **kwargs):
        self.videos.append(str(video_path))
        return SendResult(success=True, message_id="video-1")


class _RedeliveryRunner(GatewayStartupMixin):
    """Boot recovery with only the adapter lookup stubbed: the real sweep, the real redelivery
    loop and the real manifest replay run."""

    def __init__(self, adapter):
        self.adapters = {adapter.platform: adapter}

    def _authorization_adapter(self, platform, profile=None):
        return self.adapters.get(platform)


def _event(platform=PLATFORM) -> MessageEvent:
    return MessageEvent(
        text="draw me a chart",
        message_type=MessageType.TEXT,
        source=SessionSource(platform=platform, chat_id=CHAT_ID, chat_type="dm"),
        message_id="m1",
    )


async def _run_turn(adapter, response, platform=PLATFORM):
    async def handler(_event):
        return response

    adapter.set_message_handler(handler)
    event = _event(platform)
    await adapter._process_message_background(event, build_session_key(event.source))


def _rows() -> list:
    """The ledger rows, keyed by name. The SELECT is column-tolerant on purpose: on a released
    build there is no media column, and the tests must still reach their behavioural assertions
    instead of erroring on the schema."""
    present = _ledger_columns()
    names = ("obligation_id", "state", "content", "last_error", "media_manifest")
    select = ", ".join(name if name in present else f"NULL AS {name}" for name in names)
    with dl._connect() as conn:
        return [dict(zip(names, row))
                for row in conn.execute(f"SELECT {select} FROM delivery_obligations")]


def _orphan(oid: str) -> None:
    """The process that recorded the row is gone; its claim is stale (post-crash)."""
    with dl._connect() as conn:
        conn.execute("UPDATE delivery_obligations SET owner_pid=999999999, owner_started_at=1 "
                     "WHERE obligation_id=?", (oid,))


def _crash_mid_turn(oid: str, **columns) -> None:
    """Put a resolved row back into the state a crash mid-turn leaves behind: still owed, owned by
    a process that no longer exists (the row is the recovery handle the sweep will claim).

    Column names this build's table does not have are skipped, so a released build (no media
    column) still reaches the behavioural assertions instead of erroring on the schema.
    """
    present = _ledger_columns()
    columns = {name: value for name, value in columns.items() if name in present}
    assignments = ", ".join([f"{name}=?" for name in columns] + ["state='attempting'"])
    with dl._connect() as conn:
        conn.execute(
            f"UPDATE delivery_obligations SET {assignments}, owner_pid=999999999, "
            "owner_started_at=1 WHERE obligation_id=?", (*columns.values(), oid))


def _ledger_columns() -> set:
    with dl._connect() as conn:
        return {row[1] for row in conn.execute("PRAGMA table_info(delivery_obligations)")}


async def _recover(adapter) -> int:
    claimed = dl.sweep_recoverable(deliverable_platforms={PLATFORM.value})
    return await _RedeliveryRunner(adapter)._redeliver_claimed_obligations(claimed)


# ---------------------------------------------------------------- crash in the attachment window


@pytest.mark.asyncio
async def test_crash_before_attachments_redelivers_the_media(monkeypatch, tmp_path):
    """Text landed, the gateway died before the attachment went out.

    The obligation must still be owed, and recovery must resend text AND the attachment. Before
    the manifest existed the row was resolved from the text result, so the sweep had nothing to
    claim and the media was gone for good.
    """
    png = _media_file(tmp_path, monkeypatch, "chart.png", b"\x89PNG\r\n\x1a\n" + b"x" * 32)
    adapter = _RecordingAdapter()

    async def _dies_mid_attachments(*_args, **_kwargs):
        """Nothing past the text send runs — what a real crash in this window leaves behind."""
        raise ConnectionError("gateway killed in the attachment window")

    monkeypatch.setattr(adapter, "_deliver_attachments", _dies_mid_attachments)
    try:
        await _run_turn(adapter, f"Here is the chart:\nMEDIA:{png}")
    except BaseException:
        pass  # the crash is contained by the background task; only the row state matters

    rows = _rows()
    assert len(rows) == 1
    assert rows[0]["state"] == "attempting", (
        "a crash in the attachment window must leave the obligation unresolved, not delivered")
    assert rows[0]["media_manifest"] == _manifest_json(png)

    _orphan(rows[0]["obligation_id"])
    replay = _RecordingAdapter()
    assert await _recover(replay) == 1

    assert replay.images == [str(png)], (
        "recovery must resend the attachment the ledger recorded for this turn")
    assert any(dl.RECOVERED_MARKER in text for text in replay.sent), (
        "the redelivered text must carry the duplicate marker")
    assert _rows()[0]["state"] == "delivered"


def _manifest_json(path: Path) -> str:
    """The stored column value for a single batched image (canonical JSON, see the ledger)."""
    import json

    return json.dumps(_image_manifest(path), sort_keys=True, separators=(",", ":"))


# --------------------------------------------------- partial attachment delivery is a failure


@pytest.mark.asyncio
async def test_partial_attachment_failure_is_not_marked_delivered(monkeypatch, tmp_path):
    """The image went out, the document was refused: the turn is NOT delivered.

    Partial success is a failure, so the row stays owed and the next sweep resends the whole
    response (text with marker + every attachment) instead of logging the lost file.
    """
    png = _media_file(tmp_path, monkeypatch, "chart.png", b"\x89PNG\r\n\x1a\n" + b"x" * 32)
    pdf = _media_file(tmp_path, monkeypatch, "report.pdf", b"%PDF-1.4\n" + b"x" * 32)
    adapter = _RecordingAdapter(documents_ok=False)

    await _run_turn(adapter, f"Chart and report attached\nMEDIA:{png}\nMEDIA:{pdf}")

    rows = _rows()
    assert len(rows) == 1
    assert rows[0]["state"] == "failed", (
        "a turn whose attachment failed must not be marked delivered")
    assert rows[0]["last_error"] == "attachment_delivery_failed"

    _orphan(rows[0]["obligation_id"])
    replay = _RecordingAdapter()
    assert await _recover(replay) == 1

    assert replay.images == [str(png)] and replay.documents == [str(pdf)], (
        "the full response — text and every attachment — must be resent")
    assert any(dl.RECOVERED_MARKER in text for text in replay.sent)
    assert _rows()[0]["state"] == "delivered"


@pytest.mark.asyncio
async def test_failed_attachment_redelivery_keeps_the_row_owed(monkeypatch, tmp_path):
    """Recovery hits the same failure: the row must stay owed, never delivered."""
    png = _media_file(tmp_path, monkeypatch, "chart.png", b"\x89PNG\r\n\x1a\n" + b"x" * 32)
    pdf = _media_file(tmp_path, monkeypatch, "report.pdf", b"%PDF-1.4\n" + b"x" * 32)
    adapter = _RecordingAdapter(documents_ok=False)
    await _run_turn(adapter, f"Chart and report\nMEDIA:{png}\nMEDIA:{pdf}")
    assert _rows()[0]["state"] == "failed"

    _orphan(_rows()[0]["obligation_id"])

    async def _refuse(*_args, **_kwargs):
        return SendResult(success=False, error="attachment refused")

    broken = _RecordingAdapter()
    broken.send_image_file = _refuse
    assert await _recover(broken) == 0
    assert _rows()[0]["state"] == "failed", (
        "an attachment that failed again during recovery leaves the row owed, not delivered")


# ------------------------------------------------------------------------ media-only turns


@pytest.mark.asyncio
async def test_media_only_turn_records_a_recoverable_obligation(monkeypatch, tmp_path):
    """No text at all: the attachment alone is the response and it must be owed.

    Before this the text-send gate meant such a turn recorded nothing, so a crash left no
    ledger row and the sweep had nothing to claim.
    """
    png = _media_file(tmp_path, monkeypatch, "photo.png", b"\x89PNG\r\n\x1a\n" + b"x" * 32)
    adapter = _RecordingAdapter()

    await _run_turn(adapter, f"MEDIA:{png}")

    assert adapter.sent == [], "a media-only turn sends no text"
    rows = _rows()
    assert len(rows) == 1, "a media-only turn must record an obligation"
    assert rows[0]["content"] == ""
    assert rows[0]["media_manifest"] == _manifest_json(png)
    assert rows[0]["state"] == "delivered"

    # ... and the same row is the recovery handle for a crash in that window.
    _crash_mid_turn(rows[0]["obligation_id"])
    replay = _RecordingAdapter()
    assert await _recover(replay) == 1
    assert replay.images == [str(png)]
    assert replay.sent == [], "a media-only row has no text half to redeliver"


@pytest.mark.asyncio
async def test_tts_caption_delivery_records_the_reply_text(tmp_path, monkeypatch):
    """The TTS caption IS the text send (it replaces it), so it must take the same obligation.

    The caption path skips ``_send_final_text`` — the only other producer — so before this a crash
    around the voice send left the reply with no ledger row at all.
    """
    adapter = _RecordingAdapter(platform=Platform.TELEGRAM)
    voice_file = tmp_path / "reply.ogg"
    voice_file.write_bytes(b"OggS")
    monkeypatch.setattr(adapter, "_wants_auto_tts", lambda *a, **k: True)

    async def _synth(_text):
        return [str(voice_file)], None

    monkeypatch.setattr(adapter, "_synthesize_auto_tts", _synth)

    async def _play_tts(chat_id, audio_path, **kwargs):
        adapter.captions.append(kwargs.get("caption"))
        adapter.voices.append(str(audio_path))
        return SendResult(success=True, message_id="voice-1")

    monkeypatch.setattr(adapter, "play_tts", _play_tts)

    await _run_turn(adapter, "a short spoken reply", platform=Platform.TELEGRAM)

    assert adapter.voices == [str(voice_file)]
    assert adapter.captions == ["a short spoken reply"], "the caption carried the reply text"
    assert adapter.sent == [], "the caption replaced the text send"
    rows = _rows()
    assert len(rows) == 1, "the caption's text must be owed to the ledger like any final text"
    assert rows[0]["content"] == "a short spoken reply"
    assert rows[0]["state"] == "delivered"


@pytest.mark.asyncio
async def test_unreadable_manifest_is_not_reported_delivered(monkeypatch, tmp_path):
    """A manifest this build cannot replay must fail closed: no text, no 'delivered'."""
    png = _media_file(tmp_path, monkeypatch, "photo.png", b"\x89PNG\r\n\x1a\n" + b"x" * 32)
    adapter = _RecordingAdapter()
    await _run_turn(adapter, f"MEDIA:{png}")
    oid = _rows()[0]["obligation_id"]
    _crash_mid_turn(oid, media_manifest="{not json at all")

    replay = _RecordingAdapter()
    assert await _recover(replay) == 0
    assert replay.sent == [] and replay.images == []
    row = _rows()[0]
    assert row["state"] == "failed" and row["last_error"] == "media_manifest_unreadable"


# ------------------------------------------------- released builds: rows, readers, ids


_LEGACY_TABLE_DDL = """CREATE TABLE delivery_obligations (
    obligation_id TEXT PRIMARY KEY,
    session_key TEXT NOT NULL,
    platform TEXT NOT NULL,
    chat_id TEXT NOT NULL,
    thread_id TEXT,
    content TEXT NOT NULL,
    state TEXT NOT NULL,
    attempts INTEGER NOT NULL DEFAULT 0,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    owner_pid INTEGER,
    owner_started_at INTEGER,
    last_error TEXT,
    adapter_profile TEXT
)"""

# The column list a released build's sweep selects, and the INSERT a released producer writes.
_LEGACY_SELECT_COLUMNS = ("obligation_id, session_key, platform, chat_id, thread_id, content, "
                          "state, attempts, created_at, owner_pid, owner_started_at, last_error, "
                          "adapter_profile, updated_at")


def _legacy_insert(conn: sqlite3.Connection, oid: str, content: str) -> None:
    """A released build's ``record_obligation``: its own column list, no media column."""
    now = time.time()
    conn.execute(
        """INSERT OR REPLACE INTO delivery_obligations
           (obligation_id, session_key, platform, chat_id, thread_id,
            content, state, attempts, created_at, updated_at,
            owner_pid, owner_started_at, adapter_profile)
           VALUES (?, ?, ?, ?, ?, ?, 'pending', 0, ?, ?, ?, ?, ?)""",
        (oid, "agent:main:signal:dm:111", "signal", CHAT_ID, None, content, now, now, None, None,
         "default"))


@pytest.mark.asyncio
async def test_released_rows_and_readers_survive_the_new_column(tmp_path, monkeypatch):
    """An already-released build must keep working against a ledger this build wrote.

    New fields ride an ADDED nullable column, so their SELECT/INSERT column lists keep parsing
    every row (including one carrying a manifest), and a row they wrote reads back here as a
    plain text-only obligation.
    """
    adapter = _RecordingAdapter()
    png = _media_file(tmp_path, monkeypatch, "chart.png", b"\x89PNG\r\n\x1a\n" + b"x" * 32)

    conn = sqlite3.connect(dl._db_path())
    try:
        conn.execute(_LEGACY_TABLE_DDL)          # schema of a released build
        dl._initialize_schema(conn)              # this build's migration
        _legacy_insert(conn, "legacy-row", "an older build's answer")
    finally:
        conn.commit()
        conn.close()

    # A row this build writes, text + attachment.
    await _run_turn(adapter, f"Here is the chart:\nMEDIA:{png}")

    conn = sqlite3.connect(dl._db_path())
    try:
        rows = conn.execute(f"SELECT {_LEGACY_SELECT_COLUMNS} FROM delivery_obligations").fetchall()
    finally:
        conn.close()
    assert len(rows) == 2, "a released reader's column list must still parse every row"

    # The released redelivery shape (content straight into adapter.send) still works, for both.
    legacy_row = next(row for row in rows if row[0] == "legacy-row")
    assert legacy_row[5] == "an older build's answer"

    # A legacy row (no manifest column value) is a text-only obligation here.
    legacy_claimed = _claim_legacy_row("legacy-row")
    assert legacy_claimed["media_manifest"] == []
    assert legacy_claimed["content"] == "an older build's answer"


def _claim_legacy_row(oid: str) -> dict:
    with dl._connect() as conn:
        conn.execute("UPDATE delivery_obligations SET owner_pid=999999999, owner_started_at=1, "
                     "state='attempting' WHERE obligation_id=?", (oid,))
    claimed = [row for row in dl.sweep_recoverable() if row["obligation_id"] == oid]
    assert len(claimed) == 1
    return claimed[0]


def test_text_only_obligation_ids_are_unchanged():
    """A released writer and this build agree on the id of a text-only turn (idempotent re-record)."""
    released = hashlib.sha256(
        "sk1|msg1|hello".encode("utf-8", "replace")).hexdigest()[:24]
    assert dl.compute_obligation_id("sk1", "msg1", "hello") == released
    assert dl.compute_obligation_id("sk1", "msg1", "hello", None) == released
    assert dl.compute_obligation_id("sk1", "msg1", "hello", []) == released
    # Identical text, different attachments: different obligations (they must not collide).
    assert dl.compute_obligation_id("sk1", "msg1", "hello", _image_manifest(Path("/a.png"))) != released
    assert dl.compute_obligation_id("sk1", "msg1", "hello", _image_manifest(Path("/a.png"))) != \
        dl.compute_obligation_id("sk1", "msg1", "hello", _image_manifest(Path("/b.png")))
