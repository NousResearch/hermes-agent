"""Unit tests for the pure soul read/write helpers (GET/PUT /api/soul contract).

Covers the server-side core of roadmap K phase 2 (spec
2026-11-09-miri-soul-direct-editor-design.md): byte-exact round-trip, byte
cap (not char cap), hash gate, backup creation/rotation/collision,
atomic-replace failure semantics. No aiohttp here — pure filesystem.
"""

import re
import hashlib
from datetime import datetime

import pytest

from gateway.platforms import api_server_soul as soul
from gateway.platforms.api_server_soul import (
    CAP_BYTES,
    BackupCollision,
    InvalidSoulContent,
    SoulChangedSinceRead,
    SoulMissing,
    SoulNotFound,
    SoulWriteFailed,
    prune_backups,
    read_soul,
    soul_sha256,
    write_soul,
)

BACKUP_RE = re.compile(r"^SOUL\.md\.bak-\d{8}-\d{6}(-\d+)?\.md$")


def _seed(home, text: str) -> bytes:
    data = text.encode("utf-8")
    (home / "SOUL.md").write_bytes(data)
    return data


def test_read_roundtrip_byte_exact(tmp_path):
    raw = "Хто ти\r\n  • рядок 2 з юнікод 😀\r\nфінал\n".encode("utf-8")
    (tmp_path / "SOUL.md").write_bytes(raw)
    doc = read_soul(tmp_path)
    assert doc["content"] == raw.decode("utf-8")
    assert doc["sha256"] == soul_sha256(raw)
    assert doc["sha256"] == hashlib.sha256(raw).hexdigest()
    assert doc["bytes"] == len(raw)
    assert doc["path"] == str(tmp_path / "SOUL.md")
    # tz-aware ISO-8601
    datetime.fromisoformat(doc["updated_at"])


def test_read_missing_raises(tmp_path):
    with pytest.raises(SoulNotFound):
        read_soul(tmp_path)


def test_write_creates_backup_and_rotates(tmp_path, monkeypatch):
    # Every backup gets a fresh, deterministic second-stamp.
    ticks = iter(f"20260101-{h:02d}{m:02d}{s:02d}"
                 for h in range(24) for m in range(60) for s in range(60))
    monkeypatch.setattr(soul, "_backup_stamp", lambda: next(ticks))

    res = write_soul(tmp_path, "v0", None)  # create, no backup
    assert res["backup"] is None
    for i in range(1, 52):  # 51 more writes -> 51 backups made, rotation keeps 50
        write_soul(tmp_path, f"v{i}", None)

    backups = sorted(p.name for p in tmp_path.iterdir() if BACKUP_RE.match(p.name))
    assert len(backups) == 50
    newest = sorted(backups)[-1]
    assert (tmp_path / newest).read_text(encoding="utf-8") == "v50"
    assert (tmp_path / "SOUL.md").read_text(encoding="utf-8") == "v51"


def test_write_foreign_backups_untouched(tmp_path):
    _seed(tmp_path, "оригінал")
    foreign = tmp_path / "SOUL.md.bak-2026-01-01"  # the MVP agent-style backup
    foreign.write_text("чужий бэкап", encoding="utf-8")
    write_soul(tmp_path, "нове", None)
    write_soul(tmp_path, "ще новіше", None)
    assert foreign.exists()
    assert foreign.read_text(encoding="utf-8") == "чужий бэкап"
    assert prune_backups(tmp_path, keep=1) >= 1  # prunes only our pattern
    assert foreign.exists()


def test_write_base_sha_mismatch_raises(tmp_path):
    raw = _seed(tmp_path, "поточний")
    with pytest.raises(SoulChangedSinceRead) as exc:
        write_soul(tmp_path, "мій варіант", "deadbeef")
    assert exc.value.current_sha256 == soul_sha256(raw)
    assert (tmp_path / "SOUL.md").read_bytes() == raw  # untouched


def test_write_base_sha_missing_file_raises(tmp_path):
    # base provided, file vanished between GET and PUT -> refuse, not create
    with pytest.raises(SoulMissing):
        write_soul(tmp_path, "новий", "ab12cd34")
    assert not (tmp_path / "SOUL.md").exists()


def test_write_without_base_creates(tmp_path):
    res = write_soul(tmp_path, "душа з нуля", None)
    assert res["backup"] is None
    assert (tmp_path / "SOUL.md").read_text(encoding="utf-8") == "душа з нуля"
    assert res["sha256"] == soul_sha256("душа з нуля".encode("utf-8"))


def test_write_empty_or_whitespace_raises(tmp_path):
    _seed(tmp_path, "база")
    for bad in ("", "   ", "\n\t \r\n"):
        with pytest.raises(InvalidSoulContent):
            write_soul(tmp_path, bad, None)


def test_write_cap_counts_utf8_bytes_not_chars(tmp_path):
    # Review Focus 1: the cap is UTF-8 BYTES, not characters.
    over = "😀" * 16385  # 4 bytes each: 65540 bytes, 16385 chars
    assert len(over.encode("utf-8")) == 65540
    with pytest.raises(InvalidSoulContent):
        write_soul(tmp_path, over, None)
    exact = "😀" * 16384  # exactly 65536 bytes
    assert len(exact.encode("utf-8")) == CAP_BYTES
    res = write_soul(tmp_path, exact, None)
    assert res["bytes"] == CAP_BYTES


def test_write_replaces_atomically_no_tmp_leftover(tmp_path):
    _seed(tmp_path, "перший")
    write_soul(tmp_path, "другий", None)
    assert (tmp_path / "SOUL.md").exists()
    leftovers = [p for p in tmp_path.iterdir() if p.name.endswith(".tmp")]
    assert leftovers == []


def test_write_os_replace_fails_leaves_file_intact(tmp_path, monkeypatch):
    # Review Focus 3: Windows AV/file-lock mid-replace -> soul_write_failed
    raw = _seed(tmp_path, "світла душа")

    def boom(src, dst, *a, **kw):
        raise OSError(13, "Permission denied (simulated AV lock)")

    monkeypatch.setattr(soul.os, "replace", boom)
    with pytest.raises(SoulWriteFailed):
        write_soul(tmp_path, "зіпсована спроба", None)
    assert (tmp_path / "SOUL.md").read_bytes() == raw  # byte-exact intact
    ours = [p for p in tmp_path.iterdir() if BACKUP_RE.match(p.name)]
    assert len(ours) == 1  # backup was made before the failed replace
    assert ours[0].read_bytes() == raw
    assert not list(tmp_path.glob("SOUL.md.tmp-*"))  # mkstemp debris cleaned


def test_write_records_prev_sha(tmp_path):
    # Audit support (spec §59 "sha %s→%s"): the write result carries the
    # pre-replacement sha so the log line can show what changed into what.
    res0 = write_soul(tmp_path, "one", None)
    assert res0["prev_sha256"] is None  # create-from-nothing
    res1 = write_soul(tmp_path, "two", res0["sha256"])
    assert res1["prev_sha256"] == res0["sha256"]


def test_write_rejects_nonstring_base(tmp_path):
    # A non-string base_sha256 is a client bug: 400 (InvalidSoulContent),
    # not an accidental 409 on a healthy file.
    write_soul(tmp_path, "one", None)
    with pytest.raises(InvalidSoulContent):
        write_soul(tmp_path, "two", 123)  # type: ignore[arg-type]  # untrusted HTTP body reality
