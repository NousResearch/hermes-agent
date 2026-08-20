from __future__ import annotations

import importlib.util
import json
import sqlite3
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "skills"
    / "apple"
    / "apple-voice-memos"
    / "scripts"
    / "voicememos.py"
)
SKILL_PATH = SCRIPT_PATH.parents[1] / "SKILL.md"


@pytest.fixture(scope="module")
def mod():
    spec = importlib.util.spec_from_file_location("apple_voice_memos_skill", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _atom(payload: bytes) -> bytes:
    return (len(payload) + 8).to_bytes(4, "big") + b"tsrp" + payload


def _record(title: str, filename: str, uid: str, *, exists: bool = True) -> dict:
    return {
        "title": title,
        "filename": filename,
        "path": f"/recordings/{filename}",
        "exists": exists,
        "date": "2026-07-29T12:00:00-04:00",
        "date_human": "2026-07-29 12:00",
        "duration_sec": 61.2,
        "unique_id": uid,
    }


def test_extract_transcript_reads_string_runs(mod, tmp_path: Path):
    payload = json.dumps(
        {"attributedString": {"runs": ["Hello ", {"ignored": True}, "world"]}}
    ).encode()
    recording = tmp_path / "memo.m4a"
    recording.write_bytes(b"ftyp" + _atom(payload))

    assert mod.extract_transcript(recording) == "Hello world"


def test_extract_transcript_tolerates_atom_padding(mod, tmp_path: Path):
    payload = json.dumps({"attributedString": {"runs": ["padded"]}}).encode()
    recording = tmp_path / "memo.m4a"
    recording.write_bytes(_atom(b" \n" + payload + b"\x00\x00"))

    assert mod.extract_transcript(recording) == "padded"


def test_extract_transcript_skips_malformed_marker(mod, tmp_path: Path):
    payload = json.dumps({"attributedString": {"runs": ["real transcript"]}}).encode()
    malformed = b"\xff\xff\xff\xfftsrpnot-json"
    recording = tmp_path / "memo.m4a"
    recording.write_bytes(malformed + _atom(payload))

    assert mod.extract_transcript(recording) == "real transcript"


@pytest.mark.parametrize(
    "content",
    [
        b"no transcript atom",
        b"\x00\x00\x00\x08tsrp",
        _atom(b"not-json"),
        _atom(json.dumps({"attributedString": {"runs": "not-a-list"}}).encode()),
    ],
)
def test_extract_transcript_returns_none_for_invalid_content(
    mod, tmp_path: Path, content: bytes
):
    recording = tmp_path / "memo.m4a"
    recording.write_bytes(content)

    assert mod.extract_transcript(recording) is None
    assert mod.extract_transcript(tmp_path / "missing.m4a") is None


def test_best_transcript_reads_apple_text(mod, tmp_path: Path):
    recording = tmp_path / "memo.m4a"
    payload = json.dumps({"attributedString": {"runs": ["Apple text"]}}).encode()
    recording.write_bytes(_atom(payload))
    rec = _record("Memo", recording.name, "memo-id")
    rec["path"] = str(recording)

    assert mod.best_transcript(rec) == ("Apple text", "apple")


def test_best_transcript_returns_none_without_local_audio(mod):
    rec = _record("Memo", "missing.m4a", "memo-id", exists=False)

    assert mod.best_transcript(rec) == (None, None)


def test_load_recordings_reads_snapshot_newest_first(mod, tmp_path: Path, monkeypatch):
    recordings = tmp_path / "Recordings"
    recordings.mkdir()
    database = recordings / "CloudRecordings.db"
    conn = sqlite3.connect(database)
    conn.execute(
        "CREATE TABLE ZCLOUDRECORDING ("
        "ZENCRYPTEDTITLE, ZPATH, ZDATE, ZDURATION, ZUNIQUEID)"
    )
    conn.executemany(
        "INSERT INTO ZCLOUDRECORDING VALUES (?, ?, ?, ?, ?)",
        [
            ("Older", "/source/older.m4a", 100.0, 3.04, "old-id"),
            ("Newer", "/source/newer.m4a", 200.0, 61.26, "new-id"),
            ("Skipped", None, 300.0, 1.0, "skip-id"),
        ],
    )
    conn.commit()
    conn.close()
    (recordings / "newer.m4a").write_bytes(b"audio")
    original_database = database.read_bytes()
    monkeypatch.setattr(mod, "RECORDINGS_DIR", recordings)
    monkeypatch.setattr(mod, "DB_PATH", database)

    result = mod.load_recordings()

    assert [item["title"] for item in result] == ["Newer", "Older"]
    assert result[0]["filename"] == "newer.m4a"
    assert result[0]["exists"] is True
    assert result[0]["duration_sec"] == 61.3
    assert result[0]["unique_id"] == "new-id"
    assert result[0]["date"] is not None
    assert result[1]["exists"] is False
    assert database.read_bytes() == original_database


def test_load_recordings_missing_database_has_actionable_error(
    mod, tmp_path: Path, monkeypatch
):
    missing = tmp_path / "CloudRecordings.db"
    monkeypatch.setattr(mod, "DB_PATH", missing)

    with pytest.raises(mod.VoiceMemosAccessError) as exc:
        mod.load_recordings()

    message = str(exc.value)
    assert str(missing) in message
    assert str(Path(sys.executable).resolve()) in message
    assert "Full Disk Access" in message


def test_filtered_list_preserves_original_index(mod, monkeypatch, capsys):
    records = [
        _record("First", "first.m4a", "first-id"),
        _record("Target", "target.m4a", "target-id"),
        _record("Last", "last.m4a", "last-id"),
    ]
    monkeypatch.setattr(mod, "load_recordings", lambda: records)
    monkeypatch.setattr(mod, "best_transcript", lambda rec: ("text", "apple"))
    args = SimpleNamespace(search="target", with_transcript=False, limit=30, json=False)

    mod.cmd_list(args)

    assert "[  1] [A]" in capsys.readouterr().out
    assert mod._resolve_one(records, "1") is records[1]


def test_json_list_includes_original_index_without_mutating_records(
    mod, monkeypatch, capsys
):
    records = [
        _record("First", "first.m4a", "first-id"),
        _record("Target one", "target-one.m4a", "target-one-id"),
        _record("Target two", "target-two.m4a", "target-two-id"),
    ]
    monkeypatch.setattr(mod, "load_recordings", lambda: records)
    monkeypatch.setattr(mod, "best_transcript", lambda rec: ("text", "apple"))
    args = SimpleNamespace(search="target", with_transcript=False, limit=1, json=True)

    mod.cmd_list(args)
    result = json.loads(capsys.readouterr().out)

    assert result[0]["index"] == 1
    assert result[0]["transcript_source"] == "apple"
    assert result[0]["has_transcript"] is True
    assert "index" not in records[1]


def test_skill_metadata_and_section_order_follow_repository_standard():
    content = SKILL_PATH.read_text(encoding="utf-8")
    description_line = next(
        line for line in content.splitlines() if line.startswith("description: ")
    )
    description = description_line.removeprefix("description: ").strip('"')

    assert len(description) <= 60
    assert description.endswith(".")
    assert 'author: "Zach Leahan (@ZacharyLeahan)"' in content

    headings = [
        "# Apple Voice Memos Skill",
        "## When to Use",
        "## Prerequisites",
        "## How to Run",
        "## Quick Reference",
        "## Procedure",
        "## Pitfalls",
        "## Verification",
    ]
    positions = [content.index(heading) for heading in headings]
    assert positions == sorted(positions)


def test_list_distinguishes_missing_audio_from_missing_apple_text(
    mod, monkeypatch, capsys
):
    records = [
        _record("Not downloaded", "remote.m4a", "remote-id", exists=False),
        _record("No Apple text", "local.m4a", "local-id"),
    ]
    monkeypatch.setattr(mod, "load_recordings", lambda: records)
    monkeypatch.setattr(mod, "best_transcript", lambda rec: (None, None))
    args = SimpleNamespace(search=None, with_transcript=False, limit=30, json=False)

    mod.cmd_list(args)

    output = capsys.readouterr().out
    assert "[  0] [D]" in output
    assert "[  1] [ ]" in output
    assert "D = audio not downloaded locally" in output
    assert "blank = no embedded Apple transcript" in output


def test_json_list_exposes_both_unavailable_states(mod, monkeypatch, capsys):
    records = [
        _record("Not downloaded", "remote.m4a", "remote-id", exists=False),
        _record("No Apple text", "local.m4a", "local-id"),
    ]
    monkeypatch.setattr(mod, "load_recordings", lambda: records)
    monkeypatch.setattr(mod, "best_transcript", lambda rec: (None, None))
    args = SimpleNamespace(search=None, with_transcript=False, limit=30, json=True)

    mod.cmd_list(args)
    result = json.loads(capsys.readouterr().out)

    assert result[0]["exists"] is False
    assert result[0]["has_transcript"] is False
    assert result[0]["transcript_source"] is None
    assert result[1]["exists"] is True
    assert result[1]["has_transcript"] is False
    assert result[1]["transcript_source"] is None


def test_with_transcript_excludes_both_unavailable_states_and_preserves_index(
    mod, monkeypatch, capsys
):
    records = [
        _record("No Apple text", "none.m4a", "none-id"),
        _record("Not downloaded", "remote.m4a", "remote-id", exists=False),
        _record("Available", "available.m4a", "available-id"),
    ]
    monkeypatch.setattr(mod, "load_recordings", lambda: records)
    monkeypatch.setattr(
        mod,
        "best_transcript",
        lambda rec: ("Apple text", "apple")
        if rec["unique_id"] == "available-id"
        else (None, None),
    )
    args = SimpleNamespace(search=None, with_transcript=True, limit=30, json=False)

    mod.cmd_list(args)

    output = capsys.readouterr().out
    assert "[  2] [A]" in output
    assert "No Apple text" not in output
    assert "Not downloaded" not in output


def test_resolve_one_supports_index_filename_title_and_unique_id(mod):
    records = [
        _record("Standup Notes", "standup.m4a", "ABC-123"),
        _record("Planning", "planning.m4a", "DEF-456"),
    ]

    assert mod._resolve_one(records, "0") is records[0]
    assert mod._resolve_one(records, "planning.m4a") is records[1]
    assert mod._resolve_one(records, "standup") is records[0]
    assert mod._resolve_one(records, "def-456") is records[1]
    assert mod._resolve_one(records, "99") is None
    assert mod._resolve_one(records, "missing") is None


def test_dump_excludes_audio_that_is_not_downloaded(mod, monkeypatch, capsys):
    records = [
        _record("Not downloaded", "remote.m4a", "remote-id", exists=False),
        _record("Available", "available.m4a", "available-id"),
    ]
    monkeypatch.setattr(mod, "load_recordings", lambda: records)
    monkeypatch.setattr(mod, "best_transcript", lambda rec: ("Apple text", "apple"))
    args = SimpleNamespace(search=None, limit=10, only_transcribed=False, json=True)

    mod.cmd_dump(args)
    result = json.loads(capsys.readouterr().out)

    assert [item["title"] for item in result] == ["Available"]
    assert result[0]["transcript"] == "Apple text"
    assert result[0]["transcript_source"] == "apple"


def test_transcript_json_and_no_match_error(mod, monkeypatch, capsys):
    rec = _record("Memo", "memo.m4a", "memo-id")
    monkeypatch.setattr(mod, "load_recordings", lambda: [rec])
    monkeypatch.setattr(mod, "best_transcript", lambda item: ("memo text", "apple"))

    mod.cmd_transcript(SimpleNamespace(selector="memo-id", json=True))
    result = json.loads(capsys.readouterr().out)
    assert result["transcript"] == "memo text"
    assert result["transcript_source"] == "apple"

    monkeypatch.setattr(mod, "best_transcript", lambda item: (None, None))
    mod.cmd_transcript(SimpleNamespace(selector="memo-id", json=False))
    assert "Apple has not embedded a transcript" in capsys.readouterr().out

    rec["exists"] = False
    mod.cmd_transcript(SimpleNamespace(selector="memo-id", json=False))
    assert "audio is not downloaded locally" in capsys.readouterr().out

    with pytest.raises(SystemExit) as exc:
        mod.cmd_transcript(SimpleNamespace(selector="missing", json=False))
    assert exc.value.code == 1
    assert "no recording matched" in capsys.readouterr().err
