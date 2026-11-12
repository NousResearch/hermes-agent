"""Pruned-session file removal is id-scoped even when the id carries glob metacharacters.

``request_dump_<id>_*.json`` was interpolated unescaped, so an id containing ``[``/``?``/``*`` was
a PATTERN: its own dumps were left behind and another session's could be matched instead.
"""

from __future__ import annotations

from hermes_state_sessions import SessionSessionsMixin


def test_remove_session_files_escapes_glob_metacharacters(tmp_path):
    tricky, neighbour = "sess-[ab]-1", "sess-a-1"
    for session_id in (tricky, neighbour):
        (tmp_path / f"{session_id}.jsonl").write_text("{}\n", encoding="utf-8")
        (tmp_path / f"request_dump_{session_id}_0.json").write_text("{}", encoding="utf-8")

    SessionSessionsMixin._remove_session_files(tmp_path, tricky)

    assert not (tmp_path / f"{tricky}.jsonl").exists()
    assert not (tmp_path / f"request_dump_{tricky}_0.json").exists(), "own dump was not matched"
    assert (tmp_path / f"{neighbour}.jsonl").exists()
    assert (tmp_path / f"request_dump_{neighbour}_0.json").exists(), "neighbour's dump was removed"


def test_remove_session_files_removes_legacy_session_prefix_snapshot(tmp_path):
    # Older builds wrote the transcript snapshot as session_<id>.json; the
    # remover only swept <id>.json/<id>.jsonl, so the snapshot (which can
    # carry plaintext secrets) survived every delete/prune.
    sid = "abc123"
    (tmp_path / f"session_{sid}.json").write_text('{"messages": [{"content": "secret"}]}', encoding="utf-8")

    SessionSessionsMixin._remove_session_files(tmp_path, sid)

    assert not (tmp_path / f"session_{sid}.json").exists()
