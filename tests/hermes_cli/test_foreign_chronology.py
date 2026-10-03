"""Chronology contracts exercised through the real parser and SQLite store."""
import json
import math
from datetime import datetime

import pytest

from hermes_cli.foreign_sessions import import_foreign_session
from hermes_state import SessionDB


@pytest.mark.parametrize("source,tool_kind", [
    ("claude", None), ("codex", None), ("codex", "custom_tool_call"),
    ("codex", "function_call"), ("codex", "local_shell_call"),
])
@pytest.mark.parametrize("stamps", [
    ["2024-10-29T12:00:00Z", "2024-10-29T05:01:00-07:00", 1730203320.125, 1730203380],
    [1730203380, 1730203320, 1730203260, 1730203200],
    [0, 0, 1, 2],
])
def test_import_preserves_source_time_and_content_order(tmp_path, source, tool_kind, stamps):
    path = tmp_path / "source.jsonl"
    roles = ["user", "assistant", "assistant", "user"]
    records = []
    for index, (role, stamp) in enumerate(zip(roles, stamps)):
        message = {"role": role, "content": str(index)}
        records.append({"timestamp": stamp, "type": role, "sessionId": "original", "message": message}
                       if source == "claude" else
                       {"timestamp": stamp, "type": "response_item", "payload": {"type": "message", **message}})
        if index == 2 and tool_kind:
            records[-1]["payload"] = {"type": tool_kind, "name": "probe"}
    path.write_text("\n".join(map(json.dumps, records)), encoding="utf-8")
    original = path.read_bytes()
    expected = [datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()
                if isinstance(s, str) else s for s in stamps]
    with_db = SessionDB(db_path=tmp_path / "state.db")
    try:
        sid = import_foreign_session(source, path, db=with_db)
        messages = with_db.get_messages(sid)
        merged_text = "1\n\n[ran tool: probe]" if tool_kind else "1\n\n2"
        assert [m["content"] for m in messages] == ["0", merged_text, "3"]
        assert [m["timestamp"] for m in messages] == [expected[0], min(expected[1:3]), expected[3]]
        session = with_db.get_session(sid)
        assert session["started_at"] == min(expected)
        provenance = json.loads(session["origin_json"])["imported_from"]
        assert provenance["source_ended_at"] == max(expected)
        assert provenance["source_timestamps"] == [[expected[0]], expected[1:3], [expected[3]]]
        assert provenance["imported_at"] > max(expected)
        assert session["ended_at"] is None
        assert path.read_bytes() == original
    finally:
        with_db.close()


@pytest.mark.parametrize("invalid", [None, "nonsense", "2024-10-29T12:00:00", True, float("nan"), float("inf")])
@pytest.mark.parametrize("known", [None, 1730203200.125])
def test_missing_dates_and_synthetic_stub_are_estimates(tmp_path, invalid, known):
    path = tmp_path / "source.jsonl"
    rows = [("assistant", "leading", invalid), ("user", "anchor", known), ("assistant", "tail", invalid)]
    path.write_text("\n".join(json.dumps({"type": role, "timestamp": stamp,
        "message": {"role": role, "content": content}}) for role, content, stamp in rows), encoding="utf-8")
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        sid = import_foreign_session("claude", path, db=db)
        session = db.get_session(sid)
        origin = json.loads(session["origin_json"])["imported_from"]
        fallback = known if known is not None else origin["imported_at"]
        messages = db.get_messages(sid)
        assert [m["role"] for m in messages] == ["user", "assistant", "user", "assistant"]
        assert all(math.isfinite(m["timestamp"]) and m["timestamp"] == fallback for m in messages)
        assert origin["source_timestamps"] == [[], [None], [known], [None]]
        assert session["started_at"] == fallback
    finally:
        db.close()
