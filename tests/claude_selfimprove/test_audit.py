from __future__ import annotations

from claude_selfimprove import audit


def test_record_and_read_all_roundtrip(sandbox):
    audit.record("candidate_created", candidate_id="abc123", category="explicit_instruction")
    audit.record("candidate_applied", candidate_id="abc123", target="rules/foo.md")
    entries = audit.read_all()
    assert len(entries) == 2
    assert entries[0]["event"] == "candidate_created"
    assert entries[1]["event"] == "candidate_applied"
    assert "ts" in entries[0]


def test_read_all_on_missing_log_returns_empty(sandbox):
    assert audit.read_all() == []


def test_record_never_raises_on_unserializable_field(sandbox):
    class Unserializable:
        pass

    # json.dumps would ordinarily raise on the object; record() falls back
    # to str() for anything it cannot serialize directly, so the entry is
    # still written rather than silently lost.
    audit.record("weird_event", thing=Unserializable())
    entries = audit.read_all()
    assert len(entries) == 1
    assert entries[0]["event"] == "weird_event"


def test_skips_corrupt_lines_when_reading(sandbox):
    from claude_selfimprove import paths

    paths.state_dir().mkdir(parents=True, exist_ok=True)
    with paths.audit_log_path().open("w", encoding="utf-8") as fh:
        fh.write('{"event": "good_one", "ts": "x"}\n')
        fh.write("not json at all\n")
        fh.write('{"event": "also_good", "ts": "y"}\n')
    entries = audit.read_all()
    assert [e["event"] for e in entries] == ["good_one", "also_good"]
