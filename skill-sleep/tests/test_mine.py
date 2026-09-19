import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lib.task_card import TaskCard
import pipeline.mine as mine


# ── helpers ───────────────────────────────────────────────────────────────


def _session(sid="sess_1", messages=None, **kw):
    base = {"id": sid, "title": kw.pop("title", ""), "cwd": kw.pop("cwd", "/tmp/proj"), "started_at": 123.0}
    base.update(kw)
    base["messages"] = messages or []
    return base


def _msg(role, content="", **kw):
    m = {"role": role, "content": content}
    m.update(kw)
    return m


# ── TaskCard ──────────────────────────────────────────────────────────────


def test_task_card_to_dict_truncates():
    c = TaskCard("s", "id123", "x" * 5000, ["e1", "e2", "e3", "e4", "e5", "e6"], [], 1.0)
    d = c.to_dict()
    assert len(d["user_request"]) == 2000
    assert len(d["friction_evidence"]) == 5
    assert d["tool_calls"] == 0


def test_task_card_repr():
    c = TaskCard("skill-a", "abcdefghij" * 4, "req", ["user_correction: hi"], [], 0)
    assert "skill-a" in repr(c)


# ── resolve_after ─────────────────────────────────────────────────────────


def test_resolve_after_iso_passthrough():
    assert mine.resolve_after("2026-08-13T00:00:00Z") == "2026-08-13T00:00:00Z"
    assert mine.resolve_after("2026-01-02") == "2026-01-02"


def test_resolve_after_relative():
    # 7d returns YYYY-MM-DD
    out = mine.resolve_after("7d")
    assert out  # non-empty
    # 24h returns iso
    out2 = mine.resolve_after("24h")
    assert "T" in out2


def test_resolve_after_invalid_fallback():
    assert mine.resolve_after("bogus") == "7d"


# ── export_sessions (mocked subprocess) ───────────────────────────────────


def test_export_sessions_parses_jsonl():
    payload = '{"id":"a","messages":[]}\n{"id":"b","messages":[]}\n'
    mock = MagicMock(returncode=0, stdout=payload, stderr="")
    with patch("pipeline.mine.subprocess.run", return_value=mock) as sp:
        sessions = mine.export_sessions("7d", timeout=5)
        assert len(sessions) == 2
        assert sessions[0]["id"] == "a"
        # should have called with redact
        assert "--redact" in sp.call_args[0][0]


def test_export_sessions_no_redact_flag():
    mock = MagicMock(returncode=0, stdout="", stderr="")
    with patch("pipeline.mine.subprocess.run", return_value=mock) as sp:
        mine.export_sessions("7d", redact=False)
        assert "--redact" not in sp.call_args[0][0]


def test_export_sessions_timeout_returns_empty():
    with patch("pipeline.mine.subprocess.run", side_effect=subprocess.TimeoutExpired(cmd="hermes", timeout=1)):
        assert mine.export_sessions("7d", timeout=1) == []


def test_export_sessions_nonzero_returns_empty(capsys):
    mock = MagicMock(returncode=1, stdout="", stderr="boom")
    with patch("pipeline.mine.subprocess.run", return_value=mock):
        assert mine.export_sessions("7d") == []
    assert "WARN" in capsys.readouterr().err


def test_export_sessions_malformed_line_skipped(capsys):
    payload = '{"id":"ok"}\nnot-json\n{"id":"ok2"}\n'
    mock = MagicMock(returncode=0, stdout=payload, stderr="")
    with patch("pipeline.mine.subprocess.run", return_value=mock):
        sessions = mine.export_sessions("7d")
    assert len(sessions) == 2
    assert "WARN" in capsys.readouterr().err


def test_export_sessions_timeout_param_forwarded():
    mock = MagicMock(returncode=0, stdout="", stderr="")
    with patch("pipeline.mine.subprocess.run", return_value=mock) as sp:
        mine.export_sessions("7d", timeout=42)
        assert sp.call_args[1]["timeout"] == 42


# ── detect_friction ───────────────────────────────────────────────────────


def test_detect_correction_cn():
    sess = _session(messages=[_msg("user", "不对，不是这样的"), _msg("assistant", "ok")])
    cards = mine.detect_friction(sess)
    assert len(cards) == 1
    assert any("user_correction" in e for e in cards[0].friction_evidence)


def test_detect_correction_en():
    sess = _session(messages=[_msg("user", "that's wrong, try again"), _msg("assistant", "ok")])
    cards = mine.detect_friction(sess)
    assert len(cards) == 1


def test_detect_tool_error_exit_code():
    sess = _session(
        messages=[
            _msg("user", "do thing"),
            _msg("tool", json.dumps({"output": "fail", "exit_code": 1}), tool_name="terminal"),
        ]
    )
    cards = mine.detect_friction(sess)
    assert len(cards) == 1
    assert any("tool_error" in e for e in cards[0].friction_evidence)


def test_detect_tool_error_keyword():
    sess = _session(
        messages=[
            _msg("user", "run"),
            _msg("tool", "Traceback: exception foo", tool_name="terminal"),
        ]
    )
    cards = mine.detect_friction(sess)
    assert len(cards) == 1


def test_detect_retry_same_request():
    sess = _session(
        messages=[
            _msg("user", "please fix the deploy script"),
            _msg("assistant", "done"),
            _msg("user", "please fix the deploy script"),
            _msg("assistant", "done"),
            _msg("user", "please fix the deploy script"),
        ]
    )
    cards = mine.detect_friction(sess)
    assert len(cards) == 1
    assert any("retry" in e for e in cards[0].friction_evidence)


def test_no_friction_returns_empty():
    sess = _session(messages=[_msg("user", "hi there"), _msg("assistant", "hello!")])
    assert mine.detect_friction(sess) == []


def test_empty_messages_no_card():
    assert mine.detect_friction(_session(messages=[])) == []


def test_get_skill_name_from_title():
    sess = _session(title="skill: my-skill — task", messages=[_msg("user", "不对")])
    cards = mine.detect_friction(sess)
    assert cards[0].skill_name == "my-skill"


def test_tool_retry_same_tool_multiple_errors():
    sess = _session(
        messages=[
            _msg("user", "deploy"),
            _msg("tool", "error: timeout", tool_name="terminal"),
            _msg("tool", "error: timeout again", tool_name="terminal"),
        ]
    )
    cards = mine.detect_friction(sess)
    assert any("tool_retry" in e for e in cards[0].friction_evidence)


# ── deduplicate ───────────────────────────────────────────────────────────


def test_deduplicate_keeps_best():
    c1 = TaskCard("s", "a", "req", ["user_correction: a"], [], 1)
    c2 = TaskCard("s", "b", "req", ["user_correction: a", "tool_error: x"], [], 2)
    c3 = TaskCard("other", "c", "req", ["user_correction: a"], [], 3)
    out = mine.deduplicate([c1, c2, c3])
    # s::user_correction bucket keeps c2 (more evidence), other keeps c3
    assert len(out) == 2
    # find s bucket
    s_cards = [x for x in out if x.skill_name == "s"]
    assert s_cards[0].session_id == "b"


# ── write_task_cards ──────────────────────────────────────────────────────


def test_write_task_cards(tmp_path):
    cards = [TaskCard("s", "id1", "hello", ["user_correction: hi"], [], 123.0)]
    out = mine.write_task_cards(cards, str(tmp_path), total_sessions_scanned=5)
    data = json.loads(Path(out).read_text(encoding="utf-8"))
    assert data["total_cards"] == 1
    assert data["total_sessions_scanned"] == 5
    assert data["tasks"][0]["skill_name"] == "s"
    assert "generated_at" in data


# ── CLI: cron skip integration ────────────────────────────────────────────


def test_cron_sessions_skipped_in_pipeline(tmp_path):
    # Simulate export returning one cron + one friction session
    cron = _session("cron_abc", messages=[_msg("user", "不对")])
    good = _session("sess_good", messages=[_msg("user", "不对，错了")])
    payload = "\n".join([json.dumps(cron), json.dumps(good)])
    mock = MagicMock(returncode=0, stdout=payload, stderr="")
    with patch("pipeline.mine.subprocess.run", return_value=mock):
        sessions = mine.export_sessions("7d")
    user_sessions = [s for s in sessions if not str(s.get("id", "")).startswith("cron_")]
    assert len(user_sessions) == 1
    assert user_sessions[0]["id"] == "sess_good"
    cards = []
    for s in user_sessions:
        cards.extend(mine.detect_friction(s))
    assert len(cards) == 1
    out = mine.write_task_cards(mine.deduplicate(cards), str(tmp_path), total_sessions_scanned=len(user_sessions))
    assert Path(out).exists()


# ── P11: Seen Fingerprint Deduplication ────────────────────────────────────


def test_compute_fingerprint_deterministic():
    sid = "sess_100"
    ev1 = ["user_correction: 不对，重新执行", "tool_error: terminal — exit_code 1"]
    ev2 = ["user_correction: 不对，重新执行", "tool_error: terminal — exit_code 1"]
    ev3 = ["user_correction: 不同的摩擦证据"]

    fp1 = mine.compute_fingerprint(sid, ev1)
    fp2 = mine.compute_fingerprint(sid, ev2)
    fp3 = mine.compute_fingerprint(sid, ev3)
    fp_diff_sid = mine.compute_fingerprint("sess_200", ev1)

    assert fp1 == fp2
    assert fp1 != fp3
    assert fp1 != fp_diff_sid
    assert fp1.startswith("sess_100:")
    parts = fp1.split(":", 1)
    assert len(parts[1]) == 12


def test_load_and_save_seen_fingerprints(tmp_path):
    seen_file = tmp_path / "seen.json"

    # Non-existent file returns empty set
    assert mine.load_seen_fingerprints(seen_file) == set()

    # Save and reload
    fps = {"sess_1:1234567890ab", "sess_2:cdef12345678"}
    mine.save_seen_fingerprints(seen_file, fps)
    loaded = mine.load_seen_fingerprints(seen_file)
    assert loaded == fps

    # Test loading legacy/alternative list format
    seen_file.write_text(json.dumps(["fp_a", "fp_b"]), encoding="utf-8")
    assert mine.load_seen_fingerprints(seen_file) == {"fp_a", "fp_b"}

    # Corrupted JSON returns empty set without crashing
    seen_file.write_text("invalid json {", encoding="utf-8")
    assert mine.load_seen_fingerprints(seen_file) == set()


def test_filter_seen_cards():
    c1 = TaskCard("s1", "sess_1", "req1", ["user_correction: 不对"], [], 100.0)
    c2 = TaskCard("s2", "sess_2", "req2", ["tool_error: terminal — fail"], [], 200.0)

    fp1 = mine.compute_fingerprint(c1.session_id, c1.friction_evidence)

    # With fp1 in seen set
    fresh, seen = mine.filter_seen_cards([c1, c2], {fp1})
    assert len(fresh) == 1
    assert fresh[0].session_id == "sess_2"
    assert len(seen) == 1
    assert seen[0].session_id == "sess_1"


def test_mine_seen_dedup_run_twice(tmp_path):
    # Simulate a single friction session
    mock_session = _session("sess_fixed_id", title="skill: test-skill", messages=[_msg("user", "不对，错了")])

    seen_file = tmp_path / "seen.json"
    run1_dir = tmp_path / "run1"
    run2_dir = tmp_path / "run2"
    run3_dir = tmp_path / "run3"

    # Run 1: First run — card should be produced and fingerprint recorded
    with patch("pipeline.mine.export_sessions", return_value=[mock_session]):
        out1, fresh1, seen1 = mine.run_mine(
            seen_file=str(seen_file),
            output_dir=str(run1_dir),
        )
    assert len(fresh1) == 1
    assert len(seen1) == 0
    assert (run1_dir / "tasks.json").exists()
    data1 = json.loads((run1_dir / "tasks.json").read_text(encoding="utf-8"))
    assert data1["total_cards"] == 1
    assert len(data1["tasks"]) == 1
    assert seen_file.exists()
    seen_fps = mine.load_seen_fingerprints(seen_file)
    assert len(seen_fps) == 1

    # Run 2: Second run with same session — card should be marked seen and skipped
    with patch("pipeline.mine.export_sessions", return_value=[mock_session]):
        out2, fresh2, seen2 = mine.run_mine(
            seen_file=str(seen_file),
            output_dir=str(run2_dir),
        )
    assert len(fresh2) == 0
    assert len(seen2) == 1
    assert (run2_dir / "tasks.json").exists()
    data2 = json.loads((run2_dir / "tasks.json").read_text(encoding="utf-8"))
    assert data2["total_cards"] == 0
    assert data2["seen_cards_skipped"] == 1
    assert len(data2["tasks"]) == 0

    # Run 3: Third run with --reset-seen — card should be fresh again
    with patch("pipeline.mine.export_sessions", return_value=[mock_session]):
        out3, fresh3, seen3 = mine.run_mine(
            seen_file=str(seen_file),
            output_dir=str(run3_dir),
            reset_seen=True,
        )
    assert len(fresh3) == 1
    assert len(seen3) == 0
    assert (run3_dir / "tasks.json").exists()
    data3 = json.loads((run3_dir / "tasks.json").read_text(encoding="utf-8"))
    assert data3["total_cards"] == 1
    assert len(data3["tasks"]) == 1


def test_mine_parser_seen_args():
    parser = mine.build_parser()
    args = parser.parse_args(["--seen-file", "/tmp/custom-seen.json", "--reset-seen"])
    assert args.seen_file == "/tmp/custom-seen.json"
    assert args.reset_seen is True


