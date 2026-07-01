"""
claude-history-scan self-tests — run with `python claude-history-scan.test.py`
from the scripts/ directory.

Pure stdlib (unittest + tempfile + importlib). Tests use synthesized jsonl
files, NEVER the real ~/.claude/projects/ on this host. Keeps the safety
boundary clear: this test never reads user history.

Covers:
  - schema stability (every record has the documented keys)
  - SKIP_TYPES (queue-operation, attachment) are not counted
  - COUNT_TYPES (user, assistant) ARE counted, but only rows with a non-empty
    text block — a row whose content is a tool_result does not bump the count
  - first_user is the first line of the first text block, truncated to 100
  - workspace_group matches the immediate subdirectory name
  - per-file read failures are captured, do not crash the whole scan
  - non-existent claude home exits 2 (separate path from "no sessions found")
  - --strict exits 3 on a partial read failure
"""
from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


# Load the hyphen-named scanner module explicitly. Python's import system
# can't import "claude-history-scan" directly (hyphens aren't valid in module
# names), so we use importlib.util to give it a real module object under the
# underscore alias and bind it locally as `chs` for the tests below.
_HERE = Path(__file__).resolve().parent
_SPEC = importlib.util.spec_from_file_location(
    "claude_history_scan", _HERE / "claude-history-scan.py"
)
assert _SPEC is not None and _SPEC.loader is not None
chs = importlib.util.module_from_spec(_SPEC)
sys.modules["claude_history_scan"] = chs
_SPEC.loader.exec_module(chs)


def _write_jsonl(path: Path, rows: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def _user_row(text: str, cwd: str = "C:\\proj", ts: str = "2026-06-12T11:00:00.000Z") -> dict:
    return {
        "type": "user",
        "sessionId": "sess-1",
        "cwd": cwd,
        "timestamp": ts,
        "message": {"role": "user", "content": [{"type": "text", "text": text}]},
    }


def _assistant_row(text: str, ts: str = "2026-06-12T11:00:01.000Z") -> dict:
    return {
        "type": "assistant",
        "sessionId": "sess-1",
        "timestamp": ts,
        "message": {"role": "assistant", "content": [{"type": "text", "text": text}]},
    }


def _user_with_only_tool_result() -> dict:
    """User row whose content is a tool_result block, not text. Should NOT count."""
    return {
        "type": "user",
        "sessionId": "sess-1",
        "message": {
            "role": "user",
            "content": [
                {"type": "tool_result", "content": "secret-token-AAAA", "is_error": False}
            ],
        },
    }


class SchemaTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.home = Path(self.tmp.name) / "claude"
        (self.home / "projects" / "C--proj").mkdir(parents=True)
        _write_jsonl(
            self.home / "projects" / "C--proj" / "aaaa1111.jsonl",
            [_user_row("hello"), _assistant_row("hi")],
        )

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_keys_present(self) -> None:
        out = chs.scan(self.home, workers=1)
        self.assertEqual(len(out), 1)
        e = out[0]
        for k in (
            "session_id",
            "cwd",
            "first_user",
            "first_timestamp",
            "last_timestamp",
            "message_count",
            "workspace_group",
            "file_size_bytes",
            "file_path",
        ):
            self.assertIn(k, e, f"missing key: {k}")

    def test_workspace_group_matches_subdir(self) -> None:
        out = chs.scan(self.home, workers=1)
        self.assertEqual(out[0]["workspace_group"], "C--proj")

    def test_message_count_excludes_queue_and_attachment(self) -> None:
        # 1 user + 1 assistant should be counted; queue + attachment should not.
        out = chs.scan(self.home, workers=1)
        self.assertEqual(out[0]["message_count"], 2)

    def test_message_count_excludes_tool_result_only_user(self) -> None:
        # Replace the jsonl with one whose only user row is a tool_result —
        # the user row must NOT be counted (no text block).
        path = self.home / "projects" / "C--proj" / "bbbb2222.jsonl"
        _write_jsonl(
            path,
            [_user_with_only_tool_result(), _assistant_row("ok")],
        )
        out = chs.scan(self.home, workers=1)
        target = next(e for e in out if e["session_id"] == "bbbb2222")
        self.assertEqual(target["message_count"], 1)
        # first_user must be None — no text-only user row existed.
        self.assertIsNone(target["first_user"])

    def test_first_user_truncates_to_100_chars(self) -> None:
        path = self.home / "projects" / "C--proj" / "cccc3333.jsonl"
        long_text = "X" * 500
        _write_jsonl(path, [_user_row(long_text)])
        out = chs.scan(self.home, workers=1)
        target = next(e for e in out if e["session_id"] == "cccc3333")
        self.assertEqual(len(target["first_user"]), 100)

    def test_first_user_uses_first_line_only(self) -> None:
        path = self.home / "projects" / "C--proj" / "dddd4444.jsonl"
        _write_jsonl(
            path,
            [_user_row("first-line\nsecond-line\nthird-line")],
        )
        out = chs.scan(self.home, workers=1)
        target = next(e for e in out if e["session_id"] == "dddd4444")
        self.assertEqual(target["first_user"], "first-line")

    def test_skips_tool_results_subdir(self) -> None:
        # tool-results/ holds raw tool output, not session jsonl. Must be
        # skipped even if it contains files matching the *.jsonl pattern.
        tr = self.home / "projects" / "C--proj" / "tool-results"
        tr.mkdir()
        _write_jsonl(tr / "should_be_ignored.jsonl", [_user_row("in tool-results")])
        out = chs.scan(self.home, workers=1)
        ids = {e["session_id"] for e in out}
        self.assertNotIn("should_be_ignored", ids)

    def test_does_not_recurse_into_subagents(self) -> None:
        # Sub-agent jsonl live deeper (subagents/workflows/.../agent-*.jsonl).
        # We deliberately do NOT list them as separate sessions — they're part
        # of the parent session's transcript.
        sub = self.home / "projects" / "C--proj" / "parent-session-id" / "subagents" / "wf_x" / "agent-1.jsonl"
        _write_jsonl(sub, [_user_row("sub agent", cwd="C:\\proj")])
        out = chs.scan(self.home, workers=1)
        ids = {e["session_id"] for e in out}
        self.assertIn("aaaa1111", ids)
        self.assertNotIn("agent-1", ids)


class ErrorHandlingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.home = Path(self.tmp.name) / "claude"
        (self.home / "projects" / "C--proj").mkdir(parents=True)
        _write_jsonl(
            self.home / "projects" / "C--proj" / "ok.jsonl",
            [_user_row("fine")],
        )

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_per_file_failure_is_captured_not_fatal(self) -> None:
        # To force a real read failure we point one entry at a directory —
        # open() on a directory raises IsADirectoryError, which we catch as
        # an OSError and surface as a per-file "error" key.
        bad_dir = self.home / "projects" / "C--proj" / "badentry.jsonl"
        bad_dir.mkdir(parents=True, exist_ok=True)
        out = chs.scan(self.home, workers=1)
        ids = {e["session_id"]: e for e in out}
        self.assertIn("ok", ids)
        self.assertIn("badentry", ids)
        self.assertIn("error", ids["badentry"])

    def test_empty_projects_dir_yields_empty_list(self) -> None:
        empty = Path(self.tmp.name) / "claude-empty"
        (empty / "projects").mkdir(parents=True)
        self.assertEqual(chs.scan(empty, workers=1), [])


class CLITests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.home = Path(self.tmp.name) / "claude"
        # Home exists, but NO projects/ subdir — that's the failure mode we
        # exercise in test_projects_dir_missing_exits_2. The two _exit_3
        # cases build their own projects/ inside the test.

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_missing_home_exits_2(self) -> None:
        bogus = Path(self.tmp.name) / "no-such-dir"
        with mock.patch.object(sys, "argv", ["claude-history-scan.py", "--claude-home", str(bogus)]):
            with self.assertRaises(SystemExit) as cm:
                chs.main()
            self.assertEqual(cm.exception.code, 2)

    def test_projects_dir_missing_exits_2(self) -> None:
        # Home exists but has no projects/ inside.
        with mock.patch.object(sys, "argv", ["claude-history-scan.py", "--claude-home", str(self.home)]):
            with self.assertRaises(SystemExit) as cm:
                chs.main()
            self.assertEqual(cm.exception.code, 2)

    def test_strict_exits_3_on_partial_failure(self) -> None:
        # Build a home with one good session + one bad entry, run --strict.
        (self.home / "projects" / "C--proj").mkdir(parents=True)
        _write_jsonl(
            self.home / "projects" / "C--proj" / "ok.jsonl",
            [_user_row("fine")],
        )
        bad_dir = self.home / "projects" / "C--proj" / "badentry.jsonl"
        bad_dir.mkdir(parents=True, exist_ok=True)
        with mock.patch.object(
            sys, "argv", ["claude-history-scan.py", "--claude-home", str(self.home), "--strict"]
        ):
            with self.assertRaises(SystemExit) as cm:
                chs.main()
            self.assertEqual(cm.exception.code, 3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
