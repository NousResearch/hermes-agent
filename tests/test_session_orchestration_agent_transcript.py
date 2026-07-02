"""Tests for session_orchestration/agent_transcript.py."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from session_orchestration.agent_transcript import (
    discover_omp_session_file,
    read_assistant_texts,
)


def _msg(role: str, blocks: list[dict]) -> str:
    return json.dumps({"type": "message", "message": {"role": role, "content": blocks}})


def _write_transcript(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# read_assistant_texts
# ---------------------------------------------------------------------------


class TestReadAssistantTexts:
    def test_extracts_only_assistant_text_blocks(self, tmp_path):
        f = tmp_path / "s.jsonl"
        _write_transcript(f, [
            json.dumps({"type": "session", "id": "x", "cwd": "/repo"}),
            _msg("user", [{"type": "text", "text": "do the thing"}]),
            _msg("assistant", [
                {"type": "thinking", "thinking": "hmm"},
                {"type": "toolCall", "name": "bash"},
            ]),
            _msg("toolResult", [{"type": "text", "text": "build log spam"}]),
            _msg("assistant", [
                {"type": "thinking", "thinking": "ok"},
                {"type": "text", "text": "Here is the answer."},
            ]),
        ])
        texts, total = read_assistant_texts(str(f), 0)
        assert texts == ["Here is the answer."]
        assert total == 5

    def test_offset_skips_consumed_lines(self, tmp_path):
        f = tmp_path / "s.jsonl"
        _write_transcript(f, [
            _msg("assistant", [{"type": "text", "text": "first turn"}]),
            _msg("assistant", [{"type": "text", "text": "second turn"}]),
        ])
        texts1, off1 = read_assistant_texts(str(f), 0)
        assert texts1 == ["first turn", "second turn"]
        # Append a new turn; only it should be returned from the new offset.
        with open(f, "a", encoding="utf-8") as fh:
            fh.write(_msg("assistant", [{"type": "text", "text": "third turn"}]) + "\n")
        texts2, off2 = read_assistant_texts(str(f), off1)
        assert texts2 == ["third turn"]
        assert off2 == off1 + 1

    def test_malformed_lines_skipped(self, tmp_path):
        f = tmp_path / "s.jsonl"
        _write_transcript(f, [
            "not json at all {{{",
            _msg("assistant", [{"type": "text", "text": "survives"}]),
        ])
        texts, total = read_assistant_texts(str(f), 0)
        assert texts == ["survives"]
        assert total == 2

    def test_missing_file_returns_unchanged_offset(self, tmp_path):
        texts, off = read_assistant_texts(str(tmp_path / "nope.jsonl"), 7)
        assert texts == []
        assert off == 7

    def test_multiple_text_blocks_joined(self, tmp_path):
        f = tmp_path / "s.jsonl"
        _write_transcript(f, [
            _msg("assistant", [
                {"type": "text", "text": "para one"},
                {"type": "text", "text": "para two"},
            ]),
        ])
        texts, _ = read_assistant_texts(str(f), 0)
        assert texts == ["para one\npara two"]


# ---------------------------------------------------------------------------
# discover_omp_session_file
# ---------------------------------------------------------------------------


class TestDiscover:
    def _mk(self, root: Path, workdir: str, name: str) -> Path:
        d = root / workdir.rstrip("/").replace("/", "-")
        d.mkdir(parents=True, exist_ok=True)
        p = d / name
        p.write_text("{}\n", encoding="utf-8")
        return p

    def test_finds_file_near_launch_ts(self, tmp_path):
        launch = datetime(2026, 7, 2, 4, 40, 35, tzinfo=timezone.utc)
        p = self._mk(tmp_path, "/dev/z-harness",
                     "2026-07-02T04-40-31-230Z_019f2120.jsonl")
        found = discover_omp_session_file(
            "/dev/z-harness", launch, sessions_root=tmp_path
        )
        assert found == str(p)

    def test_ignores_files_outside_window(self, tmp_path):
        launch = datetime(2026, 7, 2, 12, 0, 0, tzinfo=timezone.utc)
        self._mk(tmp_path, "/dev/z-harness",
                 "2026-07-02T04-40-31-230Z_old.jsonl")
        found = discover_omp_session_file(
            "/dev/z-harness", launch, sessions_root=tmp_path
        )
        assert found is None

    def test_excludes_claimed_and_picks_closest(self, tmp_path):
        launch = datetime(2026, 7, 2, 4, 40, 35, tzinfo=timezone.utc)
        closest = self._mk(tmp_path, "/dev/z-harness",
                           "2026-07-02T04-40-36-000Z_b.jsonl")
        other = self._mk(tmp_path, "/dev/z-harness",
                         "2026-07-02T04-41-30-000Z_c.jsonl")
        # Closest already claimed by a parallel spawn -> pick the other.
        found = discover_omp_session_file(
            "/dev/z-harness", launch, claimed=[str(closest)],
            sessions_root=tmp_path,
        )
        assert found == str(other)

    def test_missing_dir_returns_none(self, tmp_path):
        launch = datetime(2026, 7, 2, tzinfo=timezone.utc)
        assert discover_omp_session_file("/no/such", launch, sessions_root=tmp_path) is None
