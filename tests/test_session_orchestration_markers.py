"""
Unit tests for session_orchestration/markers.py.

Covers:
- Round-trip append/read of every marker kind.
- Version field presence (v == MARKER_SCHEMA_VERSION).
- Malformed lines are skipped, not fatal.
- read_markers_since with non-zero offset returns only new lines.
- marker_kind_to_lifecycle maps every kind to the correct SessionLifecycle.
- MARKER_KINDS contains all expected kinds.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from session_orchestration.markers import (
    MARKER_DONE,
    MARKER_HANDOFF_CONTINUE,
    MARKER_HANDOFF_DECISION,
    MARKER_HEARTBEAT,
    MARKER_KINDS,
    MARKER_NEEDS_INPUT,
    MARKER_SCHEMA_VERSION,
    MARKER_STATUS,
    append_marker,
    marker_kind_to_lifecycle,
    read_markers_since,
)
from session_orchestration.types import SessionLifecycle


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def marker_file(tmp_path) -> Path:
    """Return a path to a fresh marker file (not yet created)."""
    return tmp_path / ".hermes" / "sessions" / "test-session.jsonl"


# ---------------------------------------------------------------------------
# Round-trip: every kind
# ---------------------------------------------------------------------------


class TestRoundTrip:
    """Append then read back each marker kind; verify envelope fields."""

    CASES: list[tuple[str, dict]] = [
        (MARKER_STATUS, {"phase": "running", "detail": "Processing step 1"}),
        (MARKER_HEARTBEAT, {"note": "still alive"}),
        (MARKER_HEARTBEAT, {"note": None}),
        (MARKER_NEEDS_INPUT, {"question": "Which branch?", "options": ["main", "dev"], "context": "git context"}),
        (MARKER_NEEDS_INPUT, {"question": "Confirm?", "options": None, "context": None}),
        (MARKER_HANDOFF_CONTINUE, {"handoff_text": "Resuming after clear checkpoint"}),
        (MARKER_HANDOFF_DECISION, {"question": "Merge now?", "handoff_text": "At merge checkpoint"}),
        (MARKER_DONE, {"summary": "Task completed", "artifacts": ["output.txt"]}),
        (MARKER_DONE, {"summary": "Task completed", "artifacts": None}),
    ]

    @pytest.mark.parametrize("kind,payload", CASES)
    def test_round_trip(self, marker_file: Path, kind: str, payload: dict):
        """append then read returns an envelope with the correct kind and payload."""
        append_marker(marker_file, kind, payload, task="test-task-1")
        markers, offset = read_markers_since(marker_file, 0)

        assert len(markers) == 1
        m = markers[0]
        assert m["kind"] == kind
        assert m["task"] == "test-task-1"
        assert m["payload"] == payload
        assert offset > 0

    def test_version_field_present_and_correct(self, marker_file: Path):
        """Every appended marker carries v == MARKER_SCHEMA_VERSION."""
        for kind, payload in self.CASES:
            append_marker(marker_file, kind, payload, task="t")

        markers, _ = read_markers_since(marker_file, 0)
        assert len(markers) == len(self.CASES)
        for m in markers:
            assert "v" in m
            assert m["v"] == MARKER_SCHEMA_VERSION

    def test_ts_field_present(self, marker_file: Path):
        """Every marker has a non-empty 'ts' string."""
        append_marker(marker_file, MARKER_STATUS, {"phase": "init", "detail": ""}, task="t")
        markers, _ = read_markers_since(marker_file, 0)
        assert markers[0]["ts"]  # non-empty string

    def test_all_kinds_appended_and_read(self, marker_file: Path):
        """All MARKER_KINDS can be appended and are read back intact."""
        payloads = {
            MARKER_STATUS: {"phase": "init", "detail": "x"},
            MARKER_HEARTBEAT: {"note": None},
            MARKER_NEEDS_INPUT: {"question": "q", "options": None, "context": None},
            MARKER_HANDOFF_CONTINUE: {"handoff_text": "ht"},
            MARKER_HANDOFF_DECISION: {"question": "q", "handoff_text": "ht"},
            MARKER_DONE: {"summary": "s", "artifacts": None},
        }
        assert MARKER_KINDS == set(payloads.keys()), "test is missing a kind"

        for kind in sorted(MARKER_KINDS):
            append_marker(marker_file, kind, payloads[kind], task="t")

        markers, _ = read_markers_since(marker_file, 0)
        read_kinds = {m["kind"] for m in markers}
        assert read_kinds == MARKER_KINDS


# ---------------------------------------------------------------------------
# Malformed-line skipping
# ---------------------------------------------------------------------------


class TestMalformedLines:
    """Malformed lines must be silently skipped; valid lines must survive."""

    def test_invalid_json_skipped(self, marker_file: Path):
        """A non-JSON line does not prevent valid markers from being read."""
        marker_file.parent.mkdir(parents=True, exist_ok=True)
        # Write a corrupt line directly, then a valid marker.
        marker_file.write_bytes(b"this is not json\n")
        append_marker(marker_file, MARKER_HEARTBEAT, {"note": "ok"}, task="t")

        markers, _ = read_markers_since(marker_file, 0)
        assert len(markers) == 1
        assert markers[0]["kind"] == MARKER_HEARTBEAT

    def test_missing_envelope_fields_skipped(self, marker_file: Path):
        """A JSON object that lacks required envelope fields is skipped."""
        marker_file.parent.mkdir(parents=True, exist_ok=True)
        # Missing 'kind', 'ts', 'task', 'payload'.
        bad = json.dumps({"v": 1, "data": "incomplete"}) + "\n"
        marker_file.write_bytes(bad.encode())
        append_marker(marker_file, MARKER_DONE, {"summary": "done", "artifacts": None}, task="t")

        markers, _ = read_markers_since(marker_file, 0)
        assert len(markers) == 1
        assert markers[0]["kind"] == MARKER_DONE

    def test_empty_lines_skipped(self, marker_file: Path):
        """Blank lines do not crash read_markers_since."""
        marker_file.parent.mkdir(parents=True, exist_ok=True)
        marker_file.write_bytes(b"\n\n\n")
        append_marker(marker_file, MARKER_STATUS, {"phase": "x", "detail": "y"}, task="t")

        markers, _ = read_markers_since(marker_file, 0)
        assert len(markers) == 1

    def test_all_malformed_returns_empty_list(self, marker_file: Path):
        """A file containing only malformed lines returns an empty markers list."""
        marker_file.parent.mkdir(parents=True, exist_ok=True)
        marker_file.write_bytes(b"bad\n{}\n{\"v\":1}\n")

        markers, _ = read_markers_since(marker_file, 0)
        assert markers == []

    def test_payload_not_dict_skipped(self, marker_file: Path):
        """A line where 'payload' is not a dict is skipped."""
        marker_file.parent.mkdir(parents=True, exist_ok=True)
        bad = json.dumps({"v": 1, "ts": "2024-01-01T00:00:00+00:00", "kind": "done", "task": "t", "payload": "string"}) + "\n"
        marker_file.write_bytes(bad.encode())
        append_marker(marker_file, MARKER_HEARTBEAT, {"note": None}, task="t")

        markers, _ = read_markers_since(marker_file, 0)
        assert len(markers) == 1
        assert markers[0]["kind"] == MARKER_HEARTBEAT


# ---------------------------------------------------------------------------
# read-since-offset returns only new lines
# ---------------------------------------------------------------------------


class TestReadSinceOffset:
    """The offset mechanism must skip already-consumed bytes."""

    def test_offset_skips_earlier_lines(self, marker_file: Path):
        """After reading with offset=0, a second call with new_offset skips old lines."""
        append_marker(marker_file, MARKER_STATUS, {"phase": "a", "detail": "first"}, task="t")
        markers1, offset1 = read_markers_since(marker_file, 0)
        assert len(markers1) == 1

        append_marker(marker_file, MARKER_HEARTBEAT, {"note": "second"}, task="t")
        markers2, offset2 = read_markers_since(marker_file, offset1)

        assert len(markers2) == 1
        assert markers2[0]["kind"] == MARKER_HEARTBEAT
        assert offset2 > offset1

    def test_repeated_offset_returns_empty(self, marker_file: Path):
        """Calling read_markers_since with the same offset twice returns empty on second call."""
        append_marker(marker_file, MARKER_DONE, {"summary": "s", "artifacts": None}, task="t")
        _, offset = read_markers_since(marker_file, 0)

        markers, new_offset = read_markers_since(marker_file, offset)
        assert markers == []
        assert new_offset == offset

    def test_offset_accumulates_across_multiple_appends(self, marker_file: Path):
        """A rolling offset over three appends yields exactly one marker per read."""
        kinds = [MARKER_STATUS, MARKER_HEARTBEAT, MARKER_DONE]
        payloads = [
            {"phase": "p", "detail": "d"},
            {"note": None},
            {"summary": "s", "artifacts": None},
        ]

        offset = 0
        for kind, payload in zip(kinds, payloads):
            append_marker(marker_file, kind, payload, task="t")
            ms, offset = read_markers_since(marker_file, offset)
            assert len(ms) == 1, f"expected exactly 1 new marker for kind={kind}"
            assert ms[0]["kind"] == kind

    def test_zero_offset_reads_all_lines(self, marker_file: Path):
        """offset=0 always reads from the beginning of the file."""
        for _ in range(5):
            append_marker(marker_file, MARKER_HEARTBEAT, {"note": None}, task="t")

        markers, _ = read_markers_since(marker_file, 0)
        assert len(markers) == 5

    def test_nonexistent_file_returns_empty_at_same_offset(self, tmp_path: Path):
        """A missing file returns ([], offset) without raising."""
        path = tmp_path / "no_such_file.jsonl"
        markers, new_offset = read_markers_since(path, 42)
        assert markers == []
        assert new_offset == 42


# ---------------------------------------------------------------------------
# marker_kind_to_lifecycle
# ---------------------------------------------------------------------------


class TestMarkerKindToLifecycle:
    """Every kind maps to the expected SessionLifecycle value."""

    EXPECTED: list[tuple[str, SessionLifecycle]] = [
        (MARKER_STATUS, SessionLifecycle.RUNNING),
        (MARKER_HEARTBEAT, SessionLifecycle.RUNNING),
        (MARKER_NEEDS_INPUT, SessionLifecycle.WAITING_USER),
        (MARKER_HANDOFF_CONTINUE, SessionLifecycle.PAUSED_HANDOFF),
        (MARKER_HANDOFF_DECISION, SessionLifecycle.PAUSED_HANDOFF),
        (MARKER_DONE, SessionLifecycle.DONE),
    ]

    @pytest.mark.parametrize("kind,expected_lifecycle", EXPECTED)
    def test_known_kind_maps_correctly(self, kind: str, expected_lifecycle: SessionLifecycle):
        assert marker_kind_to_lifecycle(kind) == expected_lifecycle

    def test_unknown_kind_returns_none(self):
        """An unrecognised kind returns None rather than raising."""
        assert marker_kind_to_lifecycle("totally_unknown_kind") is None

    def test_all_marker_kinds_are_mapped(self):
        """MARKER_KINDS has no unmapped members — table is complete."""
        for kind in MARKER_KINDS:
            result = marker_kind_to_lifecycle(kind)
            assert result is not None, f"kind={kind!r} is missing from lifecycle mapping"

    def test_lifecycle_values_are_sessionlifecycle_members(self):
        """All mapped values are members of the SessionLifecycle enum."""
        for kind in MARKER_KINDS:
            lc = marker_kind_to_lifecycle(kind)
            assert isinstance(lc, SessionLifecycle)


# ---------------------------------------------------------------------------
# Parent-directory creation
# ---------------------------------------------------------------------------


class TestParentDirCreation:
    """append_marker must create missing parent directories."""

    def test_nested_parent_created(self, tmp_path: Path):
        deep = tmp_path / "a" / "b" / "c" / "session.jsonl"
        assert not deep.parent.exists()
        append_marker(deep, MARKER_HEARTBEAT, {"note": None}, task="t")
        assert deep.exists()
        markers, _ = read_markers_since(deep, 0)
        assert len(markers) == 1
