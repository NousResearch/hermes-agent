"""Smoke tests for the ical optional skill.

The ical skill is a stdlib-only RFC 5545 parser/generator with a thin
CLI on top. These tests verify the SKILL.md frontmatter conforms to the
hardline format, the script parses as Python, and round-tripping events
through generate_ics → parse_ics preserves the meaningful fields.
"""
from __future__ import annotations

import ast
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

SKILL_DIR = Path(__file__).resolve().parents[2] / "optional-skills" / "productivity" / "ical"


@pytest.fixture(scope="module")
def frontmatter() -> dict:
    src = (SKILL_DIR / "SKILL.md").read_text()
    m = re.search(r"^---\n(.*?)\n---", src, re.DOTALL)
    assert m, "SKILL.md missing YAML frontmatter"
    return yaml.safe_load(m.group(1))


def test_skill_dir_exists() -> None:
    assert SKILL_DIR.is_dir(), f"missing skill dir: {SKILL_DIR}"


def test_skill_md_present() -> None:
    assert (SKILL_DIR / "SKILL.md").is_file()


def test_description_under_60_chars(frontmatter) -> None:
    desc = frontmatter["description"]
    assert len(desc) <= 60, f"description is {len(desc)} chars (hardline ≤60): {desc!r}"


def test_name_matches_dir(frontmatter) -> None:
    assert frontmatter["name"] == "ical"


def test_platforms_covers_all_three_os(frontmatter) -> None:
    assert set(frontmatter["platforms"]) == {"linux", "macos", "windows"}, (
        f"expected all three platforms, got {frontmatter['platforms']}"
    )


def test_author_credits_contributor(frontmatter) -> None:
    assert "Thomas Bale" in frontmatter["author"], (
        f"author should credit the contributor: {frontmatter['author']!r}"
    )


def test_license_mit(frontmatter) -> None:
    assert frontmatter["license"] == "MIT"


def test_shipped_script_parses() -> None:
    src = (SKILL_DIR / "scripts" / "ical.py").read_text()
    ast.parse(src)


# ---------------------------------------------------------------------------
# Library-level round-trip tests
# ---------------------------------------------------------------------------


def _run_ical(*args: str, stdin: str | None = None) -> subprocess.CompletedProcess:
    script = SKILL_DIR / "scripts" / "ical.py"
    return subprocess.run(
        [sys.executable, str(script), *args],
        capture_output=True,
        text=True,
        input=stdin,
        timeout=15,
    )


def test_quick_command_emits_parseable_ical() -> None:
    """The SKILL.md verification step: `quick` → pipe to `read` and confirm
    the table comes back. If either side breaks, the user notices immediately.
    """
    quick = _run_ical("quick", "smoke test", "2026-06-15T12:00:00", "2026-06-15T13:00:00")
    assert quick.returncode == 0, quick.stderr
    assert "BEGIN:VCALENDAR" in quick.stdout
    assert "END:VCALENDAR" in quick.stdout
    assert "SUMMARY:smoke test" in quick.stdout

    read = _run_ical("read", "-", stdin=quick.stdout)
    assert read.returncode == 0, read.stderr
    assert "smoke test" in read.stdout
    assert "2026-06-15 12:00:00" in read.stdout
    assert "2026-06-15 13:00:00" in read.stdout


def test_write_command_round_trip(tmp_path) -> None:
    """Generate an .ics from JSON, re-read it, and confirm the events survived."""
    events_json = tmp_path / "events.json"
    events_json.write_text(json.dumps([
        {
            "summary": "Lunch with Alex",
            "start": "2026-06-15T12:30:00",
            "end": "2026-06-15T13:30:00",
            "location": "Cafe Nord",
            "description": "Catch up; bring the spec",
        },
        {
            "summary": "Standup",
            "start": "2026-06-16T09:00:00",
            "end": "2026-06-16T09:15:00",
        },
    ]))
    ics_path = tmp_path / "out.ics"
    write_result = _run_ical("write", str(events_json), str(ics_path))
    assert write_result.returncode == 0, write_result.stderr
    assert ics_path.is_file()

    # Re-read as JSON and check fields survived.
    read_result = _run_ical("read", str(ics_path), "--json")
    assert read_result.returncode == 0, read_result.stderr
    events = json.loads(read_result.stdout)
    assert len(events) == 2
    assert events[0]["summary"] == "Lunch with Alex"
    assert events[0]["start"] == "2026-06-15T12:30:00"
    assert events[0]["end"] == "2026-06-15T13:30:00"
    assert events[0]["location"] == "Cafe Nord"
    assert events[0]["description"] == "Catch up; bring the spec"
    assert events[1]["summary"] == "Standup"


def test_validate_passes_for_well_formed_calendar(tmp_path) -> None:
    ics = tmp_path / "good.ics"
    ics.write_text(
        "BEGIN:VCALENDAR\r\n"
        "VERSION:2.0\r\n"
        "PRODID:-//test//EN\r\n"
        "BEGIN:VEVENT\r\n"
        "UID:abc@example.com\r\n"
        "DTSTAMP:20260601T120000Z\r\n"
        "DTSTART:20260615T120000Z\r\n"
        "DTEND:20260615T130000Z\r\n"
        "SUMMARY:hello\r\n"
        "END:VEVENT\r\n"
        "END:VCALENDAR\r\n",
        encoding="utf-8",
    )
    result = _run_ical("validate", str(ics))
    assert result.returncode == 0, result.stderr
    assert "OK" in result.stdout


def test_validate_fails_on_missing_calendar_envelope(tmp_path) -> None:
    ics = tmp_path / "bad.ics"
    ics.write_text("BEGIN:VEVENT\r\nSUMMARY:orphan\r\nEND:VEVENT\r\n")
    result = _run_ical("validate", str(ics))
    assert result.returncode != 0


def test_validate_fails_on_missing_required_event_fields(tmp_path) -> None:
    ics = tmp_path / "incomplete.ics"
    ics.write_text(
        "BEGIN:VCALENDAR\r\n"
        "VERSION:2.0\r\n"
        "BEGIN:VEVENT\r\n"
        "SUMMARY:no uid or dtstart\r\n"
        "END:VEVENT\r\n"
        "END:VCALENDAR\r\n",
        encoding="utf-8",
    )
    result = _run_ical("validate", str(ics))
    assert result.returncode != 0
    assert "UID" in result.stderr or "DTSTART" in result.stderr


def test_escape_round_trip_preserves_commas_and_semicolons(tmp_path) -> None:
    """RFC 5545 requires that commas / semicolons / backslashes in text
    fields be escaped on write and unescaped on read. The round-trip must
    preserve them verbatim to the user.
    """
    tricky = "Project: alpha, beta; gamma\\delta"
    events_json = tmp_path / "events.json"
    events_json.write_text(json.dumps([
        {
            "summary": tricky,
            "start": "2026-06-15T12:00:00",
            "end": "2026-06-15T13:00:00",
        }
    ]))
    ics_path = tmp_path / "out.ics"
    _run_ical("write", str(events_json), str(ics_path))

    raw = ics_path.read_text()
    # Comma / semicolon / backslash must appear escaped in the raw file
    # (RFC 5545 §3.3.11).
    assert r"\," in raw or r"\;" in raw or r"\\" in raw, (
        f"expected escaping in raw .ics output, got: {raw!r}"
    )

    # Round-trip recovers the original.
    result = _run_ical("read", str(ics_path), "--json")
    assert json.loads(result.stdout)[0]["summary"] == tricky


def test_long_lines_are_folded(tmp_path) -> None:
    """RFC 5545 §3.1 mandates line folding at 75 octets with a leading
    space on continuation lines. Generated .ics files must obey this.
    """
    long_summary = "x" * 200
    events_json = tmp_path / "events.json"
    events_json.write_text(json.dumps([
        {
            "summary": long_summary,
            "start": "2026-06-15T12:00:00",
            "end": "2026-06-15T13:00:00",
        }
    ]))
    ics_path = tmp_path / "out.ics"
    _run_ical("write", str(events_json), str(ics_path))
    raw = ics_path.read_bytes()
    # The summary line was folded: there should be a CRLF + space sequence
    # in the output (RFC 5545 continuation). Read as bytes so we don't lose
    # the CRLF to text-mode newline translation.
    assert b"\r\n " in raw, "expected folded continuation lines in output"


def test_help_exits_zero() -> None:
    result = _run_ical("--help")
    assert result.returncode == 0
    assert "read" in result.stdout
    assert "write" in result.stdout
    assert "validate" in result.stdout
