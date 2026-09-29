#!/usr/bin/env python3
"""
Robin integration selftest.

Default mode uses a temporary state directory and exercises Robin's command
surface without touching the user's real library. Passing --state-dir performs
non-destructive setup checks only.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"


class SelftestFailure(Exception):
    pass


class Reporter:
    def __init__(self) -> None:
        self.results: list[tuple[bool, str, str | None]] = []

    def check(self, label: str, func) -> Any:
        try:
            result = func()
        except Exception as exc:
            self.results.append((False, label, str(exc)))
            return None
        self.results.append((True, label, None))
        return result

    def print_report(self, state_dir: Path) -> None:
        print(f"Robin integration check: {state_dir}")
        for passed, label, error in self.results:
            status = "PASS" if passed else "FAIL"
            print(f"  [{status}] {label}")
            if error:
                print(f"         {error}")
        passed = sum(1 for item in self.results if item[0])
        total = len(self.results)
        print(f"  {passed}/{total} passed")

    @property
    def ok(self) -> bool:
        return all(passed for passed, _, _ in self.results)


def _script(name: str) -> str:
    return str(SCRIPTS / name)


def _run_json(args: list[str], *, expect_success: bool = True) -> dict | list:
    proc = subprocess.run(
        [sys.executable, *args],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if expect_success and proc.returncode != 0:
        raise SelftestFailure(f"Command failed: {' '.join(args)}\n{proc.stderr}{proc.stdout}")
    if not expect_success and proc.returncode == 0:
        raise SelftestFailure(f"Command unexpectedly succeeded: {' '.join(args)}\n{proc.stdout}")
    output = proc.stdout.strip() or proc.stderr.strip()
    try:
        return json.loads(output)
    except json.JSONDecodeError as exc:
        raise SelftestFailure(
            f"Command did not return valid JSON: {' '.join(args)}\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        ) from exc


def _require_keys(payload: dict, keys: set[str]) -> None:
    missing = sorted(keys - set(payload))
    if missing:
        raise SelftestFailure(f"Missing keys: {', '.join(missing)}")


def _fail_selftest(message: str) -> None:
    raise SelftestFailure(message)


def _write_config(state_dir: Path) -> None:
    (state_dir / "topics").mkdir(parents=True, exist_ok=True)
    (state_dir / "media").mkdir(parents=True, exist_ok=True)
    config = {
        "topics_dir": "topics",
        "media_dir": "media",
        "min_items_before_review": 1,
        "review_cooldown_days": 0,
    }
    (state_dir / "robin-config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")


def _check_setup(state_dir: Path) -> None:
    config_path = state_dir / "robin-config.json"
    if not config_path.exists():
        raise SelftestFailure(f"Missing {config_path}")
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SelftestFailure(f"{config_path} is invalid JSON: {exc}") from exc
    if not isinstance(config, dict):
        raise SelftestFailure(f"{config_path} must contain a JSON object")
    for dirname in (config.get("topics_dir", "topics"), config.get("media_dir", "media")):
        path = state_dir / str(dirname)
        if not path.is_dir():
            raise SelftestFailure(f"Missing directory: {path}")


def _check_topics_json(state_dir: Path) -> list[dict]:
    payload = _run_json([_script("topics.py"), "--state-dir", str(state_dir), "--json"])
    if not isinstance(payload, list):
        raise SelftestFailure("topics.py --json did not return a list")
    for item in payload:
        if not isinstance(item, dict):
            raise SelftestFailure("topics.py --json returned a non-object topic item")
        _require_keys(item, {"topic", "filename", "entries", "rated", "unrated"})
    return payload


def _add_text_entry(state_dir: Path, *, allow_duplicate: bool = False) -> dict:
    payload = _run_json(
        [
            _script("add_entry.py"),
            "--state-dir",
            str(state_dir),
            "--topic",
            "Robin Selftest",
            "--content",
            "Robin selftest distinctive phrase alpha bravo.",
            "--description",
            "A temporary selftest entry used to verify Robin's write path.",
            "--tags",
            "selftest,robin",
            *(["--allow-duplicate"] if allow_duplicate else []),
            "--json",
        ]
    )
    if not isinstance(payload, dict):
        raise SelftestFailure("add_entry.py --json did not return an object")
    _require_keys(payload, {"id", "topic", "filename", "entry_type", "media_source", "description"})
    if payload["topic"] != "robin-selftest" or payload["entry_type"] != "text":
        raise SelftestFailure(f"Unexpected add payload: {payload}")
    return payload


def _check_topic_added(state_dir: Path) -> None:
    topics = _check_topics_json(state_dir)
    match = next((item for item in topics if item.get("topic") == "robin-selftest"), None)
    if not match or match.get("entries") != 1:
        raise SelftestFailure(f"Expected robin-selftest topic with one entry, got {topics}")


def _check_search_finds_entry(state_dir: Path, entry_id: str) -> None:
    payload = _run_json(
        [
            _script("search.py"),
            "--state-dir",
            str(state_dir),
            "alpha bravo",
            "--json",
        ]
    )
    if not isinstance(payload, dict):
        raise SelftestFailure("search.py --json did not return an object")
    _require_keys(payload, {"count", "entries"})
    entries = payload.get("entries")
    if not isinstance(entries, list) or not any(entry.get("id") == entry_id for entry in entries):
        raise SelftestFailure(f"Search did not return the added entry: {payload}")


def _check_review_and_rate(state_dir: Path) -> None:
    payload = _run_json([_script("review.py"), "--state-dir", str(state_dir), "--active-review", "--json"])
    if not isinstance(payload, dict):
        raise SelftestFailure("review.py --json did not return an object")
    _require_keys(
        payload,
        {
            "status",
            "id",
            "topic",
            "date_added",
            "entry_type",
            "media_kind",
            "media_source",
            "source",
            "description",
            "creator",
            "published_at",
            "summary",
            "tags",
            "body",
            "rating",
            "times_surfaced",
        },
    )
    if payload["status"] != "ok":
        raise SelftestFailure(f"Expected review status ok, got {payload}")

    rated = _run_json(
        [
            _script("review.py"),
            "--state-dir",
            str(state_dir),
            "--rate",
            payload["id"],
            "4",
            "--json",
        ]
    )
    if not isinstance(rated, dict):
        raise SelftestFailure("review.py --rate --json did not return an object")
    _require_keys(rated, {"id", "topic", "date", "rating", "last_surfaced", "times_surfaced", "_awaiting_rating"})
    if rated["rating"] != 4 or rated["_awaiting_rating"] is not False:
        raise SelftestFailure(f"Unexpected rate payload: {rated}")


def _check_duplicate_rejected(state_dir: Path) -> None:
    payload = _run_json(
        [
            _script("add_entry.py"),
            "--state-dir",
            str(state_dir),
            "--topic",
            "Robin Selftest",
            "--content",
            "Robin selftest distinctive phrase alpha bravo.",
            "--description",
            "A temporary selftest entry used to verify Robin's write path.",
            "--tags",
            "selftest,robin",
            "--json",
        ],
        expect_success=False,
    )
    if not isinstance(payload, dict) or "duplicates" not in payload:
        raise SelftestFailure(f"Expected duplicate error payload, got {payload}")


def _check_duplicate_allowed(state_dir: Path) -> None:
    payload = _add_text_entry(state_dir, allow_duplicate=True)
    if not payload.get("id"):
        raise SelftestFailure(f"Duplicate add did not return an id: {payload}")
    topics = _check_topics_json(state_dir)
    match = next((item for item in topics if item.get("topic") == "robin-selftest"), None)
    if not match or match.get("entries") < 2:
        raise SelftestFailure(f"Expected duplicate add to create a second entry, got {topics}")


def _check_entries_move_and_delete(state_dir: Path) -> None:
    payload = _run_json(
        [
            _script("add_entry.py"),
            "--state-dir",
            str(state_dir),
            "--topic",
            "Entry Management",
            "--content",
            "Robin selftest entry management phrase.",
            "--description",
            "A temporary selftest entry used to verify Robin's entry management path.",
            "--json",
        ]
    )
    if not isinstance(payload, dict) or not payload.get("id"):
        raise SelftestFailure(f"Expected add payload for entry management check, got {payload}")
    entry_id = payload["id"]

    moved = _run_json([_script("entries.py"), "--state-dir", str(state_dir), "--move", entry_id, "--topic", "Moved Entries", "--json"])
    if not isinstance(moved, dict) or moved.get("status") != "moved" or moved.get("to_topic") != "moved-entries":
        raise SelftestFailure(f"Expected move payload, got {moved}")

    deleted = _run_json([_script("entries.py"), "--state-dir", str(state_dir), "--delete", entry_id, "--json"])
    if not isinstance(deleted, dict) or deleted.get("status") != "deleted":
        raise SelftestFailure(f"Expected delete payload, got {deleted}")


def _check_local_video_rejected(state_dir: Path) -> None:
    video = state_dir / "clip.mp4"
    video.write_bytes(b"fake video")
    payload = _run_json(
        [
            _script("add_entry.py"),
            "--state-dir",
            str(state_dir),
            "--topic",
            "Robin Selftest",
            "--entry-type",
            "video",
            "--media-path",
            str(video),
            "--description",
            "A rejection-path selftest entry.",
            "--creator",
            "Robin",
            "--published-at",
            "2026",
            "--summary",
            "A fake local video path.",
            "--json",
        ],
        expect_success=False,
    )
    if not isinstance(payload, dict) or "error" not in payload:
        raise SelftestFailure(f"Expected error payload, got {payload}")


def _check_missing_media_metadata_rejected(state_dir: Path) -> None:
    payload = _run_json(
        [
            _script("add_entry.py"),
            "--state-dir",
            str(state_dir),
            "--topic",
            "Robin Selftest",
            "--entry-type",
            "video",
            "--media-url",
            "https://example.com/watch?v=selftest",
            "--description",
            "A rejection-path selftest entry.",
            "--published-at",
            "2026",
            "--summary",
            "Missing creator should reject this media entry.",
            "--json",
        ],
        expect_success=False,
    )
    if not isinstance(payload, dict) or "error" not in payload:
        raise SelftestFailure(f"Expected error payload, got {payload}")


def _check_doctor_json(state_dir: Path) -> None:
    payload = _run_json([_script("doctor.py"), "--state-dir", str(state_dir), "--json"])
    if not isinstance(payload, dict):
        raise SelftestFailure("doctor.py --json did not return an object")
    _require_keys(payload, {"ok", "errors", "warnings", "diagnostics"})
    if payload["ok"] is not True or payload["errors"] != 0:
        raise SelftestFailure(f"Expected healthy doctor payload, got {payload}")


def _check_reindex_after_cleanup(state_dir: Path) -> None:
    for filename in ("robin-selftest.md", "entry-management.md", "moved-entries.md"):
        topic_file = state_dir / "topics" / filename
        if topic_file.exists():
            topic_file.unlink()
    payload = _run_json([_script("reindex.py"), "--state-dir", str(state_dir), "--json"])
    if not isinstance(payload, dict):
        raise SelftestFailure("reindex.py --json did not return an object")
    _require_keys(payload, {"entries_found", "items_indexed", "rated", "unrated"})
    if payload["entries_found"] != 0 or payload["items_indexed"] != 0:
        raise SelftestFailure(f"Expected empty index after cleanup, got {payload}")


def _run_setup_checks(reporter: Reporter, state_dir: Path) -> None:
    reporter.check("robin-config.json exists and parses", lambda: _check_setup(state_dir))
    reporter.check("topics.py returns valid JSON", lambda: _check_topics_json(state_dir))
    reporter.check("doctor.py returns healthy JSON", lambda: _check_doctor_json(state_dir))


def _run_full_selftest(reporter: Reporter, state_dir: Path) -> None:
    # This order is intentional: review/rate validates the single-entry path
    # before the duplicate acceptance check adds a second entry.
    reporter.check("temporary state directory initialized", lambda: _write_config(state_dir))
    _run_setup_checks(reporter, state_dir)
    added = reporter.check("add text entry returns expected shape", lambda: _add_text_entry(state_dir))
    reporter.check("topics.py reports added topic", lambda: _check_topic_added(state_dir))
    if isinstance(added, dict):
        reporter.check("search finds added entry", lambda: _check_search_finds_entry(state_dir, added["id"]))
    else:
        reporter.check("search finds added entry", lambda: _fail_selftest("add step failed"))
    reporter.check("review surfaces an item and rate updates state", lambda: _check_review_and_rate(state_dir))
    reporter.check("duplicate text entry is rejected by default", lambda: _check_duplicate_rejected(state_dir))
    reporter.check("duplicate text entry is accepted with override", lambda: _check_duplicate_allowed(state_dir))
    reporter.check("entries.py moves and deletes an entry", lambda: _check_entries_move_and_delete(state_dir))
    reporter.check("video with local file is rejected", lambda: _check_local_video_rejected(state_dir))
    reporter.check("media entry missing creator is rejected", lambda: _check_missing_media_metadata_rejected(state_dir))
    reporter.check("reindex after cleanup succeeds", lambda: _check_reindex_after_cleanup(state_dir))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Robin integration checks")
    parser.add_argument(
        "--state-dir",
        help="Existing Robin state directory to validate non-destructively. Full functional tests use a temp dir.",
    )
    parser.add_argument("--keep-temp", action="store_true", help="Keep the temporary selftest state directory")
    args = parser.parse_args()

    reporter = Reporter()
    temp_dir: str | None = None
    state_dir = Path(args.state_dir).expanduser().resolve() if args.state_dir else None
    try:
        if state_dir:
            _run_setup_checks(reporter, state_dir)
        else:
            temp_dir = tempfile.mkdtemp(prefix="robin-selftest-")
            state_dir = Path(temp_dir) / "data" / "robin"
            _run_full_selftest(reporter, state_dir)
    finally:
        if state_dir is not None:
            reporter.print_report(state_dir)
        if temp_dir:
            if reporter.ok and not args.keep_temp:
                shutil.rmtree(temp_dir, ignore_errors=True)
            else:
                print(f"Temporary files kept at: {temp_dir}")

    raise SystemExit(0 if reporter.ok else 1)


if __name__ == "__main__":
    main()
