#!/usr/bin/env python3
"""Read Apple Voice Memos metadata and transcripts on macOS.

Transcripts are produced on-device by Voice Memos (macOS 15+/iOS 18+) and
embedded directly in each ``.m4a`` file inside a QuickTime ``tsrp`` user-data
atom as JSON. This script reads them straight out of the file -- no audio
processing, no transcription API, no third-party dependencies (stdlib only).

Recording metadata (title, date, duration, filename) comes from the Voice
Memos SQLite database, ``CloudRecordings.db``.

Subcommands:
  list         List recordings (newest first) with transcript availability.
  transcript   Print the transcript for one recording (by index/filename/search).
  dump         Print metadata + transcript for many recordings at once.

All commands accept --json for machine-readable output.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import re
import shutil
import sqlite3
import sys
import tempfile
from pathlib import Path

# Core Data / NSDate epoch is 2001-01-01 UTC; add this to get a Unix timestamp.
COCOA_EPOCH = 978307200

RECORDINGS_DIR = Path(
    os.path.expanduser(
        "~/Library/Group Containers/"
        "group.com.apple.VoiceMemos.shared/Recordings"
    )
)
DB_PATH = RECORDINGS_DIR / "CloudRecordings.db"
WHISPER_TRANSCRIPTS_DIR = Path(
    os.path.expanduser("~/.voicememo-whisper/transcripts")
)

# The binary that needs macOS Full Disk Access for any process reading the
# (TCC-protected) Voice Memos container — surfaced in error messages so a
# failed cron run says exactly what to fix.
_FDA_HINT = (
    "Voice Memos is a macOS privacy-protected location. The running process "
    "needs Full Disk Access. Grant it to the gateway's Python:\n"
    "  System Settings > Privacy & Security > Full Disk Access > +\n"
    "  add  ~/.hermes/hermes-agent/venv/bin/python  (Cmd+Shift+G to paste the path)\n"
    "then: hermes gateway restart"
)


class VoiceMemosAccessError(RuntimeError):
    """Raised when the Voice Memos DB/container can't be read (usually TCC/FDA)."""


def _die(msg: str) -> None:
    print(f"error: {msg}", file=sys.stderr)
    sys.exit(1)


def extract_transcript(m4a_path: Path) -> str | None:
    """Pull Apple's on-device transcript out of an .m4a's ``tsrp`` atom.

    Returns the plain-text transcript, or None if the file has no transcript
    (older recordings, very short clips, or transcription not yet finished).
    """
    try:
        data = m4a_path.read_bytes()
    except OSError:
        return None
    idx = data.find(b"tsrp")
    if idx < 4:
        return None
    # QuickTime atom layout: [4-byte big-endian size][4-byte type][payload].
    # The size and type sit just before the payload, so the 4 bytes preceding
    # "tsrp" are the atom size (covering header + payload).
    size = int.from_bytes(data[idx - 4 : idx], "big")
    payload = data[idx + 4 : idx - 4 + size]
    try:
        obj = json.loads(payload.decode("utf-8", "replace"))
        runs = obj["attributedString"]["runs"]
    except (ValueError, KeyError, TypeError):
        # Fall back to a tolerant raw_decode in case of trailing atom padding.
        try:
            obj, _ = json.JSONDecoder().raw_decode(
                payload.decode("utf-8", "replace")
            )
            runs = obj["attributedString"]["runs"]
        except Exception:
            return None
    text = "".join(x for x in runs if isinstance(x, str)).strip()
    return text or None


def _safe_cache_key(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._") or "recording"


def whisper_transcript_path(rec: dict) -> Path:
    """Return the cache path for a local Whisper transcript of this recording."""
    key = rec.get("unique_id") or Path(rec["filename"]).stem
    return WHISPER_TRANSCRIPTS_DIR / f"{_safe_cache_key(str(key))}.txt"


def read_whisper_transcript(rec: dict) -> str | None:
    """Read a cached local Whisper transcript, if the cron worker made one."""
    try:
        text = whisper_transcript_path(rec).read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return text or None


def best_transcript(rec: dict) -> tuple[str | None, str | None]:
    """Return transcript text and source, preferring Apple's embedded text."""
    apple = extract_transcript(Path(rec["path"])) if rec["exists"] else None
    if apple:
        return apple, "apple"
    whisper = read_whisper_transcript(rec) if rec["exists"] else None
    if whisper:
        return whisper, "whisper-medium"
    return None, None


def load_recordings() -> list[dict]:
    """Read recording metadata from CloudRecordings.db, newest first.

    The DB (and its -wal/-shm sidecars) are copied to a private temp dir
    before opening, so we (a) never touch Apple's live database and (b) read a
    consistent snapshot even while Voice Memos has the WAL open. Any failure to
    read the protected container is raised as VoiceMemosAccessError with a
    Full-Disk-Access remediation hint rather than a bare sqlite error.
    """
    try:
        if not DB_PATH.exists():
            raise VoiceMemosAccessError(
                f"Voice Memos database not found at {DB_PATH}\n{_FDA_HINT}"
            )
        tmpdir = tempfile.mkdtemp(prefix="hermes_vm_")
        local_db = os.path.join(tmpdir, "CloudRecordings.db")
        # Copy the main db plus WAL/SHM so recently-recorded rows aren't missed.
        for suffix in ("", "-wal", "-shm"):
            src = Path(str(DB_PATH) + suffix)
            if src.exists():
                shutil.copy2(src, local_db + suffix)
    except (OSError, PermissionError) as e:
        # errno 1 (EPERM) / 13 (EACCES) here are the TCC denial signature.
        raise VoiceMemosAccessError(f"cannot read Voice Memos data: {e}\n{_FDA_HINT}")

    try:
        conn = sqlite3.connect(local_db)
        try:
            rows = conn.execute(
                "SELECT ZENCRYPTEDTITLE, ZPATH, ZDATE, ZDURATION, ZUNIQUEID "
                "FROM ZCLOUDRECORDING ORDER BY ZDATE DESC"
            ).fetchall()
        finally:
            conn.close()
    except sqlite3.OperationalError as e:
        raise VoiceMemosAccessError(f"cannot open Voice Memos database: {e}\n{_FDA_HINT}")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    out: list[dict] = []
    for title, path, zdate, duration, uid in rows:
        if not path:
            continue
        fname = os.path.basename(path)
        full = RECORDINGS_DIR / fname
        when = None
        if zdate is not None:
            when = _dt.datetime.fromtimestamp(
                zdate + COCOA_EPOCH, _dt.timezone.utc
            ).astimezone()
        out.append(
            {
                "title": title or "Recording",
                "filename": fname,
                "path": str(full),
                "exists": full.exists(),
                "date": when.isoformat() if when else None,
                "date_human": when.strftime("%Y-%m-%d %H:%M") if when else "?",
                "duration_sec": round(duration, 1) if duration else 0.0,
                "unique_id": uid,
            }
        )
    return out


def _fmt_dur(sec: float) -> str:
    m, s = divmod(int(round(sec)), 60)
    return f"{m}:{s:02d}"


def _match(rec: dict, needle: str) -> bool:
    n = needle.lower()
    return (
        n in rec["title"].lower()
        or n in rec["filename"].lower()
        or n == (rec["unique_id"] or "").lower()
    )


def cmd_list(args) -> None:
    recs = load_recordings()
    if args.search:
        recs = [r for r in recs if _match(r, args.search)]
    if args.with_transcript:
        recs = [r for r in recs if r["exists"] and best_transcript(r)[0]]
    if args.limit:
        recs = recs[: args.limit]

    if args.json:
        for r in recs:
            _text, source = best_transcript(r)
            r["has_transcript"] = bool(source)
            r["transcript_source"] = source
            r["whisper_transcript_path"] = str(whisper_transcript_path(r))
        print(json.dumps(recs, indent=2))
        return

    if not recs:
        print("No matching recordings.")
        return
    for i, r in enumerate(recs):
        _text, source = best_transcript(r)
        has = "A" if source == "apple" else "W" if source == "whisper-medium" else " "
        print(
            f"[{i:>3}] [{has}] {r['date_human']}  "
            f"{_fmt_dur(r['duration_sec']):>6}  {r['title']}"
        )
    print(
        "\n(A = Apple transcript, W = local Whisper medium transcript. "
        "Use: voicememos.py transcript <index|search>)"
    )


def _resolve_one(recs: list[dict], selector: str) -> dict | None:
    if selector.isdigit():
        i = int(selector)
        return recs[i] if 0 <= i < len(recs) else None
    hits = [r for r in recs if _match(r, selector)]
    return hits[0] if hits else None


def cmd_transcript(args) -> None:
    recs = load_recordings()
    rec = _resolve_one(recs, args.selector)
    if rec is None:
        _die(f"no recording matched {args.selector!r}")
    text, source = best_transcript(rec)
    if args.json:
        print(json.dumps({**rec, "transcript": text, "transcript_source": source}, indent=2))
        return
    print(f"# {rec['title']}")
    print(f"{rec['date_human']}  ({_fmt_dur(rec['duration_sec'])})  {rec['filename']}\n")
    if text:
        if source == "whisper-medium":
            print("(local Whisper medium transcript)\n")
        print(text)
    else:
        print("(no Apple or cached Whisper transcript)")


def cmd_dump(args) -> None:
    recs = load_recordings()
    if args.search:
        recs = [r for r in recs if _match(r, args.search)]
    recs = [r for r in recs if r["exists"]]
    if args.limit:
        recs = recs[: args.limit]

    results = []
    for r in recs:
        text, source = best_transcript(r)
        if args.only_transcribed and not text:
            continue
        results.append({**r, "transcript": text, "transcript_source": source})

    if args.json:
        print(json.dumps(results, indent=2))
        return
    for r in results:
        source = f" [{r['transcript_source']}]" if r.get("transcript_source") else ""
        print(f"## {r['title']}  —  {r['date_human']}  ({_fmt_dur(r['duration_sec'])}){source}")
        print(r["transcript"] or "(no transcript)")
        print()


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Read Apple Voice Memos transcripts.")
    sub = p.add_subparsers(dest="cmd", required=True)

    pl = sub.add_parser("list", help="list recordings (newest first)")
    pl.add_argument("--limit", type=int, default=30)
    pl.add_argument("--search", help="filter by title/filename")
    pl.add_argument(
        "--with-transcript",
        action="store_true",
        help="only show recordings that have a transcript",
    )
    pl.add_argument("--json", action="store_true")
    pl.set_defaults(func=cmd_list)

    pt = sub.add_parser("transcript", help="print one transcript")
    pt.add_argument("selector", help="list index, filename, unique id, or title search")
    pt.add_argument("--json", action="store_true")
    pt.set_defaults(func=cmd_transcript)

    pd = sub.add_parser("dump", help="print many transcripts at once")
    pd.add_argument("--limit", type=int, default=10)
    pd.add_argument("--search", help="filter by title/filename")
    pd.add_argument(
        "--only-transcribed",
        action="store_true",
        help="skip recordings without a transcript",
    )
    pd.add_argument("--json", action="store_true")
    pd.set_defaults(func=cmd_dump)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
