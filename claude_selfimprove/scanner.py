"""Incremental, checkpointed discovery of Claude Code transcript turns.

The transcript corpus this pipeline reads from is large (670+ MB under
``~/.claude/projects`` alone) and grows continuously while Claude Code is in
daily use. Re-reading it from scratch every run does not scale, so this
module never does that:

* A directory walk finds transcript files by *path* only (cheap — no file
  content is read during discovery).
* Each file's ``(size, mtime)`` is compared against the last checkpoint.
  Unchanged files are skipped entirely.
* A changed file is read from its stored **byte offset** onward, not from
  the start — a transcript file is append-only while its session is live,
  so re-reading from the last watermark is sufficient and correct.
* Only complete lines (terminated by ``\\n``) advance the watermark. A
  partial trailing line — the file mid-write during a live session — is
  left for the next run.
* A file that shrank below its stored offset (rotated, truncated, or
  replaced) is treated as new and re-read from byte 0.

All parsing is defensive: a malformed JSON line, an unexpected shape, or a
read error is skipped for that line/file, never raised. A single corrupt
transcript must not abort the whole scan (requirement: fail safely).

Every text value handed back to the caller is treated as **untrusted data**
from here on — this module does not redact (that is the caller's job before
persistence or model review), and it never evaluates or executes anything
found in a transcript.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from . import atomic, paths

# Transcript entry types that carry genuine natural-language text. Everything
# else (queue-operation, attachment, last-prompt, tool bookkeeping, ...) is
# skipped without being parsed further.
_TEXT_ENTRY_TYPES = frozenset({"user", "assistant"})


@dataclass(frozen=True)
class TranscriptTurn:
    """One human- or assistant-authored text turn from a transcript."""

    source: str  # "claude" | "claude-hermes"
    project: str  # the project directory name under projects/
    session_id: str
    file_path: str
    line_no: int
    role: str  # "user" | "assistant"
    text: str
    timestamp: str | None


class CheckpointStore:
    """Per-file scan watermarks, persisted as one JSON file.

    Shape: ``{"files": {"<abs path>": {"size": int, "mtime": float,
    "offset": int}}}``. Loaded once, mutated in memory during a scan pass,
    and saved atomically at the end (or after every file, when the caller
    wants crash-safety mid-pass — see :meth:`save`).
    """

    def __init__(self, path: Path | None = None) -> None:
        self.path = path or paths.checkpoints_path()
        self._data: dict[str, Any] = {"files": {}}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            # A corrupt checkpoint file must not crash the pipeline. Worst
            # case, the next run re-reads everything once and rebuilds it.
            return
        if isinstance(raw, dict) and isinstance(raw.get("files"), dict):
            self._data = raw

    def get(self, file_path: str) -> dict[str, Any] | None:
        return self._data["files"].get(file_path)

    def set(self, file_path: str, *, size: int, mtime: float, offset: int) -> None:
        self._data["files"][file_path] = {
            "size": size,
            "mtime": mtime,
            "offset": offset,
        }

    def save(self) -> None:
        atomic.atomic_write_json(self.path, self._data)

    def known_file_count(self) -> int:
        return len(self._data["files"])


def _extract_text(content: Any) -> str | None:
    """Pull genuine natural-language text out of one message's ``content``.

    A plain string is real, human- or model-authored text: return it as-is.
    A list is Anthropic-format content blocks; keep only ``type: "text"``
    blocks (skip ``tool_use`` / ``tool_result`` / anything else — those are
    synthesized structure, not authored prose, and including them would let
    raw tool inputs/outputs leak into candidate mining, which requirement 4
    forbids). Returns ``None`` when there is no text to extract.
    """
    if isinstance(content, str):
        stripped = content.strip()
        return stripped or None
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                text = block.get("text")
                if isinstance(text, str) and text.strip():
                    parts.append(text.strip())
        joined = "\n".join(parts).strip()
        return joined or None
    return None


def _parse_line(line: str) -> dict[str, Any] | None:
    line = line.strip()
    if not line:
        return None
    try:
        obj = json.loads(line)
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def iter_turns_from_file(
    file_path: Path, *, start_offset: int, source: str, project: str
) -> tuple[list[TranscriptTurn], int]:
    """Read *file_path* from *start_offset*, return (turns, new_offset).

    ``new_offset`` only advances past complete lines, so a partial trailing
    line is safely re-read on the next call.
    """
    turns: list[TranscriptTurn] = []
    try:
        with file_path.open("rb") as fh:
            fh.seek(start_offset)
            chunk = fh.read()
    except OSError:
        return [], start_offset

    if not chunk:
        return [], start_offset

    consumed = 0
    for raw_line in chunk.splitlines(keepends=True):
        if not raw_line.endswith(b"\n"):
            # Partial final line — do not consume it.
            break
        consumed += len(raw_line)
        try:
            text_line = raw_line.decode("utf-8", errors="replace")
        except Exception:
            continue
        obj = _parse_line(text_line)
        if obj is None:
            continue
        if obj.get("type") not in _TEXT_ENTRY_TYPES:
            continue
        message = obj.get("message")
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        if role not in ("user", "assistant"):
            continue
        text = _extract_text(message.get("content"))
        if not text:
            continue
        session_id = str(obj.get("sessionId") or "")
        timestamp = obj.get("timestamp")
        turns.append(
            TranscriptTurn(
                source=source,
                project=project,
                session_id=session_id,
                file_path=str(file_path),
                line_no=len(turns),  # informational only
                role=role,
                text=text,
                timestamp=str(timestamp) if timestamp else None,
            )
        )

    return turns, start_offset + consumed


def seed_all(checkpoints: CheckpointStore) -> int:
    """Mark every currently-existing transcript file as fully read.

    Sets each file's checkpoint offset to its *current* size without
    opening it for parsing — no candidates are mined from this content.
    This is a one-time operation for a pipeline's first activation: the
    real corpus this pipeline reads from is hundreds of megabytes of
    history that predates the pipeline's existence, and reading all of it
    through the classifier on day one would be slow, expensive, and not
    what "nightly incremental scan" is meant to mean. The same idea
    already has a name and a precedent in this profile —
    ``self_improvement_watch.sh``'s ``seed_watermarks()`` — for exactly
    this reason: "First run must not replay history."

    Returns the number of files seeded.
    """
    count = 0
    for source, project, file_path in discover_files():
        try:
            stat = file_path.stat()
        except OSError:
            continue
        checkpoints.set(str(file_path), size=stat.st_size, mtime=stat.st_mtime, offset=stat.st_size)
        count += 1
    return count


def discover_files() -> list[tuple[str, str, Path]]:
    """List every transcript file across all source roots, recursively.

    Returns ``(source_name, project_name, file_path)`` tuples. This only
    walks directory entries — no file content is read here.

    Recursive on purpose: a session that spawns a subagent (the Task/Agent
    tool) gets its own transcript under
    ``projects/<project>/<session-id>/subagents/agent-*.jsonl`` — one level
    deeper than the session's own file. On George's real profile these
    subagent transcripts are roughly half of all transcript files by
    count, so a shallow, one-level glob would silently skip about half the
    corpus this pipeline is meant to observe.
    """
    found: list[tuple[str, str, Path]] = []
    for root in paths.source_roots():
        try:
            project_dirs = [d for d in root.projects_dir.iterdir() if d.is_dir()]
        except OSError:
            continue
        for project_dir in project_dirs:
            try:
                files = sorted(project_dir.rglob("*.jsonl"))
            except OSError:
                continue
            for f in files:
                found.append((root.name, project_dir.name, f))
    return found


def scan_new_turns(
    checkpoints: CheckpointStore,
) -> Iterator[TranscriptTurn]:
    """Yield every not-yet-seen turn across all source roots, incrementally.

    Skips any file whose ``(size, mtime)`` matches the last checkpoint —
    the fast path that makes a nightly run cheap even across a corpus with
    hundreds of thousands of lines. A file that shrank below its stored
    offset is re-read from byte 0 (rotated/replaced).
    """
    for source, project, file_path in discover_files():
        try:
            stat = file_path.stat()
        except OSError:
            continue

        key = str(file_path)
        prior = checkpoints.get(key)
        if prior and prior.get("size") == stat.st_size and prior.get("mtime") == stat.st_mtime:
            continue  # unchanged — skip without opening the file

        start_offset = 0
        if prior:
            prior_offset = int(prior.get("offset", 0))
            if prior_offset <= stat.st_size:
                start_offset = prior_offset

        turns, new_offset = iter_turns_from_file(
            file_path, start_offset=start_offset, source=source, project=project
        )
        for turn in turns:
            yield turn

        checkpoints.set(key, size=stat.st_size, mtime=stat.st_mtime, offset=new_offset)
