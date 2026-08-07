from __future__ import annotations

import re
from dataclasses import dataclass, replace
from pathlib import Path

from robin.config import state_dir, topics_path
from robin.files import atomic_write_text
from robin.index import ensure_entry_in_index
from robin.media import is_remote_reference
from robin.models import Entry
from robin.parser import RobinEntryParseError, SEPARATOR, parse_entry, topic_slug, topic_to_filename
from robin.serializer import serialize_entry


@dataclass(slots=True)
class EntryMatch:
    entry: Entry
    filepath: Path
    chunk_index: int
    raw_chunk: str


class EntryOperationError(ValueError):
    pass


def normalize_body(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip().lower()


def duplicate_candidates(entries: list[Entry], entry: Entry) -> list[Entry]:
    source = entry.source.strip().lower()
    media_source = entry.media_source.strip().lower()
    body = normalize_body(entry.body)

    for candidate in entries:
        if candidate.entry_id == entry.entry_id:
            continue
        if source and candidate.source.strip().lower() == source:
            return [candidate]
        if media_source and candidate.media_source.strip().lower() == media_source:
            return [candidate]
        if body and normalize_body(candidate.body) == body:
            return [candidate]

    return []


def find_duplicate_candidates(config: dict, explicit_state_dir: str | None, entry: Entry) -> list[Entry]:
    for filepath in _topic_files(config, explicit_state_dir):
        for match in _load_topic_chunks(filepath):
            duplicate = duplicate_candidates([match.entry], entry)
            if duplicate:
                return duplicate
    return []


def duplicate_payload(matches: list[Entry]) -> list[dict]:
    return [
        {
            "id": entry.entry_id,
            "topic": entry.topic,
            "date_added": entry.date_added,
            "source": entry.source,
            "media_source": entry.media_source,
            "description": entry.description,
        }
        for entry in matches
    ]


def _topic_files(config: dict, explicit_state_dir: str | None) -> list[Path]:
    base = topics_path(config, explicit_state_dir)
    if not base.exists():
        return []
    return sorted(base.glob("*.md"))


def _load_topic_chunks(filepath: Path) -> list[EntryMatch]:
    content = filepath.read_text(encoding="utf-8")
    if not content.strip():
        return []

    matches: list[EntryMatch] = []
    for chunk_index, chunk in enumerate(content.split(SEPARATOR), start=1):
        if not chunk.strip():
            continue
        raw_chunk = chunk.lstrip()
        try:
            entry = parse_entry(raw_chunk, filepath.stem)
        except ValueError as exc:
            raise RobinEntryParseError(filepath, chunk_index, str(exc)) from exc
        matches.append(EntryMatch(entry=entry, filepath=filepath, chunk_index=chunk_index, raw_chunk=raw_chunk))
    return matches


def _find_entry_matches(config: dict, explicit_state_dir: str | None, entry_id: str) -> list[EntryMatch]:
    matches: list[EntryMatch] = []
    for filepath in _topic_files(config, explicit_state_dir):
        for chunk in _load_topic_chunks(filepath):
            if chunk.entry.entry_id == entry_id:
                matches.append(chunk)
                if len(matches) >= 2:
                    return matches
    return matches


def append_entry_to_file(filepath: Path, serialized: str) -> None:
    if filepath.exists():
        content = filepath.read_text(encoding="utf-8")
        if content.endswith("\n"):
            content = content[:-1]
        out = content + SEPARATOR + serialized if content.strip() else serialized
    else:
        out = serialized
    atomic_write_text(filepath, out + "\n")


def _write_chunks(filepath: Path, chunks: list[str]) -> None:
    if chunks:
        atomic_write_text(filepath, SEPARATOR.join(chunks) + "\n")
    elif filepath.exists():
        filepath.unlink()


def _remove_match(match: EntryMatch) -> None:
    chunks = _load_topic_chunks(match.filepath)
    remaining = [chunk.raw_chunk for chunk in chunks if chunk.entry.entry_id != match.entry.entry_id]
    _write_chunks(match.filepath, remaining)


def _require_single_match(config: dict, explicit_state_dir: str | None, entry_id: str) -> EntryMatch:
    matches = _find_entry_matches(config, explicit_state_dir, entry_id)
    if not matches:
        raise EntryOperationError(f"Entry '{entry_id}' not found.")
    if len(matches) > 1:
        raise EntryOperationError(f"Entry '{entry_id}' appears more than once. Run robin-doctor before mutating entries.")
    return matches[0]


def validate_destination_topic(topic: str) -> str:
    slug = topic_slug(topic)
    if not topic.strip() or slug == "untitled":
        raise EntryOperationError("Move requires a non-empty topic that normalizes to a valid filename.")
    return slug


def delete_entry(config: dict, explicit_state_dir: str | None, index: dict, entry_id: str) -> dict:
    match = _require_single_match(config, explicit_state_dir, entry_id)
    _remove_match(match)
    index.setdefault("items", {}).pop(entry_id, None)
    return {
        "status": "deleted",
        "id": match.entry.entry_id,
        "topic": match.entry.topic,
        "filename": match.filepath.name,
    }


def move_entry(config: dict, explicit_state_dir: str | None, index: dict, entry_id: str, topic: str) -> dict:
    dest_topic = validate_destination_topic(topic)
    match = _require_single_match(config, explicit_state_dir, entry_id)
    from_filename = match.filepath.name
    to_filename = topic_to_filename(topic)

    if match.entry.topic != dest_topic:
        moved = replace(match.entry, topic=dest_topic)
        dest_path = topics_path(config, explicit_state_dir) / to_filename
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            serialized = serialize_entry(moved)
        except ValueError as exc:
            raise EntryOperationError(str(exc)) from exc
        _remove_match(match)
        append_entry_to_file(dest_path, serialized)
    else:
        moved = match.entry

    ensure_entry_in_index(moved, index)
    index["items"][entry_id]["topic"] = moved.topic
    index["items"][entry_id]["date"] = moved.date_added
    return {
        "status": "moved",
        "id": moved.entry_id,
        "from_topic": match.entry.topic,
        "to_topic": moved.topic,
        "from_filename": from_filename,
        "to_filename": to_filename,
    }


def remove_new_media_if_present(explicit_state_dir: str | None, media_source: str) -> None:
    media_source = media_source.strip()
    if not media_source or is_remote_reference(media_source):
        return
    base = state_dir(explicit_state_dir)
    media_path = (base / media_source).resolve()
    try:
        media_path.relative_to(base)
    except ValueError:
        return
    try:
        if media_path.is_file():
            media_path.unlink()
    except OSError:
        return
