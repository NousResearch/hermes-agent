from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from robin.config import LEGACY_SPLIT_LAYOUT_KEY, config_path, index_path, media_path, state_dir, topics_path
from robin.index import is_legacy_key
from robin.media import is_remote_reference
from robin.parser import RobinEntryParseError, load_topic_entries
from robin.review_logic import parse_timestamp

MAX_TOPIC_ENTRIES = 100
MAX_TOPIC_BYTES = 1024 * 1024


@dataclass(slots=True)
class Diagnostic:
    level: str
    code: str
    message: str
    path: str | None = None
    entry_id: str | None = None
    topic: str | None = None
    media_source: str | None = None

    def to_dict(self) -> dict:
        return {key: value for key, value in asdict(self).items() if value is not None}


def _diag(
    level: str,
    code: str,
    message: str,
    *,
    path: Path | str | None = None,
    entry_id: str | None = None,
    topic: str | None = None,
    media_source: str | None = None,
) -> Diagnostic:
    return Diagnostic(
        level=level,
        code=code,
        message=message,
        path=str(path) if path is not None else None,
        entry_id=entry_id,
        topic=topic,
        media_source=media_source,
    )


def _is_relative_to(path: Path, base: Path) -> bool:
    try:
        path.relative_to(base)
    except ValueError:
        return False
    return True


def _load_config(explicit_state_dir: str | None, diagnostics: list[Diagnostic]) -> tuple[Path | None, dict | None]:
    try:
        base = state_dir(explicit_state_dir)
    except SystemExit as exc:
        diagnostics.append(_diag("error", "missing_state_dir", str(exc)))
        return None, None

    if not base.exists():
        diagnostics.append(_diag("error", "missing_state_dir", f"State directory does not exist: {base}", path=base))
        return base, None
    if not base.is_dir():
        diagnostics.append(_diag("error", "invalid_state_dir", f"State path is not a directory: {base}", path=base))
        return base, None

    path = config_path(explicit_state_dir)
    try:
        with path.open(encoding="utf-8") as f:
            config = json.load(f)
    except FileNotFoundError:
        diagnostics.append(_diag("error", "missing_config", f"Config not found at {path}", path=path))
        return base, None
    except json.JSONDecodeError as exc:
        diagnostics.append(_diag("error", "invalid_config_json", f"Config at {path} is invalid JSON: {exc}", path=path))
        return base, None

    if not isinstance(config, dict):
        diagnostics.append(_diag("error", "invalid_config", f"Config at {path} must be a JSON object.", path=path))
        return base, None

    if LEGACY_SPLIT_LAYOUT_KEY in config:
        diagnostics.append(
            _diag(
                "error",
                "legacy_split_layout",
                "Config uses the old split-layout field for a separate content root.",
                path=path,
            )
        )
    return base, config


def _load_entries(config: dict, explicit_state_dir: str | None, diagnostics: list[Diagnostic]) -> tuple[list, set[str]]:
    base = topics_path(config, explicit_state_dir)
    if not base.exists():
        diagnostics.append(_diag("error", "missing_topics_dir", f"Topics directory does not exist: {base}", path=base))
        return [], set()
    if not base.is_dir():
        diagnostics.append(_diag("error", "invalid_topics_dir", f"Topics path is not a directory: {base}", path=base))
        return [], set()

    entries = []
    ids_seen: dict[str, Path] = {}
    duplicate_ids: set[str] = set()
    for filepath in sorted(base.glob("*.md")):
        size = filepath.stat().st_size
        if size > MAX_TOPIC_BYTES:
            diagnostics.append(
                _diag(
                    "warning",
                    "large_topic_file",
                    f"Topic file is {size} bytes; consider splitting very large topic files.",
                    path=filepath,
                    topic=filepath.stem,
                )
            )
        try:
            topic_entries = load_topic_entries(filepath)
        except RobinEntryParseError as exc:
            diagnostics.append(_diag("error", "invalid_topic_entry", str(exc), path=filepath))
            continue
        if len(topic_entries) > MAX_TOPIC_ENTRIES:
            diagnostics.append(
                _diag(
                    "warning",
                    "large_topic_entry_count",
                    f"Topic file contains {len(topic_entries)} entries; consider splitting very large topic files.",
                    path=filepath,
                    topic=filepath.stem,
                )
            )
        entries.extend(topic_entries)
        for entry in topic_entries:
            previous_path = ids_seen.get(entry.entry_id)
            if previous_path is not None:
                duplicate_ids.add(entry.entry_id)
                diagnostics.append(
                    _diag(
                        "error",
                        "duplicate_entry_id",
                        f"Entry id {entry.entry_id} appears in both {previous_path} and {filepath}.",
                        path=filepath,
                        entry_id=entry.entry_id,
                        topic=entry.topic,
                    )
                )
            else:
                ids_seen[entry.entry_id] = filepath
    return entries, set(ids_seen) | duplicate_ids


def _check_media(
    config: dict,
    explicit_state_dir: str | None,
    entries: list,
    diagnostics: list[Diagnostic],
) -> None:
    base = state_dir(explicit_state_dir)
    media_dir = media_path(config, explicit_state_dir)
    referenced_files: set[Path] = set()

    for entry in entries:
        media_source = entry.media_source.strip()
        if not media_source or is_remote_reference(media_source):
            continue

        media_file = (base / media_source).resolve()
        if not _is_relative_to(media_file, base):
            diagnostics.append(
                _diag(
                    "error",
                    "media_outside_state_dir",
                    f"Local media reference points outside the Robin state directory: {media_source}",
                    path=media_file,
                    entry_id=entry.entry_id,
                    topic=entry.topic,
                    media_source=media_source,
                )
            )
            continue

        referenced_files.add(media_file)
        if not media_file.exists():
            diagnostics.append(
                _diag(
                    "error",
                    "missing_media",
                    f"Local media file is missing: {media_source}",
                    path=media_file,
                    entry_id=entry.entry_id,
                    topic=entry.topic,
                    media_source=media_source,
                )
            )
        elif not media_file.is_file():
            diagnostics.append(
                _diag(
                    "error",
                    "invalid_media",
                    f"Local media reference is not a file: {media_source}",
                    path=media_file,
                    entry_id=entry.entry_id,
                    topic=entry.topic,
                    media_source=media_source,
                )
            )

    if not media_dir.exists():
        diagnostics.append(_diag("warning", "missing_media_dir", f"Media directory does not exist: {media_dir}", path=media_dir))
        return
    if not media_dir.is_dir():
        diagnostics.append(_diag("error", "invalid_media_dir", f"Media path is not a directory: {media_dir}", path=media_dir))
        return

    for filepath in sorted(path for path in media_dir.rglob("*") if path.is_file()):
        if filepath.resolve() not in referenced_files:
            diagnostics.append(_diag("warning", "orphan_media", f"Media file is not referenced by any entry: {filepath}", path=filepath))


def _check_index(
    explicit_state_dir: str | None,
    entry_ids: set[str],
    diagnostics: list[Diagnostic],
) -> None:
    path = index_path(explicit_state_dir)
    if not path.exists():
        if entry_ids:
            diagnostics.append(
                _diag(
                    "warning",
                    "missing_index",
                    "Review index is missing; run robin-reindex to rebuild review state from topic files.",
                    path=path,
                )
            )
        return

    try:
        with path.open(encoding="utf-8") as f:
            index = json.load(f)
    except json.JSONDecodeError as exc:
        diagnostics.append(_diag("error", "invalid_index_json", f"Review index at {path} is invalid JSON: {exc}", path=path))
        return

    if not isinstance(index, dict) or not isinstance(index.get("items"), dict):
        diagnostics.append(_diag("error", "invalid_index", f"Review index at {path} must contain an items object.", path=path))
        return

    index_items = index["items"]
    for entry_id, item in index_items.items():
        if not isinstance(item, dict):
            diagnostics.append(
                _diag("error", "invalid_index_item", f"Review index item {entry_id} must be an object.", path=path, entry_id=entry_id)
            )
            continue
        if entry_id not in entry_ids:
            if is_legacy_key(entry_id):
                message = f"Legacy-format review item {entry_id}; run robin-reindex to migrate review history."
            else:
                message = f"Review index item {entry_id} has no matching topic entry."
            diagnostics.append(
                _diag(
                    "warning",
                    "orphan_index_item",
                    message,
                    path=path,
                    entry_id=entry_id,
                )
            )
        rating = item.get("rating")
        if rating is not None and (not isinstance(rating, int) or rating < 1 or rating > 5):
            diagnostics.append(
                _diag("error", "invalid_rating", f"Review index item {entry_id} has an invalid rating.", path=path, entry_id=entry_id)
            )
        times_surfaced = item.get("times_surfaced", 0)
        if not isinstance(times_surfaced, int) or times_surfaced < 0:
            diagnostics.append(
                _diag(
                    "error",
                    "invalid_times_surfaced",
                    f"Review index item {entry_id} has an invalid times_surfaced value.",
                    path=path,
                    entry_id=entry_id,
                )
            )
        last_surfaced = item.get("last_surfaced")
        if last_surfaced:
            try:
                parse_timestamp(str(last_surfaced))
            except ValueError:
                diagnostics.append(
                    _diag(
                        "error",
                        "invalid_last_surfaced",
                        f"Review index item {entry_id} has an invalid last_surfaced timestamp.",
                        path=path,
                        entry_id=entry_id,
                    )
                )

    for entry_id in sorted(entry_ids - set(index_items)):
        diagnostics.append(
            _diag(
                "warning",
                "missing_index_item",
                f"Entry {entry_id} is missing from the review index; run robin-reindex to rebuild it.",
                path=path,
                entry_id=entry_id,
            )
        )


def run_doctor(explicit_state_dir: str | None = None) -> dict:
    diagnostics: list[Diagnostic] = []
    _, config = _load_config(explicit_state_dir, diagnostics)
    entries = []
    entry_ids: set[str] = set()

    if config is not None:
        entries, entry_ids = _load_entries(config, explicit_state_dir, diagnostics)
        _check_media(config, explicit_state_dir, entries, diagnostics)
        _check_index(explicit_state_dir, entry_ids, diagnostics)

    errors = sum(1 for item in diagnostics if item.level == "error")
    warnings = sum(1 for item in diagnostics if item.level == "warning")
    return {
        "ok": errors == 0,
        "errors": errors,
        "warnings": warnings,
        "diagnostics": [item.to_dict() for item in diagnostics],
    }
