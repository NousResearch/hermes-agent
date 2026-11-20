"""The media verbs of the admission reference.

``hosted_document_references`` comes from ``session_rules.media``, the media root passed in as data. Transcript text is a row's ``content`` as the store keeps it: a string as is, structured content as its JSON text. A stored media reference without a ``path`` is corrupt store data and raises ``PermanentStoreError``.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from persistence.authority.admissions import UnheldMedia
from persistence.core.errors import PermanentStoreError
from session_rules.media import hosted_document_references

from . import admissions_rows as rows


def _media(payload: Any, holder: str) -> list[Any]:
    value = payload.get(holder) if isinstance(payload, dict) else None
    media = value.get("media") if isinstance(value, dict) else None
    return list(media) if isinstance(media, list) else []


def _has_media(payload: Any, holder: str) -> bool:
    value = payload.get(holder) if isinstance(payload, dict) else None
    return isinstance(value, dict) and "media" in value


def _content_text(content: Any) -> str:
    if isinstance(content, bytes):
        return content.decode("utf-8", "replace")
    return content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)


def _paths(references: Sequence[Any]) -> set[str]:
    try:
        return {reference["path"] for reference in references}
    except (KeyError, TypeError) as exc:
        raise PermanentStoreError(f"admission media reference is corrupt: {exc!r}") from exc


class AdmissionMediaMixin:
    """Media holds, transcript mentions and the durable retirement candidates; mixed into ``InMemoryAdmissionRepository``."""

    def _mentions(self, references: Sequence[Mapping[str, Any]], after_id: int = 0) -> tuple[int, set[str]]:
        needles: dict[str, str] = {}
        for reference in references:
            raw = reference["path"]
            for tail in (reference["sha256"] + sep + Path(raw).name for sep in ("/", "\\")):
                needles[tail] = needles[json.dumps(tail)[1:-1]] = raw
        transcripts = self._tables().get("messages", {}).get("transcripts", {})
        watermark, mentioned = 0, set()
        for transcript in transcripts.values():
            for row in transcript.rows:
                watermark = max(watermark, row.id)
                if row.id <= after_id or not needles:
                    continue
                text = _content_text(row.message.content)
                mentioned.update(raw for needle, raw in needles.items() if needle in text)
        return watermark, mentioned

    def history_mentions(self, references: Sequence[Mapping[str, Any]]) -> tuple[int, frozenset[str]]:
        watermark, mentioned = self._mentions(references)
        return watermark, frozenset(mentioned)

    def _held_paths(self, media_root: str) -> set[str]:
        """Media paths admissions still hold: API-turn media of every row, plus attachment, native-text and hosted-document media of non-terminal rows."""
        held: set[str] = set()
        for row in self._committed()["admissions"].values():
            payload = row["payload"]
            references = _media(payload, "api_turn_v1")
            if row["status"] != "terminal":
                references += _media(payload, "attachments_v1") + _media(payload, "native_text_v1")
                references += hosted_document_references(row["request_id"], payload, media_root)
            held |= _paths(references)
        return held

    def unheld_media(self, references: Sequence[Mapping[str, Any]], *, after_id: int, retain_history: bool, media_root: str) -> UnheldMedia:
        held = self._held_paths(media_root)
        if retain_history:
            held |= self._mentions(references, after_id)[1]
        candidates = tuple(dict(reference) for reference in references if reference["path"] not in held)
        return UnheldMedia(candidates=candidates, held=frozenset(held))

    def record_retired_media(self, references: Sequence[Mapping[str, Any]]) -> None:
        state = self._state()
        with state.transaction() as staged:
            meta = staged.authority["meta"]
            for reference in references:
                meta.setdefault(rows.MEDIA_PREFIX + hashlib.sha256(reference["path"].encode()).hexdigest(), json.dumps(reference))

    def drop_retired_candidates(self, keys: Sequence[str]) -> None:
        state = self._state()
        with state.transaction() as staged:
            for key in keys:
                staged.authority["meta"].pop(key, None)

    def settled_media(self) -> list[Sequence[Mapping[str, Any]]]:
        out = []
        for row in rows.rows_by_seq(self._committed()):
            payload = row["payload"]
            if row["status"] == "terminal" and (_has_media(payload, "attachments_v1") or _has_media(payload, "native_text_v1")):
                out.append(_media(payload, "attachments_v1") + _media(payload, "native_text_v1"))
        return out

    def retired_candidates(self, kind: str) -> list[tuple[str, Any]]:
        prefix = rows.MEDIA_PREFIX if kind == "media" else rows.SPILLOVER_PREFIX
        meta = self._committed()["meta"]
        return [(key, json.loads(meta[key])) for key in sorted(k for k in meta if isinstance(k, str) and k.startswith(prefix))]
