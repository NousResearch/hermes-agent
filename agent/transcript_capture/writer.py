from __future__ import annotations

import hashlib
import os
import re
import tempfile
from pathlib import Path

from .config import TranscriptCaptureConfig
from .sanitize import force_redact_text

_SAFE_COMPONENT_RE = re.compile(r"[^A-Za-z0-9_-]+")


def stable_short_hash(value: str, *, length: int = 16) -> str:
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()[:length]


def _safe_component(value: str) -> str:
    cleaned = _SAFE_COMPONENT_RE.sub("-", (value or "unknown").strip().lower()).strip("-")
    return cleaned or "unknown"


def _fsync_dir(path: Path) -> None:
    try:
        fd = os.open(str(path), os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


class TranscriptWriter:
    def __init__(self, config: TranscriptCaptureConfig):
        self.config = config

    def _redact_configured_identifiers(self, body: str, *, session_key: str, session_id: str) -> str:
        replacements = []
        replacements.extend((value, "[REDACTED CHAT ID]") for value in self.config.chat_allowlist)
        replacements.extend((value, "[REDACTED SESSION ID]") for value in self.config.session_allowlist)
        replacements.extend((value, "[REDACTED TRANSCRIPT ID]") for value in self.config.denylist)
        replacements.extend(
            (value, "[REDACTED SESSION ID]")
            for value in (session_key, session_id)
            if value
        )
        redacted = body
        for raw, marker in sorted(replacements, key=lambda item: len(str(item[0])), reverse=True):
            raw_text = str(raw)
            if not raw_text:
                continue
            redacted = redacted.replace(raw_text, marker)
        return redacted

    def final_path(self, date_prefix: str, platform: str, session_key: str, session_id: str) -> Path:
        filename = (
            f"{_safe_component(date_prefix)}-{_safe_component(platform)}-"
            f"{stable_short_hash(session_key)}-{stable_short_hash(session_id)}.txt"
        )
        # Flat corpus only: filename is always a single safe path component.
        return self.config.corpus_dir / filename

    def publish(self, date_prefix: str, platform: str, session_key: str, session_id: str, body: str) -> Path:
        body = force_redact_text(body)
        body = self._redact_configured_identifiers(body, session_key=session_key, session_id=session_id)
        if not body.rstrip().endswith("END_SESSION"):
            raise ValueError("finalized transcript must end with END_SESSION")
        self.config.active_dir.mkdir(parents=True, exist_ok=True)
        self.config.corpus_dir.mkdir(parents=True, exist_ok=True)
        final = self.final_path(date_prefix, platform, session_key, session_id)
        if final.exists():
            existing = final.read_text(encoding="utf-8")
            if existing == body:
                return final
            raise FileExistsError(f"final transcript already exists with different content: {final}")
        fd, tmp_name = tempfile.mkstemp(prefix=final.stem + ".", suffix=".part", dir=str(self.config.active_dir), text=True)
        tmp = Path(tmp_name)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(body)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp, final)
            _fsync_dir(self.config.corpus_dir)
            _fsync_dir(self.config.active_dir)
        finally:
            if tmp.exists():
                tmp.unlink()
        return final
