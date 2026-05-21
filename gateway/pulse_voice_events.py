"""Best-effort local voice event bridge for Pulse/Aegis.

The messaging gateway streams assistant deltas/commentary to platform adapters,
but those interim chunks are not persisted in ``state.db`` until a turn finishes.
Aegis needs the live chunks to speak Jarvis-style acknowledgements immediately,
so this module appends sanitized JSONL events under ``$HERMES_HOME/pulse``.

This is intentionally best-effort: failures must never affect chat delivery.
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any

from hermes_constants import get_hermes_home

_LOCK = threading.Lock()
_MAX_BYTES = 2_000_000


def _enabled() -> bool:
    value = str(os.getenv("HERMES_PULSE_VOICE_EVENTS", "1")).strip().lower()
    return value not in {"0", "false", "no", "off"}


def voice_events_path() -> Path:
    return get_hermes_home() / "pulse" / "voice-events.jsonl"


def _trim_if_needed(path: Path) -> None:
    try:
        if not path.exists() or path.stat().st_size <= _MAX_BYTES:
            return
        data = path.read_bytes()[-(_MAX_BYTES // 2):]
        # Start at the next newline so consumers never see a partial JSON row.
        first_newline = data.find(b"\n")
        if first_newline >= 0:
            data = data[first_newline + 1 :]
        path.write_bytes(data)
    except OSError:
        return


def publish_voice_event(kind: str, text: str, **metadata: Any) -> None:
    """Append a voice event for local Pulse/Aegis subscribers.

    ``kind`` is usually ``delta`` or ``commentary``. ``text`` should already be
    user-visible assistant text; empty/whitespace events are ignored.
    """
    if not _enabled():
        return
    text = str(text or "")
    if not text.strip():
        return
    try:
        path = voice_events_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        event = {
            "id": f"{time.time_ns()}",
            "ts": time.time(),
            "kind": str(kind or "delta"),
            "text": text,
            **{k: v for k, v in metadata.items() if v is not None},
        }
        line = json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n"
        with _LOCK:
            _trim_if_needed(path)
            with path.open("a", encoding="utf-8") as fh:
                fh.write(line)
    except Exception:
        # Voice events are an observability side-channel; never break gateway.
        return
