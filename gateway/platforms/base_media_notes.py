"""Media-delivery failure notices: one for the user, one for the model.

Split out of :mod:`gateway.platforms.base` to keep that facade's length down; behaviour is
unchanged.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from agent.i18n import t

_MEDIA_KIND_KEYS = frozenset({"audio", "video", "file", "image"})


def media_failure_text(kind: str, file_name: str | None = None) -> str:
    """User-facing "couldn't deliver" notice; ``file_name`` is the only name ever shown."""
    kind_label = t(f"gateway.notify.media_kind.{kind}") if kind in _MEDIA_KIND_KEYS else kind
    if file_name:
        return t("gateway.notify.media_delivery_failed_with_name", kind=kind_label, file_name=file_name)
    return t("gateway.notify.media_delivery_failed", kind=kind_label)


def stage_media_failure_note(adapter: Any, event: Any, media_path: str, result: Any) -> None:
    """Tell the model, on its next run, that this attachment never reached the user.

    Post-turn uploads finish after the agent's turn ended, so without this the model believes a
    file it attached was delivered and never retries. Rides the runner's existing one-shot sidecar
    notes (same channel as ``[Voice channel now: …]``); nothing is written to the transcript, so
    user/assistant alternation stays intact. A no-op when the adapter has no runner (tests,
    standalone use).
    """
    runner = getattr(adapter, "gateway_runner", None)
    stage = getattr(runner, "_set_pending_turn_sidecar_notes", None)
    if stage is None:
        return
    # ponytail: last failure wins — sidecar notes are staged as a whole list and only readable
    # once. Merge per-file notes here if multi-file partial failures start to matter.
    stage(adapter._source_session_key(event.source), [
        f"[Attachment NOT delivered] {Path(media_path).name}: "
        f"{getattr(result, 'error', 'unknown error')}. If the user still needs it, send it again.",
    ])
