"""Cached images from passive Telegram group context."""

from typing import Any, Dict, List, Optional

# Cap on cached observed-group images re-attached to a later addressed turn.
# Observed context accumulates for the whole session, so without a bound a busy
# group would re-upload every photo it has ever seen on every single reply.
_MAX_OBSERVED_IMAGE_ATTACHMENTS = 4


def _collect_observed_image_paths(
    history: Optional[List[Dict[str, Any]]],
    limit: int = _MAX_OBSERVED_IMAGE_ATTACHMENTS,
) -> List[str]:
    """Cached image paths from the *trailing* observed rows, oldest first.

    Scoped to observed chatter that arrived since the last real exchange, so a
    photo is attached on the turn that follows it and not re-uploaded on every
    later reply — the text note continues to carry it in context after that.

    Only rows written by the observe path carry ``media_urls``; transcripts
    written before that field existed yield nothing and keep their text note,
    so this is a no-op on existing sessions.
    """
    from agent.image_routing import _IMAGE_EXTS

    trailing: List[Dict[str, Any]] = []
    for msg in reversed(history or []):
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        if role in {"session_meta", "system"}:
            continue
        if not msg.get("observed"):
            break
        trailing.append(msg)
    trailing.reverse()

    paths: List[str] = []
    for msg in trailing:
        if msg.get("role") != "user":
            continue
        urls = msg.get("media_urls") or []
        types = msg.get("media_types") or []
        for i, path in enumerate(urls):
            if not path:
                continue
            # Trust the recorded MIME; fall back to the extension only when the
            # observe path could not determine one, mirroring
            # ``_event_media_is_image`` for live events.
            mime = str(types[i]) if i < len(types) else ""
            if mime:
                if not mime.startswith("image/"):
                    continue
            elif not str(path).lower().endswith(_IMAGE_EXTS):
                continue
            paths.append(str(path))
    return paths[-limit:] if limit and len(paths) > limit else paths


