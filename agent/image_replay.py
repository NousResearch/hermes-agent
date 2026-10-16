"""Re-attach user images on turns whose history was rebuilt from the session DB.

The DB stores user rows as text (``session_persistence._durable_content``): an image part becomes a
``[screenshot]`` line at its position (``[screenshot detail=low]`` when the caller chose a detail).
A turn rebuilt from the DB (every API-server request, ``--resume``, a gateway restart) therefore
showed the model that placeholder instead of the photo the user asked about one message earlier.
The bytes are not gone: the session-authority attach paths (CLI, the shared-gateway TUI, messaging
gateway, API server) commit them under the profile's content-addressed ``native-inputs/<sha256>/<name>``
and name that path in the same row (``[Image attached at: <path>]``,
written by ``image_routing.build_native_content_parts`` / ``session_api_media.restore_api_images``).

On the WIRE COPY only, when this turn sends images natively, the ``vision.replay_recent_images``
newest images (default 3, 0 = off) go back as ``image_url`` parts. The data URL comes from the same
``_file_to_data_url`` the live turn used, so the rebuilt row is byte-identical to the request that
first sent it and the prompt-cache prefix holds. Only retained bytes under this profile's
``native-inputs`` whose digest still matches are replayed; remote URLs (they can expire and 400
every later turn), missing files (the gateway's settled-media sweep collects them after
``MEDIA_CACHE_MAX_AGE_HOURS``) and anything outside the window keep their ``[screenshot]`` text.
"""

from __future__ import annotations

import hashlib
import logging
import re
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

DEFAULT_REPLAY_RECENT_IMAGES = 3
_HINT_RE = re.compile(r"^\[Image attached(?P<local> at)?: (?P<ref>.+)\]$")
# ``session_persistence._image_marker``: the live part's ``detail`` rides on its marker.
_MARKER_RE = re.compile(r"^\[screenshot(?: detail=(?P<detail>low|high|auto))?\]$")
_IMAGE_PART_TYPES = {"image", "image_url", "input_image"}


def resolve_replay_recent_images(cfg: dict) -> int:
    """``vision.replay_recent_images`` as an int >= 0; an unusable value warns and disables replay."""
    vision = (cfg or {}).get("vision")
    raw = (vision if isinstance(vision, dict) else {}).get("replay_recent_images", DEFAULT_REPLAY_RECENT_IMAGES)
    if raw is None:
        return DEFAULT_REPLAY_RECENT_IMAGES
    if isinstance(raw, int) and not isinstance(raw, bool) and raw >= 0:
        return raw
    logger.warning("vision.replay_recent_images=%r is not a non-negative integer (0 = off, default %d); "
                   "not replaying earlier images", raw, DEFAULT_REPLAY_RECENT_IMAGES)
    return 0


def _split_rebuilt_row(content: str) -> Optional[list[tuple[str, Any]]]:
    """A DB-rebuilt user row as its original parts, in order: ``("text", str)`` for each text run and
    ``("image", (ref, detail, marker_line))`` per ``[screenshot]`` line, the markers pairing 1:1 in order
    with the row's ``[Image attached…]`` hints wherever the image sat (API callers put it first;
    turn-time memory/hook context follows it). ``None`` when the counts disagree (a typed marker)."""
    lines = content.split("\n")
    markers = [(i, m.group("detail")) for i, line in enumerate(lines) if (m := _MARKER_RE.match(line))]
    hints = [m for line in lines if (m := _HINT_RE.match(line))]
    if not markers or len(hints) != len(markers):
        return None
    parts: list[tuple[str, Any]] = []
    start = 0
    for (i, detail), hint in zip(markers, hints):
        if i > start:
            parts.append(("text", "\n".join(lines[start:i])))
        parts.append(("image", (hint.group("ref") if hint.group("local") else "", detail, lines[i])))
        start = i + 1
    if start < len(lines):
        parts.append(("text", "\n".join(lines[start:])))
    return parts


def _retained_data_url(ref: str) -> Optional[str]:
    """Data URL for a retained local image this profile still holds unchanged, else ``None``."""
    if not ref:
        return None
    from agent.image_routing import _file_to_data_url
    from gateway.session_ingress_media import native_inputs_root

    path = Path(ref)
    try:
        if path.parent.parent != native_inputs_root() or path.is_symlink() or not path.is_file():
            return None
        if hashlib.sha256(path.read_bytes()).hexdigest() != path.parent.name:
            return None
    except OSError:
        return None
    return _file_to_data_url(path)


def replay_recent_images(api_messages: list[dict[str, Any]], keep: int) -> int:
    """Rebuild the newest ``keep`` images in place on ``api_messages`` (wire copies); images still
    live in the conversation count toward the window. Returns the number re-attached."""
    budget, plan = keep, []
    for msg in reversed(api_messages):
        if budget <= 0:
            break
        if msg.get("role") != "user":
            continue
        content = msg.get("content")
        if isinstance(content, list):
            budget -= sum(1 for p in content if isinstance(p, dict) and p.get("type") in _IMAGE_PART_TYPES)
            continue
        parts = _split_rebuilt_row(content) if isinstance(content, str) else None
        if parts is None:
            continue
        images = [i for i, (kind, _) in enumerate(parts) if kind == "image"]
        urls: dict[int, str] = {}
        for i in reversed(images):  # a row's latest image first
            if budget > 0 and (url := _retained_data_url(parts[i][1][0])):
                urls[i] = url
                budget -= 1
        if urls:
            plan.append((msg, parts, urls))
    attached = 0
    for msg, parts, urls in plan:
        rebuilt: list[dict[str, Any]] = []
        for i, (kind, value) in enumerate(parts):
            if kind == "text":
                rebuilt.append({"type": "text", "text": value})
            elif i in urls:
                image = {"url": urls[i], **({"detail": value[1]} if value[1] else {})}
                rebuilt.append({"type": "image_url", "image_url": image})
            else:
                rebuilt.append({"type": "text", "text": value[2]})
        msg["content"] = rebuilt
        attached += len(urls)
    return attached


def _turn_replay_window(agent: Any) -> int:
    """Images to replay this turn: 0 unless the model takes images natively. Resolved once per turn
    (keyed on the turn's frozen admission clock and the model), not per tool-loop request."""
    provider = (getattr(agent, "provider", "") or "").strip()
    model = (getattr(agent, "model", "") or "").strip()
    key = (getattr(agent, "_current_turn_timestamp", None), provider, model)
    cached = getattr(agent, "_image_replay_window", None)
    if cached is not None and cached[0] == key:
        return cached[1]
    from agent.image_routing import decide_image_input_mode
    from hermes_cli.config import load_config_readonly

    cfg = load_config_readonly()
    keep = resolve_replay_recent_images(cfg)
    if keep > 0 and decide_image_input_mode(provider, model, cfg) != "native":
        keep = 0
    agent._image_replay_window = (key, keep)
    return keep


def replay_for_agent(agent: Any, api_messages: list[dict[str, Any]]) -> None:
    """``build_api_messages`` hook: replay only when this turn's model takes images natively."""
    if not any(m.get("role") == "user" and isinstance(m.get("content"), str)
               and "[screenshot" in m["content"] for m in api_messages):
        return
    if (keep := _turn_replay_window(agent)) <= 0:
        return
    if attached := replay_recent_images(api_messages, keep):
        logger.info("image_replay: re-attached %d earlier image(s) to the request", attached)
