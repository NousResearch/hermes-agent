"""Re-attach user images on turns whose history was rebuilt from the session DB.

The DB stores user rows as text (``session_persistence._durable_content``): an image part becomes a
trailing ``[screenshot]`` line. A turn rebuilt from the DB (every API-server request, ``--resume``,
a gateway restart) therefore showed the model that placeholder instead of the photo the user asked
about one message earlier. The bytes are not gone: every native attach path (CLI, TUI, gateway,
API server) commits them under the profile's content-addressed ``native-inputs/<sha256>/<name>``
and names that path in the same row (``[Image attached at: <path>]``, written by
``image_routing.build_native_content_parts`` / ``session_api_media.restore_api_images``).

On the WIRE COPY only, when this turn sends images natively, the ``vision.replay_recent_images``
newest images (default 3, 0 = off) go back as ``image_url`` parts. The data URL comes from the same
``_file_to_data_url`` the live turn used, so the rebuilt row is byte-identical to the request that
first sent it and the prompt-cache prefix holds. Only retained bytes under this profile's
``native-inputs`` whose digest still matches are replayed; remote URLs (they can expire and 400
every later turn), missing files and anything outside the window keep their ``[screenshot]`` text.
"""

from __future__ import annotations

import hashlib
import logging
import re
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

DEFAULT_REPLAY_RECENT_IMAGES = 3
SCREENSHOT_MARKER = "[screenshot]"
_HINT_RE = re.compile(r"^\[Image attached(?P<local> at)?: (?P<ref>.+)\]$")
_IMAGE_PART_TYPES = {"image", "image_url", "input_image"}


def resolve_replay_recent_images(cfg: dict) -> int:
    """``vision.replay_recent_images`` as an int >= 0; an unusable value warns and disables replay."""
    raw = ((cfg or {}).get("vision") or {}).get("replay_recent_images", DEFAULT_REPLAY_RECENT_IMAGES)
    if raw is None:
        return DEFAULT_REPLAY_RECENT_IMAGES
    if isinstance(raw, int) and not isinstance(raw, bool) and raw >= 0:
        return raw
    logger.warning("vision.replay_recent_images=%r is not a non-negative integer (0 = off, default %d); "
                   "not replaying earlier images", raw, DEFAULT_REPLAY_RECENT_IMAGES)
    return 0


def _split_rebuilt_row(content: str) -> Optional[tuple[str, list[str]]]:
    """``(text, refs)`` for a DB-rebuilt user row whose trailing ``[screenshot]`` lines pair 1:1 with
    its ``[Image attached…]`` hints; ``None`` for anything else (a typed marker never pairs)."""
    lines = content.split("\n")
    n = 0
    while n < len(lines) and lines[len(lines) - 1 - n] == SCREENSHOT_MARKER:
        n += 1
    if not n:
        return None
    text_lines = lines[: len(lines) - n]
    refs: list[str] = []
    for line in text_lines[len(text_lines) - n:] if len(text_lines) > n else ():
        m = _HINT_RE.match(line)
        if m is None:
            return None
        refs.append(m.group("ref") if m.group("local") else "")
    return ("\n".join(text_lines), refs) if len(refs) == n else None


def _retained_data_url(ref: str) -> Optional[str]:
    """Data URL for a retained local image this profile still holds unchanged, else ``None``."""
    if not ref:
        return None
    from agent.image_routing import _file_to_data_url
    from gateway.session_ingress_media import _media_root

    path = Path(ref)
    try:
        if path.parent.parent != _media_root() or path.is_symlink() or not path.is_file():
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
        split = _split_rebuilt_row(content) if isinstance(content, str) else None
        if split is None:
            continue
        text, refs = split
        urls: list[Optional[str]] = [None] * len(refs)
        for i in range(len(refs) - 1, -1, -1):  # a row's latest image first
            if budget > 0 and (url := _retained_data_url(refs[i])):
                urls[i] = url
                budget -= 1
        if any(urls):
            plan.append((msg, text, urls))
    attached = 0
    for msg, text, urls in plan:
        msg["content"] = [{"type": "text", "text": text}, *(
            {"type": "image_url", "image_url": {"url": u}} if u else {"type": "text", "text": SCREENSHOT_MARKER}
            for u in urls)]
        attached += sum(1 for u in urls if u)
    return attached


def replay_for_agent(agent: Any, api_messages: list[dict[str, Any]]) -> None:
    """``build_api_messages`` hook: replay only when this turn's model takes images natively."""
    if not any(m.get("role") == "user" and isinstance(m.get("content"), str)
               and m["content"].endswith(SCREENSHOT_MARKER) for m in api_messages):
        return
    try:
        from agent.image_routing import decide_image_input_mode
        from hermes_cli.config import load_config

        cfg = load_config()
        keep = resolve_replay_recent_images(cfg)
        if keep <= 0:
            return
        provider = (getattr(agent, "provider", "") or "").strip()
        model = (getattr(agent, "model", "") or "").strip()
        if decide_image_input_mode(provider, model, cfg) != "native":
            return
        if attached := replay_recent_images(api_messages, keep):
            logger.info("image_replay: re-attached %d earlier image(s) to the request", attached)
    except Exception:
        logger.warning("image_replay: re-attaching earlier images failed", exc_info=True)
