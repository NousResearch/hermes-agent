"""Routing helpers for inbound user-attached images.

Two modes:

  native  — attach images as OpenAI-style ``image_url`` content parts on the
            user turn. Provider adapters (Anthropic, Gemini, Bedrock, Codex,
            OpenAI chat.completions) already translate these into their
            vendor-specific multimodal formats.

  text    — run ``vision_analyze`` on each image up-front and prepend the
            description to the user's text. The model never sees the pixels;
            it only sees a lossy text summary. This is the pre-existing
            behaviour and still the right choice for non-vision models.

The decision is made once per message turn by :func:`decide_image_input_mode`.
It reads ``agent.image_input_mode`` from config.yaml (``auto`` | ``native``
| ``text``, default ``auto``) and the active model's capability metadata.

In ``auto`` mode:
  - If the user has explicitly configured ``auxiliary.vision.provider``
    (i.e. not ``auto`` and not empty), we assume they want the text pipeline
    regardless of the main model — they've opted in to a specific vision
    backend for a reason (cost, quality, local-only, etc.).
  - Otherwise, if the active model reports ``supports_vision=True`` in its
    models.dev metadata, we attach natively.
  - Otherwise (non-vision model, no explicit override), we fall back to text.

This keeps ``vision_analyze`` surfaced as a tool in every session — skills
and agent flows that chain it (browser screenshots, deeper inspection of
URL-referenced images, style-gating loops) keep working. The routing only
affects *how user-attached images on the current turn* are presented to the
main model.
"""

from __future__ import annotations

import base64
import logging
import mimetypes
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


_VALID_MODES = frozenset({"auto", "native", "text"})


def _coerce_mode(raw: Any) -> str:
    """Normalize a config value into one of the valid modes."""
    if not isinstance(raw, str):
        return "auto"
    val = raw.strip().lower()
    if val in _VALID_MODES:
        return val
    return "auto"


def _explicit_aux_vision_override(cfg: Optional[Dict[str, Any]]) -> bool:
    """True when the user configured a specific auxiliary vision backend.

    An explicit override means the user *wants* the text pipeline (they're
    paying for a dedicated vision model), so we don't silently bypass it.
    """
    if not isinstance(cfg, dict):
        return False
    aux = cfg.get("auxiliary") or {}
    if not isinstance(aux, dict):
        return False
    vision = aux.get("vision") or {}
    if not isinstance(vision, dict):
        return False

    provider = str(vision.get("provider") or "").strip().lower()
    model = str(vision.get("model") or "").strip()
    base_url = str(vision.get("base_url") or "").strip()

    # "auto" / "" / blank = not explicit
    if provider in ("", "auto") and not model and not base_url:
        return False
    return True


def _lookup_supports_vision(provider: str, model: str) -> Optional[bool]:
    """Return True/False if we can resolve caps, None if unknown."""
    if not provider or not model:
        return None
    try:
        from agent.models_dev import get_model_capabilities
        caps = get_model_capabilities(provider, model)
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("image_routing: caps lookup failed for %s:%s — %s", provider, model, exc)
        return None
    if caps is None:
        return None
    return bool(caps.supports_vision)


def decide_image_input_mode(
    provider: str,
    model: str,
    cfg: Optional[Dict[str, Any]],
) -> str:
    """Return ``"native"`` or ``"text"`` for the given turn.

    Args:
      provider: active inference provider ID (e.g. ``"anthropic"``, ``"openrouter"``).
      model:    active model slug as it would be sent to the provider.
      cfg:      loaded config.yaml dict, or None. When None, behaves as auto.
    """
    mode_cfg = "auto"
    if isinstance(cfg, dict):
        agent_cfg = cfg.get("agent") or {}
        if isinstance(agent_cfg, dict):
            mode_cfg = _coerce_mode(agent_cfg.get("image_input_mode"))

    if mode_cfg == "native":
        return "native"
    if mode_cfg == "text":
        return "text"

    # auto
    if _explicit_aux_vision_override(cfg):
        return "text"

    supports = _lookup_supports_vision(provider, model)
    if supports is True:
        return "native"
    return "text"


# Image size handling is REACTIVE rather than proactive: we attempt native
# attachment at full size regardless of provider, and rely on
# ``run_agent._try_shrink_image_parts_in_messages`` to shrink + retry if
# the provider rejects the request (e.g. Anthropic's hard 5 MB per-image
# ceiling returned as HTTP 400 "image exceeds 5 MB maximum").
#
# Why reactive: our knowledge of provider ceilings is partial and evolving
# (OpenAI accepts 49 MB+, Anthropic 5 MB, Gemini 100 MB, others unknown).
# A proactive per-provider table would be stale the moment a provider raises
# or lowers its limit, and silently degrading quality for users on providers
# that would have accepted the full image is the worse failure mode.
# The shrink-on-reject path loses 1 API call + maybe 1s of Pillow work when
# it fires, which is cheaper than permanent quality loss.


def _sniff_mime_from_bytes(raw: bytes) -> Optional[str]:
    """Detect image MIME from magic bytes. Returns None if unrecognised.

    Filename-based detection (``mimetypes.guess_type``) is unreliable when
    upstream platforms lie about content-type. Discord, for example, can
    serve a PNG with ``content_type=image/webp`` for proxied/animated
    stickers, custom emoji previews, or images uploaded via certain bots.
    Anthropic strictly validates that declared media_type matches the
    actual bytes and returns HTTP 400 on mismatch, so we sniff to be safe.
    """
    if not raw:
        return None
    # PNG: 89 50 4E 47 0D 0A 1A 0A
    if raw.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    # JPEG: FF D8 FF
    if raw.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    # GIF87a / GIF89a
    if raw[:6] in (b"GIF87a", b"GIF89a"):
        return "image/gif"
    # WEBP: "RIFF" .... "WEBP"
    if len(raw) >= 12 and raw[:4] == b"RIFF" and raw[8:12] == b"WEBP":
        return "image/webp"
    # BMP: "BM"
    if raw.startswith(b"BM"):
        return "image/bmp"
    # HEIC/HEIF: ftypheic / ftypheix / ftypmif1 / ftypmsf1 etc.
    if len(raw) >= 12 and raw[4:8] == b"ftyp" and raw[8:12] in (
        b"heic", b"heix", b"hevc", b"hevx", b"mif1", b"msf1", b"heim", b"heis",
    ):
        return "image/heic"
    return None


def _guess_mime(path: Path, raw: Optional[bytes] = None) -> str:
    """Return image MIME type for *path*.

    If *raw* bytes are provided, magic-byte sniffing wins (authoritative).
    Otherwise we fall back to ``mimetypes`` then suffix-based defaults.
    """
    if raw is not None:
        sniffed = _sniff_mime_from_bytes(raw)
        if sniffed:
            return sniffed
    mime, _ = mimetypes.guess_type(str(path))
    if mime and mime.startswith("image/"):
        return mime
    # mimetypes on some Linux distros mis-maps .jpg; default to jpeg when
    # the suffix looks imagey.
    suffix = path.suffix.lower()
    return {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".webp": "image/webp",
        ".bmp": "image/bmp",
    }.get(suffix, "image/jpeg")


def _file_to_data_url(path: Path) -> Optional[str]:
    """Encode a local image as a base64 data URL at its native size.

    Size limits are NOT enforced here — the agent retry loop
    (``run_agent._try_shrink_image_parts_in_messages``) shrinks on the
    provider's first rejection. Keeping this simple means providers that
    accept large images (OpenAI 49 MB+, Gemini 100 MB) don't pay a silent
    quality tax just because one other provider is stricter.

    Returns None only if the file can't be read (missing, permission
    denied, etc.); the caller reports those paths in ``skipped``.
    """
    try:
        raw = path.read_bytes()
    except Exception as exc:
        logger.warning("image_routing: failed to read %s — %s", path, exc)
        return None
    mime = _guess_mime(path, raw=raw)
    b64 = base64.b64encode(raw).decode("ascii")
    return f"data:{mime};base64,{b64}"


def build_native_content_parts(
    user_text: str,
    image_paths: List[str],
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Build an OpenAI-style ``content`` list for a user turn.

    Shape:
      [{"type": "text", "text": "...\\n\\n[Image attached at: /local/path]"},
       {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}},
       ...]

    The local path of each successfully attached image is appended to the
    text part as ``[Image attached at: <path>]``. The model still sees the
    pixels via the ``image_url`` part (full native vision); the path note
    just gives it a string handle so MCP/skill tools that take an image
    path or URL argument can be invoked on the same image without an
    extra round-trip. This parallels the text-mode hint produced by
    ``Runner._enrich_message_with_vision`` (``vision_analyze using image_url:
    <path>``) so behaviour is consistent across both image input modes.

    Images are attached at their native size. If a provider rejects the
    request because an image is too large (e.g. Anthropic's 5 MB per-image
    ceiling), the agent's retry loop transparently shrinks and retries
    once — see ``run_agent._try_shrink_image_parts_in_messages``.

    Returns (content_parts, skipped_paths). Skipped paths are files that
    couldn't be read from disk and are NOT advertised in the path hints.
    """
    skipped: List[str] = []
    image_parts: List[Dict[str, Any]] = []
    attached_paths: List[str] = []

    for raw_path in image_paths:
        p = Path(raw_path)
        if not p.exists() or not p.is_file():
            skipped.append(str(raw_path))
            continue
        data_url = _file_to_data_url(p)
        if not data_url:
            skipped.append(str(raw_path))
            continue
        image_parts.append({
            "type": "image_url",
            "image_url": {"url": data_url},
        })
        attached_paths.append(str(raw_path))

    text = (user_text or "").strip()

    # If at least one image attached, build a single text part that combines
    # the user's caption (or a neutral default) with one path hint per image.
    if attached_paths:
        base_text = text or "What do you see in this image?"
        path_hints = "\n".join(
            f"[Image attached at: {p}]" for p in attached_paths
        )
        combined_text = f"{base_text}\n\n{path_hints}"
        parts: List[Dict[str, Any]] = [{"type": "text", "text": combined_text}]
        parts.extend(image_parts)
        return parts, skipped

    # No images successfully attached — fall back to plain text-only behaviour.
    parts = []
    if text:
        parts.append({"type": "text", "text": text})
    return parts, skipped


__all__ = [
    "decide_image_input_mode",
    "build_native_content_parts",
]
