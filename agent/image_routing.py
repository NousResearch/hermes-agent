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
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


_VALID_MODES = frozenset({"auto", "native", "text"})


# Image extensions used by extract_image_refs(). Kept tight on purpose — we
# only auto-attach things the model can actually see. Documents/archives are
# excluded because the gateway's broader extract_local_files() also routes
# them differently (send_document), and we don't want to attach a PDF as a
# vision part.
_IMAGE_EXTS = (
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tiff", ".tif", ".heic",
)
_IMAGE_EXT_PATTERN = "|".join(e.lstrip(".") for e in _IMAGE_EXTS)

# Absolute / home-relative local image path. Matches the same shape gateway's
# extract_local_files() uses: anchors to ``~/`` or ``/``, ignores matches inside
# URLs (the ``(?<![/:\w.])`` lookbehind), and case-insensitive on the extension.
_LOCAL_IMAGE_PATH_RE = re.compile(
    r"(?<![/:\w.])(?:~/|/)(?:[\w.\-]+/)*[\w.\-]+\.(?:" + _IMAGE_EXT_PATTERN + r")\b",
    re.IGNORECASE,
)

# http(s) URL ending in an image extension (optionally followed by a
# query string). Case-insensitive on the extension. Strict ``http(s)://``
# scheme so we don't accidentally grab ``file://`` URLs or other shapes.
_IMAGE_URL_RE = re.compile(
    r"https?://[^\s<>\"']+?\.(?:" + _IMAGE_EXT_PATTERN + r")(?:\?[^\s<>\"']*)?",
    re.IGNORECASE,
)


def extract_image_refs(text: str) -> Tuple[List[str], List[str]]:
    """Scan free-form text for image references the model should see.

    Returns ``(local_paths, urls)``:

      * ``local_paths`` — absolute (``/``) or home-relative (``~/``) paths
        whose suffix is an image extension AND whose expanded form exists
        on disk as a file. Order-preserving, deduplicated.
      * ``urls`` — ``http(s)://…`` URLs whose path ends in an image
        extension (a ``?query`` is allowed after the extension).
        Order-preserving, deduplicated.

    Matches inside fenced code blocks (``` ``` ```) and inline backticks
    (`` `…` ``) are skipped so that snippets pasted into a task body for
    reference aren't mistaken for live attachments. This mirrors the
    behaviour of ``gateway.platforms.base.BaseAdapter.extract_local_files``.

    Local paths are validated against the filesystem; URLs are not
    (the provider fetches them at request time).
    """
    if not isinstance(text, str) or not text:
        return [], []

    # Build spans covered by fenced code blocks and inline code so we can
    # ignore references the author embedded purely as example text.
    code_spans: list[tuple[int, int]] = []
    for m in re.finditer(r"```[^\n]*\n.*?```", text, re.DOTALL):
        code_spans.append((m.start(), m.end()))
    for m in re.finditer(r"`[^`\n]+`", text):
        code_spans.append((m.start(), m.end()))

    def _in_code(pos: int) -> bool:
        return any(s <= pos < e for s, e in code_spans)

    local_paths: list[str] = []
    seen_paths: set[str] = set()
    for match in _LOCAL_IMAGE_PATH_RE.finditer(text):
        if _in_code(match.start()):
            continue
        raw = match.group(0)
        expanded = os.path.expanduser(raw)
        try:
            if not os.path.isfile(expanded):
                continue
        except OSError:
            # ENAMETOOLONG / EINVAL on pathological inputs — skip rather than crash.
            continue
        if expanded in seen_paths:
            continue
        seen_paths.add(expanded)
        local_paths.append(expanded)

    urls: list[str] = []
    seen_urls: set[str] = set()
    for match in _IMAGE_URL_RE.finditer(text):
        if _in_code(match.start()):
            continue
        url = match.group(0)
        # Strip trailing punctuation that's almost certainly prose, not part
        # of the URL (e.g. "see https://x.com/a.png." or "/a.png)").
        url = url.rstrip(".,;:!?)]>")
        if url in seen_urls:
            continue
        seen_urls.add(url)
        urls.append(url)

    return local_paths, urls


# Strict YAML/JSON boolean coercion for capability overrides.
#
# ``bool("false")`` is True in Python because non-empty strings are truthy, so
# a user writing ``supports_vision: "false"`` (quoted — a common YAML mistake)
# would silently enable native vision routing on a model that can't actually
# handle it. Accept only the values YAML 1.1 / 1.2 treat as booleans, plus
# real ``bool`` and integer 0/1. Anything else returns None so the caller
# falls through to models.dev rather than honouring garbage.
_TRUE_TOKENS = frozenset({"true", "yes", "on", "1"})
_FALSE_TOKENS = frozenset({"false", "no", "off", "0"})


def _coerce_capability_bool(raw: Any) -> Optional[bool]:
    """Return True/False for recognised boolean values, None otherwise."""
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, int):
        if raw in (0, 1):
            return bool(raw)
        return None
    if isinstance(raw, str):
        s = raw.strip().lower()
        if s in _TRUE_TOKENS:
            return True
        if s in _FALSE_TOKENS:
            return False
    return None


def _supports_vision_override(
    cfg: Optional[Dict[str, Any]],
    provider: str,
    model: str,
) -> Optional[bool]:
    """Resolve user-declared vision capability from config.yaml.

    Resolution order, first hit wins:
      1. ``model.supports_vision`` (top-level shortcut for the active model)
      2. ``providers.<provider>.models.<model>.supports_vision``
         (named custom providers — ``provider`` may be the runtime-resolved
         value ``"custom"`` and/or the user-declared name under
         ``model.provider``; both are tried)

    Returns None when no override is set, so the caller falls through to
    models.dev. Returns False explicitly only when the user wrote a
    recognised boolean false token.
    """
    if not isinstance(cfg, dict):
        return None

    # 1. Top-level shortcut
    model_cfg_raw = cfg.get("model")
    model_cfg: Dict[str, Any] = model_cfg_raw if isinstance(model_cfg_raw, dict) else {}
    top = _coerce_capability_bool(model_cfg.get("supports_vision"))
    if top is not None:
        return top

    # 2. Per-provider, per-model. Named custom providers (e.g. "my-vllm")
    # get rewritten to provider="custom" at runtime
    # (hermes_cli/runtime_provider.py:_resolve_named_custom_runtime), so the
    # config still holds the user-declared name under model.provider. Try
    # both as candidate provider keys.
    config_provider = str(model_cfg.get("provider") or "").strip()
    providers_raw = cfg.get("providers")
    providers_cfg: Dict[str, Any] = providers_raw if isinstance(providers_raw, dict) else {}
    for p in dict.fromkeys(filter(None, (provider, config_provider))):
        entry_raw = providers_cfg.get(p)
        entry: Dict[str, Any] = entry_raw if isinstance(entry_raw, dict) else {}
        models_raw = entry.get("models")
        models_cfg: Dict[str, Any] = models_raw if isinstance(models_raw, dict) else {}
        per_model_raw = models_cfg.get(model)
        per_model: Dict[str, Any] = per_model_raw if isinstance(per_model_raw, dict) else {}
        coerced = _coerce_capability_bool(per_model.get("supports_vision"))
        if coerced is not None:
            return coerced

    # 2b. Legacy list-style custom_providers. Entries are dicts with a
    # "name" key and a nested "models" dict. Match by provider name (which
    # may appear as the raw name or "custom:<name>" at runtime).
    custom_providers = cfg.get("custom_providers")
    if isinstance(custom_providers, list):
        # Build candidate names: the provider value and the config provider
        # value, both raw and with "custom:" prefix stripped/added.
        candidate_names: set = set()
        for p in filter(None, (provider, config_provider)):
            candidate_names.add(p)
            if p.startswith("custom:"):
                candidate_names.add(p[len("custom:"):])
            else:
                candidate_names.add(f"custom:{p}")
        for entry_raw in custom_providers:
            if not isinstance(entry_raw, dict):
                continue
            entry_name = str(entry_raw.get("name") or "").strip()
            if entry_name not in candidate_names:
                continue
            models_raw = entry_raw.get("models")
            models_cfg = models_raw if isinstance(models_raw, dict) else {}
            per_model_raw = models_cfg.get(model)
            per_model = per_model_raw if isinstance(per_model_raw, dict) else {}
            coerced = _coerce_capability_bool(per_model.get("supports_vision"))
            if coerced is not None:
                return coerced

    return None


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
    if provider in {"", "auto"} and not model and not base_url:
        return False
    return True


def _lookup_supports_vision(
    provider: str,
    model: str,
    cfg: Optional[Dict[str, Any]] = None,
) -> Optional[bool]:
    """Return True/False if we can resolve caps, None if unknown.

    Consults the user's ``supports_vision`` override in config.yaml first
    (so custom/local models declared as vision-capable don't fall through to
    text routing in ``auto`` mode), then falls back to models.dev.
    """
    override = _supports_vision_override(cfg, provider, model)
    if override is not None:
        return override
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

    supports = _lookup_supports_vision(provider, model, cfg)
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
    if raw[:6] in {b"GIF87a", b"GIF89a"}:
        return "image/gif"
    # WEBP: "RIFF" .... "WEBP"
    if len(raw) >= 12 and raw[:4] == b"RIFF" and raw[8:12] == b"WEBP":
        return "image/webp"
    # BMP: "BM"
    if raw.startswith(b"BM"):
        return "image/bmp"
    # HEIC/HEIF: ftypheic / ftypheix / ftypmif1 / ftypmsf1 etc.
    if len(raw) >= 12 and raw[4:8] == b"ftyp" and raw[8:12] in {
        b"heic", b"heix", b"hevc", b"hevx", b"mif1", b"msf1", b"heim", b"heis",
    }:
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


_NATIVE_IMAGE_MIME_TYPES = {"image/jpeg", "image/png", "image/gif", "image/webp"}


def _file_to_data_url(path: Path) -> Optional[str]:
    """Encode a local image as a base64 data URL at its native size.

    Size limits are NOT enforced here — the agent retry loop
    (``run_agent._try_shrink_image_parts_in_messages``) shrinks on the
    provider's first rejection. Keeping this simple means providers that
    accept large images (OpenAI 49 MB+, Gemini 100 MB) don't pay a silent
    quality tax just because one other provider is stricter.

    Returns None if the file can't be read or if its bytes are not one of
    the natively supported image formats.  This deliberately treats magic
    bytes as authoritative: platform adapters can occasionally pass the
    wrong path into the image list (for example a Telegram Ogg/Opus voice
    note from a mixed media burst).  Falling back to a ``.jpg`` suffix for
    those bytes produces ``data:image/jpeg`` payloads that providers reject
    with HTTP 400 ``invalid_image`` errors.
    """
    try:
        raw = path.read_bytes()
    except Exception as exc:
        logger.warning("image_routing: failed to read %s — %s", path, exc)
        return None
    mime = _sniff_mime_from_bytes(raw)
    if mime not in _NATIVE_IMAGE_MIME_TYPES:
        logger.warning(
            "image_routing: skipped non-native-image file %s (detected=%s)",
            path,
            mime or "unknown",
        )
        return None
    b64 = base64.b64encode(raw).decode("ascii")
    return f"data:{mime};base64,{b64}"


def build_native_content_parts(
    user_text: str,
    image_paths: List[str],
    image_urls: Optional[List[str]] = None,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Build an OpenAI-style ``content`` list for a user turn.

    Shape:
      [{"type": "text", "text": "...\\n\\n[Image attached at: /local/path]"},
       {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}},
       {"type": "image_url", "image_url": {"url": "https://example.com/a.png"}},
       ...]

    Local paths are read from disk and embedded as base64 ``data:`` URLs.
    Remote URLs (``http(s)://``) are passed through verbatim — the provider
    fetches them server-side. The model still sees the pixels either way.

    For each successfully attached image, a hint is appended to the text
    part:

      * local path → ``[Image attached at: <path>]``
      * URL        → ``[Image attached: <url>]``

    The hint gives the model a string handle so MCP/skill tools that take
    an image path or URL argument can be invoked on the same image without
    an extra round-trip. This parallels the text-mode hint produced by
    ``Runner._enrich_message_with_vision`` (``vision_analyze using image_url:
    <path>``) so behaviour is consistent across both image input modes.

    Images are attached at their native size. If a provider rejects the
    request because an image is too large (e.g. Anthropic's 5 MB per-image
    ceiling), the agent's retry loop transparently shrinks and retries
    once — see ``run_agent._try_shrink_image_parts_in_messages``.

    Returns (content_parts, skipped). Skipped entries are local paths
    that couldn't be read from disk; URLs are never skipped (they're
    not validated here).
    """
    skipped: List[str] = []
    image_parts: List[Dict[str, Any]] = []
    attached_paths: List[str] = []
    attached_urls: List[str] = []

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

    for url in image_urls or []:
        url = (url or "").strip()
        if not url:
            continue
        image_parts.append({
            "type": "image_url",
            "image_url": {"url": url},
        })
        attached_urls.append(url)

    text = (user_text or "").strip()

    # If at least one image attached, build a single text part that combines
    # the user's caption (or a neutral default) with one hint per image.
    if attached_paths or attached_urls:
        base_text = text or "What do you see in this image?"
        hint_lines: List[str] = []
        hint_lines.extend(f"[Image attached at: {p}]" for p in attached_paths)
        hint_lines.extend(f"[Image attached: {u}]" for u in attached_urls)
        combined_text = f"{base_text}\n\n" + "\n".join(hint_lines)
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
    "extract_image_refs",
]
