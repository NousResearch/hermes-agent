"""Routing helpers for inbound user-attached images.

``native`` attaches images as OpenAI-style ``image_url`` parts; ``text`` runs
``vision_analyze`` up-front and prepends the lossy description (right for
non-vision models). :func:`decide_image_input_mode` picks once per turn from
``agent.image_input_mode`` (``auto`` | ``native`` | ``text``): in ``auto`` an
explicit ``auxiliary.vision`` backend forces ``text`` even for vision-capable
main models (``native`` is the absolute override); else ``supports_vision``
(config override or catalog) decides. ``vision_analyze`` stays a tool regardless.
"""

from __future__ import annotations

import base64
import logging
import mimetypes
import os
import re
from contextlib import suppress
from io import BytesIO
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

logger = logging.getLogger(__name__)


_VALID_MODES = frozenset({"auto", "native", "text"})


# Extensions extract_image_refs() auto-attaches. Documents/archives are excluded:
# the gateway routes them via send_document and a PDF must never become a vision part.
_IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tiff", ".tif", ".heic")
_IMAGE_EXT_PATTERN = "|".join(e.lstrip(".") for e in _IMAGE_EXTS)
# Local path: same shape as gateway extract_local_files() — anchored to ``~/`` or
# ``/``, lookbehind skips matches inside URLs. URL: strict ``http(s)://`` so
# ``file://`` and other schemes are not grabbed; optional query string.
_LOCAL_IMAGE_PATH_RE = re.compile(
    r"(?<![/:\w.])(?:~/|/)(?:[\w.\-]+/)*[\w.\-]+\.(?:" + _IMAGE_EXT_PATTERN + r")\b", re.IGNORECASE,
)
_IMAGE_URL_RE = re.compile(
    r"https?://[^\s<>\"']+?\.(?:" + _IMAGE_EXT_PATTERN + r")(?:\?[^\s<>\"']*)?", re.IGNORECASE,
)
_CODE_SPAN_RES = (re.compile(r"```[^\n]*\n.*?```", re.DOTALL), re.compile(r"`[^`\n]+`"))


def _matches_outside_code(pattern: re.Pattern, text: str) -> Iterable[str]:
    """Yield ``pattern`` matches whose start is not inside a fenced block or inline backticks."""
    spans = [(m.start(), m.end()) for p in _CODE_SPAN_RES for m in p.finditer(text)]
    return (m.group(0) for m in pattern.finditer(text) if not any(s <= m.start() < e for s, e in spans))


def _existing_file(candidate: str) -> Optional[str]:
    """Expanded path when it is a regular file; None otherwise (incl. OSError on pathological input)."""
    expanded = os.path.expanduser(candidate)
    try:
        return expanded if os.path.isfile(expanded) else None
    except OSError:
        return None


def extract_image_refs(text: str) -> Tuple[List[str], List[str]]:
    """Scan free-form text for image references → ``(local_paths, urls)``, each
    ordered and deduplicated. Local paths must exist as files; URLs are not
    validated (the provider fetches them). Code spans are skipped so pasted
    snippets aren't live attachments (mirrors ``BaseAdapter.extract_local_files``)."""
    if not isinstance(text, str) or not text:
        return [], []
    local_paths = dict.fromkeys(
        p for p in map(_existing_file, _matches_outside_code(_LOCAL_IMAGE_PATH_RE, text)) if p
    )
    # Trailing punctuation is almost certainly prose ("see https://x/a.png.").
    urls = dict.fromkeys(u.rstrip(".,;:!?)]>") for u in _matches_outside_code(_IMAGE_URL_RE, text))
    return list(local_paths), list(urls)


_BOOL_TOKENS = {
    **dict.fromkeys(("true", "yes", "on", "1"), True),
    **dict.fromkeys(("false", "no", "off", "0"), False),
}


def _coerce_capability_bool(raw: Any) -> Optional[bool]:
    """Strict boolean coercion for capability overrides: real bools, 0/1 and YAML
    boolean tokens only; anything else is None so the caller falls through to
    models.dev. ``bool("false")`` is True, so a quoted ``supports_vision: "false"``
    would otherwise silently enable native routing on a model that can't handle it."""
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, int):
        return bool(raw) if raw in (0, 1) else None
    return _BOOL_TOKENS.get(raw.strip().lower()) if isinstance(raw, str) else None


def _dict_or_empty(raw: Any) -> Dict[str, Any]:
    return raw if isinstance(raw, dict) else {}


def _clean_str(raw: Any) -> str:
    return str(raw or "").strip()


def _runtime_main(key: str) -> str:
    """Stripped context-local main-runtime value, or "" when unavailable."""
    try:
        from agent.auxiliary_client import _runtime_main_value

        return _clean_str(_runtime_main_value(key))
    except Exception:
        return ""


def _model_supports_vision_override(models_cfg: Any, model: str) -> Optional[bool]:
    """Per-model ``supports_vision`` (or ``vision`` alias) from a ``models`` mapping."""
    per_model = _dict_or_empty(_dict_or_empty(models_cfg).get(model))
    return _coerce_capability_bool(per_model.get("supports_vision", per_model.get("vision")))


def _custom_provider_entries(cfg: Dict[str, Any], names: Iterable[str]) -> Iterable[Dict[str, Any]]:
    """Yield legacy ``custom_providers`` entries matching ``names`` (case-insensitive);
    ``names`` is the outer loop so list order cannot let a persisted default shadow the live route."""
    entries = _custom_provider_list(cfg)
    for wanted in (n.strip().lower() for n in names):
        yield from (e for e in entries if _clean_str(e.get("name")).lower() == wanted)


def _custom_provider_list(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Dict entries of the legacy ``custom_providers`` list (empty when absent/malformed)."""
    raw = cfg.get("custom_providers")
    return [e for e in raw if isinstance(e, dict)] if isinstance(raw, list) else []


def _supports_vision_override(
    cfg: Optional[Dict[str, Any]],
    provider: str,
    model: str,
    *,
    requested_provider: str = "",
) -> Optional[bool]:
    """Resolve user-declared vision capability from config.yaml; None when unset.

    First hit wins: ``model.supports_vision`` → ``providers.<p>.models.<model>``
    → legacy ``custom_providers[].models.<model>``. Named custom providers are
    rewritten to ``provider="custom"`` at runtime while config keeps the user's
    name under ``model.provider``, so the requested, runtime and config
    identities are all tried, plus the bare ``<name>`` of any ``custom:<name>``.
    """
    if not isinstance(cfg, dict):
        return None
    model_cfg = _dict_or_empty(cfg.get("model"))
    top = _coerce_capability_bool(model_cfg.get("supports_vision"))
    if top is not None:
        return top

    candidates: List[str] = []
    for candidate in filter(None, (requested_provider, provider, _clean_str(model_cfg.get("provider")))):
        candidates.append(candidate)
        if candidate.startswith("custom:") and candidate[len("custom:"):]:
            candidates.append(candidate[len("custom:"):])
    candidates = list(dict.fromkeys(candidates))

    providers_cfg = _dict_or_empty(cfg.get("providers"))
    model_maps = [_dict_or_empty(providers_cfg.get(p)).get("models") for p in candidates]
    model_maps += [entry.get("models") for entry in _custom_provider_entries(cfg, candidates)]
    return next((v for v in (_model_supports_vision_override(m, model) for m in model_maps) if v is not None), None)


def _resolve_inference_value(
    cfg: Optional[Dict[str, Any]],
    provider: str,
    key: str,
    *,
    runtime_ok: Callable[[str], bool],
) -> str:
    """``base_url`` / ``api_key`` of the active inference provider. Order: runtime
    value (when ``runtime_ok`` accepts it) → ``model.<key>`` → ``providers.<name>.<key>``
    → ``custom_providers[].<key>``, ``<name>`` covering the provider and
    ``model.provider`` in both bare and ``custom:``-prefixed forms."""
    runtime = _runtime_main(key)
    if runtime and runtime_ok(runtime):
        return runtime
    if not isinstance(cfg, dict):
        return ""
    model_cfg = _dict_or_empty(cfg.get("model"))
    value = _clean_str(model_cfg.get(key))
    if value:
        return value

    names: set[str] = set()
    for p in filter(None, (provider, _clean_str(model_cfg.get("provider")))):
        names.add(p)
        names.add(p.split(":", 1)[1] if p.lower().startswith("custom:") else f"custom:{p}")
    lowered = {n.lower() for n in names}
    providers_cfg = _dict_or_empty(cfg.get("providers"))
    entries = [e for e in map(providers_cfg.get, names) if isinstance(e, dict)]
    entries += [
        e for e in _custom_provider_list(cfg)
        if _clean_str(e.get("name")) in names or _clean_str(e.get("name")).lower() in lowered
    ]
    return next((v for v in (_clean_str(e.get(key)) for e in entries) if v), "")


def _resolve_inference_base_url(cfg: Optional[Dict[str, Any]], provider: str) -> str:
    """Best-effort base URL for the active inference provider; the runtime value is
    only trusted when it belongs to the requested provider (or none was requested)."""
    requested = _clean_str(provider).lower()
    return _resolve_inference_value(
        cfg, provider, "base_url",
        runtime_ok=lambda _: not requested or requested == _runtime_main("provider").lower(),
    )


def _resolve_inference_api_key(cfg: Optional[Dict[str, Any]], provider: str) -> str:
    """Best-effort API key, resolved like :func:`_resolve_inference_base_url` so it
    matches the base URL actually probed; otherwise the local server-type probe hits
    a keyed remote endpoint without Authorization and sprays 401s on every image turn.

    Mirrors :func:`_resolve_inference_base_url`'s resolution order (runtime value, then ``model.api_key``,
    then the providers blocks) so the key matches the base URL actually being probed. See #89863.
    """
    return _resolve_inference_value(cfg, provider, "api_key", runtime_ok=lambda _: True)


def _should_probe_ollama_vision(provider: str, base_url: str, api_key: str = "") -> bool:
    """True when the active provider likely fronts a local Ollama server. Fingerprint
    probing is only valid for LOCAL endpoints: remote OpenAI-compatible APIs (sglang,
    vLLM) expose Ollama-compat routes that can misidentify, and probing them without
    an api_key returns 401 on every leg."""
    if _clean_str(provider).lower() == "ollama":
        return True
    if not base_url:
        return False
    try:
        from agent.model_metadata import detect_local_server_type, is_local_endpoint

        # Forward the key: an unauthorized probe can never produce a positive verdict.
        return bool(is_local_endpoint(base_url)) and detect_local_server_type(base_url, api_key=api_key) == "ollama"
    except Exception:
        return False


def _coerce_mode(raw: Any) -> str:
    """Normalize a config value into one of the valid modes (default ``auto``)."""
    mode = raw.strip().lower() if isinstance(raw, str) else ""
    return mode if mode in _VALID_MODES else "auto"


def _explicit_aux_vision_override(cfg: Optional[Dict[str, Any]]) -> bool:
    """True when the user configured a specific ``auxiliary.vision`` backend — the
    de-facto image route in ``auto`` mode even when the main model has native vision.
    ``auto``/empty provider with no model and no base_url is not explicit."""
    vision = _dict_or_empty(_dict_or_empty(_dict_or_empty(cfg).get("auxiliary")).get("vision"))
    return bool(vision) and not (
        _clean_str(vision.get("provider")).lower() in {"", "auto"}
        and not _clean_str(vision.get("model"))
        and not _clean_str(vision.get("base_url"))
    )


def _probe_managed_runtime(provider: str, model: str, cfg: Optional[Dict[str, Any]]) -> Optional[bool]:
    """Managed local runtime verdict: the server receiving the image is the authority
    on whether it can see (its /props reports modalities). Cloud catalogs have never
    heard of a local GGUF, so without this every local model reads as text-only and
    screenshots detour to a cloud auxiliary."""
    from hermes_cli.local_runtime.capabilities import is_managed_provider, managed_model_supports_vision

    managed = is_managed_provider(provider, _resolve_inference_base_url(cfg, provider) or "")
    return managed_model_supports_vision(model) if managed else None


def _probe_models_dev(provider: str, model: str, cfg: Optional[Dict[str, Any]]) -> Optional[bool]:
    """models.dev catalog verdict. ``allow_network=True`` on purpose: this runs only
    when an image needs routing, and the text-only-main guard depends on catalog
    data — a cold cache returning "unknown" would reintroduce attempting the call.
    The fetch is cached (4h TTL) and backoff-limited."""
    from agent.models_dev import get_model_capabilities

    # allow_network=True on purpose: vision-capability lookup runs when an image actually needs routing (not
    # per turn), and the #31179 text-only-main guard depends on catalog data — a cold cache returning
    # "unknown" would fall back to attempting the call and reintroduce the bug. This preserves the
    # historical network-on-cold-cache behavior for this one path; the fetch is cached (4h TTL) and
    # backoff-limited after failures.
    caps = get_model_capabilities(provider, model, allow_network=True)
    return None if caps is None else bool(caps.supports_vision)


def _probe_ollama(provider: str, model: str, cfg: Optional[Dict[str, Any]]) -> Optional[bool]:
    """Ollama ``/api/show`` verdict for local endpoints (see :func:`_should_probe_ollama_vision`)."""
    base_url = _resolve_inference_base_url(cfg, provider)
    if not base_url and _clean_str(provider).lower() == "ollama":
        base_url = "http://localhost:11434/v1"
    api_key = _resolve_inference_api_key(cfg, provider)
    if not _should_probe_ollama_vision(provider, base_url, api_key=api_key):
        return None
    from agent.model_metadata import query_ollama_supports_vision

    return query_ollama_supports_vision(model, base_url, api_key=api_key)


# Capability probes after the config override, in priority order; each returns
# True/False or None (unknown → next probe). Exceptions are logged and treated as None.
_VISION_PROBES: Tuple[Tuple[str, Callable[..., Optional[bool]]], ...] = (
    ("managed-runtime caps lookup", _probe_managed_runtime),
    ("caps lookup", _probe_models_dev),
    ("ollama vision probe", _probe_ollama),
)


def _lookup_supports_vision(
    provider: str,
    model: str,
    cfg: Optional[Dict[str, Any]] = None,
    *,
    requested_provider: str = "",
) -> Optional[bool]:
    """Return True/False if vision capability can be resolved, None if unknown.

    Order: config ``supports_vision`` override → :data:`_VISION_PROBES`
    (managed local runtime → models.dev catalog → Ollama probe).
    """
    # Named custom providers are canonicalized to ``provider="custom"``; the
    # original name lives in the context-local main runtime. Borrow it only on an
    # exact provider+model match so background/auxiliary lookups never take
    # another turn's identity.
    if (
        not requested_provider
        and _runtime_main("provider").lower() == _clean_str(provider).lower()
        and _runtime_main("model") == _clean_str(model)
    ):
        requested_provider = _runtime_main("requested_provider")

    override = _supports_vision_override(cfg, provider, model, requested_provider=requested_provider)
    if override is not None:
        return override
    if not provider or not model:
        return None

    for label, probe in _VISION_PROBES:
        try:
            verdict = probe(provider, model, cfg)
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("image_routing: %s failed for %s:%s — %s", label, provider, model, exc)
            continue
        if verdict is not None:
            return verdict
    return None


def decide_image_input_mode(
    provider: str,
    model: str,
    cfg: Optional[Dict[str, Any]],
    *,
    requested_provider: str = "",
) -> str:
    """Return ``"native"`` or ``"text"`` for the given turn (``cfg`` None behaves as
    auto; ``requested_provider`` is the identity before runtime canonicalization)."""
    mode_cfg = _coerce_mode(_dict_or_empty(_dict_or_empty(cfg).get("agent")).get("image_input_mode"))
    if mode_cfg != "auto":
        return mode_cfg
    if _explicit_aux_vision_override(cfg):  # auto: an explicit auxiliary.vision backend wins
        return "text"
    # Keep the three-argument call contract for callers/tests that replace the lookup hook.
    extra = {"requested_provider": requested_provider} if requested_provider else {}
    return "native" if _lookup_supports_vision(provider, model, cfg, **extra) is True else "text"


# Image size handling is REACTIVE: attach at full size and let
# ``run_agent._try_shrink_image_parts_in_messages`` shrink + retry on rejection
# (e.g. Anthropic's 5 MB ceiling as HTTP 400). Provider ceilings are partial and
# evolving; a proactive table would go stale and silently degrade quality.

# Magic-byte signatures as ((offset, bytes), ...) conjunctions, checked in order.
# Platforms lie about content-type (Discord serves PNG as ``image/webp`` for
# proxied stickers) and Anthropic rejects a mismatched media_type with HTTP 400.
# ISO-BMFF family (HEIC/HEIF/AVIF): 'ftyp' at 4..8, major brand at 8..12.
_FTYP_BRANDS = {
    **dict.fromkeys((b"avif", b"avis"), "image/avif"),
    **dict.fromkeys((b"heic", b"heix", b"hevc", b"hevx", b"mif1", b"msf1", b"heim", b"heis"), "image/heic"),
}
_MAGIC: Tuple[Tuple[Tuple[Tuple[int, bytes], ...], str], ...] = (
    (((0, b"\x89PNG\r\n\x1a\n"),), "image/png"),
    (((0, b"\xff\xd8\xff"),), "image/jpeg"),
    (((0, b"GIF87a"),), "image/gif"), (((0, b"GIF89a"),), "image/gif"),
    (((0, b"RIFF"), (8, b"WEBP")), "image/webp"),
    (((0, b"BM"),), "image/bmp"),
    *((((4, b"ftyp"), (8, brand)), mime) for brand, mime in _FTYP_BRANDS.items()),
    (((0, b"II*\x00"),), "image/tiff"), (((0, b"MM\x00*"),), "image/tiff"),
    (((0, b"\x00\x00\x01\x00"),), "image/x-icon"),
)


def _sniff_mime_from_bytes(raw: bytes) -> Optional[str]:
    """Detect image MIME from magic bytes; None if unrecognised."""
    if not raw:
        return None
    for conditions, mime in _MAGIC:
        if all(raw[off:off + len(sig)] == sig for off, sig in conditions):
            return mime
    # SVG is text: look for an <svg tag near the start (skip BOM/whitespace).
    head = raw[:512].lstrip().lower()
    return "image/svg+xml" if head.startswith((b"<?xml", b"<svg")) and b"<svg" in head else None


# Formats every major vision provider accepts natively. Anything else is transcoded
# to PNG before declaring media_type or the provider returns HTTP 400 and the turn
# fails; chat platforms freely accept AVIF (Chromium screenshots), HEIC (iPhone),
# TIFF, BMP and ICO. SVG is vector — Pillow cannot rasterize it — so it is skipped.
_UNIVERSALLY_SUPPORTED_MIMES = frozenset({"image/png", "image/jpeg", "image/gif", "image/webp"})


def _transcode_to_png(raw: bytes) -> Optional[bytes]:
    """Decode with Pillow and re-encode as PNG; None when impossible. HEIC/HEIF and
    AVIF need optional Pillow plugins, registered on demand; a missing plugin just
    looks like "can't decode" so the caller skips the image and the turn proceeds."""
    try:
        from PIL import Image
    except ImportError:
        logger.info(
            "image_routing: Pillow not installed; cannot transcode "
            "non-standard image format to PNG. Install with `pip install Pillow` "
            "(and `pillow-heif` / `pillow-avif-plugin` for those formats)."
        )
        return None
    with suppress(Exception):
        import pillow_heif  # type: ignore

        pillow_heif.register_heif_opener()
    with suppress(Exception):
        import pillow_avif  # type: ignore  # noqa: F401  -- registers AVIF on import
    try:
        with Image.open(BytesIO(raw)) as im:
            # Normalise exotic modes to RGBA so PNG can serialise and transparency survives.
            if im.mode not in {"RGB", "RGBA", "L", "LA", "P"}:
                im = im.convert("RGBA")
            buf = BytesIO()
            im.save(buf, format="PNG", optimize=False)
            return buf.getvalue()
    except Exception as exc:
        logger.info("image_routing: Pillow could not transcode image to PNG -- %s", exc)
        return None


_SUFFIX_MIMES = {
    ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
    ".gif": "image/gif", ".webp": "image/webp", ".bmp": "image/bmp",
}


def _guess_mime(path: Path, raw: Optional[bytes] = None) -> str:
    """Image MIME for *path*: magic bytes (authoritative) → ``mimetypes`` → suffix → jpeg."""
    sniffed = _sniff_mime_from_bytes(raw) if raw is not None else None
    mime = sniffed or mimetypes.guess_type(str(path))[0]
    if mime and mime.startswith("image/"):
        return mime
    # mimetypes on some Linux distros mis-maps .jpg; default to jpeg.
    return _SUFFIX_MIMES.get(path.suffix.lower(), "image/jpeg")


def _accepted_mimes() -> frozenset:
    """Provider-accepted MIME set for the current main runtime. The managed local
    server decodes fewer formats (no WebP — and a WebP part fails SILENTLY: the model
    confabulates a description), so its narrower set transcodes those here."""
    try:
        from agent.auxiliary_client import _runtime_main_value
        from hermes_cli.local_runtime.capabilities import ACCEPTED_IMAGE_MIMES, is_managed_provider

        if is_managed_provider(str(_runtime_main_value("provider") or ""), str(_runtime_main_value("base_url") or "")):
            return ACCEPTED_IMAGE_MIMES
    except Exception:  # noqa: BLE001 — best-effort narrowing only
        pass
    return _UNIVERSALLY_SUPPORTED_MIMES


def _file_to_data_url(path: Path) -> Optional[str]:
    """Encode a local image as a base64 data URL at native size (the agent retry
    loop shrinks on rejection, so lenient providers pay no silent quality tax);
    MIMEs outside the accepted set are transcoded to PNG. None when unreadable,
    blocked by the read guard, or untranscodable — the caller reports ``skipped``."""
    try:
        from agent.file_safety import raise_if_read_blocked

        raise_if_read_blocked(str(path))
    except ValueError as exc:
        logger.warning("image_routing: blocked local image attachment %s -- %s", path, exc)
        return None
    except Exception:
        pass  # Keep attachment routing best-effort if the guard itself is unavailable.
    try:
        raw = path.read_bytes()
    except Exception as exc:
        logger.warning("image_routing: failed to read %s — %s", path, exc)
        return None
    mime = _guess_mime(path, raw=raw)
    if mime not in _accepted_mimes():
        if (transcoded := _transcode_to_png(raw)) is None:
            logger.warning(
                "image_routing: %s is %s which is not accepted by the active provider "
                "and could not be transcoded to PNG; skipping this attachment.", path, mime,
            )
            return None
        logger.info("image_routing: transcoded %s (%s) -> image/png for provider compatibility", path.name, mime)
        raw, mime = transcoded, "image/png"
    return f"data:{mime};base64,{base64.b64encode(raw).decode('ascii')}"


def build_native_content_parts(
    user_text: str,
    image_paths: List[str],
    image_urls: Optional[List[str]] = None,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Build an OpenAI-style ``content`` list for a user turn.

    Local paths become base64 ``data:`` URLs; remote URLs pass through verbatim.
    When any image attaches, one text part combines the caption (or a neutral
    default) with a ``[Image attached at: <path>]`` / ``[Image attached: <url>]``
    hint per image — a string handle for tools taking an image path/URL, mirroring
    ``Runner._enrich_message_with_vision``. Returns ``(content_parts, skipped)``;
    ``skipped`` holds unreadable local paths (URLs are never skipped).
    """
    skipped: List[str] = []
    attached: List[Tuple[str, str]] = []  # (url, hint)
    for raw_path in image_paths:
        p = Path(raw_path)
        data_url = _file_to_data_url(p) if p.exists() and p.is_file() else None
        if data_url:
            attached.append((data_url, f"[Image attached at: {raw_path}]"))
        else:
            skipped.append(str(raw_path))
    attached += [(u, f"[Image attached: {u}]") for u in ((u or "").strip() for u in image_urls or []) if u]

    text = (user_text or "").strip()
    if not attached:
        return ([{"type": "text", "text": text}] if text else []), skipped
    combined_text = f"{text or 'What do you see in this image?'}\n\n" + "\n".join(h for _, h in attached)
    image_parts = [{"type": "image_url", "image_url": {"url": u}} for u, _ in attached]
    return [{"type": "text", "text": combined_text}, *image_parts], skipped


__all__ = ["decide_image_input_mode", "build_native_content_parts", "extract_image_refs"]
