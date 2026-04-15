"""Helpers for chat providers that also serve TTS / image / vision natively.

When a chat provider exposes those non-chat capabilities on the same API
base and credential, the setup wizard wires them as the default tool
backends and the tool dispatchers route through this module instead of
the generic FAL / auxiliary-client paths.

Mirrors the shape of `hermes_cli.nous_subscription`: one apply hook for
the wizard, plus runtime helpers consumed by the tool files.

Adding a provider:

  1. Add the canonical id to `MINIMAX_PROVIDERS` (or add a new
     `NATIVE_TOOLS_BY_PROVIDER` row + provider set).
  2. Implement the request helpers in this file (mirror
     `_minimax_image_request` / `_minimax_vlm_request`).
  3. Route to them from `generate_image` / `analyze_image` / etc.
"""

from __future__ import annotations

import base64
import json
import logging
import mimetypes
import os
import ssl
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Set, Tuple

logger = logging.getLogger(__name__)


#: Canonical provider id → tool categories the provider serves natively.
NATIVE_TOOLS_BY_PROVIDER: Dict[str, Tuple[str, ...]] = {
    "minimax":    ("tts", "image_gen", "vision"),
    "minimax-cn": ("tts", "image_gen", "vision"),
}

MINIMAX_PROVIDERS: Set[str] = {"minimax", "minimax-cn"}


# ─── Active-provider lookup ──────────────────────────────────────────────


def _active_provider_id(config: Dict[str, Any]) -> str:
    """Canonical id of the active chat provider, or `""`.

    Reads `model.provider` (dict shape) or strips the prefix off the
    legacy `model: "provider/name"` string.  Normalises through
    `hermes_cli.providers.normalize_provider` so hand-edited aliases
    resolve to the canonical id.
    """
    model = config.get("model")
    raw = ""
    if isinstance(model, dict):
        raw = str(model.get("provider") or "").strip().lower()
    elif isinstance(model, str) and "/" in model:
        raw = model.split("/", 1)[0].strip().lower()
    if not raw:
        return ""
    try:
        from hermes_cli.providers import normalize_provider
        return normalize_provider(raw)
    except Exception:
        return raw


def get_native_tools(config: Dict[str, Any]) -> Tuple[str, ...]:
    """Native tool categories declared for the active provider, or `()`."""
    return NATIVE_TOOLS_BY_PROVIDER.get(_active_provider_id(config), ())


def provider_has_native_tool(tool: str, config: Dict[str, Any]) -> bool:
    """True when the active chat provider serves `tool` natively."""
    return tool in get_native_tools(config)


def active_provider_api_root(config: Dict[str, Any]) -> str:
    """API root for the active chat provider, or `""`.

    Reads `model.base_url` and strips the `/anthropic` chat-compat suffix
    when present, so non-chat endpoints (`/v1/image_generation`,
    `/v1/t2a_v2`, `/v1/coding_plan/vlm`) hang off the returned root.
    """
    model = config.get("model")
    if not isinstance(model, dict):
        return ""
    base = str(model.get("base_url") or "").strip().rstrip("/")
    if not base:
        return ""
    return base[: -len("/anthropic")] if base.endswith("/anthropic") else base


# ─── Setup-time defaults ─────────────────────────────────────────────────
#
# Mirrors `apply_nous_provider_defaults`: hardcodes the few config slots
# we touch, only overrides built-in defaults, and returns the set of
# slots actually changed so the wizard can print a summary.


_OVERRIDABLE_TTS    = {"", "edge"}
_OVERRIDABLE_IMAGE  = {"", "auto", "fal"}
_OVERRIDABLE_VISION = {"", "auto", "main"}


def apply_provider_native_tool_defaults(config: Dict[str, Any]) -> Set[str]:
    """Wire `tts.provider` / `image_gen.provider` for native providers.

    Returns the set of categories newly wired (empty when the provider
    isn't native or everything is already set).  Vision is reported when
    something else also changed; no config value is persisted because
    the dispatcher in `tools/vision_tools.py` activates per session.
    """
    native = get_native_tools(config)
    if not native:
        return set()

    provider = _active_provider_id(config)
    changed: Set[str] = set()

    if "tts" in native:
        cfg = config.setdefault("tts", {})
        if isinstance(cfg, dict) and (str(cfg.get("provider") or "").strip().lower() in _OVERRIDABLE_TTS):
            cfg["provider"] = provider
            changed.add("tts")

    if "image_gen" in native:
        cfg = config.setdefault("image_gen", {})
        if isinstance(cfg, dict) and (str(cfg.get("provider") or "").strip().lower() in _OVERRIDABLE_IMAGE):
            cfg["provider"] = provider
            changed.add("image_gen")

    if "vision" in native and changed:
        aux = config.get("auxiliary") if isinstance(config.get("auxiliary"), dict) else {}
        v = aux.get("vision") if isinstance(aux.get("vision"), dict) else {}
        if str(v.get("provider") or "").strip().lower() in _OVERRIDABLE_VISION:
            changed.add("vision")

    if changed:
        logger.info("Applied provider-native tool defaults for %r: %s",
                    provider, sorted(changed))
    return changed


_SUMMARY: Dict[str, Dict[str, str]] = {
    "minimax": {
        "tts":       "TTS → speech-2.6-hd (30+ voices)",
        "image_gen": "Image generation → image-01",
        "vision":    "Vision analysis → MiniMax-VL-01",
    },
}
_SUMMARY["minimax-cn"] = _SUMMARY["minimax"]


def describe_changes(changed: Iterable[str], config: Dict[str, Any]) -> str:
    """Bullet-list summary used by the setup wizard."""
    items = sorted(changed)
    if not items:
        return "No changes — existing tool choices were preserved."
    phrasing = _SUMMARY.get(_active_provider_id(config), {})
    return "\n".join(
        f"  • {phrasing.get(k, k)}" for k in items
    )


# ─── Runtime dispatchers (consumed by tool files) ────────────────────────


def generate_image(
    *,
    prompt: str,
    aspect_ratio: str,
    num_images: int,
    output_format: str,
    config: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    """Generate via the active provider's native image backend, or `None`.

    Tool files call this before falling through to their existing path.
    The return shape matches `image_generate_tool` so callers can return
    the result verbatim.
    """
    cfg = config if config is not None else _safe_load_config()
    if _active_provider_id(cfg) in MINIMAX_PROVIDERS:
        return _minimax_image_request(
            prompt=prompt, aspect_ratio=aspect_ratio,
            num_images=num_images, output_format=output_format,
            config=cfg,
        )
    return None


def analyze_image(
    image_source: str,
    user_prompt: str,
    *,
    config: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    """Describe via the active provider's native VLM backend, or `None`."""
    cfg = config if config is not None else _safe_load_config()
    if _active_provider_id(cfg) in MINIMAX_PROVIDERS:
        return _minimax_vlm_request(image_source, user_prompt, cfg)
    return None


def native_credential_present(
    tool: str,
    config: Optional[Dict[str, Any]] = None,
) -> bool:
    """True when the active provider serves `tool` natively *and* its
    credential is configured.  Used by tool-level `check_fn` gates."""
    cfg = config if config is not None else _safe_load_config()
    if not provider_has_native_tool(tool, cfg):
        return False
    if _active_provider_id(cfg) in MINIMAX_PROVIDERS:
        return bool(_minimax_credential(cfg))
    return False


def minimax_endpoint_and_key(
    subpath: str,
    config: Optional[Dict[str, Any]] = None,
) -> Tuple[str, str]:
    """Return ``(url, api_key)`` for a MiniMax subpath (e.g. ``/v1/t2a_v2``)
    derived from the active provider's ``model.base_url`` + the
    region-appropriate credential.  Both values are empty strings when
    the active provider isn't MiniMax, so callers can probe with a
    single call and fall through.
    """
    cfg = config if config is not None else _safe_load_config()
    if _active_provider_id(cfg) not in MINIMAX_PROVIDERS:
        return "", ""
    root = active_provider_api_root(cfg).rstrip("/")
    key = _minimax_credential(cfg)
    if not root or not key:
        return "", ""
    return f"{root}{subpath}", key


# ─── Internal: shared helpers ────────────────────────────────────────────


def _safe_load_config() -> Dict[str, Any]:
    try:
        from hermes_cli.config import load_config
        cfg = load_config()
        return cfg if isinstance(cfg, dict) else {}
    except Exception as exc:
        logger.debug("provider_native_tools: config load failed: %s", exc)
        return {}


def _post_json(url: str, payload: Dict[str, Any], key: str, *, timeout: int = 120) -> Dict[str, Any]:
    """Bearer-auth POST → parsed JSON.  Raises urllib / json errors."""
    req = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        method="POST",
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    ctx = ssl.create_default_context()
    with urllib.request.urlopen(req, timeout=timeout, context=ctx) as resp:
        return json.loads(resp.read().decode("utf-8", errors="replace"))


def _download(url: str, output_path: Path, *, timeout: int = 120) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ctx = ssl.create_default_context()
    with urllib.request.urlopen(url, timeout=timeout, context=ctx) as resp:
        with open(output_path, "wb") as fh:
            while True:
                chunk = resp.read(65536)
                if not chunk:
                    break
                fh.write(chunk)


# ─── MiniMax bindings ────────────────────────────────────────────────────
#
# Per-provider request helpers.  Mirror the per-provider TTS helpers in
# `tools/tts_tool.py` (`_generate_minimax_tts` etc.).  Adding another
# provider is parallel: drop helpers below and route to them above.


def _minimax_credential(config: Optional[Dict[str, Any]] = None) -> str:
    """Pick the key matching the active provider's region.

    `minimax-cn` prefers `MINIMAX_CN_API_KEY`; `minimax` (international)
    prefers `MINIMAX_API_KEY`.  Falls back to the other on absence so a
    user with only one key still works for both regions (at their own
    entitlement risk).
    """
    cfg = config if config is not None else {}
    is_cn = _active_provider_id(cfg) == "minimax-cn"
    order = ("MINIMAX_CN_API_KEY", "MINIMAX_API_KEY") if is_cn \
            else ("MINIMAX_API_KEY", "MINIMAX_CN_API_KEY")
    for var in order:
        v = os.environ.get(var)
        if v and v.strip():
            return v.strip()
    return ""


def _minimax_image_cache_dir() -> Path:
    try:
        from hermes_constants import get_hermes_dir
        return Path(get_hermes_dir("cache/images", "image_cache"))
    except Exception:
        return Path.home() / ".hermes" / "image_cache"


_MINIMAX_RATIO_ALIAS = {"landscape": "16:9", "portrait": "9:16", "square": "1:1"}
_MINIMAX_VALID_RATIOS = {"1:1", "16:9", "9:16", "4:3", "3:4", "3:2", "2:3", "21:9"}


def _minimax_image_request(
    *, prompt: str, aspect_ratio: str, num_images: int, output_format: str,
    config: Dict[str, Any],
) -> str:
    api_root = active_provider_api_root(config).rstrip("/")
    if not api_root:
        return json.dumps({"success": False,
            "error": "image_gen.provider=minimax but model.base_url is unset"})
    key = _minimax_credential(config)
    if not key:
        return json.dumps({"success": False,
            "error": "MINIMAX_API_KEY (or _CN) required"})

    ratio = _MINIMAX_RATIO_ALIAS.get((aspect_ratio or "").lower(), aspect_ratio)
    if ratio not in _MINIMAX_VALID_RATIOS:
        ratio = "16:9"

    payload = {
        "model": "image-01",
        "prompt": (prompt or "").strip(),
        "aspect_ratio": ratio,
        "n": max(1, min(4, int(num_images or 1))),
        "response_format": "url",
    }
    try:
        body = _post_json(f"{api_root}/v1/image_generation", payload, key)
    except urllib.error.HTTPError as exc:
        return json.dumps({"success": False,
            "error": f"image API HTTP {exc.code}", "status": exc.code})
    except (urllib.error.URLError, json.JSONDecodeError) as exc:
        return json.dumps({"success": False, "error": str(exc)})

    base = body.get("base_resp") or {}
    if isinstance(base, dict) and base.get("status_code") not in (0, None):
        return json.dumps({"success": False,
            "error": f"image API error {base.get('status_code')}: "
                     f"{base.get('status_msg', '')}",
            "status": base.get("status_code")})

    data = body.get("data") or {}
    urls: list = []
    if isinstance(data, dict):
        for k in ("image_urls", "urls"):
            v = data.get(k)
            if isinstance(v, list):
                urls = [u for u in v if isinstance(u, str) and u.startswith("http")]
                break
    if not urls:
        return json.dumps({"success": False, "error": "image API returned no URL(s)"})

    out_dir = _minimax_image_cache_dir()
    ts = time.strftime("%Y%m%d-%H%M%S")
    ext = "jpeg" if (output_format or "").lower() in ("jpeg", "jpg") else "png"
    saved: list = []
    for i, url in enumerate(urls):
        path = out_dir / (f"image_{ts}.{ext}" if i == 0 else f"image_{ts}_{i + 1}.{ext}")
        try:
            _download(url, path)
            saved.append(str(path))
        except Exception as exc:
            logger.warning("image download failed for %s: %s", url, exc)
    if not saved:
        return json.dumps({"success": False, "error": "all image downloads failed"})
    return json.dumps({
        "success": True,
        "provider": _active_provider_id(config),
        "model": "image-01",
        "aspect_ratio": ratio,
        "paths": saved,
        "urls": urls,
    }, ensure_ascii=False)


_VLM_SUPPORTED_MIME = {"image/jpeg", "image/png", "image/webp"}
_VLM_MAGIC = (
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"\xff\xd8\xff",      "image/jpeg"),
)


def _coerce_vlm_image_url(image_source: str) -> Optional[str]:
    """Normalise into a value the VLM endpoint accepts (data URI / URL),
    or `None` for unsupported / missing files."""
    src = (image_source or "").strip()
    if not src:
        return None
    if src.startswith(("data:", "http://", "https://")):
        return src
    if src.startswith("file://"):
        src = src[len("file://"):]
    path = Path(src).expanduser()
    if not path.is_file():
        return None
    mime, _ = mimetypes.guess_type(str(path))
    if not mime:
        head = path.read_bytes()[:12]
        mime = next((m for magic, m in _VLM_MAGIC if head.startswith(magic)), None)
        if mime is None and head.startswith(b"RIFF") and head[8:12] == b"WEBP":
            mime = "image/webp"
    if mime not in _VLM_SUPPORTED_MIME:
        return None
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode()}"


def _minimax_vlm_request(image_source: str, user_prompt: str,
                         config: Dict[str, Any]) -> Optional[str]:
    api_root = active_provider_api_root(config).rstrip("/")
    if not api_root:
        return None
    key = _minimax_credential(config)
    if not key:
        return None
    image_url = _coerce_vlm_image_url(image_source)
    if not image_url:
        return None

    try:
        body = _post_json(
            f"{api_root}/v1/coding_plan/vlm",
            {"prompt": user_prompt, "image_url": image_url},
            key,
            timeout=60,
        )
    except (urllib.error.HTTPError, urllib.error.URLError, json.JSONDecodeError) as exc:
        logger.debug("VLM call failed: %s", exc)
        return None

    base = body.get("base_resp") or {}
    if isinstance(base, dict) and base.get("status_code") not in (0, None):
        # Surface provider-side errors verbatim (e.g. 1008 insufficient
        # balance) instead of masking them as fallback triggers.
        return json.dumps({
            "success": False,
            "error": f"VLM error {base.get('status_code')}: {base.get('status_msg', '')}",
            "status": base.get("status_code"),
        }, ensure_ascii=False)

    content = body.get("content")
    if not isinstance(content, str) or not content.strip():
        return None
    return json.dumps({
        "success": True,
        "analysis": content,
        "provider": _active_provider_id(config),
        "model": "MiniMax-VL-01",
        "endpoint": "/v1/coding_plan/vlm",
    }, ensure_ascii=False)
