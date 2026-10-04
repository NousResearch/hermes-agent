"""MiniMax Music 3.0 backend for Hermes.

In-tree music counterpart of ``plugins/image_gen/krea`` and
``plugins/video_gen/fal``. Registers a ``generate_music`` tool (toolset
``music_gen``) that turns a style prompt + optional lyrics into a full
produced song via MiniMax's hosted ``POST /v1/music_generation`` endpoint
(model ``music-3.0``).

Credential order: portal credits via the managed Nous gateway
(``minimax`` vendor route) once tool-gateway onboards MiniMax — verified
live by a reachability probe before steering calls at it; BYOK via
``MINIMAX_API_KEY`` until then — the same key the built-in MiniMax TTS
provider uses. The tool surfaces whenever either credential is resolvable.

Why the blocking shape: MiniMax has no public async job API for music — the
endpoint is synchronous and returns either hex-encoded audio bytes or a
24h-expiring URL in one JSON response. A full song take is slow (tens of
seconds to a few minutes), so the tool blocks with a generous timeout and
always materializes the result to disk before returning.

Cover generation (``model=music-cover``) and the lyrics helper
(``POST /v1/lyrics_generation``) are exposed through the same tool via
``mode=``; the two-step cover preprocess endpoint is intentionally left
out of the agent surface (it belongs in a studio UI flow, not a chat
tool).
"""

from __future__ import annotations

import base64
import json
import logging
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, Optional

from agent.secret_scope import get_secret
from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)

# Versionless base — the /v1 prefix lives in the request paths so the same
# path shape works against the managed gateway origin (which allowlists
# POST /v1/music_generation exactly, with no /v1 prefix on the origin).
API_BASE_DIRECT = "https://api.minimax.io"
DEFAULT_MODEL = "music-3.0"
# Music generation of a full multi-minute song is slow; the endpoint is
# synchronous, so we need a long socket timeout but still bounded.
REQUEST_TIMEOUT_S = 600

ENV_KEY = "MINIMAX_API_KEY"
# Managed-gateway vendor slug (mirrors resolve_managed_tool_gateway("fal-queue")
# / krea). The route + meter land in Nous infra; when they do, this plugin
# serves music on portal credits with zero keys, same as fal video today.
_GATEWAY_VENDOR = "minimax"


def _resolve_key() -> Optional[str]:
    return (get_secret(ENV_KEY) or "").strip() or None


def _resolve_gateway():
    """Nous managed gateway config for MiniMax, or None when the user isn't on
    the managed path (not signed in / managed tools off / no token). Kept as
    the preferred credential when a MINIMAX_API_KEY is present — portal-credit
    service is the default; the direct key is an explicit BYOK override."""
    try:  # helper may be unavailable in stripped-down embeddings
        from tools.managed_tool_gateway import resolve_managed_tool_gateway

        return resolve_managed_tool_gateway(_GATEWAY_VENDOR)
    except Exception:  # noqa: BLE001
        return None


def _music_cache_dir() -> Path:
    d = get_hermes_home() / "cache" / "music"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _slug(text: str, limit: int = 40) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", (text or "").lower()).strip("-")
    return (s[:limit].strip("-") or "song")


def _gateway_host_reachable(origin: str) -> bool:
    """The managed route exists iff the vendor gateway host resolves + answers.
    Until tool-gateway onboards MiniMax, the minimax vendor host is NXDOMAIN
    and we must not steer calls at it — that would break a working direct-key
    setup the moment the user signs into Nous. Cheap probe (DNS only); any
    resolvable host counts as reachable (401 = fronted, auth'd at app layer)."""
    import socket
    from urllib.parse import urlparse

    try:
        host = urlparse(origin).hostname or ""
        socket.getaddrinfo(host, 443)
    except OSError:
        return False
    return True


def _auth_token() -> Dict[str, Any]:
    """Resolve credential + base. Order:
      1. `MINIMAX_GATEWAY_URL` override (staging) → nous token (trusted as-is)
      2. managed gateway, ONLY if the vendor route resolves → portal token
      3. direct MINIMAX_API_KEY                    → vendor token
    Portal-credit service is the default ONCE THE ROUTE EXISTS; the direct key
    is the live fallback until then, and is never shadowed by a dead route."""
    override = os.environ.get("MINIMAX_GATEWAY_URL", "").strip().rstrip("/")
    gw = _resolve_gateway()
    if override and gw is not None:
        return {"base": override, "token": gw.nous_user_token, "managed": True}
    if gw is not None and gw.gateway_origin:
        origin = gw.gateway_origin.rstrip("/")
        if _gateway_host_reachable(origin):
            return {"base": origin, "token": gw.nous_user_token, "managed": True}
        # Route not onboarded yet — fall through to the direct key.
    key = _resolve_key()
    if key:
        return {"base": API_BASE_DIRECT, "token": key, "managed": False}
    return {"base": None, "token": None, "managed": False}


def _post(path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """POST JSON to MiniMax (direct or via the managed Nous gateway) and
    return the parsed body.

    Auth and request-shape failures come back as HTTP 200 with a non-zero
    ``base_resp.status_code`` — MiniMax does not use HTTP status for
    application errors. We therefore never raise on HTTP 200 bodies; we
    surface status_code/status_msg to the caller instead.
    """
    auth = _auth_token()
    if not auth["token"]:
        raise RuntimeError(
            f"No MiniMax credential. Set {ENV_KEY} in your Hermes .env for BYOK, "
            f"or sign in to Nous (`hermes auth add nous --type oauth`) to use the "
            f"managed gateway once the MiniMax route is live."
        )
    req = urllib.request.Request(
        f"{auth['base']}{path}",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {auth['token']}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT_S) as resp:
            body = resp.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "replace")[:500]
        raise RuntimeError(f"MiniMax HTTP {e.code}: {detail}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"MiniMax request failed: {e.reason}") from e
    try:
        return json.loads(body)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"MiniMax returned non-JSON body: {body[:300]}") from e


_STATUS_HINTS: Dict[int, str] = {
    1002: "Rate limited — slow down and retry.",
    1004: "Auth failed — check MINIMAX_API_KEY in your Hermes .env.",
    1008: "Insufficient balance — top up at platform.minimax.io.",
    1026: "Content rejected by MiniMax moderation — reword the prompt/lyrics.",
    2013: "Invalid parameters — check prompt/lyrics lengths (lyrics 1-3500 chars for music-3.0).",
    2049: "Invalid API key format — check MINIMAX_API_KEY.",
}


def _check_base_resp(result: Dict[str, Any]) -> Optional[str]:
    base = result.get("base_resp") or {}
    code = base.get("status_code", 0)
    if code in (0, None):
        return None
    msg = base.get("status_msg") or "unknown error"
    hint = _STATUS_HINTS.get(code, "")
    return f"MiniMax error {code}: {msg}" + (f" ({hint})" if hint else "")


def _save_audio(result: Dict[str, Any], prompt: str, fmt: str) -> Dict[str, Any]:
    """Materialize the audio from a successful response; return file info."""
    data = result.get("data") or {}
    extra = result.get("extra_info") or {}
    ts = time.strftime("%Y%m%d-%H%M%S")
    stem = f"{ts}-{_slug(prompt)}.{fmt}"

    # Preferred: server-hosted URL (output_format=url). Fallback: hex bytes.
    audio_url = data.get("audio") or ""
    out_path = _music_cache_dir() / stem
    if isinstance(audio_url, str) and audio_url.startswith("http"):
        # data.status 1 (in progress) means the URL pre-exists the file; poll
        # briefly before giving up. status 2 = completed.
        if data.get("status") == 1:
            deadline = time.time() + 60
            body = None
            while time.time() < deadline:
                try:
                    with urllib.request.urlopen(audio_url, timeout=120) as r:
                        body = r.read()
                    break
                except Exception:  # noqa: BLE001
                    time.sleep(3)
            if body is None:
                return {"error": "audio still rendering at result URL after 60s", "audio_url": audio_url}
            out_path.write_bytes(body)
        else:
            try:
                with urllib.request.urlopen(audio_url, timeout=120) as r:
                    out_path.write_bytes(r.read())
            except Exception as e:  # noqa: BLE001
                return {"error": f"generated OK but download from result URL failed: {e}", "audio_url": audio_url}
    elif isinstance(audio_url, str) and audio_url:
        # hex-encoded
        try:
            out_path.write_bytes(bytes.fromhex(audio_url))
        except ValueError:
            out_path.write_bytes(base64.b64decode(audio_url))
    else:
        return {"error": "response contained no audio payload", "raw_keys": list(data.keys())}

    return {
        "file": str(out_path),
        "audio_url": audio_url if audio_url.startswith("http") else None,
        "duration_s": round((extra.get("music_duration") or 0) / 1000.0, 2) or None,
        "sample_rate": extra.get("music_sample_rate"),
        "channels": extra.get("music_channel"),
        "bitrate": extra.get("bitrate"),
        "size_bytes": extra.get("music_size") or out_path.stat().st_size,
        "trace_id": result.get("trace_id"),
    }


def generate_music(
    prompt: str,
    lyrics: Optional[str] = None,
    mode: str = "song",
    reference_audio_url: Optional[str] = None,
    fmt: str = "mp3",
    sample_rate: int = 44100,
    bitrate: int = 256000,
    model: Optional[str] = None,
) -> Dict[str, Any]:
    """Generate music with MiniMax.

    mode:
      - "song"         (default): full song. lyrics optional (auto-written if absent).
      - "instrumental": no vocals; lyrics ignored.
      - "lyrics":      text-only — return generated lyrics without rendering audio.
      - "cover":       reinterpret reference_audio_url (6s-6min, <=50MB) in the style
                       of `prompt`. lyrics optional (ASR-extracted if absent).
    """
    prompt = (prompt or "").strip()
    if not prompt:
        return {"error": "prompt is required (describe genre, mood, instruments, vocals, mix)."}

    if mode == "lyrics":
        try:
            result = _post("/v1/lyrics_generation", {"mode": "write_full_song", "prompt": prompt})
            err = _check_base_resp(result)
            data = result.get("data") or {}
            return {"mode": "lyrics", "lyrics": data.get("lyrics") or "",
                    "error": err, "trace_id": result.get("trace_id"),
                    "raw_keys": list(data.keys())}
        except RuntimeError as e:
            return {"error": str(e)}

    resolved_model = model or DEFAULT_MODEL
    payload: Dict[str, Any] = {
        "model": resolved_model,
        "prompt": prompt,
        "audio_setting": {"sample_rate": sample_rate, "bitrate": bitrate, "format": fmt},
        "output_format": "url",
    }

    if mode == "instrumental":
        payload["is_instrumental"] = True
    elif mode == "cover":
        payload["model"] = "music-cover"
        if not reference_audio_url:
            return {"error": "mode='cover' requires reference_audio_url (http(s) URL, 6s-6min, <=50MB)."}
        payload["audio_url"] = reference_audio_url
        if lyrics:
            payload["lyrics"] = lyrics
    else:  # song
        if lyrics and lyrics.strip():
            payload["lyrics"] = lyrics.strip()
        else:
            # MiniMax 2013s on lyrics_optimizer:true sent with an explicit
            # empty lyrics string — the field must be ABSENT for auto-write.
            payload["lyrics_optimizer"] = True

    try:
        result = _post("/v1/music_generation", payload)
    except RuntimeError as e:
        return {"error": str(e), "mode": mode, "model": payload.get("model")}

    err = _check_base_resp(result)
    if err:
        return {"error": err, "mode": mode, "model": payload.get("model"),
                "trace_id": result.get("trace_id")}

    saved = _save_audio(result, prompt, fmt)
    saved.update({"mode": mode, "model": payload.get("model"), "prompt": prompt})
    return saved


# ---------------------------------------------------------------------------
# Tool surface
# ---------------------------------------------------------------------------


def _check_music_tools() -> bool:
    """Surface the tool only when a credential CAN resolve — keeps the core
    toolset narrow for everyone else. A signed-in Nous sub counts: the managed
    gateway path needs no MINIMAX_API_KEY."""
    return bool(_resolve_key() or _resolve_gateway())


_TOOL_SCHEMA: Dict[str, Any] = {
    "name": "generate_music",
    "description": (
        "Compose and produce a full song (or instrumental) with MiniMax Music 3.0. "
        "Give a style prompt (genre, mood, tempo, instruments, vocal character, mix) and "
        "optionally structured lyrics ([Verse]/[Chorus]/[Bridge] tags); the model writes "
        "lyrics itself when omitted. mode='lyrics' returns lyrics only, mode='cover' "
        "reinterprets a reference audio URL in a new style. Saves the audio file locally "
        "and returns its path plus duration/format metadata."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "prompt": {
                "type": "string",
                "description": "Style description: genre, mood, BPM, key, instruments, vocal character, mix. Be specific — this drives composition and production.",
            },
            "lyrics": {
                "type": "string",
                "description": "Optional lyrics with structure tags ([Verse] [Chorus] [Bridge] [Outro]). Omit to have lyrics written from the prompt (mode='song'). 1-3500 chars.",
            },
            "mode": {
                "type": "string",
                "enum": ["song", "instrumental", "lyrics", "cover"],
                "default": "song",
                "description": "song=full production, instrumental=no vocals, lyrics=text-only lyrics generation, cover=restyle reference_audio_url.",
            },
            "reference_audio_url": {
                "type": "string",
                "description": "Required for mode='cover': public URL of reference audio (6s-6min, <=50MB).",
            },
            "format": {
                "type": "string",
                "enum": ["mp3", "wav", "pcm"],
                "default": "mp3",
                "description": "Output container. mp3 default; wav for further production; pcm raw.",
            },
        },
        "required": ["prompt"],
    },
}


def _generate_music_handler(args: Dict[str, Any]) -> str:
    result = generate_music(
        prompt=args.get("prompt") or "",
        lyrics=args.get("lyrics"),
        mode=args.get("mode") or "song",
        reference_audio_url=args.get("reference_audio_url"),
        fmt=args.get("format") or "mp3",
    )
    if result.get("file"):
        # Agent-facing surface mirrors text_to_speech: MEDIA: path so the
        # desktop app / gateways can attach and play it natively.
        result["MEDIA"] = result["file"]
    return json.dumps(result, ensure_ascii=False, default=str)


def register(ctx) -> None:
    """Hermes plugin entry point."""
    ctx.register_tool(
        name="generate_music",
        toolset="music_gen",
        schema=_TOOL_SCHEMA,
        handler=_generate_music_handler,
        check_fn=_check_music_tools,
        requires_env=[ENV_KEY],
        description="Generate a full song or instrumental with MiniMax Music 3.0",
        emoji="🎵",
    )
