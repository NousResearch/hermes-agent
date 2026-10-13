"""Routing helpers for inbound user-attached voice / audio clips.

``native`` attaches clips to the user turn as OpenAI-style ``input_audio`` parts —
``{"type": "input_audio", "input_audio": {"data": <b64>, "format": "wav"|"mp3"}}`` —
so the main model hears the clip itself instead of a lossy STT transcript (or, with
``stt.enabled: false``, a path pointer it cannot open). The gateway's path-pointing
text note stays in the turn: it is the degrade path whenever the wire cannot take
``input_audio`` (backend gate in :func:`strip_unsupported_audio_parts`) and whenever
a clip is too large, too long, or untranscodable — no turn ever depends on the part
reaching the provider.

Config ``media.native_audio`` (see ``hermes_cli.config_defaults.DEFAULT_CONFIG``):

* ``auto`` (default) — attach on an OpenAI-compatible chat-completions backend only;
  Anthropic / Gemini / Codex / Bedrock wires and an unknown ``api_mode`` degrade to
  the text note, so a non-OpenAI backend never sees a part it would 400 on.
* ``on``  — always attach (explicit override). A backend that already rejected audio
  this session still gets the text form: an explicit override must not wedge the turn
  in a retry loop (``agent._audio_rejecting_models``).
* ``off`` — never attach; voice rides the pre-existing text note only.

Clips follow the image path end to end: buffered per session by the gateway,
consumed once at ``run_conversation``, dropped from the wire on unsupported backends,
and replaced by a text placeholder in historical media stripping after compression.
"""

from __future__ import annotations

import base64
import logging
import shutil
import subprocess
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


#: Content-part types that carry audio (OpenAI chat ``input_audio`` and the generic
#: ``audio`` spelling some adapters emit).
AUDIO_PART_TYPES = frozenset({"input_audio", "audio"})

_VALID_MODES = frozenset({"auto", "on", "off"})
#: Accepted spellings for ``media.native_audio`` — boolean/YAML tokens and the
#: image-routing vocabulary (``native``/``text``) map onto on/off.
_MODE_ALIASES = {
    "auto": "auto",
    "on": "on", "native": "on", "true": "on", "yes": "on", "1": "on",
    "off": "off", "text": "off", "false": "off", "no": "off", "0": "off",
}

#: Wire shapes that have an OpenAI ``input_audio`` part. Every other api_mode
#: (``anthropic_messages``, ``codex_responses``, ``bedrock_converse``,
#: ``codex_app_server``, a plugin's own dialect) has no such part and rejects it.
_NATIVE_AUDIO_API_MODES = frozenset({"chat_completions"})

#: OpenAI-compatible chat-completions routes whose upstream still does not take
#: audio input (Gemini's/Anthropic's OpenAI shims, Bedrock proxies). Provider slugs
#: only — the first-party Anthropic/Codex wires are already excluded by api_mode.
_AUTO_EXCLUDED_PROVIDERS = frozenset({
    "anthropic", "gemini", "google", "bedrock", "vertex", "vertexai", "openai-codex",
})

# Hard ceilings for one native clip: 8 MB of base64 is ~2M chars of payload (the
# size class where providers start 413), and 10 minutes of audio dwarfs a turn.
# Past either, the text note alone carries the message.
MAX_NATIVE_AUDIO_BYTES = 8 * 1024 * 1024
MAX_NATIVE_AUDIO_SECONDS = 600.0

# Formats the ``input_audio`` part accepts verbatim; anything else is transcoded
# with ffmpeg before encoding (chat platforms deliver voice as ogg/opus/silk/amr).
_NATIVE_AUDIO_FORMATS = frozenset({"wav", "mp3"})
_NORMALIZE_FORMAT = "mp3"
_FFMPEG_TIMEOUT_S = 60
_FFPROBE_TIMEOUT_S = 10


# ─── config / backend gate ─────────────────────────────────────────────────

def _coerce_audio_mode(raw: Any) -> str:
    """Normalize a ``media.native_audio`` value into ``auto`` | ``on`` | ``off``."""
    if isinstance(raw, str):
        return _MODE_ALIASES.get(raw.strip().lower(), "auto")
    if isinstance(raw, bool):
        return "on" if raw else "off"
    return "auto"


def audio_input_mode(cfg: Any) -> str:
    """``media.native_audio`` from a config mapping (``auto`` when absent/malformed)."""
    media = cfg.get("media") if isinstance(cfg, dict) else None
    return _coerce_audio_mode(media.get("native_audio") if isinstance(media, dict) else None)


def native_audio_supported(
    cfg: Any = None, *, api_mode: str = "", provider: str = ""
) -> bool:
    """True when ``input_audio`` parts may ride this backend's requests.

    ``media.native_audio`` decides first (``off`` never, ``on`` always), then ``auto``
    narrows to the OpenAI-compatible chat-completions wire — an unknown ``api_mode``
    counts as unsupported, because guessing wrong is a 4xx on every turn.
    """
    mode = audio_input_mode(cfg)
    if mode == "off":
        return False
    if mode == "on":
        return True
    if str(api_mode or "").strip().lower() not in _NATIVE_AUDIO_API_MODES:
        return False
    return str(provider or "").strip().lower() not in _AUTO_EXCLUDED_PROVIDERS


def _load_cfg_readonly() -> Any:
    """Cached read-only config; ``{}`` when the config layer is unavailable."""
    try:
        from hermes_cli.config import load_config_readonly

        return load_config_readonly()
    except Exception:  # the gate must never fail a request
        return {}


# ─── clip preparation ──────────────────────────────────────────────────────

def _read_head(path: Path, size: int = 16) -> bytes:
    try:
        with open(path, "rb") as fh:
            return fh.read(size)
    except OSError:
        return b""


def sniff_audio_format(path: Any) -> Optional[str]:
    """``"wav"`` / ``"mp3"`` from magic bytes, then the extension; None when neither.

    Extensions are only a fallback: platforms serve voice under whatever suffix they
    like, and a wrong ``format`` field is a provider 400.
    """
    p = Path(path)
    head = _read_head(p)
    if head[:4] == b"RIFF" and head[8:12] == b"WAVE":
        return "wav"
    if head[:3] == b"ID3" or (len(head) > 1 and head[0] == 0xFF and (head[1] & 0xE0) == 0xE0):
        return "mp3"
    suffix = p.suffix.lower()
    if suffix in (".wav", ".mp3"):
        return suffix.lstrip(".")
    return None


def probe_duration_seconds(path: Any) -> Optional[float]:
    """Clip length in seconds; None when it cannot be probed (never raises).

    ``.wav`` is read with the stdlib ``wave`` module (no subprocess); everything
    else goes through ``ffprobe`` when it is installed.
    """
    p = Path(path)
    if p.suffix.lower() == ".wav":
        try:
            import wave

            with wave.open(str(p), "rb") as wf:
                rate = wf.getframerate() or 1
                return wf.getnframes() / float(rate)
        except Exception:  # malformed wav falls through to ffprobe
            pass
    if shutil.which("ffprobe") is None:
        return None
    try:
        proc = subprocess.run(
            [
                "ffprobe", "-v", "error", "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1", str(p),
            ],
            capture_output=True, text=True, timeout=_FFPROBE_TIMEOUT_S, check=False,
        )
        return float(proc.stdout.strip()) if proc.returncode == 0 else None
    except Exception:  # a probe failure only skips the duration gate
        logger.debug("audio_routing: ffprobe duration failed for %s", p, exc_info=True)
        return None


def transcode_with_ffmpeg(path: Any) -> Optional[bytes]:
    """Re-encode a non-wav/mp3 clip to mp3 on stdout; None when ffmpeg cannot.

    Mono 16 kHz at 48 kbps: speech is the payload, and a 10-minute clip then lands
    under the byte ceiling instead of 19 MB as PCM wav.
    """
    if shutil.which("ffmpeg") is None:
        logger.info(
            "audio_routing: %s is not wav/mp3 and ffmpeg is not installed; "
            "attaching the text note only.", path,
        )
        return None
    try:
        proc = subprocess.run(
            [
                "ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(path),
                "-vn", "-ac", "1", "-ar", "16000", "-b:a", "48k",
                "-f", _NORMALIZE_FORMAT, "-",
            ],
            capture_output=True, timeout=_FFMPEG_TIMEOUT_S, check=False,
        )
    except Exception:  # degrade to the text note, never fail the turn
        logger.warning("audio_routing: ffmpeg transcode failed for %s", path, exc_info=True)
        return None
    if proc.returncode != 0 or not proc.stdout:
        logger.warning(
            "audio_routing: ffmpeg could not transcode %s: %s",
            path, (proc.stderr or b"")[:200].decode("utf-8", "replace"),
        )
        return None
    return proc.stdout


def _prepare_clip(path: Any) -> Optional[tuple[bytes, str]]:
    """``(raw_bytes, format)`` for one clip, or None when it must not be attached.

    Order matters: cheap stat/size/duration gates first so an oversized clip never
    pays for a read (or a transcode).
    """
    p = Path(path)
    try:
        from agent.file_safety import raise_if_read_blocked

        raise_if_read_blocked(str(p))
    except ValueError as exc:
        logger.warning("audio_routing: blocked local audio attachment %s -- %s", p, exc)
        return None
    except Exception:  # attachment stays best-effort without the guard
        pass
    try:
        if not p.is_file():
            return None
        size = p.stat().st_size
    except OSError:
        return None
    if size <= 0 or size > MAX_NATIVE_AUDIO_BYTES:
        if size > MAX_NATIVE_AUDIO_BYTES:
            logger.info("audio_routing: %s is %.1f MB, over the %.0f MB native limit; text only",
                        p, size / (1024 * 1024), MAX_NATIVE_AUDIO_BYTES / (1024 * 1024))
        return None
    duration = probe_duration_seconds(p)
    if duration is not None and duration > MAX_NATIVE_AUDIO_SECONDS:
        logger.info("audio_routing: %s is %.0fs, over the %.0fs native limit; text only",
                    p, duration, MAX_NATIVE_AUDIO_SECONDS)
        return None

    fmt = sniff_audio_format(p)
    if fmt in _NATIVE_AUDIO_FORMATS:
        try:
            data = p.read_bytes()
        except OSError:
            return None
    else:
        data = transcode_with_ffmpeg(p)
        fmt = _NORMALIZE_FORMAT
        if data is None:
            return None
    if not data or len(data) > MAX_NATIVE_AUDIO_BYTES:
        return None
    return data, fmt


def build_native_audio_parts(
    user_text: str, audio_paths: list[str]
) -> tuple[list[dict[str, Any]], list[str]]:
    """Build an OpenAI-style ``content`` list for a user turn carrying voice clips.

    Mirrors ``agent.image_routing.build_native_content_parts``: one text part holds
    the caption plus a ``[Voice message attached as audio: <path>]`` handle per clip,
    followed by one ``input_audio`` part per clip. Returns ``(content_parts,
    skipped)`` where ``skipped`` holds the paths that could not be attached (missing,
    over the size/duration ceiling, untranscodable) — their path-pointing note in the
    caller's text is the model's only handle, exactly as before native routing.
    """
    clipped: list[tuple[bytes, str, str]] = []  # (raw, format, path)
    skipped: list[str] = []
    for raw_path in dict.fromkeys(audio_paths or []):
        prepared = _prepare_clip(raw_path)
        if prepared is None:
            skipped.append(str(raw_path))
        else:
            clipped.append((prepared[0], prepared[1], str(raw_path)))

    text = (user_text or "").strip()
    if not clipped:
        return ([{"type": "text", "text": text}] if text else []), skipped

    hints = "\n".join(f"[Voice message attached as audio: {path}]" for _, _, path in clipped)
    combined = f"{text}\n\n{hints}" if text else hints
    parts: list[dict[str, Any]] = [{"type": "text", "text": combined}]
    parts.extend(
        {"type": "input_audio", "input_audio": {"data": base64.b64encode(raw).decode("ascii"), "format": fmt}}
        for raw, fmt, _ in clipped
    )
    return parts, skipped


# ─── send path ─────────────────────────────────────────────────────────────

def is_audio_part(part: Any) -> bool:
    """True when ``part`` is an ``input_audio`` / ``audio`` content part."""
    return isinstance(part, dict) and part.get("type") in AUDIO_PART_TYPES


def _has_text_part(parts: list[dict[str, Any]]) -> bool:
    return any(
        isinstance(p, dict) and p.get("type") in ("text", "input_text") and str(p.get("text") or "").strip()
        for p in parts
    )


def strip_unsupported_audio_parts(agent: Any, api_messages: Any) -> int:
    """Drop ``input_audio`` parts the current backend cannot accept; returns rows rewritten.

    Runs per attempt in ``turn_api_request.build_api_request`` — the api_mode is that
    of the wire actually being used, so a fallback to Anthropic/Codex mid-turn strips
    the parts the primary accepted. Rewrites the per-call copy only; history keeps the
    parts so a later switch to an audio-capable backend hears the clip again. A row
    left without any text gets a one-line placeholder (an audio-only user row would
    otherwise be dropped as empty by the empty-turn repair).
    """
    if not isinstance(api_messages, list) or not api_messages:
        return 0
    try:
        cfg = _load_cfg_readonly()
    except Exception:  # a config hiccup must not change wire behavior
        cfg = {}
    api_mode = str(getattr(agent, "api_mode", "") or "")
    provider = str(getattr(agent, "provider", "") or "")
    # A model that already rejected audio this session keeps the text form even under
    # ``media.native_audio: on`` — the override is a routing preference, not a licence
    # to loop on a known 4xx (same per-(provider, model) keying as images).
    try:
        from agent.vision_message_prep import _provider_model_key

        rejecting = _provider_model_key(agent) in (getattr(agent, "_audio_rejecting_models", None) or set())
    except Exception:  # without the key the config gate still decides
        rejecting = bool(getattr(agent, "_audio_rejecting_models", None))
    if native_audio_supported(cfg, api_mode=api_mode, provider=provider) and not rejecting:
        return 0

    rewritten = 0
    for msg in api_messages:
        if not isinstance(msg, dict):
            continue
        content = msg.get("content")
        if not isinstance(content, list) or not any(is_audio_part(p) for p in content):
            continue
        remaining = [p for p in content if not is_audio_part(p)]
        if not remaining:
            remaining = [
                {"type": "text", "text": "[audio attachment omitted: this backend does not accept audio input]"}
            ]
        elif not _has_text_part(remaining):
            remaining.insert(0, {
                "type": "text",
                "text": "[audio attachment omitted: this backend does not accept audio input]",
            })
        msg["content"] = remaining
        rewritten += 1
    if rewritten:
        logger.debug(
            "audio_routing: stripped input_audio from %d message(s) for %s/%s",
            rewritten, provider or "?", api_mode or "?",
        )
    return rewritten


__all__ = [
    "AUDIO_PART_TYPES",
    "MAX_NATIVE_AUDIO_BYTES",
    "MAX_NATIVE_AUDIO_SECONDS",
    "audio_input_mode",
    "build_native_audio_parts",
    "is_audio_part",
    "native_audio_supported",
    "probe_duration_seconds",
    "sniff_audio_format",
    "strip_unsupported_audio_parts",
    "transcode_with_ffmpeg",
]
