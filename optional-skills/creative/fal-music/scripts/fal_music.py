#!/usr/bin/env python3
"""Generate music or sound effects through fal.ai's queue API.

Usage:
    python fal_music.py --list
    python fal_music.py --model elevenlabs-music --prompt "warm lofi hip hop, vinyl crackle" --duration 30 -o /tmp/track.mp3
    python fal_music.py --model minimax-music-3 --prompt "upbeat synthpop, female vocals" \
        --lyrics "[verse]\\nNeon lights..." --duration 90 -o /tmp/song.wav
    python fal_music.py --model elevenlabs-sfx --prompt "heavy wooden door creaks open" --duration 4 -o /tmp/door.mp3
    python fal_music.py --model lyria3 --prompt "cinematic orchestral swell" --dry-run   # print payload, no call

Requires ``FAL_KEY`` (https://fal.ai/dashboard/keys) and the ``fal-client`` package
(``pip install fal-client==0.13.1``). Each model's payload builder only emits keys its
fal OpenAPI schema declares; unsupported knobs are dropped, never forwarded.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request
from typing import Any, Callable, Dict, Optional

# ── Model catalog ────────────────────────────────────────────────────────────
# ``build`` maps the generic CLI knobs onto the endpoint's declared input schema.
# ``duration`` is (min, max) seconds or None when the endpoint has no length knob.


def _elevenlabs_music(a: argparse.Namespace) -> Dict[str, Any]:
    payload: Dict[str, Any] = {"prompt": a.prompt, "output_format": "mp3_44100_128"}
    if a.duration is not None:
        payload["music_length_ms"] = int(a.duration * 1000)
    if a.instrumental:
        payload["force_instrumental"] = True
    return payload


def _minimax_music_3(a: argparse.Namespace) -> Dict[str, Any]:
    # ``lyrics`` is required; an instrumental track is expressed as an [inst] body.
    payload: Dict[str, Any] = {"prompt": a.prompt, "lyrics": a.lyrics or "[inst]"}
    if a.duration is not None:
        payload["duration"] = a.duration
    if a.seed is not None:
        payload["seed"] = a.seed
    return payload


def _minimax_music_26(a: argparse.Namespace) -> Dict[str, Any]:
    payload: Dict[str, Any] = {"prompt": a.prompt}
    if a.lyrics:
        payload["lyrics"] = a.lyrics
    if a.instrumental:
        payload["is_instrumental"] = True
    return payload


def _lyria3(a: argparse.Namespace) -> Dict[str, Any]:
    return {"prompt": a.prompt}  # ~30s fixed-length clips; no duration/seed/negative knobs


def _stable_audio_3(a: argparse.Namespace) -> Dict[str, Any]:
    payload: Dict[str, Any] = {"prompt": a.prompt, "output_format": "mp3"}
    if a.duration is not None:
        payload["duration"] = a.duration
    if a.seed is not None:
        payload["seed"] = a.seed
    if a.negative_prompt:
        payload["negative_prompt"] = a.negative_prompt
    return payload


def _ace_step(a: argparse.Namespace) -> Dict[str, Any]:
    # ``tags`` carries the style prompt; lyrics default to instrumental.
    payload: Dict[str, Any] = {"tags": a.prompt, "lyrics": a.lyrics or "[inst]"}
    if a.duration is not None:
        payload["duration"] = a.duration
    if a.seed is not None:
        payload["seed"] = a.seed
    return payload


def _elevenlabs_sfx(a: argparse.Namespace) -> Dict[str, Any]:
    payload: Dict[str, Any] = {"text": a.prompt}
    if a.duration is not None:
        payload["duration_seconds"] = a.duration
    if a.loop:
        payload["loop"] = True
    return payload


MODELS: Dict[str, Dict[str, Any]] = {
    "elevenlabs-music": {
        "endpoint": "elevenlabs/music/v2.5", "build": _elevenlabs_music, "duration": (10, 300), "ext": "mp3",
        "kind": "music (vocals or instrumental)", "lyrics": False,
        "notes": "Best overall quality; $0.60 per output minute (rounded up). Prompt-driven; use --instrumental to forbid vocals.",
    },
    "minimax-music-3": {
        "endpoint": "minimax/music-3", "build": _minimax_music_3, "duration": (1, 300), "ext": "wav",
        "kind": "full songs from lyrics + style", "lyrics": True,
        "notes": "Up to 5 min, 44.1 kHz WAV, seedable. Lyrics support [intro]/[verse]/[chorus] tags; pass none for instrumental.",
    },
    "minimax-music-2.6": {
        "endpoint": "fal-ai/minimax-music/v2.6", "build": _minimax_music_26, "duration": None, "ext": "mp3",
        "kind": "songs from lyrics + style", "lyrics": True,
        "notes": "Previous MiniMax generation; no duration knob, --instrumental supported.",
    },
    "lyria3": {
        "endpoint": "fal-ai/lyria3", "build": _lyria3, "duration": None, "ext": "mp3",
        "kind": "~30s music clips", "lyrics": False,
        "notes": "Google Lyria 3. Prompt only (no duration, seed or negative prompt). Also returns generated lyrics when applicable.",
    },
    "lyria3-pro": {
        "endpoint": "fal-ai/lyria3/pro", "build": _lyria3, "duration": None, "ext": "mp3",
        "kind": "~30s music clips, higher fidelity", "lyrics": False,
        "notes": "Google Lyria 3 Pro. Same prompt-only surface as lyria3.",
    },
    "stable-audio-3": {
        "endpoint": "fal-ai/stable-audio-3/medium/text-to-audio", "build": _stable_audio_3, "duration": (1, 380), "ext": "mp3",
        "kind": "instrumental music, long form", "lyrics": False,
        "notes": "Stability AI, 1.4B, licensed training data, up to ~6 min. Supports --negative-prompt and --seed.",
    },
    "ace-step": {
        "endpoint": "fal-ai/ace-step", "build": _ace_step, "duration": (5, 240), "ext": "wav",
        "kind": "cheap songs from tags + lyrics", "lyrics": True,
        "notes": "Open-source ACE-Step; $0.0002 per second (~83 min per $1). Prompt is a comma-separated tag list.",
    },
    "elevenlabs-sfx": {
        "endpoint": "fal-ai/elevenlabs/sound-effects/v2", "build": _elevenlabs_sfx, "duration": (0.5, 22), "ext": "mp3",
        "kind": "sound effects", "lyrics": False,
        "notes": "0.5-22 s effects; --loop for seamless loops. Duration inferred from the prompt when omitted.",
    },
}

DEFAULT_MODEL = "elevenlabs-music"


def clamp_duration(model: str, duration: Optional[float]) -> Optional[float]:
    """Clamp into the model's declared range; None when the model has no length knob."""
    rng = MODELS[model]["duration"]
    if duration is None or rng is None:
        return None
    lo, hi = rng
    return max(lo, min(hi, duration))


def build_payload(model: str, args: argparse.Namespace) -> Dict[str, Any]:
    meta = MODELS[model]
    requested = args.duration
    args.duration = clamp_duration(model, args.duration)
    payload = meta["build"](args)
    # Knobs the endpoint never declared are dropped, not forwarded; say so once so a user notices.
    ignored = [flag for flag, value, present in (
        ("duration", requested, requested is not None and not any(k in payload for k in ("duration", "duration_seconds", "music_length_ms"))),
        ("seed", args.seed, args.seed is not None and "seed" not in payload),
        ("negative-prompt", args.negative_prompt, bool(args.negative_prompt) and "negative_prompt" not in payload),
        ("instrumental", args.instrumental, args.instrumental and not any(k in payload for k in ("force_instrumental", "is_instrumental"))),
        ("loop", args.loop, args.loop and "loop" not in payload),
    ) if present]
    if ignored:
        print(f"note: {model} has no knob for --{', --'.join(ignored)}; ignored", file=sys.stderr)
    return payload


def _audio_url(result: Dict[str, Any]) -> str:
    audio = result.get("audio") or result.get("audio_file") or {}
    if isinstance(audio, dict) and audio.get("url"):
        return audio["url"]
    if isinstance(audio, str):
        return audio
    raise SystemExit(f"No audio URL in result: {json.dumps(result)[:400]}")


def generate(model: str, payload: Dict[str, Any], output: str) -> Dict[str, Any]:
    try:
        import fal_client
    except ImportError:
        raise SystemExit("fal-client not installed: pip install fal-client==0.13.1")
    if not os.environ.get("FAL_KEY"):
        raise SystemExit("FAL_KEY is not set (https://fal.ai/dashboard/keys)")
    try:
        result = fal_client.subscribe(MODELS[model]["endpoint"], arguments=payload, with_logs=False)
    except Exception as exc:  # fal_client raises its own HTTP error types; surface the message, not a traceback
        raise SystemExit(f"fal request failed ({MODELS[model]['endpoint']}): {exc}") from exc
    url = _audio_url(result)
    with urllib.request.urlopen(url, timeout=120) as resp, open(output, "wb") as fh:
        fh.write(resp.read())
    extra = {k: v for k, v in result.items() if k not in ("audio", "audio_file")}
    return {"output": output, "url": url, "model": model, "endpoint": MODELS[model]["endpoint"], **extra}


def main(argv: Optional[list] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--list", action="store_true", help="list models and exit")
    p.add_argument("--model", default=DEFAULT_MODEL, choices=sorted(MODELS), help=f"model id (default {DEFAULT_MODEL})")
    p.add_argument("--prompt", help="style / content description (tags for ace-step, effect text for elevenlabs-sfx)")
    p.add_argument("--lyrics", help="lyrics with [verse]/[chorus] structure tags (models with lyrics support)")
    p.add_argument("--duration", type=float, help="target length in seconds (clamped to the model's range)")
    p.add_argument("--seed", type=int, help="seed (models that declare one)")
    p.add_argument("--negative-prompt", help="qualities to avoid (stable-audio-3 only)")
    p.add_argument("--instrumental", action="store_true", help="no vocals (elevenlabs-music, minimax-music-2.6)")
    p.add_argument("--loop", action="store_true", help="seamless loop (elevenlabs-sfx only)")
    p.add_argument("--dry-run", action="store_true", help="print the endpoint + payload and exit without calling fal")
    p.add_argument("-o", "--output", help="output file path (default /tmp/fal-music-<model>.<ext>)")
    a = p.parse_args(argv)

    if a.list:
        for mid, m in MODELS.items():
            rng = m["duration"]
            span = f"{rng[0]}-{rng[1]}s" if rng else "fixed"
            print(f"{mid:20} {m['endpoint']:46} {span:10} lyrics={'y' if m['lyrics'] else 'n'}  {m['kind']}")
            print(f"{'':20} {m['notes']}")
        return 0
    if not a.prompt:
        p.error("--prompt is required")
    if a.lyrics and not MODELS[a.model]["lyrics"]:
        print(f"note: {a.model} ignores --lyrics (prompt-only model)", file=sys.stderr)

    payload = build_payload(a.model, a)
    if a.dry_run:
        print(json.dumps({"endpoint": MODELS[a.model]["endpoint"], "payload": payload}, indent=2))
        return 0
    output = a.output or f"/tmp/fal-music-{a.model}.{MODELS[a.model]['ext']}"
    print(json.dumps(generate(a.model, payload, output), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
