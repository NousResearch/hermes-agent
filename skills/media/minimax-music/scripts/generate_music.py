#!/usr/bin/env python3
"""Generate music with the MiniMax regional Music API."""

import argparse
import json
import os
from pathlib import Path
from urllib.request import Request, urlopen

ENDPOINTS = {
    "global": "https://api.minimax.io/v1/music_generation",
    "cn": "https://api.minimaxi.com/v1/music_generation",
}
MODELS = {
    "music-3.0", "music-2.6", "music-3.0-free", "music-2.6-free",
    "music-cover", "music-cover-free",
}


def generate(args, opener=urlopen):
    api_key = os.environ.get("MINIMAX_API_KEY")
    if not api_key:
        raise SystemExit("MINIMAX_API_KEY is required")
    if args.model not in MODELS:
        raise SystemExit(f"unsupported model: {args.model}")
    payload = {
        "model": args.model,
        "prompt": args.prompt,
        "stream": False,
        "output_format": args.output_format,
        "audio_setting": {"format": args.audio_format},
        "is_instrumental": args.instrumental,
    }
    for key in ("lyrics", "audio_url", "audio_base64", "cover_feature_id"):
        value = getattr(args, key, None)
        if value is not None:
            payload[key] = value
    if args.region == "cn" and args.aigc_watermark:
        payload["aigc_watermark"] = True
    request = Request(
        ENDPOINTS[args.region],
        data=json.dumps(payload).encode(),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    with opener(request, timeout=300) as response:
        result = json.load(response)
    if result.get("base_resp", {}).get("status_code") != 0:
        raise RuntimeError("MiniMax music generation failed")
    data = result.get("data", {})
    if data.get("status") != 2 or not data.get("audio"):
        raise RuntimeError("MiniMax music generation did not complete")
    output = Path(args.output)
    if args.output_format == "hex":
        output.write_bytes(bytes.fromhex(data["audio"]))
    else:
        with opener(data["audio"], timeout=60) as response:
            output.write_bytes(response.read())
    return output


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prompt", required=True)
    p.add_argument("--lyrics")
    p.add_argument("--model", default="music-3.0")
    p.add_argument("--region", choices=ENDPOINTS, default="global")
    p.add_argument("--output", required=True)
    p.add_argument("--output-format", choices=("url", "hex"), default="url")
    p.add_argument("--audio-format", choices=("mp3", "wav", "pcm"), default="mp3")
    p.add_argument("--instrumental", action="store_true")
    p.add_argument("--aigc-watermark", action="store_true")
    p.add_argument("--audio-url")
    p.add_argument("--audio-base64")
    p.add_argument("--cover-feature-id")
    return p


if __name__ == "__main__":
    print(generate(parser().parse_args()))

