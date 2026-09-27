#!/usr/bin/env python3
"""Generate 3D meshes (GLB) from a text prompt or a single image through fal.ai's queue API.

Usage:
    python fal_3d.py --list
    python fal_3d.py --model hunyuan-3.1-rapid --prompt "low-poly wooden treasure chest" -o chest.glb
    python fal_3d.py --model tripo-p2 --image ./mug.png --quad --faces 20000 -o mug.glb
    python fal_3d.py --model meshy-7.1 --prompt "cartoon fox character" --pbr --seed 7 -o fox.glb
    python fal_3d.py --model trellis-2 --image https://example.com/chair.jpg -o chair.glb
    python fal_3d.py --model tripo-p2 --prompt "ceramic owl" --no-texture --dry-run   # payload only, no call

Requires ``FAL_KEY`` (https://fal.ai/dashboard/keys) and the ``fal-client`` package
(``pip install fal-client==0.13.1``). A local ``--image`` path is uploaded to fal's CDN
first; a URL is forwarded as-is. Each model's payload builder only emits keys its fal
OpenAPI schema declares; unsupported knobs are dropped with a stderr note, never forwarded.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request
from typing import Any, Dict, Optional

# ── Model catalog ────────────────────────────────────────────────────────────
# ``build(args, image_url)`` maps the generic CLI knobs onto the endpoint's declared input
# schema. ``text`` / ``image`` are the endpoints per modality (None when unsupported).
# ``faces`` is the (min, max) face/polygon range the endpoint accepts, or None.


def _tripo_p2(a: argparse.Namespace, image_url: Optional[str]) -> Dict[str, Any]:
    payload: Dict[str, Any] = {"image_url": image_url} if image_url else {"prompt": a.prompt}
    if a.no_texture:
        payload["texture"] = False
        payload["pbr"] = False
    else:
        payload["texture"] = True
        payload["pbr"] = bool(a.pbr)
        if a.texture_quality:
            payload["texture_quality"] = a.texture_quality
    if a.quad:
        payload["quad"] = True
    if a.faces is not None:
        payload["face_limit"] = a.faces
    if a.seed is not None:
        payload["model_seed"] = a.seed
    if a.negative_prompt and not image_url:
        payload["negative_prompt"] = a.negative_prompt
    return payload


def _meshy_71(a: argparse.Namespace, image_url: Optional[str]) -> Dict[str, Any]:
    payload: Dict[str, Any] = {"image_url": image_url} if image_url else {"prompt": a.prompt}
    if image_url:
        payload["should_texture"] = not a.no_texture
    else:
        payload["mode"] = "preview" if a.no_texture else "full"
    if a.pbr and not a.no_texture:
        payload["enable_pbr"] = True
    if a.quad:
        payload["topology"] = "quad"
    if a.faces is not None:
        payload["target_polycount"] = a.faces
    if a.seed is not None and not image_url:
        payload["seed"] = a.seed
    return payload


def _hunyuan_pro(a: argparse.Namespace, image_url: Optional[str]) -> Dict[str, Any]:
    payload: Dict[str, Any] = {"input_image_url": image_url} if image_url else {"prompt": a.prompt}
    payload["generate_type"] = "Geometry" if a.no_texture else "Normal"
    if a.pbr and not a.no_texture:
        payload["enable_pbr"] = True
    if a.faces is not None:
        payload["face_count"] = a.faces
    return payload


def _hunyuan_rapid(a: argparse.Namespace, image_url: Optional[str]) -> Dict[str, Any]:
    payload: Dict[str, Any] = {"input_image_url": image_url} if image_url else {"prompt": a.prompt}
    if a.no_texture:
        payload["enable_geometry"] = True
    elif a.pbr:
        payload["enable_pbr"] = True
    return payload


def _trellis_2(a: argparse.Namespace, image_url: Optional[str]) -> Dict[str, Any]:
    payload: Dict[str, Any] = {"image_url": image_url}
    if a.faces is not None:
        payload["decimation_target"] = a.faces
    if a.seed is not None:
        payload["seed"] = a.seed
    return payload


MODELS: Dict[str, Dict[str, Any]] = {
    "hunyuan-3.1-rapid": {
        "text": "fal-ai/hunyuan-3d/v3.1/rapid/text-to-3d", "image": "fal-ai/hunyuan-3d/v3.1/rapid/image-to-3d",
        "build": _hunyuan_rapid, "faces": None, "seed": False,
        "kind": "fast, cheapest textured mesh",
        "notes": "Tencent Hunyuan 3D 3.1 Rapid. $0.225 per model (+$0.15 with --pbr). Prompt max 200 chars. Returns OBJ+MTL; GLB when available.",
    },
    "hunyuan-3.1-pro": {
        "text": "fal-ai/hunyuan-3d/v3.1/pro/text-to-3d", "image": "fal-ai/hunyuan-3d/v3.1/pro/image-to-3d",
        "build": _hunyuan_pro, "faces": (40_000, 1_500_000), "seed": False,
        "kind": "high-poly textured mesh",
        "notes": "Tencent Hunyuan 3D 3.1 Pro. $0.375 per model. --faces 40k-1.5M (default 500k); --no-texture gives a white geometry-only model.",
    },
    "tripo-p2": {
        "text": "tripo3d/p2/text-to-3d", "image": "tripo3d/p2/image-to-3d",
        "build": _tripo_p2, "faces": (1, 25_000), "seed": True,
        "kind": "production PBR assets, quad topology",
        "notes": "Tripo P2 (Sep 2026). $1.00 untextured, $1.10 standard/fast, $1.20 detailed, $1.30 extreme textures. --quad, --faces (quad meshes up to 25k), --negative-prompt (text only).",
    },
    "meshy-7.1": {
        "text": "meshy/v7.1/text-to-3d", "image": "meshy/v7.1/image-to-3d",
        "build": _meshy_71, "faces": (100, 300_000), "seed": True,
        "kind": "game-ready assets, optional rigging upstream",
        "notes": "Meshy 7.1 (Sep 2026). $0.80 untextured (--no-texture), $1.20 textured; --pbr adds metallic/roughness/normal maps. Prompt max 600 chars. Also returns FBX/USDZ/OBJ URLs.",
    },
    "trellis-2": {
        "text": None, "image": "fal-ai/trellis-2",
        "build": _trellis_2, "faces": (5_000, 2_000_000), "seed": True,
        "kind": "image-to-3D only, open model",
        "notes": "Microsoft TRELLIS.2 (open weights, GPU-time billing). --faces is the vertex decimation target (default 500k); no text-to-3D endpoint.",
    },
}

DEFAULT_MODEL = "hunyuan-3.1-rapid"
FACE_KEYS = ("face_limit", "target_polycount", "face_count", "decimation_target")
SEED_KEYS = ("seed", "model_seed")
TEXTURE_OFF_MARKERS = (("texture", False), ("mode", "preview"), ("should_texture", False),
                       ("generate_type", "Geometry"), ("enable_geometry", True))


def clamp_faces(model: str, faces: Optional[int]) -> Optional[int]:
    """Clamp into the model's declared face range; None when the model has no face knob."""
    rng = MODELS[model]["faces"]
    if faces is None or rng is None:
        return None
    lo, hi = rng
    return max(lo, min(hi, faces))


def endpoint_for(model: str, image_url: Optional[str]) -> str:
    meta = MODELS[model]
    ep = meta["image"] if image_url else meta["text"]
    if not ep:
        raise SystemExit(f"{model} has no {'image' if image_url else 'text'}-to-3D endpoint"
                         + ("" if image_url else "; pass --image"))
    return ep


def build_payload(model: str, args: argparse.Namespace, image_url: Optional[str]) -> Dict[str, Any]:
    meta = MODELS[model]
    requested_faces = args.faces
    args.faces = clamp_faces(model, args.faces)
    payload = meta["build"](args, image_url)
    # Knobs the endpoint never declared are dropped, not forwarded; say so once so a user notices.
    ignored = [flag for flag, present in (
        ("faces", requested_faces is not None and not any(k in payload for k in FACE_KEYS)),
        ("seed", args.seed is not None and not any(k in payload for k in SEED_KEYS)),
        ("negative-prompt", bool(args.negative_prompt) and "negative_prompt" not in payload),
        ("quad", args.quad and "quad" not in payload and payload.get("topology") != "quad"),
        ("pbr", args.pbr and not args.no_texture and "enable_pbr" not in payload and "pbr" not in payload),
        ("texture-quality", bool(args.texture_quality) and not args.no_texture and "texture_quality" not in payload),
        ("no-texture", args.no_texture and not any(payload.get(k) == v for k, v in TEXTURE_OFF_MARKERS)),
    ) if present]
    if ignored:
        print(f"note: {model} has no knob for --{', --'.join(ignored)}; ignored", file=sys.stderr)
    return payload


def _file_url(value: Any) -> Optional[str]:
    if isinstance(value, dict) and value.get("url"):
        return value["url"]
    if isinstance(value, str) and value.startswith("http"):
        return value
    return None


def pick_mesh(result: Dict[str, Any]) -> tuple[str, str]:
    """Return (url, extension) of the primary mesh: GLB first, then whatever the endpoint shipped."""
    for key in ("model_glb", "model_mesh"):
        url = _file_url(result.get(key))
        if url:
            return url, "glb" if key == "model_glb" or url.lower().endswith(".glb") else url.rsplit(".", 1)[-1][:4]
    urls = result.get("model_urls") or {}
    if isinstance(urls, dict):
        for fmt in ("glb", "fbx", "obj", "usdz"):
            url = _file_url(urls.get(fmt))
            if url:
                return url, fmt
    url = _file_url(result.get("model_obj"))
    if url:
        return url, "obj"
    raise SystemExit(f"No mesh URL in result: {json.dumps(result)[:400]}")


def resolve_image(image: Optional[str], dry_run: bool) -> Optional[str]:
    """Forward URLs untouched; upload a local file to fal's CDN (skipped under --dry-run)."""
    if not image:
        return None
    if image.startswith(("http://", "https://", "data:")):
        return image
    path = os.path.abspath(image)
    if not os.path.isfile(path):
        raise SystemExit(f"--image not found: {path}")
    if dry_run:
        return f"<upload:{path}>"
    import pathlib

    import fal_client
    return fal_client.upload_file(pathlib.Path(path))


def generate(model: str, endpoint: str, payload: Dict[str, Any], output: str) -> Dict[str, Any]:
    try:
        import fal_client
    except ImportError:
        raise SystemExit("fal-client not installed: pip install fal-client==0.13.1")
    try:
        result = fal_client.subscribe(endpoint, arguments=payload, with_logs=False)
    except Exception as exc:  # fal_client raises its own HTTP error types; surface the message, not a traceback
        raise SystemExit(f"fal request failed ({endpoint}): {exc}") from exc
    url, ext = pick_mesh(result)
    if not os.path.splitext(output)[1]:
        output = f"{output}.{ext}"
    with urllib.request.urlopen(url, timeout=300) as resp, open(output, "wb") as fh:
        fh.write(resp.read())
    preview = _file_url(result.get("thumbnail") or result.get("rendered_image"))
    raw_urls = result.get("model_urls")
    urls: Dict[str, Any] = raw_urls if isinstance(raw_urls, dict) else {}
    return {
        "output": output, "url": url, "format": ext, "model": model, "endpoint": endpoint,
        "preview_url": preview, "seed": result.get("seed"),
        "other_formats": {k: _file_url(v) for k, v in urls.items() if _file_url(v)},
    }


def main(argv: Optional[list] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--list", action="store_true", help="list models and exit")
    p.add_argument("--model", default=DEFAULT_MODEL, choices=sorted(MODELS), help=f"model id (default {DEFAULT_MODEL})")
    p.add_argument("--prompt", help="what the object is (text-to-3D); ignored when --image is given except as a note")
    p.add_argument("--image", help="reference image: local path (uploaded to fal) or URL (image-to-3D)")
    p.add_argument("--no-texture", action="store_true", help="geometry only, no textures (cheaper on tripo/meshy)")
    p.add_argument("--pbr", action="store_true", help="request PBR material maps (metallic/roughness/normal)")
    p.add_argument("--quad", action="store_true", help="quad topology instead of triangles (tripo-p2, meshy-7.1)")
    p.add_argument("--faces", type=int, help="target face / polygon count (clamped to the model's range)")
    p.add_argument("--texture-quality", choices=["fast", "standard", "detailed", "extreme"], help="tripo-p2 texture tier")
    p.add_argument("--seed", type=int, help="seed (models that declare one)")
    p.add_argument("--negative-prompt", help="features to avoid (tripo-p2 text-to-3D only)")
    p.add_argument("--dry-run", action="store_true", help="print the endpoint + payload and exit without calling fal")
    p.add_argument("-o", "--output", help="output file path (default ./fal-3d-<model>.glb in the current directory)")
    a = p.parse_args(argv)

    if a.list:
        for mid, m in MODELS.items():
            rng = m["faces"]
            span = f"{rng[0]}-{rng[1]}" if rng else "model-decided"
            modes = "+".join(k for k in ("text", "image") if m[k])
            print(f"{mid:18} {modes:10} faces={span:18} seed={'y' if m['seed'] else 'n'}  {m['kind']}")
            print(f"{'':18} text:  {m['text'] or '-'}")
            print(f"{'':18} image: {m['image'] or '-'}")
            print(f"{'':18} {m['notes']}")
        return 0
    if not a.prompt and not a.image:
        p.error("--prompt or --image is required")

    image_url = resolve_image(a.image, a.dry_run)
    endpoint = endpoint_for(a.model, image_url)
    payload = build_payload(a.model, a, image_url)
    if a.dry_run:
        print(json.dumps({"endpoint": endpoint, "payload": payload}, indent=2))
        return 0
    output = a.output or os.path.join(os.getcwd(), f"fal-3d-{a.model}.glb")
    print(json.dumps(generate(a.model, endpoint, payload, output), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
