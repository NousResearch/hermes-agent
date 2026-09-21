"""FAL.ai client for image and video generation.
Supports both Hermes native image_gen and direct API for video.

IMPORTANT: Every API call costs credits. Image generation is ~£0.005 each.
Video generation is ~£0.16-0.40 per 5-second clip. Use sparingly.
"""
import _no_proxy  # noqa: F401  (strips dead Privoxy from env on import)
import base64
import os
import time
import uuid
from pathlib import Path
from typing import Optional

try:
    import requests
except ImportError:
    requests = None

FAL_KEY = os.getenv("FAL_KEY", "")

# Model endpoints on fal.ai
IMAGE_MODELS = {
    "z_image": "fal-ai/z-image/turbo",           # ~£0.008/img, fast + detailed
    "flux_klein": "fal-ai/flux-2/klein/9b",      # £0.0048/img, fast
    "krea_medium": "fal-ai/krea/v2/medium/text-to-image",  # ~£0.02/img, aesthetic detail
    "krea_large": "fal-ai/krea/v2/large/text-to-image",    # ~£0.03/img, max detail
    "flux_pro": "fal-ai/flux-2-pro",             # £0.024/img, quality
    "ideogram": "fal-ai/ideogram/v3",            # £0.024/img, typography
    "qwen": "fal-ai/qwen-image",                  # ~£0.016/img, strong text (LLM-arch), cheapest text-capable
    "recraft": "fal-ai/recraft/v3/text-to-image", # ~£0.032/img, #1 text/typography benchmark
    "flux_ultra": "fal-ai/flux-pro/v1.1-ultra",            # ~£0.048/img, 2K/4MP high detail
    "flux_kontext": "fal-ai/flux-pro/kontext/text-to-image", # ~£0.032/img
    "qwen2_pro": "fal-ai/qwen-image-2/pro/text-to-image",  # ~£0.06/img, #1 AI-Arena text
    "seedream": "fal-ai/bytedance/seedream/v4/text-to-image",       # ~£0.024, strong text
    "seedream45": "fal-ai/bytedance/seedream/v4.5/text-to-image",   # ~£0.032, best text/value
    "nano_banana": "fal-ai/nano-banana-pro",     # £0.12/img, text+reasoning
    "nano_banana_2": "fal-ai/nano-banana-2/edit", # $0.08/img, Google Nano Banana 2 edit
    "gpt_image_2": "fal-ai/gpt-image-2",         # ~£0.04/img, SOTA text/layout (hero fallback)
}

# Approx GBP per image, for budget gating of paid hero models.
MODEL_COST_GBP = {
    "z_image": 0.004, "flux_klein": 0.004, "krea_medium": 0.016,
    "krea_large": 0.024, "flux_pro": 0.024, "ideogram": 0.024,
    "gpt_image_2": 0.04, "nano_banana": 0.12, "nano_banana_2": 0.064, "qwen": 0.016, "recraft": 0.032,
    "flux_ultra": 0.048, "flux_kontext": 0.032, "qwen2_pro": 0.06,
    "seedream": 0.024, "seedream45": 0.032,
}

VIDEO_MODELS = {
    "wan_22_480p": "fal-ai/wan/v2.2/480p",       # ~£0.032/sec
    "wan_22_720p": "fal-ai/wan/v2.2/720p",       # ~£0.064/sec
    "wan_27": "fal-ai/wan/v2.7",                 # ~£0.08/sec premium
    "seedance_fast": "fal-ai/seedance/v2.0/fast", # ~£0.022/sec cheapest
}


def _headers() -> dict:
    return {"Authorization": f"Key {FAL_KEY}", "Content-Type": "application/json"}


def _poll_queue_result(model_id: str, request_id: str, timeout: int = 120) -> Optional[dict]:
    """Poll async queue endpoint for result."""
    if not FAL_KEY or requests is None:
        return None
    poll_url = f"https://queue.fal.run/{model_id}/requests/{request_id}"
    start = time.time()
    while time.time() - start < timeout:
        try:
            r = requests.get(poll_url, headers=_headers(), timeout=15)
            if r.status_code == 200:
                data = r.json()
                status = data.get("status", "")
                if status == "COMPLETED":
                    return data
                elif status in ("FAILED", "CANCELLED"):
                    print(f"FAL queue failed: {status}")
                    return None
            time.sleep(3)
        except Exception as e:
            print(f"FAL poll error: {e}")
            return None
    print("FAL queue: polling timeout")
    return None


# Platform-native sizes. IG feed is 4:5 and TikTok/Reels 9:16 — generating
# square and centre-cropping later destroys the composition, so the base must
# be generated at the final aspect. Custom width/height dicts are accepted by
# the flux/krea/seedream families; models that only take preset enums fall
# back to the nearest preset via _image_size_for.
_ASPECT_MAP = {
    "square": "square_hd",
    "landscape": "landscape_16_9",
    "portrait": "portrait_16_9",
    "portrait_4_5": {"width": 1024, "height": 1280},
    "portrait_9_16": {"width": 1080, "height": 1920},
}

# Models that reject custom size objects and need a preset enum instead.
_PRESET_ONLY_MODELS = {"nano_banana"}
_PRESET_FALLBACK = {"portrait_4_5": "portrait_4_3", "portrait_9_16": "portrait_16_9"}


def _image_size_for(model: str, aspect: str):
    if model in _PRESET_ONLY_MODELS and aspect in _PRESET_FALLBACK:
        return _PRESET_FALLBACK[aspect]
    return _ASPECT_MAP.get(aspect, "square_hd")


def _images_from(result: dict) -> list:
    images = result.get("images", [])
    if not images and "output" in result:
        images = result["output"].get("images", [])
    if not images and "image" in result:
        images = [result["image"]]
    return images


def _download(url: str, out_dir: str, model: str, filename: Optional[str]) -> Optional[str]:
    img_r = requests.get(url, timeout=90)
    if img_r.status_code != 200:
        print(f"FAL image download failed: {img_r.status_code}")
        return None
    out_path = Path(out_dir) / "fal_images"
    out_path.mkdir(parents=True, exist_ok=True)
    fpath = out_path / (filename or f"{model}_{str(uuid.uuid4())[:8]}.png")
    fpath.write_bytes(img_r.content)
    print(f"FAL image saved: {fpath} ({len(img_r.content)} bytes)")
    return str(fpath)


def generate_image(
    prompt: str,
    model: str = "krea_medium",
    aspect: str = "square",
    output_dir: str = "/home/kensei/repos/KenseiAgent/content_engine/output",
    filename: Optional[str] = None,
    negative_prompt: str = "",
    attempts: int = 3,
) -> Optional[str]:
    """Generate an image via fal.ai with retries and queue-first for slow models.

    Slow models (nano-banana, krea-large, flux-pro) always use the async queue,
    which is the only reliable path from this box; fast models try sync then fall
    back to the queue. Network timeouts are retried with backoff rather than
    silently failing.
    """
    if not FAL_KEY or requests is None:
        print("FAL_KEY not set or requests unavailable. Skipping image generation.")
        return None

    try:
        from model_config import SLOW_MODELS
    except Exception:
        SLOW_MODELS = {"nano_banana", "krea_large", "flux_pro", "ideogram"}

    model_id = IMAGE_MODELS.get(model, IMAGE_MODELS["krea_medium"])
    payload = {"prompt": prompt, "image_size": _image_size_for(model, aspect), "num_images": 1}
    # negative_prompt only helps diffusion models; reasoning models reject/ignore it.
    if negative_prompt and model not in ("nano_banana",):
        payload["negative_prompt"] = negative_prompt

    # The sync endpoint is the reliable path from this box; the async queue is
    # flaky/slow here, so it is only a fallback. Slow models just get a longer
    # sync timeout rather than being forced onto the queue.
    sync_timeout = 220 if model in SLOW_MODELS else 100

    for attempt in range(1, attempts + 1):
        try:
            r = requests.post(f"https://fal.run/{model_id}", json=payload,
                              headers=_headers(), timeout=sync_timeout)
            if r.status_code == 200:
                imgs = _images_from(r.json())
                if imgs:
                    url = imgs[0].get("url") if isinstance(imgs[0], dict) else imgs[0]
                    return _download(url, output_dir, model, filename) if url else None
                print(f"FAL sync: no images (keys {list(r.json().keys())[:8]})")
                return None
            # Auth/permission/billing errors are not transient: retrying just
            # multiplies identical noise into #content and burns time. Fail fast
            # with a single concise line and stop attempting this model.
            if r.status_code in (401, 402, 403):
                print(f"FAL {model}: {r.status_code} auth/billing rejected (check FAL_KEY / spend cap)")
                return None
            if r.status_code not in (202, 429):
                print(f"FAL sync failed: {r.status_code} {r.text[:160]}")
            # 202/429/other -> queue fallback below

            qr = requests.post(f"https://queue.fal.run/{model_id}", json=payload,
                              headers=_headers(), timeout=30)
            if qr.status_code != 200:
                print(f"FAL queue submit failed: {qr.status_code} {qr.text[:120]}")
                raise RuntimeError("queue submit non-200")
            rid = qr.json().get("request_id")
            if not rid:
                raise RuntimeError("no request_id")
            result = _poll_queue_result(model_id, rid, timeout=240)
            if not result:
                raise RuntimeError("poll returned nothing")
            imgs = _images_from(result)
            if not imgs:
                print(f"FAL queue: no images (keys {list(result.keys())[:8]})")
                return None
            url = imgs[0].get("url") if isinstance(imgs[0], dict) else imgs[0]
            return _download(url, output_dir, model, filename) if url else None

        except Exception as e:  # noqa: BLE001 (retry transient failures)
            wait = 2 ** attempt
            print(f"FAL {model} attempt {attempt}/{attempts} error: {e}; retry in {wait}s")
            if attempt < attempts:
                time.sleep(wait)

    print(f"FAL {model}: all {attempts} attempts failed")
    return None


def upload_file(path, content_type: str = "image/jpeg") -> Optional[str]:
    """Upload a local file to FAL storage and return a public URL.

    Used to hand reference anchors to the edit endpoint. Falls back to a base64
    data URI when the storage API is unavailable, so anchoring still works.
    """
    p = Path(path)
    if not FAL_KEY or requests is None or not p.exists():
        return None
    try:
        init = requests.post(
            "https://rest.alpha.fal.ai/storage/upload/initiate",
            headers={"Authorization": f"Key {FAL_KEY}", "Content-Type": "application/json"},
            json={"content_type": content_type, "file_name": f"{uuid.uuid4().hex}{p.suffix}"},
            timeout=30,
        )
        if init.status_code == 200:
            d = init.json()
            put = requests.put(d["upload_url"], data=p.read_bytes(),
                               headers={"Content-Type": content_type}, timeout=90)
            if put.status_code in (200, 201):
                return d["file_url"]
    except Exception as e:  # noqa: BLE001 (degrade to data URI)
        print(f"FAL upload error: {e}")
    suffix = p.suffix.lower().lstrip(".")
    mime = {"png": "image/png", "webp": "image/webp"}.get(suffix, "image/jpeg")
    return f"data:{mime};base64," + base64.b64encode(p.read_bytes()).decode()


def generate_image_edit(
    prompt: str,
    image_urls: list,
    model: str = "fal-ai/nano-banana-pro/edit",
    aspect: str = "4:5",
    resolution: str = "2K",
    output_dir: str = "/home/kensei/repos/KenseiAgent/content_engine/output",
    filename: Optional[str] = None,
    attempts: int = 2,
) -> Optional[str]:
    """Reference-anchored (img2img/edit) generation via the nano-banana-pro/edit
    endpoint. ``image_urls`` are anchor URLs (or data URIs). Sync first, queue
    fallback. Returns a local PNG path or None.
    """
    if not FAL_KEY or requests is None:
        print("FAL_KEY not set or requests unavailable. Skipping edit generation.")
        return None
    payload = {"prompt": prompt, "image_urls": list(image_urls),
               "aspect_ratio": aspect, "num_images": 1, "resolution": resolution}
    headers = _headers()
    for attempt in range(1, attempts + 1):
        try:
            r = requests.post(f"https://fal.run/{model}", json=payload,
                              headers=headers, timeout=240)
            if r.status_code == 200:
                imgs = _images_from(r.json())
                if imgs:
                    url = imgs[0].get("url") if isinstance(imgs[0], dict) else imgs[0]
                    return _download(url, output_dir, "edit", filename) if url else None
                print(f"FAL edit: no images (keys {list(r.json().keys())[:8]})")
                return None
            if r.status_code in (401, 402, 403):
                print(f"FAL edit: {r.status_code} auth/billing rejected (check FAL_KEY / spend cap)")
                return None
            if r.status_code not in (202, 429):
                print(f"FAL edit failed: {r.status_code} {r.text[:160]}")
            qr = requests.post(f"https://queue.fal.run/{model}", json=payload,
                              headers=headers, timeout=30)
            if qr.status_code != 200:
                raise RuntimeError(f"queue submit {qr.status_code}")
            rid = qr.json().get("request_id")
            if not rid:
                raise RuntimeError("no request_id")
            result = _poll_queue_result(model, rid, timeout=300)
            if not result:
                raise RuntimeError("poll returned nothing")
            imgs = _images_from(result)
            if not imgs:
                print("FAL edit queue: no images")
                return None
            url = imgs[0].get("url") if isinstance(imgs[0], dict) else imgs[0]
            return _download(url, output_dir, "edit", filename) if url else None
        except Exception as e:  # noqa: BLE001 (retry transient failures)
            wait = 2 ** attempt
            print(f"FAL edit attempt {attempt}/{attempts} error: {e}; retry in {wait}s")
            if attempt < attempts:
                time.sleep(wait)
    print("FAL edit: all attempts failed")
    return None


def generate_video(
    prompt: str,
    model: str = "wan_22_480p",
    duration: int = 5,
    output_dir: str = "/home/kensei/repos/KenseiAgent/content_engine/output",
    filename: Optional[str] = None,
) -> Optional[str]:
    """Generate video via fal.ai. Returns file path or None."""
    if not FAL_KEY or requests is None:
        print("FAL_KEY not set. Skipping video generation.")
        return None

    model_id = VIDEO_MODELS.get(model, VIDEO_MODELS["wan_22_480p"])
    # Video models are slower — use queue endpoint
    queue_url = f"https://queue.fal.run/{model_id}"
    payload = {
        "prompt": prompt,
        "duration": duration,
        "aspect_ratio": "16:9",
    }

    try:
        qr = requests.post(queue_url, json=payload, headers=_headers(), timeout=30)
        if qr.status_code != 200:
            print(f"FAL video queue failed: {qr.status_code} {qr.text[:200]}")
            return None

        qdata = qr.json()
        request_id = qdata.get("request_id")
        if not request_id:
            print("FAL video: no request_id")
            return None

        print(f"FAL video queued: {request_id}, polling...")
        result = _poll_queue_result(model_id, request_id, timeout=300)
        if not result:
            return None

        # Extract video URL
        video_url = result.get("video", {}).get("url")
        if not video_url:
            video_url = result.get("output", {}).get("video", {}).get("url")
        if not video_url:
            print(f"FAL video: no URL. Keys: {list(result.keys())[:10]}")
            return None

        vid_r = requests.get(video_url, timeout=120)
        if vid_r.status_code != 200:
            return None

        out_path = Path(output_dir) / "fal_videos"
        out_path.mkdir(parents=True, exist_ok=True)
        fname = filename or f"{model}_{str(uuid.uuid4())[:8]}.mp4"
        fpath = out_path / fname
        fpath.write_bytes(vid_r.content)
        print(f"FAL video saved: {fpath} ({len(vid_r.content)} bytes)")
        return str(fpath)

    except Exception as e:
        print(f"FAL video generation error: {e}")
        return None


def generate_image_from_text_card(
    brand: str,
    body_text: str,
    output_dir: str = "/home/kensei/repos/KenseiAgent/content_engine/output",
) -> Optional[str]:
    """DEPRECATED. Never embed literal post text in a diffusion prompt -- the
    model renders it as gibberish. Use draft_media.generate_post_image(), which
    generates a textless branded base and overlays the headline with Pillow.
    Kept as a thin shim so any stray caller still produces a usable image.
    """
    from draft_media import generate_post_image

    return generate_post_image(
        {"brand": brand, "body_text": body_text, "platform": "twitter", "id": ""},
        output_dir=output_dir,
    )
