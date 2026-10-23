"""Kling image-to-video on Replicate.

Two things here are the whole reason this pipeline works, and both are easy to
miss:

1. `end_image` set to the START image. Kling then returns to its opening pose,
   which is what makes the clip loop instead of ping-pong. Without it you are
   left cross-fading, and a cross-fade on a rigid object reads as a glitch.

2. The still is composited onto a KNOWN flat backing before it is sent. Kling
   will not accept alpha, and the colour you choose is not cosmetic: because
   you know it exactly, the matte stage can solve for the true foreground
   instead of estimating it. Mid-grey is used rather than a chroma-key green
   or magenta, which spill onto glossy edges and destroy soft shadows.
"""
import os
import time
import urllib.request

from . import config, http

API = "https://api.replicate.com/v1"
# Mid-grey: far enough from most art to matte cleanly, neutral enough that any
# spill it does leave is colourless rather than a green or magenta fringe.
BACKING = (158, 158, 158)

# Every clause here is load-bearing, and one earlier version of this string
# silently killed the animation: it said "the object keeps its exact shape and
# proportions throughout", which reads as an instruction to stay still. Kling
# obeyed, and returned 122 frames of the input image. Constrain the CAMERA and
# the SCENE as hard as you like; never constrain the object's shape.
LOOP_RULES = (
    "The motion must be clearly visible and true to what this object really "
    "does — never a generic animation applied to it. "
    "The camera is locked off and must not move, pan, zoom or push in. "
    "The background stays flat, empty and completely static. "
    "Nothing enters the frame from outside it — no hands, no text, no captions, "
    "no watermarks, no props. "
    "Everything stays comfortably inside the frame at all times: no part of the "
    "object, and nothing it produces, may touch, reach or cross any edge of the "
    "frame at any point. Keep a clear margin on all four sides throughout. "
    # Deliberately narrower than it used to be. This clause once banned "new
    # objects or effects" outright, which also banned anything the object
    # itself produced — and for an object that does not move on its own, that
    # emission is the entire animation. Constrain where things come FROM, not
    # whether the object may produce them.
    "It stays the same object in the same material and palette throughout — its "
    "identity, colours and finish do not change."
)


# How much of the canvas the object occupies when it is sent. The image model
# returns art that fills its frame, and sending that straight on leaves motion
# nowhere to go: anything that stretches, splashes or throws a piece runs into
# the canvas edge and is clipped there, permanently, in the source render. The
# clip then survives every later stage, and framing normalization quietly hides
# it by scaling the amputated result inward.
#
# So the object is padded down to this fraction first. The margin is the room
# the animation gets to move into.
FILL = 0.68


def composite(still_path, out_path, size=1024, fill=FILL):
    """Flatten a transparent PNG onto the known backing, padded, ready to send."""
    from PIL import Image
    im = Image.open(still_path).convert("RGBA")
    box = im.getbbox()  # trim whatever transparent margin the art already has
    if box:
        im = im.crop(box)
    scale = (size * fill) / max(im.size)
    im = im.resize((max(1, round(im.width * scale)), max(1, round(im.height * scale))),
                   Image.LANCZOS)
    bg = Image.new("RGBA", (size, size), BACKING + (255,))
    bg.alpha_composite(im, ((size - im.width) // 2, (size - im.height) // 2))
    bg.convert("RGB").save(out_path)
    return out_path


def resolve_version(model=None):
    model = model or config.opt("ICONLOOP_KLING_MODEL", "kwaivgi/kling-v2.5-turbo-pro")
    k = config.key("REPLICATE_API_TOKEN", "run Kling image-to-video on Replicate")
    # Note: Replicate's ?search= is semantic and does not reliably surface Kling.
    # Hitting the model endpoint directly is what actually works.
    d = http.get_json(f"{API}/models/{model}", {"Authorization": f"Bearer {k}"})
    v = d.get("latest_version", {}).get("id")
    if not v:
        raise SystemExit(f"Could not resolve a version for {model}.")
    return model, v


OR_API = "https://openrouter.ai/api/v1/videos"

# Seedance rather than Kling 3.0, for a boring reason that cost a run to find:
# Kling's API caps the prompt at 2500 characters and the composed motion clause
# runs past 3000. It rejects the request outright (ret:1201) rather than
# truncating, so the better model is simply unusable here until the prompt is
# shorter. Seedance takes the whole thing, supports the first/last frame loop,
# and produced the better result anyway.
DEFAULT_VIDEO_MODEL = "bytedance/seedance-2.0"

# Models known to reject long prompts, with their limit.
PROMPT_LIMITS = {"kwaivgi/kling-v3.0-pro": 2500, "kwaivgi/kling-v3.0-std": 2500,
                 "kwaivgi/kling-video-o1": 2500}


def animate_openrouter(image_path, motion_prompt, out_path, duration=5,
                       model=None, poll=10):
    """Same job through OpenRouter, which carries models Replicate does not.

    Replicate's kwaivgi account stops at kling-v2.5-turbo-pro; OpenRouter lists
    Kling 3.0, Seedance 2.x, Veo 3.1, Hailuo 3 and Wan 2.7, fourteen of which
    take a first AND last frame. Same loop trick, wider choice of model, and
    one key for the still and the motion.
    """
    import base64
    k = config.key("OPENROUTER_API_KEY", "run image-to-video via OpenRouter")
    model = model or config.opt("ICONLOOP_VIDEO_MODEL", DEFAULT_VIDEO_MODEL)
    data_uri = "data:image/png;base64," + base64.b64encode(open(image_path, "rb").read()).decode()
    frame = lambda t: {"type": "image_url", "image_url": {"url": data_uri}, "frame_type": t}

    prompt = f"{motion_prompt.strip().rstrip('.')}. {LOOP_RULES}"
    cap = PROMPT_LIMITS.get(model)
    if cap and len(prompt) > cap:
        raise SystemExit(
            f"\n{model} caps prompts at {cap} characters and this one is "
            f"{len(prompt)}.\nIt will reject the request rather than truncate. "
            f"Use --model {DEFAULT_VIDEO_MODEL} (no cap, same first/last-frame "
            f"loop), or shorten the motion clause.\n")
    job = http.post_json(OR_API, {
        "model": model,
        "prompt": prompt,
        # The loop trick again: the last frame is the first frame.
        "frame_images": [frame("first_frame"), frame("last_frame")],
        "duration": duration,
    }, {"Authorization": f"Bearer {k}"})

    url = job.get("polling_url") or f"{OR_API}/{job['id']}"
    while job.get("status") in ("queued", "pending", "running", "in_progress", "processing"):
        time.sleep(poll)
        job = http.get_json(url, {"Authorization": f"Bearer {k}"})
        print(f"  {model}: {job.get('status')}", flush=True)

    if job.get("status") != "completed":
        raise RuntimeError(f"{model} failed: {job.get('error') or job.get('status')}")

    vid = job["unsigned_urls"][0]
    req = urllib.request.Request(vid, headers={"Authorization": f"Bearer {k}",
                                               "User-Agent": http.UA})
    with urllib.request.urlopen(req, timeout=600) as r, open(out_path, "wb") as f:
        f.write(r.read())
    cost = (job.get("usage") or {}).get("cost")
    print(f"  {model}: {os.path.getsize(out_path)/1024/1024:.1f} MB"
          f"{f' · ${cost}' if cost else ''} -> {out_path}", flush=True)
    return model, job.get("generation_id", "")


def animate(image_path, motion_prompt, out_path, duration=5, model=None, poll=10):
    """Run one image-to-video job and download the mp4. Returns (model, version)."""
    k = config.key("REPLICATE_API_TOKEN", "run Kling image-to-video on Replicate")
    model, version = resolve_version(model)
    import base64
    data_uri = "data:image/png;base64," + base64.b64encode(open(image_path, "rb").read()).decode()

    prompt = f"{motion_prompt.strip().rstrip('.')}. {LOOP_RULES}"
    pred = http.post_json(f"{API}/predictions", {
        "version": version,
        "input": {
            "prompt": prompt,
            "start_image": data_uri,
            # The loop trick: end where you began.
            "end_image": data_uri,
            "duration": duration,
        },
    }, {"Authorization": f"Bearer {k}", "Prefer": "wait"})

    url = pred.get("urls", {}).get("get")
    while pred.get("status") in ("starting", "processing"):
        time.sleep(poll)
        pred = http.get_json(url, {"Authorization": f"Bearer {k}"})
        print(f"  kling: {pred.get('status')}", flush=True)

    if pred.get("status") != "succeeded":
        raise RuntimeError(f"Kling failed: {pred.get('error') or pred.get('status')}")

    out = pred["output"]
    http.download(out if isinstance(out, str) else out[0], out_path)
    print(f"  kling: {os.path.getsize(out_path)/1024/1024:.1f} MB -> {out_path}", flush=True)
    return model, version
