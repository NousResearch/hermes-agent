#!/usr/bin/env python3
"""deapi.py — helper CLI for the deAPI v2 REST API (https://docs.deapi.ai).

Standard library only (Python 3.9+). Authentication via the DEAPI_API_KEY
environment variable. All generation endpoints are asynchronous: the script
submits a job, polls GET /api/v2/jobs/{request_id} until it finishes, then
prints text results to stdout or downloads file results to disk.

Model slugs are never hardcoded: when --model is omitted the script fetches
the live model list from GET /api/v2/models and picks a sensible default for
the task (preference patterns only — the final choice always comes from the
live list).

Usage: python3 deapi.py <command> [options]   (see --help per command)
"""

import argparse
import io
import json
import mimetypes
import os
import random
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid

API_BASE = os.environ.get("DEAPI_BASE_URL", "https://api.deapi.ai").rstrip("/")
POLL_INTERVAL = float(os.environ.get("DEAPI_POLL_INTERVAL", "5"))
POLL_TIMEOUT = float(os.environ.get("DEAPI_POLL_TIMEOUT", "900"))
HTTP_TIMEOUT = 120

# Canonical task -> accepted inference_type spellings (the API and its docs
# use both short (txt2img) and kebab-case (text-to-image) forms).
INFERENCE_ALIASES = {
    "image": {"txt2img", "text-to-image"},
    "edit": {"img2img", "image-to-image"},
    "tts": {"txt2audio", "txt2speech", "text-to-speech", "tts"},
    "stt": {"audio2text", "video2text", "audio_file2text", "video_file2text",
            "aud2txt", "vid2txt", "audio-to-text", "video-to-text", "transcription"},
    "ocr": {"img2txt", "image-to-text", "ocr"},
    "rmbg": {"img-rmbg", "background-removal", "rmbg"},
    "upscale": {"img-upscale", "image-upscale", "upscale"},
    "music": {"txt2music", "text-to-music", "txt2audio-music"},
    "video": {"txt2video", "text-to-video"},
    "animate": {"img2video", "image-to-video"},
    "embed": {"txt2embedding", "text-to-embedding", "embedding"},
}

# Preference patterns (regex, case-insensitive) used ONLY to rank the live
# model list when --model is omitted. Never used as literal slugs.
MODEL_PREFERENCES = {
    "image": [r"klein", r"flux"],
    "edit": [r"klein", r"qwen"],
    "tts": [r"kokoro"],
    "stt": [r"whisper"],
    "ocr": [r"nanonets", r"ocr"],
    "rmbg": [r"ben"],
    "upscale": [r"esrgan"],
    "music": [r"ace"],
    "video": [r"ltx"],
    "animate": [r"ltx"],
    "embed": [r"bge"],
}


class ApiError(Exception):
    pass


def log(msg):
    sys.stderr.write(msg + "\n")
    sys.stderr.flush()


def api_key():
    key = os.environ.get("DEAPI_API_KEY", "").strip()
    if not key:
        raise ApiError(
            "DEAPI_API_KEY is not set. Create a key at https://app.deapi.ai "
            "(API Keys section, format dpn-sk-...) and export it: "
            "export DEAPI_API_KEY=dpn-sk-..."
        )
    # The native REST API (api.deapi.ai) authenticates with the raw token;
    # the dpn-sk- prefix is only required by the OpenAI-compatible gateway
    # (oai.deapi.ai). Accept the key in either form.
    if key.startswith("dpn-sk-"):
        key = key[len("dpn-sk-"):]
    return key


def _request(method, url, body=None, headers=None):
    req = urllib.request.Request(url, data=body, method=method)
    req.add_header("Authorization", "Bearer " + api_key())
    req.add_header("Accept", "application/json")
    # Cloudflare on api.deapi.ai bans the default Python-urllib user agent.
    req.add_header("User-Agent", "deapi-skill/1.0 (+https://github.com/deapi-ai/skills)")
    for name, value in (headers or {}).items():
        req.add_header(name, value)
    try:
        with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")
        try:
            detail = json.dumps(json.loads(detail), indent=2)
        except ValueError:
            pass
        hint = ""
        if exc.code in (401, 403):
            hint = " (check DEAPI_API_KEY and your balance at https://app.deapi.ai)"
        elif exc.code == 429:
            hint = " (rate limited; Retry-After: %s)" % exc.headers.get("Retry-After", "?")
        raise ApiError("HTTP %d for %s %s%s\n%s" % (exc.code, method, url, hint, detail))
    except urllib.error.URLError as exc:
        raise ApiError("Network error for %s: %s" % (url, exc.reason))


def api_json(method, path, payload=None, query=None):
    url = API_BASE + path
    if query:
        url += "?" + urllib.parse.urlencode(query)
    body = None
    headers = {}
    if payload is not None:
        body = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    return _request(method, url, body, headers)


def api_multipart(path, fields, files):
    """POST multipart/form-data. files: list of (field_name, file_path)."""
    boundary = "----deapi-" + uuid.uuid4().hex
    buf = io.BytesIO()

    def part(header, payload):
        buf.write(("--%s\r\n%s\r\n\r\n" % (boundary, header)).encode("utf-8"))
        buf.write(payload)
        buf.write(b"\r\n")

    for name, value in fields.items():
        if value is None:
            continue
        if isinstance(value, bool):
            value = "1" if value else "0"
        part('Content-Disposition: form-data; name="%s"' % name,
             str(value).encode("utf-8"))

    for name, file_path in files:
        if not os.path.isfile(file_path):
            raise ApiError("File not found: %s" % file_path)
        ctype = mimetypes.guess_type(file_path)[0] or "application/octet-stream"
        with open(file_path, "rb") as handle:
            data = handle.read()
        part('Content-Disposition: form-data; name="%s"; filename="%s"\r\n'
             'Content-Type: %s' % (name, os.path.basename(file_path), ctype),
             data)

    buf.write(("--%s--\r\n" % boundary).encode("utf-8"))
    headers = {"Content-Type": "multipart/form-data; boundary=" + boundary}
    return _request("POST", API_BASE + path, buf.getvalue(), headers)


# ---------------------------------------------------------------- models ---

def fetch_models():
    """Fetch the full live model list (paginated)."""
    models, page = [], 1
    while page <= 20:
        data = api_json("GET", "/api/v2/models", query={"per_page": 100, "page": page})
        batch = data.get("data") or []
        if isinstance(batch, dict):
            batch = batch.get("models") or batch.get("data") or []
        if not batch:
            break
        models.extend(batch)
        meta = data.get("meta") or {}
        last = meta.get("last_page") or meta.get("total_pages")
        if last and page >= int(last):
            break
        if not last and len(batch) < 100:
            break
        page += 1
    if not models:
        raise ApiError("GET /api/v2/models returned no models.")
    return models


def models_for_task(models, task):
    aliases = INFERENCE_ALIASES[task]
    found = []
    for model in models:
        types = {str(t).lower() for t in (model.get("inference_types") or [])}
        if types & aliases:
            found.append(model)
    return found


def resolve_model(task, explicit_slug):
    """Return (slug, model_dict_or_None). Never hardcodes: defaults come from
    the live /api/v2/models list, optionally ranked by preference patterns."""
    models = None
    if explicit_slug:
        try:
            models = fetch_models()
        except ApiError:
            return explicit_slug, None  # can't fetch metadata; trust the caller
        for model in models:
            if model.get("slug") == explicit_slug:
                return explicit_slug, model
        log("warning: model '%s' not found in the live model list; sending anyway"
            % explicit_slug)
        return explicit_slug, None

    models = fetch_models()
    candidates = models_for_task(models, task)
    if not candidates:
        raise ApiError(
            "No live model supports task '%s'. Run `deapi.py models` to inspect "
            "the current list." % task)
    for pattern in MODEL_PREFERENCES.get(task, []):
        for model in candidates:
            haystack = "%s %s" % (model.get("slug", ""), model.get("name", ""))
            if re.search(pattern, haystack, re.IGNORECASE):
                log("auto-selected model: %s (from live GET /api/v2/models)"
                    % model["slug"])
                return model["slug"], model
    model = candidates[0]
    log("auto-selected model: %s (first live match for %s)" % (model["slug"], task))
    return model["slug"], model


def model_defaults(model):
    info = (model or {}).get("info") or {}
    defaults = info.get("defaults") or {}
    limits = info.get("limits") or {}
    features = info.get("features") or {}
    return defaults, limits, features


def clamp(value, limits, low_key, high_key):
    low, high = limits.get(low_key), limits.get(high_key)
    if low is not None:
        value = max(int(low), value)
    if high is not None:
        value = min(int(high), value)
    return value


# ------------------------------------------------------------------ jobs ---

def submit(path, payload=None, fields=None, files=None):
    if payload is not None:
        data = api_json("POST", path, payload=payload)
    else:
        data = api_multipart(path, fields or {}, files or [])
    body = data.get("data") or data
    request_id = body.get("request_id") or body.get("id")
    if not request_id:
        # Some endpoints can answer synchronously (e.g. return_result_in_response).
        return None, body
    return request_id, body


def wait_for_job(request_id):
    log("job %s submitted; polling /api/v2/jobs/... every %ds (timeout %ds)"
        % (request_id, POLL_INTERVAL, POLL_TIMEOUT))
    deadline = time.time() + POLL_TIMEOUT
    while time.time() < deadline:
        data = api_json("GET", "/api/v2/jobs/%s" % request_id)
        job = data.get("data") or data
        status = str(job.get("status", "")).lower()
        progress = job.get("progress")
        if progress is not None:
            sys.stderr.write("\rstatus=%s progress=%s%%   " % (status, progress))
            sys.stderr.flush()
        if status == "done":
            sys.stderr.write("\n")
            return job
        if status in ("error", "failed"):
            sys.stderr.write("\n")
            raise ApiError("Job %s failed: %s" % (request_id, json.dumps(job, indent=2)))
        time.sleep(POLL_INTERVAL)
    raise ApiError("Timed out after %ds waiting for job %s" % (POLL_TIMEOUT, request_id))


def download(url, output_path):
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT) as resp:
        payload = resp.read()
    with open(output_path, "wb") as handle:
        handle.write(payload)
    return output_path


def deliver(job, output, default_name):
    """Print text results to stdout; download file results (URLs expire ~24h)."""
    text = job.get("result")
    url = job.get("result_url")
    if text and not url:
        if output:
            with open(output, "w", encoding="utf-8") as handle:
                handle.write(text if isinstance(text, str) else json.dumps(text, indent=2))
            log("saved: %s" % output)
        else:
            print(text if isinstance(text, str) else json.dumps(text, indent=2))
        return
    if url:
        url_path = urllib.parse.urlparse(url).path
        if not output and url_path.lower().endswith((".txt", ".json", ".srt", ".vtt")):
            # Text result (transcript/OCR): print instead of saving under a UUID name.
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT) as resp:
                sys.stdout.write(resp.read().decode("utf-8", "replace"))
            sys.stdout.write("\n")
            return
        if not output:
            output = os.path.basename(url_path) or default_name
        download(url, output)
        print(output)
        return
    print(json.dumps(job, indent=2))


def run_job(path, payload=None, fields=None, files=None, output=None, default_name="deapi-result.bin"):
    request_id, body = submit(path, payload=payload, fields=fields, files=files)
    if request_id is None:
        deliver(body, output, default_name)
        return
    job = wait_for_job(request_id)
    deliver(job, output, default_name)


# -------------------------------------------------------------- commands ---

def cmd_models(args):
    models = fetch_models()
    if args.type:
        wanted = args.type.lower()
        task = None
        for key, aliases in INFERENCE_ALIASES.items():
            if wanted == key or wanted in aliases:
                task = key
                break
        if task:
            models = models_for_task(models, task)
        else:
            models = [m for m in models
                      if wanted in {str(t).lower() for t in (m.get("inference_types") or [])}]
    if args.json:
        print(json.dumps(models, indent=2))
        return
    for model in models:
        print("%-40s %s" % (model.get("slug", "?"),
                            ",".join(model.get("inference_types") or [])))
    log("%d models (source: GET /api/v2/models)" % len(models))


def cmd_image(args):
    slug, model = resolve_model("image", args.model)
    defaults, limits, features = model_defaults(model)
    steps = args.steps if args.steps is not None else int(defaults.get("steps", 4))
    payload = {
        "prompt": args.prompt,
        "model": slug,
        "width": clamp(args.width, limits, "min_width", "max_width"),
        "height": clamp(args.height, limits, "min_height", "max_height"),
        "steps": clamp(steps, limits, "min_steps", "max_steps"),
        "seed": args.seed if args.seed is not None else random.randint(0, 2**31 - 1),
    }
    if features.get("supports_guidance", True):
        payload["guidance"] = (args.guidance if args.guidance is not None
                               else float(defaults.get("guidance", 3.5)))
    if args.negative and features.get("supports_negative_prompt", True):
        payload["negative_prompt"] = args.negative
    run_job("/api/v2/images/generations", payload=payload,
            output=args.output, default_name="deapi-image.png")


def cmd_edit(args):
    slug, _ = resolve_model("edit", args.model)
    fields = {"prompt": args.prompt, "model": slug}
    if args.steps is not None:
        fields["steps"] = args.steps
    if args.guidance is not None:
        fields["guidance"] = args.guidance
    fields["seed"] = args.seed if args.seed is not None else random.randint(0, 2**31 - 1)
    if len(args.image) == 1:
        files = [("image", args.image[0])]
    else:
        files = [("images[]", path) for path in args.image[:3]]
    run_job("/api/v2/images/edits", fields=fields, files=files,
            output=args.output, default_name="deapi-edit.png")


def cmd_tts(args):
    slug, model = resolve_model("tts", args.model)
    defaults, _, _ = model_defaults(model)
    fmt = args.format or defaults.get("format", "mp3")
    fields = {
        "text": args.text,
        "model": slug,
        "lang": args.lang or defaults.get("lang", "en-us"),
        "speed": args.speed if args.speed is not None else defaults.get("speed", 1),
        "format": fmt,
        "sample_rate": args.sample_rate or defaults.get("sample_rate", 24000),
    }
    files = []
    if args.clone_audio and args.instruct:
        raise ApiError("Use either --clone-audio (voice_clone) or --instruct "
                       "(voice_design), not both.")
    if args.clone_audio:
        fields["mode"] = "voice_clone"
        files.append(("ref_audio", args.clone_audio))
        if args.ref_text:
            fields["ref_text"] = args.ref_text
    elif args.instruct:
        fields["mode"] = "voice_design"
        fields["instruct"] = args.instruct
    else:
        fields["mode"] = "custom_voice"
        voice = args.voice or defaults.get("voice")
        if voice:
            fields["voice"] = voice
    run_job("/api/v2/audio/speech", fields=fields, files=files,
            output=args.output, default_name="deapi-speech." + fmt)


def cmd_stt(args):
    if bool(args.url) == bool(args.file):
        raise ApiError("Provide exactly one of --url or --file.")
    slug, _ = resolve_model("stt", args.model)
    fields = {"model": slug, "include_ts": args.timestamps}
    files = []
    if args.url:
        fields["source_url"] = args.url
    else:
        files.append(("source_file", args.file))
    run_job("/api/v2/audio/transcriptions", fields=fields, files=files,
            output=args.output, default_name="deapi-transcript.txt")


def cmd_ocr(args):
    slug, _ = resolve_model("ocr", args.model)
    fields = {"model": slug, "format": args.format}
    if args.language:
        fields["language"] = args.language
    run_job("/api/v2/images/ocr", fields=fields, files=[("image", args.image)],
            output=args.output, default_name="deapi-ocr.txt")


def cmd_rmbg(args):
    slug, _ = resolve_model("rmbg", args.model)
    run_job("/api/v2/images/background-removals", fields={"model": slug},
            files=[("image", args.image)],
            output=args.output, default_name="deapi-rmbg.png")


def cmd_upscale(args):
    slug, _ = resolve_model("upscale", args.model)
    fields = {"model": slug}
    if args.scale:
        fields["scale"] = args.scale
    run_job("/api/v2/images/upscales", fields=fields, files=[("image", args.image)],
            output=args.output, default_name="deapi-upscaled.png")


def cmd_music(args):
    slug, model = resolve_model("music", args.model)
    defaults, _, _ = model_defaults(model)
    fields = {
        "caption": args.caption,
        "model": slug,
        "lyrics": args.lyrics,
        "duration": args.duration,
        "inference_steps": args.steps if args.steps is not None
        else int(defaults.get("inference_steps", 8)),
        "guidance_scale": args.guidance if args.guidance is not None
        else float(defaults.get("guidance_scale", 7.5)),
        "seed": args.seed if args.seed is not None else random.randint(0, 2**31 - 1),
        "format": args.format,
    }
    if args.bpm:
        fields["bpm"] = args.bpm
    if args.keyscale:
        fields["keyscale"] = args.keyscale
    run_job("/api/v2/audio/music", fields=fields, files=[],
            output=args.output, default_name="deapi-music." + args.format)


def cmd_video(args):
    slug, model = resolve_model("video", args.model)
    defaults, limits, features = model_defaults(model)
    payload = {
        "prompt": args.prompt,
        "model": slug,
        "width": clamp(args.width, limits, "min_width", "max_width"),
        "height": clamp(args.height, limits, "min_height", "max_height"),
        "steps": args.steps if args.steps is not None else int(defaults.get("steps", 8)),
        "seed": args.seed if args.seed is not None else random.randint(0, 2**31 - 1),
        "frames": args.frames if args.frames is not None
        else int(defaults.get("frames", 97)),
    }
    if features.get("supports_guidance", True):
        payload["guidance"] = (args.guidance if args.guidance is not None
                               else float(defaults.get("guidance", 3.0)))
    if args.fps:
        payload["fps"] = args.fps
    if args.negative and features.get("supports_negative_prompt", True):
        payload["negative_prompt"] = args.negative
    run_job("/api/v2/videos/generations", payload=payload,
            output=args.output, default_name="deapi-video.mp4")


def cmd_animate(args):
    slug, model = resolve_model("animate", args.model)
    defaults, limits, features = model_defaults(model)
    frames = args.frames if args.frames is not None else int(defaults.get("frames", 97))
    fields = {
        "prompt": args.prompt,
        "model": slug,
        "width": clamp(args.width, limits, "min_width", "max_width"),
        "height": clamp(args.height, limits, "min_height", "max_height"),
        "steps": clamp(args.steps if args.steps is not None
                       else int(defaults.get("steps", 8)),
                       limits, "min_steps", "max_steps"),
        "guidance": args.guidance if args.guidance is not None
        else float(defaults.get("guidance", 3.0)),
        "seed": args.seed if args.seed is not None else random.randint(0, 2**31 - 1),
        "frames": clamp(frames, limits, "min_frames", "max_frames"),
        # fps is required by the live API even though the spec marks it optional
        "fps": clamp(args.fps if args.fps is not None
                     else int(defaults.get("fps", 24)),
                     limits, "min_fps", "max_fps"),
    }
    if args.negative and features.get("supports_negative_prompt", True):
        fields["negative_prompt"] = args.negative
    files = [("first_frame_image", args.image)]
    if args.last_image:
        files.append(("last_frame_image", args.last_image))
    run_job("/api/v2/videos/animations", fields=fields, files=files,
            output=args.output, default_name="deapi-animation.mp4")


def cmd_boost(args):
    task_to_type = {
        "image": "images.generations",
        "edit": "images.edits",
        "video": "videos.generations",
        "animate": "videos.animations",
        "music": "audio.music",
    }
    boost_type = task_to_type.get(args.type, args.type)
    fields = {"prompt": args.prompt, "type": boost_type}
    if args.model:
        fields["model_slug"] = args.model
    else:
        task = args.type if args.type in INFERENCE_ALIASES else None
        if task:
            slug, _ = resolve_model(task, None)
            fields["model_slug"] = slug
    if args.negative:
        fields["negative_prompt"] = args.negative
    files = [("image", args.image)] if args.image else []
    run_job("/api/v2/prompts/enhancements", fields=fields, files=files,
            output=args.output, default_name="deapi-boosted-prompt.txt")


def cmd_embed(args):
    slug, _ = resolve_model("embed", args.model)
    request_id, body = submit("/api/v2/embeddings",
                              payload={"input": args.input, "model": slug})
    if request_id:
        body = wait_for_job(request_id)
    print(json.dumps(body, indent=2))


def cmd_balance(_args):
    data = api_json("GET", "/api/v2/account/balance")
    print(json.dumps(data.get("data") or data, indent=2))


# ------------------------------------------------------------------- cli ---

def build_parser():
    parser = argparse.ArgumentParser(
        prog="deapi.py",
        description="deAPI v2 helper: media generation and processing. "
                    "Requires DEAPI_API_KEY in the environment.")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("models", help="list live models from GET /api/v2/models")
    p.add_argument("--type", help="filter by task or inference type "
                                  "(image, tts, stt, ocr, music, video, ... or raw type slug)")
    p.add_argument("--json", action="store_true", help="dump full model objects")
    p.set_defaults(func=cmd_models)

    p = sub.add_parser("image", help="text-to-image generation")
    p.add_argument("--prompt", required=True)
    p.add_argument("--model")
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--height", type=int, default=1024)
    p.add_argument("--steps", type=int)
    p.add_argument("--guidance", type=float)
    p.add_argument("--seed", type=int)
    p.add_argument("--negative", help="negative prompt (if the model supports it)")
    p.add_argument("--output", help="output file path")
    p.set_defaults(func=cmd_image)

    p = sub.add_parser("edit", help="image editing / img2img")
    p.add_argument("--prompt", required=True)
    p.add_argument("--image", action="append", required=True,
                   help="input image path (repeat up to 3 times)")
    p.add_argument("--model")
    p.add_argument("--steps", type=int)
    p.add_argument("--guidance", type=float)
    p.add_argument("--seed", type=int)
    p.add_argument("--output")
    p.set_defaults(func=cmd_edit)

    p = sub.add_parser("tts", help="text-to-speech")
    p.add_argument("--text", required=True)
    p.add_argument("--model")
    p.add_argument("--voice", help="voice slug; discover via models --type tts --json "
                                   "(languages field). Default: model's default voice")
    p.add_argument("--clone-audio", dest="clone_audio",
                   help="reference audio file -> voice_clone mode")
    p.add_argument("--ref-text", dest="ref_text",
                   help="transcript of the reference audio (improves cloning)")
    p.add_argument("--instruct",
                   help='voice description -> voice_design mode, e.g. '
                        '"a warm female voice with a British accent"')
    p.add_argument("--lang", help="language slug, e.g. en-us (default: model default)")
    p.add_argument("--speed", type=float)
    p.add_argument("--format", help="audio format (default: model default, usually mp3)")
    p.add_argument("--sample-rate", type=int, dest="sample_rate")
    p.add_argument("--output")
    p.set_defaults(func=cmd_tts)

    p = sub.add_parser("stt", help="transcribe audio/video (URL or local file)")
    p.add_argument("--url", help="YouTube / X / Twitch / Kick / TikTok / media URL")
    p.add_argument("--file", help="local audio (<=20MB) or video (<=50MB) file")
    p.add_argument("--model")
    p.add_argument("--timestamps", action="store_true", help="include timestamps")
    p.add_argument("--output")
    p.set_defaults(func=cmd_stt)

    p = sub.add_parser("ocr", help="extract text from an image")
    p.add_argument("--image", required=True)
    p.add_argument("--model")
    p.add_argument("--language")
    p.add_argument("--format", default="text", choices=["text", "json"])
    p.add_argument("--output")
    p.set_defaults(func=cmd_ocr)

    p = sub.add_parser("rmbg", help="remove image background")
    p.add_argument("--image", required=True)
    p.add_argument("--model")
    p.add_argument("--output")
    p.set_defaults(func=cmd_rmbg)

    p = sub.add_parser("upscale", help="upscale an image")
    p.add_argument("--image", required=True)
    p.add_argument("--model")
    p.add_argument("--scale", type=int, help="scale factor if the model supports it")
    p.add_argument("--output")
    p.set_defaults(func=cmd_upscale)

    p = sub.add_parser("music", help="music generation")
    p.add_argument("--caption", required=True, help="style/genre description")
    p.add_argument("--lyrics", default="[Instrumental]",
                   help='lyrics text; default "[Instrumental]"')
    p.add_argument("--duration", type=int, default=60, help="seconds (10-600)")
    p.add_argument("--model")
    p.add_argument("--steps", type=int)
    p.add_argument("--guidance", type=float)
    p.add_argument("--seed", type=int)
    p.add_argument("--bpm", type=int)
    p.add_argument("--keyscale", help='e.g. "C major"')
    p.add_argument("--format", default="mp3")
    p.add_argument("--output")
    p.set_defaults(func=cmd_music)

    p = sub.add_parser("video", help="text-to-video generation")
    p.add_argument("--prompt", required=True)
    p.add_argument("--model")
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--height", type=int, default=576)
    p.add_argument("--frames", type=int)
    p.add_argument("--fps", type=int)
    p.add_argument("--steps", type=int)
    p.add_argument("--guidance", type=float)
    p.add_argument("--seed", type=int)
    p.add_argument("--negative")
    p.add_argument("--output")
    p.set_defaults(func=cmd_video)

    p = sub.add_parser("animate", help="image-to-video (animate a still image)")
    p.add_argument("--prompt", required=True, help="motion/scene prompt")
    p.add_argument("--image", required=True, help="first frame image")
    p.add_argument("--last-image", dest="last_image",
                   help="optional last frame image (if the model supports it)")
    p.add_argument("--model")
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--height", type=int, default=576)
    p.add_argument("--frames", type=int)
    p.add_argument("--fps", type=int)
    p.add_argument("--steps", type=int)
    p.add_argument("--guidance", type=float)
    p.add_argument("--seed", type=int)
    p.add_argument("--negative")
    p.add_argument("--output")
    p.set_defaults(func=cmd_animate)

    p = sub.add_parser("boost", help="enhance a prompt via the deAPI prompt booster")
    p.add_argument("--prompt", required=True, help="prompt to enhance")
    p.add_argument("--type", default="image",
                   help="target task (image, edit, video, animate, music) or raw "
                        "v2 dot-notation type like images.generations")
    p.add_argument("--model", help="target model slug (default: auto from live list)")
    p.add_argument("--negative", help="negative prompt to enhance alongside")
    p.add_argument("--image", help="reference image (required for edit/animate types)")
    p.add_argument("--output")
    p.set_defaults(func=cmd_boost)

    p = sub.add_parser("embed", help="text embeddings")
    p.add_argument("--input", required=True)
    p.add_argument("--model")
    p.set_defaults(func=cmd_embed)

    p = sub.add_parser("balance", help="show account balance")
    p.set_defaults(func=cmd_balance)

    return parser


def main():
    args = build_parser().parse_args()
    try:
        args.func(args)
    except ApiError as exc:
        log("error: %s" % exc)
        sys.exit(1)
    except KeyboardInterrupt:
        log("\ninterrupted")
        sys.exit(130)


if __name__ == "__main__":
    main()
