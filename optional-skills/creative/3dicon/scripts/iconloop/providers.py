"""Image generation across three backends, with the model resolved at runtime.

Pinning a model id in source is how these pipelines rot: the id you hardcode is
retired, or a better one ships and nobody notices. Each backend therefore asks
the provider what it actually has and picks the newest, unless the caller
overrides with ICONLOOP_IMAGE_MODEL.
"""
import base64
import os
import re

from . import config, http

PORTRAIT = "1024x1024"

# Same-version tie-breaks, best first. Only consulted among models that share
# the highest version number the account can see.
OPENAI_PREFERENCE = ("sunburst", "flare")


def _newest_openai(models):
    """Pick the highest-versioned gpt-image-* id, ignoring dated snapshots.

    Ids look like gpt-image-2.5-sunburst and gpt-image-2.5-sunburst-2026-09-08.
    The undated alias is the one to use: it keeps following the family.
    """
    best, best_ver = None, ()
    for m in models:
        if not m.startswith("gpt-image-"):
            continue
        if re.search(r"-\d{4}-\d{2}-\d{2}$", m):  # dated snapshot
            continue
        mv = re.match(r"gpt-image-(\d+(?:\.\d+)?)", m)
        if not mv:
            continue
        ver = tuple(int(x) for x in mv.group(1).split("."))
        suffix = m[mv.end():].lstrip("-")
        rank = OPENAI_PREFERENCE.index(suffix) if suffix in OPENAI_PREFERENCE else len(OPENAI_PREFERENCE)
        cand = (ver, -rank)
        if best is None or cand > best_ver:
            best, best_ver = m, cand
    return best


def resolve_model(backend):
    override = config.opt("ICONLOOP_IMAGE_MODEL")
    if override:
        return override

    if backend == "openai":
        k = config.key("OPENAI_API_KEY", "generate the still with GPT Image")
        data = http.get_json("https://api.openai.com/v1/models",
                             {"Authorization": f"Bearer {k}"})
        ids = [m["id"] for m in data.get("data", [])]
        pick = _newest_openai(ids)
        if not pick:
            raise SystemExit("No gpt-image-* model is available on this account.")
        return pick

    if backend == "gemini":
        k = config.key("GOOGLE_API_KEY", "generate the still with Gemini")
        data = http.get_json(
            f"https://generativelanguage.googleapis.com/v1beta/models?key={k}&pageSize=200")
        ids = [m["name"].split("/")[-1] for m in data.get("models", [])]
        cands = [i for i in ids if "image" in i and i.startswith("gemini")]
        # Highest generation number wins; "pro" beats "flash" at equal generation.
        def rank(i):
            g = re.search(r"gemini-(\d+)", i)
            return (int(g.group(1)) if g else 0, "pro" in i)
        if not cands:
            raise SystemExit("No Gemini image model is available on this key.")
        return sorted(cands, key=rank)[-1]

    if backend == "openrouter":
        # This used to refuse and ask the caller to name a model, which just
        # moved the guessing one level up — and a guessed id fails at the API
        # rather than here. OpenRouter publishes its image catalogue, so pick
        # from what actually exists.
        k = config.key("OPENROUTER_API_KEY", "generate the still via OpenRouter")
        data = http.get_json("https://openrouter.ai/api/v1/images/models",
                             {"Authorization": f"Bearer {k}"})
        ids = [m.get("id") or m.get("slug") or "" for m in (data.get("data") or data.get("models") or [])]
        gpt = [i for i in ids if i.startswith("openai/gpt-image-")]
        if gpt:
            pick = _newest_openai([i.split("/", 1)[1] for i in gpt])
            if pick:
                return "openai/" + pick
        gem = [i for i in ids if i.startswith("google/") and "image" in i]
        if gem:
            return sorted(gem)[-1]
        raise SystemExit(
            "No gpt-image or Gemini image model is listed on this OpenRouter "
            "account. Set ICONLOOP_IMAGE_MODEL explicitly — the catalogue is at "
            "https://openrouter.ai/collections/image-models")

    raise SystemExit(f"Unknown image backend: {backend}")


def generate(prompt, out_path, references=(), backend=None, model=None, size=PORTRAIT):
    """Generate one image, optionally guided by reference images. Returns the model id."""
    backend = backend or config.opt("ICONLOOP_IMAGE_BACKEND", "openai")
    model = model or resolve_model(backend)
    fn = {"openai": _openai, "gemini": _gemini, "openrouter": _openrouter}[backend]
    b64 = fn(prompt, references, model, size)
    with open(out_path, "wb") as f:
        f.write(base64.b64decode(b64))
    return model


def _openai(prompt, references, model, size):
    k = config.key("OPENAI_API_KEY", "generate the still with GPT Image")
    h = {"Authorization": f"Bearer {k}"}
    if references:
        # /images/edits is the endpoint that accepts input images, and
        # input_fidelity=high is what holds an object's identity across a run.
        fields = {"model": model, "prompt": prompt, "size": size,
                  "quality": "high", "input_fidelity": "high", "background": "transparent"}
        files = [("image[]", p) for p in references]
        d = http.post_multipart("https://api.openai.com/v1/images/edits", fields, files, h)
    else:
        d = http.post_json("https://api.openai.com/v1/images/generations", {
            "model": model, "prompt": prompt, "size": size,
            "quality": "high", "background": "transparent",
        }, h)
    return d["data"][0]["b64_json"]


def _gemini(prompt, references, model, size):
    k = config.key("GOOGLE_API_KEY", "generate the still with Gemini")
    parts = []
    for p in references:
        mime = "image/jpeg" if p.lower().endswith((".jpg", ".jpeg")) else "image/png"
        parts.append({"inline_data": {"mime_type": mime,
                                      "data": base64.b64encode(open(p, "rb").read()).decode()}})
    parts.append({"text": prompt})
    url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
           f"{model}:generateContent?key={k}")
    d = http.post_json(url, {
        "contents": [{"parts": parts}],
        "generationConfig": {"responseModalities": ["IMAGE"],
                             "imageConfig": {"aspectRatio": "1:1", "imageSize": "2K"}},
    })
    for cand in d.get("candidates", []):
        for part in cand.get("content", {}).get("parts", []):
            if "inlineData" in part:
                return part["inlineData"]["data"]
            if "inline_data" in part:
                return part["inline_data"]["data"]
    raise RuntimeError("Gemini returned no image (safety block, or a text-only reply).")


def _openrouter(prompt, references, model, size):
    k = config.key("OPENROUTER_API_KEY", "generate the still via OpenRouter")
    payload = {"model": model, "prompt": prompt}
    if references:
        payload["input_references"] = [
            "data:image/png;base64," + base64.b64encode(open(p, "rb").read()).decode()
            for p in references
        ]
    d = http.post_json("https://openrouter.ai/api/v1/images", payload,
                       {"Authorization": f"Bearer {k}"})
    # The unified Image API returns base64 payloads; shapes vary a little by
    # provider, so accept the two documented spellings rather than one.
    for item in d.get("data", []):
        if item.get("b64_json"):
            return item["b64_json"]
        if item.get("image_url", {}).get("url", "").startswith("data:"):
            return item["image_url"]["url"].split(",", 1)[1]
    raise RuntimeError(f"OpenRouter returned no image payload: {str(d)[:300]}")
