"""Free Gemini vision client (Google AI Studio key) for two pipeline features.

1. ``describe_screenshot`` — image in -> structured topic out (screenshot
   ingestion). Drop a screenshot in the inbox and it becomes a captioned,
   high-priority topic for the personal brands.
2. ``qa_image`` — generated image in -> pass/fail visual critique (the visual-QA
   gate after a transplant render: garbled text, off-brief, wrong palette).

Image OUTPUT is paid-only on this tier (validated 2026-06-17); only vision INPUT
(image -> text) is free, which is all these features need. Generation stays on
FAL. The key is read from GEMINI_API_KEY / GOOGLE_AI_API_KEY at call time;
nothing is hardcoded. Every public function degrades to None / a neutral pass on
any failure so it never crashes the cron.
"""
from __future__ import annotations

import base64
import json
import os
import re
import sys
from typing import Optional

import config as cfg

_API_ROOT = "https://generativelanguage.googleapis.com/v1beta/models"

_MIME = {
    ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
    ".webp": "image/webp", ".gif": "image/gif", ".heic": "image/heic",
}

# Allowed pillars Gemini may pick for an ingested screenshot, per brand. Keeps
# the model's free-text suggestion mapped onto the real pillar vocabulary.
_SCREENSHOT_PILLARS = {
    "sahil_twitter": ("build_in_public", "ai_tools", "tutorial", "AI Patterns", "sly_product"),
    "sahil_linkedin": ("indie", "ai", "leadership", "pm_thought", "AI Engineering Notes"),
}
_DEFAULT_PILLARS = ("build_in_public", "tutorial", "product", "insight")


def _key() -> Optional[str]:
    """The Google AI Studio key, GEMINI_API_KEY preferred over GOOGLE_AI_API_KEY."""
    return (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_AI_API_KEY") or "").strip() or None


def available() -> bool:
    """True when vision is enabled and a key is present."""
    return bool(cfg.GEMINI_VISION_ENABLED and _key())


def _b64(path: str) -> Optional[tuple[str, str]]:
    """Return (mime, base64) for an image path, or None if unreadable."""
    try:
        ext = os.path.splitext(path)[1].lower()
        mime = _MIME.get(ext, "image/png")
        with open(path, "rb") as f:
            return mime, base64.b64encode(f.read()).decode("ascii")
    except Exception as exc:  # noqa: BLE001
        print(f"[gemini_vision] read failed {path}: {exc}", file=sys.stderr)
        return None


def _strip_json(text: str) -> str:
    """Pull a JSON object out of a model reply that may be fenced or chatty."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE).strip()
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start:end + 1]
    return text


def _call_vision(prompt: str, image_paths: list[str], *, model: Optional[str] = None,
                 timeout: int = 60, max_tokens: int = 2048) -> Optional[str]:
    """One Gemini generateContent call with image(s) + text. Returns text or None.

    Uses a proxy-free session: ~/.hermes/.env exports a Privoxy HTTP(S)_PROXY
    that is not always up and must not wrap this public API (the same footgun
    that silently broke llm_generate). trust_env=False ignores it.
    """
    key = _key()
    if not key:
        return None
    try:
        import requests
    except ImportError:
        return None

    parts: list[dict] = [{"text": prompt}]
    for p in image_paths:
        enc = _b64(p)
        if not enc:
            return None
        mime, data = enc
        parts.append({"inlineData": {"mimeType": mime, "data": data}})

    session = requests.Session()
    session.trust_env = False
    url = f"{_API_ROOT}/{model or cfg.GEMINI_VISION_MODEL}:generateContent"
    try:
        r = session.post(
            url,
            headers={"x-goog-api-key": key, "Content-Type": "application/json"},
            json={
                "contents": [{"parts": parts}],
                "generationConfig": {
                    "temperature": 0.2,
                    "maxOutputTokens": max_tokens,
                    # 2.5-flash spends output budget on hidden "thinking" first,
                    # which truncates the JSON we ask for. These are structured
                    # extraction tasks, not reasoning, so thinking is disabled.
                    "thinkingConfig": {"thinkingBudget": 0},
                    "responseMimeType": "application/json",
                },
            },
            timeout=timeout,
        )
        if r.status_code != 200:
            print(f"[gemini_vision] HTTP {r.status_code}: {r.text[:160]}", file=sys.stderr)
            return None
        cand = (r.json().get("candidates") or [{}])[0]
        out = "".join(
            p.get("text", "") for p in cand.get("content", {}).get("parts", [])
        )
        return out.strip() or None
    except Exception as exc:  # noqa: BLE001 (vision must degrade, not crash the cron)
        print(f"[gemini_vision] call failed: {exc}", file=sys.stderr)
        return None


# ── Feature 1: screenshot -> topic ────────────────────────────────────

def _map_pillar(suggested: str, brand: str) -> str:
    """Map the model's free-text pillar onto the brand's real vocabulary."""
    allowed = _SCREENSHOT_PILLARS.get(brand, _DEFAULT_PILLARS)
    s = (suggested or "").strip().lower()
    for p in allowed:
        if p.lower() == s or p.lower() in s or s in p.lower():
            return p
    return allowed[0]


def describe_screenshot(image_path: str, brand: str = "sahil_twitter") -> Optional[dict]:
    """Caption a screenshot into a topic dict (educational shape, screenshot context).

    Returns None when vision is unavailable or the call fails — the caller then
    simply skips this screenshot and uses its normal topic sources.
    """
    if not available():
        return None
    allowed = _SCREENSHOT_PILLARS.get(brand, _DEFAULT_PILLARS)
    prompt = (
        "You are a content strategist reviewing a screenshot a developer dropped "
        "in to turn into a social post. Look at the image and reply with ONLY a "
        "JSON object, no prose, with these keys:\n"
        '  "title": a short, specific post angle (<=70 chars, no hashtags)\n'
        '  "summary": 1-2 sentences describing what the screenshot actually shows '
        "(the real UI/feature/data visible — be concrete, do not invent metrics)\n"
        '  "pillar": one of ' + ", ".join(allowed) + "\n"
        '  "usable": true only if the screenshot has a clear, postable subject; '
        "false if it is blank, corrupt, or purely personal/sensitive.\n"
        "Ground everything in what is visible. Never invent numbers."
    )
    raw = _call_vision(prompt, [image_path])
    if not raw:
        return None
    try:
        data = json.loads(_strip_json(raw))
    except Exception as exc:  # noqa: BLE001
        print(f"[gemini_vision] bad screenshot JSON: {exc}", file=sys.stderr)
        return None
    if not isinstance(data, dict) or not data.get("usable", True):
        return None
    title = (data.get("title") or "").strip()
    summary = (data.get("summary") or "").strip()
    if not title and not summary:
        return None
    pillar = _map_pillar(data.get("pillar", ""), brand)
    return {
        "pillar": pillar,
        "topic": (title or summary)[:120],
        "title": title[:120],
        "educational": True,
        "context": summary,
        "kb_snippets": [],
        "source": "screenshot",
        "screenshot_path": image_path,
    }


# ── Feature 2: visual QA of a generated image ─────────────────────────

def qa_image(image_path: str, brief: dict) -> dict:
    """Critique a finished render against its brief.

    Returns {"passed", "score" (0-10), "issues" [str], "available" bool, "raw"}.
    When vision is unavailable the result is a neutral PASS (available=False) so
    the pipeline is never blocked by a missing key.
    """
    neutral = {"passed": True, "score": 10, "issues": [], "available": False, "raw": ""}
    if not available():
        return neutral
    title = (brief.get("title") or "").strip()
    content = (brief.get("content") or brief.get("concept") or "").strip()
    palette = (brief.get("palette") or "").strip()
    kind = brief.get("kind", "infographic")
    prompt = (
        f"You are a strict art director QA-ing a generated {kind} for a personal "
        "tech brand. Judge ONLY the attached image against this brief and reply "
        "with ONLY a JSON object, no prose:\n"
        '  "score": integer 0-10 overall quality\n'
        '  "issues": array of short strings for any real problems\n'
        "Check, in order of severity:\n"
        "1. Text legibility — any garbled, misspelled, or nonsense lettering is a "
        "major fault (list it).\n"
        f"2. On-brief — does it depict: title \"{title}\"; content: {content}.\n"
        f"3. Palette — should sit within: {palette}.\n"
        "4. No anime mascot, no identifiable real person, not generic flat stock.\n"
        "5. Craft — composition, type hierarchy, intentional density.\n"
        "Be harsh on garbled text and off-brief; lenient on subjective taste."
    )
    raw = _call_vision(prompt, [image_path])
    if not raw:
        return neutral
    try:
        data = json.loads(_strip_json(raw))
        score = int(data.get("score", 10))
        issues = [str(i) for i in (data.get("issues") or []) if str(i).strip()]
    except Exception as exc:  # noqa: BLE001
        # The call ran but the reply was unparseable; don't block the pipeline,
        # but report it honestly as available so it is not mistaken for "no key".
        print(f"[gemini_vision] bad QA JSON: {exc}", file=sys.stderr)
        return {"passed": True, "score": 10, "issues": ["QA reply unparseable"],
                "available": True, "raw": raw}
    score = max(0, min(10, score))
    return {
        "passed": score >= cfg.IMAGERY_QA_MIN_SCORE,
        "score": score,
        "issues": issues,
        "available": True,
        "raw": raw,
    }
