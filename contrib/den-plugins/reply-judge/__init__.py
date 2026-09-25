"""reply-judge — keep chat replies short and plain.

Fires on ``transform_llm_output`` (once per turn, after tools). If the reply is
over the length budget on a chat platform, an auxiliary-model judge rewrites it
into <=5 plain-English sentences, result/blocker first. Short replies, code
blocks, tables and MEDIA: paths pass through untouched.

Fail-open: any judge error returns None (original text delivered).
"""
from __future__ import annotations

import logging
import re
import time

logger = logging.getLogger(__name__)

_DEFAULT_MAX_CHARS = 700
_DEFAULT_PLATFORMS = ("telegram",)
_TASK_KEY = "reply_judge"
_STATS = {"seen": 0, "rewritten": 0, "skipped": 0, "errors": 0}

_JUDGE_SYSTEM = (
    "You are a strict editor for a busy engineering founder who says: "
    "'please reply short, I don't have time to read all this fluff'.\n"
    "Rewrite the assistant reply below into 1-5 short plain-English sentences (or <=5 bullets).\n"
    "Rules: keep the verified result or concrete blocker first; keep every command, URL, "
    "PR/task id, number and MEDIA: path exactly as written; drop explanations of mechanisms, "
    "history, apologies, hedging and 'what this means' prose; no headers; no preamble. "
    "If the reply asks the user a question or offers choices, keep that in one final line. "
    "Output only the rewritten reply."
)


def _cfg():
    try:
        from hermes_cli.config import load_config  # type: ignore
        cfg = load_config() or {}
        section = cfg.get("reply_judge") or {}
        max_chars = int(section.get("max_chars", _DEFAULT_MAX_CHARS))
        platforms = section.get("platforms") or list(_DEFAULT_PLATFORMS)
        if isinstance(platforms, str):
            platforms = [platforms]
        enabled = bool(section.get("enabled", True))
        return enabled, max_chars, {p.lower() for p in platforms}
    except Exception:
        return True, _DEFAULT_MAX_CHARS, set(_DEFAULT_PLATFORMS)


def _protected(text: str) -> bool:
    """Content the judge must not touch: code blocks, tables, media, long URLs lists."""
    if "```" in text:
        return True
    if re.search(r"^\s*\|.*\|\s*$", text, flags=re.M):
        return True
    if "MEDIA:" in text:
        return True
    return False


def _extract(response) -> str:
    try:
        return (response.choices[0].message.content or "").strip()
    except Exception:
        pass
    try:
        return (response["choices"][0]["message"]["content"] or "").strip()
    except Exception:
        return ""


def _judge(text: str) -> str:
    from agent.auxiliary_client import call_llm

    resp = call_llm(
        task=_TASK_KEY,
        messages=[
            {"role": "system", "content": _JUDGE_SYSTEM},
            {"role": "user", "content": text},
        ],
        temperature=0,
        max_tokens=400,
        timeout=25,
    )
    return _extract(resp)


def _on_transform(response_text: str = "", platform: str = "", **_kw):
    _STATS["seen"] += 1
    enabled, max_chars, platforms = _cfg()
    if not enabled or not response_text:
        return None
    if platform and platform.lower() not in platforms:
        return None
    if len(response_text) <= max_chars or _protected(response_text):
        _STATS["skipped"] += 1
        return None
    t0 = time.time()
    try:
        short = _judge(response_text)
    except Exception as exc:  # fail open
        _STATS["errors"] += 1
        logger.warning("reply-judge: judge failed (%s); delivering original", exc)
        return None
    # Sanity: must be shorter and non-trivial, and keep any URLs/ids from the original.
    if not short or len(short) >= len(response_text) or len(short) < 20:
        _STATS["skipped"] += 1
        return None
    refs = re.findall(r"https?://\S+|\bt_[0-9a-f]{8}\b|#\d{3,6}\b", response_text)
    missing = [r for r in dict.fromkeys(refs) if r not in short]
    if missing:
        short = short.rstrip() + "\nRefs: " + " ".join(missing)
    _STATS["rewritten"] += 1
    logger.info(
        "reply-judge: rewrote %d -> %d chars in %.1fs (platform=%s)",
        len(response_text), len(short), time.time() - t0, platform,
    )
    return short


def register(ctx):
    ctx.register_hook("transform_llm_output", _on_transform)
