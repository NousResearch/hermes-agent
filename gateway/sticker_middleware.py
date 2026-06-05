"""
Sticker tag middleware for Weixin platform.

Scans LLM response text for %emotion% tags, selects a random sticker image
from the corresponding mood directory, and returns cleaned text + image path.

Design: LLM embeds tags like %愉快% in its reply. The gateway send layer
intercepts, strips the tag, and optionally sends a sticker image.
This avoids the LLM tool-calling loop that caused repeated speech issues.
"""

from __future__ import annotations

import glob
import logging
import os
import random
import re
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

STICKER_DIR = os.path.expanduser("~/.hermes/stickers")
TAG_PATTERN = re.compile(r"%([一-鿿]+)%")

# Chinese tag → mood directory mapping
MOOD_MAP: dict[str, str] = {
    "愉快": "joy",
    "开心": "joy",
    "高兴": "joy",
    "难过": "sad",
    "伤心": "sad",
    "遗憾": "sad",
    "无语": "speechless",
    "无奈": "speechless",
    "惊讶": "surprised",
    "震惊": "surprised",
    "疑惑": "doubt",
    "困惑": "doubt",
    "安慰": "comfort",
    "鼓励": "comfort",
    "害羞": "shy",
    "不好意思": "shy",
    "撒娇": "shy",
}

# Probability of actually sending a sticker (0.0 = never, 1.0 = always)
SEND_PROBABILITY = 0.5


def scan_sticker_tags(text: str) -> Tuple[str, Optional[str]]:
    """
    Scan text for %emotion% sticker tags.

    Returns:
        (cleaned_text, sticker_path_or_None)

    If a valid tag is found and a sticker image exists, there is a 50% chance
    the sticker path is returned. The tag is always stripped from the text.
    """
    match = TAG_PATTERN.search(text)
    if not match:
        return text, None

    tag = match.group(1)
    mood_dir = MOOD_MAP.get(tag)
    if not mood_dir:
        logger.debug("Unknown sticker tag: %s", tag)
        return text, None

    sticker_path = _pick_random_sticker(mood_dir)
    if not sticker_path:
        logger.debug("No stickers available for mood: %s", mood_dir)
        cleaned = TAG_PATTERN.sub("", text).strip()
        return cleaned, None

    # Always strip the tag from text
    cleaned = TAG_PATTERN.sub("", text).strip()

    # 50% probability gate
    if random.random() > SEND_PROBABILITY:
        logger.debug("Sticker suppressed by probability gate (%s)", tag)
        return cleaned, None

    logger.debug("Sticker selected: %s -> %s", tag, sticker_path)
    return cleaned, sticker_path


def _pick_random_sticker(mood: str) -> Optional[str]:
    """Pick a random image file from the given mood directory."""
    mood_dir = os.path.join(STICKER_DIR, mood)
    if not os.path.isdir(mood_dir):
        return None

    files: list[str] = []
    for ext in ("*.jpg", "*.jpeg", "*.gif", "*.png", "*.webp"):
        files.extend(glob.glob(os.path.join(mood_dir, ext)))

    return random.choice(files) if files else None
