"""
Sticker tag middleware for Weixin platform.

Scans LLM response text for %emotion% tags, selects a random sticker image
from the corresponding mood directory, and returns cleaned text + image path.

Design: LLM embeds tags like %愉快% in its reply. The gateway send layer
intercepts, strips the tag, and optionally sends a sticker image.
This avoids the LLM tool-calling loop that caused repeated speech issues.

Two sticker sources (auto-discovered):
  1. ~/.hermes/stickers/{mood}/     — new directory structure (subdirs)
  2. ~/.hermes/output/stickers/     — old flat files (mood_N.ext)
"""

from __future__ import annotations

import glob
import logging
import os
import random
import re
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

# Sticker base directory — all stickers stored as subdirectories by mood
STICKER_DIR = os.path.expanduser("~/.hermes/stickers")
TAG_PATTERN = re.compile(r"%([一-鿿]+)%")

# Probability of actually sending a sticker (0.0 = never, 1.0 = always)
SEND_PROBABILITY = 0.5


def _discover_moods() -> dict[str, list[str]]:
    """Dynamically scan sticker directory and build mood -> files map.

    Returns dict of mood_name -> list of image file paths.
    Each subdirectory under STICKER_DIR is a mood category.
    """
    moods: dict[str, list[str]] = {}
    image_exts = (".jpg", ".jpeg", ".gif", ".png", ".webp")

    if not os.path.isdir(STICKER_DIR):
        return moods

    for entry in os.listdir(STICKER_DIR):
        subdir = os.path.join(STICKER_DIR, entry)
        if not os.path.isdir(subdir):
            continue
        files = []
        for ext in image_exts:
            files.extend(glob.glob(os.path.join(subdir, f"*{ext}")))
        if files:
            moods[entry] = files

    return moods


def scan_sticker_tags(text: str) -> Tuple[str, Optional[str]]:
    """
    Scan text for %emotion% sticker tags.

    Returns:
        (cleaned_text, sticker_path_or_None)

    If a valid tag is found and a sticker image exists, there is a 50% chance
    the sticker path is returned. The tag is always stripped from the text.
    Mood directories are discovered dynamically — no hardcoded mapping.
    """
    match = TAG_PATTERN.search(text)
    if not match:
        return text, None

    tag = match.group(1)

    # Dynamic discovery — finds all moods from both directories
    moods = _discover_moods()

    # Try exact match first, then partial match
    files = moods.get(tag)
    if not files:
        # Partial match: tag contained in mood name or vice versa
        for mood_name, mood_files in moods.items():
            if tag in mood_name or mood_name in tag:
                files = mood_files
                break

    if not files:
        logger.debug("No stickers found for tag: %s (available: %s)", tag, list(moods.keys()))
        cleaned = TAG_PATTERN.sub("", text).strip()
        return cleaned, None

    sticker_path = random.choice(files)

    # Always strip the tag from text
    cleaned = TAG_PATTERN.sub("", text).strip()

    # 50% probability gate
    if random.random() > SEND_PROBABILITY:
        logger.debug("Sticker suppressed by probability gate (%s)", tag)
        return cleaned, None

    logger.debug("Sticker selected: %s -> %s", tag, sticker_path)
    return cleaned, sticker_path


def is_animated_gif(path: str) -> bool:
    """Check if a GIF file has multiple frames (is animated)."""
    if not path.lower().endswith(".gif"):
        return False
    try:
        with open(path, "rb") as f:
            data = f.read()
        # GIF89a header + multiple image blocks = animated
        # Simple heuristic: count GIF image descriptor markers (0x2C)
        return data.count(b"\x2c") > 1
    except Exception:
        return False
