"""Article delivery — paste-ready bundle on Discord approval.

On approve (manual reply on the preview card or a future programmatic hook),
the orchestrator calls `ready(bundle)` to post the body in chunks + attach
the illustrations. The user pastes the body into X's article composer and
uploads the images manually. On reject, `discard(bundle)` removes the bundle
from disk.
"""
from __future__ import annotations
import os
import shutil
from pathlib import Path
from typing import Optional

import discord_digest as dd
from discord_digest import _post, _channel_type, _token, ArticleBundle


CONTENT_LIMIT = 1900  # shared with discord_digest; the article delivery sends chunks.


def _chunk(text: str, n: int) -> list[str]:
    """Split text into <= n-char chunks on paragraph boundaries where possible."""
    if len(text) <= n:
        return [text]
    chunks: list[str] = []
    remaining = text
    while len(remaining) > n:
        cut = remaining.rfind("\n\n", 0, n)
        if cut == -1:
            cut = remaining.rfind("\n", 0, n)
        if cut == -1:
            cut = n
        chunks.append(remaining[:cut].rstrip())
        remaining = remaining[cut:].lstrip()
    if remaining:
        chunks.append(remaining)
    return chunks


def ready(bundle: ArticleBundle, channel_id: str = dd.DEFAULT_CHANNEL) -> Optional[str]:
    """Post the paste-ready body and all images. Returns the target channel/thread id.

    On a forum channel (type 15), the calls happen inside a new thread so
    the paste-ready text doesn't clutter the main forum index.
    """
    if not _token():
        print("[article_delivery] DISCORD_BOT_TOKEN not set, cannot deliver.")
        return None

    target = channel_id
    if _channel_type(channel_id) == 15:
        title = f"Ready to publish · {bundle.title[:60]}"
        preview = f"**{bundle.title}** is approved. Body in this thread."
        thread_id = dd._create_forum_thread(channel_id, title, preview)
        if not thread_id:
            return None
        target = thread_id
    else:
        _post(target, f"📰 **Ready to publish** · **{bundle.title}**")

    # Send the body in chunks.
    for chunk in _chunk(bundle.article_md, CONTENT_LIMIT):
        _post(target, chunk)

    # Attach each image as a follow-up message.
    for img in bundle.image_paths or []:
        if Path(img).exists():
            _post(target, f"Image: {Path(img).name}", file_path=str(img))

    # Final pointer.
    _post(target, f"Bundle on disk: `{bundle.dir}`  ·  article.md: `{bundle.article_md_path}`")
    return target


def discard(bundle: ArticleBundle) -> None:
    """Remove the bundle from disk (reject path)."""
    if not bundle.dir:
        return
    try:
        shutil.rmtree(bundle.dir, ignore_errors=True)
        print(f"[article_delivery] discarded bundle: {bundle.dir}")
    except Exception as exc:  # noqa: BLE001
        print(f"[article_delivery] discard failed: {exc}")
