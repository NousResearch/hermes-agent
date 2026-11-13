"""Download side of ``SlackAdapter.send_multiple_images``: one ``files_upload_v2`` batch."""

import asyncio
import logging
import os
from typing import Any

from hermes_agent.gateway.platforms.base import read_httpx_body_with_limit, ssrf_redirect_guard

logger = logging.getLogger("plugins.platforms.slack.adapter")

# content-type substring → upload filename extension (first match wins; default png).
_IMAGE_CT_EXTS = (("jpeg", "jpg"), ("jpg", "jpg"), ("gif", "gif"), ("webp", "webp"))


async def collect_image_uploads(
    chunk: list[tuple[str, str]], unquote_fn, is_safe_url_fn, client_factory, safe_url_for_log
) -> tuple[list[dict[str, Any]], list[str], list[tuple[str, str]]]:
    """``files_upload_v2`` entries for one batch: ``file://`` by path, remote via the SSRF-safe
    client. Returns ``(file_uploads, alt_texts, refused)``; ``refused`` holds the remote
    ``(url, alt)`` pairs that were unsafe, failed or over ``gateway.max_inbound_media_bytes``,
    so the caller can keep them as text instead of losing them."""
    file_uploads: list[dict[str, Any]] = []
    initial_comment_parts: list[str] = []
    refused: list[tuple[str, str]] = []
    async with client_factory(
        timeout=30.0, follow_redirects=True, event_hooks={"response": [ssrf_redirect_guard]}
    ) as http_client:
        for image_url, alt_text in chunk:
            if image_url.startswith("file://"):
                if alt_text:
                    initial_comment_parts.append(alt_text)
                local_path = unquote_fn(image_url[7:])
                if not await asyncio.to_thread(os.path.exists, local_path):
                    logger.warning("[Slack] Skipping missing image: %s", local_path)
                    continue
                file_uploads.append({"file": local_path, "filename": os.path.basename(local_path)})
                continue
            if not is_safe_url_fn(image_url):
                logger.warning("[Slack] Blocked unsafe image URL in batch")
                refused.append((image_url, alt_text))
                continue
            try:
                async with http_client.stream("GET", image_url) as response:
                    response.raise_for_status()
                    ct = response.headers.get("content-type", "")
                    content = await read_httpx_body_with_limit(response, media_type="image")
            except Exception as dl_err:
                logger.warning("[Slack] Download failed for %s: %s", safe_url_for_log(image_url), dl_err, exc_info=True)
                refused.append((image_url, alt_text))
                continue
            if alt_text:
                initial_comment_parts.append(alt_text)
            ext = next((e for k, e in _IMAGE_CT_EXTS if k in ct), "png")
            file_uploads.append({"content": content, "filename": f"image_{len(file_uploads)}.{ext}"})
    return file_uploads, initial_comment_parts, refused
