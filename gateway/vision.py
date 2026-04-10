"""
Shared vision-enrichment helper for gateway adapters.

Converts a list of image paths/URLs into text descriptions using the
vision_analyze tool, then prepends them to the user's message text.
This is used by platform adapters (Telegram, WhatsApp, etc.) and the
API server to give the conversation model image context without requiring
multimodal LLM support.
"""

import logging
from typing import List

logger = logging.getLogger(__name__)

_ANALYSIS_PROMPT = (
    "Describe everything visible in this image in thorough detail. "
    "Include any text, code, data, objects, people, layout, colors, "
    "and any other notable visual information."
)


async def enrich_message_with_vision(user_text: str, media_urls: List[str]) -> str:
    """Analyze images and prepend descriptions to *user_text*.

    Args:
        user_text:  The user's original caption / message text.
        media_urls: Local file paths or HTTP URLs to images.

    Returns:
        Enriched message string with vision descriptions prepended,
        or the original *user_text* if no images could be processed.
    """
    if not media_urls:
        return user_text

    from tools.vision_tools import vision_analyze_tool
    import json as _json

    enriched_parts: List[str] = []
    for url in media_urls:
        try:
            logger.debug("Auto-analyzing user image: %s", url)
            result_json = await vision_analyze_tool(image_url=url, user_prompt=_ANALYSIS_PROMPT)
            result = _json.loads(result_json)
            if result.get("success"):
                description = result.get("analysis", "")
                enriched_parts.append(
                    f"[The user sent an image~ Here's what I can see:\n{description}]\n"
                    f"[If you need a closer look, use vision_analyze with image_url: {url} ~]"
                )
            else:
                enriched_parts.append(
                    "[The user sent an image but I couldn't quite see it "
                    f"this time (>_<) You can try looking at it yourself "
                    f"with vision_analyze using image_url: {url}]"
                )
        except Exception as exc:
            logger.error("Vision auto-analysis error for %s: %s", url, exc)
            enriched_parts.append(
                f"[The user sent an image but something went wrong when I "
                f"tried to look at it~ You can try examining it yourself "
                f"with vision_analyze using image_url: {url}]"
            )

    if not enriched_parts:
        return user_text

    prefix = "\n\n".join(enriched_parts)
    return f"{prefix}\n\n{user_text}" if user_text else prefix
