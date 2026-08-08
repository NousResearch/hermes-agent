# ---------------------------------------------------------------------------
# Anthropic-compatible endpoint detection + image block conversion
# ---------------------------------------------------------------------------

# Providers that use Anthropic-compatible endpoints (via OpenAI SDK wrapper).
# Their image content blocks must use Anthropic format, not OpenAI format.
_ANTHROPIC_COMPAT_PROVIDERS = frozenset({"minimax", "minimax-oauth", "minimax-cn"})


def _is_anthropic_compat_endpoint(provider: str, base_url: str) -> bool:
    """Detect if an endpoint expects Anthropic-format content blocks.

    Returns True for known Anthropic-compatible providers (MiniMax) and
    any endpoint whose URL contains ``/anthropic`` in the path.
    """
    if provider in _ANTHROPIC_COMPAT_PROVIDERS:
        return True
    url_lower = (base_url or "").lower()
    return "/anthropic" in url_lower


def _convert_openai_images_to_anthropic(messages: list) -> list:
    """Convert OpenAI ``image_url``/``video_url`` blocks to Anthropic format.

    Converts:
    - ``image_url`` blocks to Anthropic ``image`` blocks
    - ``video_url`` blocks to Anthropic ``video`` blocks (MiniMax M3 compat)

    Only touches messages that have list-type content with ``image_url`` or
    ``video_url`` blocks; plain text messages pass through unchanged.
    """
    converted = []
    for msg in messages:
        content = msg.get("content")
        if not isinstance(content, list):
            converted.append(msg)
            continue
        new_content = []
        changed = False
        for block in content:
            if block.get("type") == "image_url":
                image_url_val = (block.get("image_url") or {}).get("url", "")
                if image_url_val.startswith("data:"):
                    # Parse data URI: data:<media_type>;base64,<data>
                    header, _, b64data = image_url_val.partition(",")
                    media_type = "image/png"
                    if ":" in header and ";" in header:
                        media_type = header.split(":", 1)[1].split(";", 1)[0]
                    new_content.append({
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": b64data,
                        },
                    })
                else:
                    # URL-based image
                    new_content.append({
                        "type": "image",
                        "source": {
                            "type": "url",
                            "url": image_url_val,
                        },
                    })
                changed = True
            elif block.get("type") == "video_url":
                # MiniMax's Anthropic-compatible endpoint expects a "video"
                # block (not OpenAI's "video_url", and not "input_video").
                # See https://platform.minimax.io/docs/api-reference/text-anthropic-api
                # — the Messages-field table lists type="video" (M3 only,
                # URL/base64/mm_file://). The source shape mirrors the "image"
                # block: base64 → {type:"base64", media_type, data}, URL →
                # {type:"url", url}.
                video_url_val = (block.get("video_url") or {}).get("url", "")
                if video_url_val.startswith("data:"):
                    # Parse data URI: data:<media_type>;base64,<data>
                    header, _, b64data = video_url_val.partition(",")
                    media_type = "video/mp4"
                    if ":" in header and ";" in header:
                        media_type = header.split(":", 1)[1].split(";", 1)[0]
                    new_content.append({
                        "type": "video",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": b64data,
                        },
                    })
                else:
                    # URL-based video
                    new_content.append({
                        "type": "video",
                        "source": {
                            "type": "url",
                            "url": video_url_val,
                        },
                    })
                changed = True
            else:
                new_content.append(block)
        converted.append({**msg, "content": new_content} if changed else msg)
    return converted
