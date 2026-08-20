"""Zulip upload parsing, downloading, and agent tool support.

Zulip represents files in messages as ``/user_uploads/...`` Markdown links.
The link is not directly usable by the agent: retrieving its bytes requires
the bot's API credentials to obtain a short-lived signed URL first.  This
module keeps that credential-bearing flow inside the platform plugin.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from urllib.parse import unquote

logger = logging.getLogger(__name__)

_USER_UPLOAD_LINK_RE = re.compile(
    r"\(((?:https?://[^()\s]+)?/user_uploads/[^()\s]+)\)"
)

# Keep inbound downloads bounded. This is deliberately an implementation
# safeguard, not a user-facing environment setting; platform behaviour is
# configured in config.yaml rather than a new HERMES_* variable.
MAX_UPLOAD_DOWNLOAD_BYTES = 25 * 1024 * 1024


@dataclass(frozen=True)
class DownloadedUpload:
    """A Zulip upload materialized into Hermes's media cache."""

    path: str
    filename: str
    mime_type: str
    size_bytes: int


def extract_upload_paths(content: str) -> list[str]:
    """Return unique, normalized Zulip ``/user_uploads/`` paths in order.

    Absolute URLs are reduced to a path before use, so the authenticated first
    hop always targets the configured Zulip realm rather than a host embedded
    in user-controlled Markdown.
    """
    paths: list[str] = []
    seen: set[str] = set()
    for match in _USER_UPLOAD_LINK_RE.finditer(content or ""):
        target = match.group(1)
        path = target[target.find("/user_uploads/"):].split("?", 1)[0]
        if not is_valid_upload_path(path) or path in seen:
            continue
        seen.add(path)
        paths.append(path)
    return paths


def is_valid_upload_path(path: str) -> bool:
    """Return whether *path* is a non-traversing Zulip upload path."""
    if not isinstance(path, str) or not path.startswith("/user_uploads/"):
        return False
    segments = unquote(path).split("/")
    # Expected shape: /user_uploads/<realm_id>/<filename-or-subpath>.
    return (
        len(segments) >= 4
        and segments[1] == "user_uploads"
        and all(segment not in {"", ".", ".."} for segment in segments[1:])
    )


def upload_filename(path: str) -> str:
    """Derive a safe display filename from a normalized upload path."""
    name = Path(unquote(path.rsplit("/", 1)[-1])).name
    return name or "attachment"


async def download_zulip_upload(
    *,
    site_url: str,
    bot_email: str,
    api_key: str,
    path: str,
    max_bytes: int = MAX_UPLOAD_DOWNLOAD_BYTES,
) -> DownloadedUpload:
    """Fetch one upload through Zulip's authenticated temporary-URL API.

    The returned path is already translated for the active Hermes terminal
    backend by :func:`gateway.platforms.base.cache_media_bytes`.
    """
    if not is_valid_upload_path(path):
        raise ValueError("Not a Zulip user-upload path")

    import httpx
    from gateway.platforms.base import cache_media_bytes

    filename = upload_filename(path)
    async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
        signed_response = await client.get(
            f"{site_url.rstrip('/')}/api/v1{path}",
            auth=(bot_email, api_key),
        )
        signed_response.raise_for_status()
        signed_url = (signed_response.json() or {}).get("url", "")
        if not signed_url:
            raise ValueError("Zulip did not return a temporary upload URL")
        if signed_url.startswith("/"):
            signed_url = f"{site_url.rstrip('/')}{signed_url}"

        # Stream the signed response so a malicious or incorrectly declared
        # upload cannot make the gateway buffer an unbounded file in memory.
        async with client.stream("GET", signed_url) as response:
            response.raise_for_status()
            chunks: list[bytes] = []
            size_bytes = 0
            async for chunk in response.aiter_bytes():
                size_bytes += len(chunk)
                if size_bytes > max_bytes:
                    raise ValueError(
                        f"Attachment exceeds the {max_bytes // (1024 * 1024)} MiB download limit"
                    )
                chunks.append(chunk)
            data = b"".join(chunks)

    content_type = ""
    headers = getattr(response, "headers", None)
    if headers:
        content_type = str(headers.get("content-type", "")).split(";", 1)[0].strip().lower()
    cached = cache_media_bytes(data, filename=filename, mime_type=content_type)
    if cached is None:
        raise ValueError(f"Could not cache attachment {filename!r}")

    return DownloadedUpload(
        path=cached.path,
        filename=filename,
        mime_type=cached.media_type,
        size_bytes=len(data),
    )


def _check_zulip_attachment_requirements() -> bool:
    from .search_tool import _check_zulip_search_requirements

    return _check_zulip_search_requirements()


async def zulip_download_attachment(
    message_id: int,
    filename: Optional[str] = None,
    attachment_index: int = 0,
    *,
    task_id: Optional[str] = None,
) -> str:
    """Download an attachment from a Zulip message into the local media cache.

    A Zulip gateway session is restricted to the active stream/topic or DM.
    Taking a message ID rather than an arbitrary upload URL is important: the
    bot can have access to conversations that the person asking it cannot.
    """
    del task_id
    try:
        import zulip
    except ImportError:
        return json.dumps({"error": "zulip package not installed"})

    from .search_tool import _get_session_narrow, _get_zulip_credentials

    site_url, bot_email, api_key = _get_zulip_credentials()
    if not site_url or not bot_email or not api_key:
        return json.dumps({"error": "Zulip credentials are not configured"})
    if (
        isinstance(message_id, bool)
        or not isinstance(message_id, int)
        or message_id <= 0
    ):
        return json.dumps({"error": "message_id must be a positive integer"})
    if filename is not None and not isinstance(filename, str):
        return json.dumps({"error": "filename must be a string"})
    if (
        isinstance(attachment_index, bool)
        or not isinstance(attachment_index, int)
        or attachment_index < 0
    ):
        return json.dumps({"error": "attachment_index must be a non-negative integer"})

    # Fail closed if this is a Zulip session whose source cannot be narrowed.
    try:
        from gateway.session_context import get_session_env

        in_zulip_session = get_session_env("HERMES_SESSION_PLATFORM", "") == "zulip"
    except Exception:
        in_zulip_session = False
    session_narrow = _get_session_narrow()
    if in_zulip_session and session_narrow is None:
        return json.dumps({"error": "Could not determine the current Zulip conversation"})

    client = zulip.Client(site=site_url, email=bot_email, api_key=api_key)
    try:
        result = client.get_messages({
            "anchor": str(message_id),
            "num_before": 0,
            "num_after": 1,
            "narrow": session_narrow or None,
            "apply_markdown": False,
        })
    except Exception as exc:
        logger.warning("Zulip attachment lookup failed: %s", exc)
        return json.dumps({"error": f"Zulip API error: {exc}"})

    if result.get("result") != "success":
        return json.dumps({"error": result.get("msg", "Could not fetch Zulip message")})
    message = next(
        (item for item in result.get("messages", []) if item.get("id") == message_id),
        None,
    )
    if message is None:
        return json.dumps({"error": "Message was not found in the allowed conversation"})

    paths = extract_upload_paths(message.get("content") or "")
    if filename:
        requested_name = Path(filename).name
        paths = [path for path in paths if upload_filename(path) == requested_name]
    if not paths:
        return json.dumps({"error": "No matching attachments were found on that message"})
    if attachment_index >= len(paths):
        return json.dumps({
            "error": (
                "attachment_index is out of range "
                f"(message has {len(paths)} matching attachment(s))"
            ),
        })

    try:
        downloaded = await download_zulip_upload(
            site_url=site_url,
            bot_email=bot_email,
            api_key=api_key,
            path=paths[attachment_index],
        )
    except Exception as exc:
        logger.warning("Zulip attachment download failed for message %s: %s", message_id, exc)
        return json.dumps({"error": f"Could not download attachment: {exc}"})

    return json.dumps({
        "message_id": message_id,
        "filename": downloaded.filename,
        "mime_type": downloaded.mime_type,
        "size_bytes": downloaded.size_bytes,
        "path": downloaded.path,
    })


_ZULIP_DOWNLOAD_ATTACHMENT_SCHEMA = {
    "name": "zulip_download_attachment",
    "description": (
        "Download a file attached to a Zulip message into the local media cache. "
        "Use after zulip_search_messages finds a message with an attachment. "
        "The returned path can be read with file or terminal tools. In a Zulip "
        "conversation, the message must be in the current stream/topic or DM."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "message_id": {
                "type": "integer",
                "description": "ID of the Zulip message containing the attachment.",
            },
            "filename": {
                "type": "string",
                "description": "Optional exact attachment filename to select.",
            },
            "attachment_index": {
                "type": "integer",
                "description": "Zero-based index among matching attachments; defaults to 0.",
                "default": 0,
                "minimum": 0,
            },
        },
        "required": ["message_id"],
    },
}


async def _handle_zulip_download_attachment(args, **kw):
    return await zulip_download_attachment(
        message_id=args.get("message_id"),
        filename=args.get("filename"),
        attachment_index=args.get("attachment_index", 0),
        task_id=kw.get("task_id"),
    )


def register_zulip_attachment_tool(ctx) -> None:
    """Register the history-scoped attachment materialization tool."""
    ctx.register_tool(
        name="zulip_download_attachment",
        toolset="zulip-history",
        schema=_ZULIP_DOWNLOAD_ATTACHMENT_SCHEMA,
        handler=_handle_zulip_download_attachment,
        check_fn=_check_zulip_attachment_requirements,
        is_async=True,
        emoji="📎",
    )
