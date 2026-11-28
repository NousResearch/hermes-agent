"""ACP prompt content blocks -> Hermes/OpenAI user-content payloads (text, images, resources)."""

from __future__ import annotations

import base64
import logging
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from acp.schema import (
    AudioContentBlock, BlobResourceContents, EmbeddedResourceContentBlock, ImageContentBlock,
    ResourceContentBlock, TextContentBlock, TextResourceContents,
)

logger = logging.getLogger("acp_adapter.server")

PromptBlock = (
    TextContentBlock | ImageContentBlock | AudioContentBlock | ResourceContentBlock | EmbeddedResourceContentBlock
)

_MAX_ACP_RESOURCE_BYTES = 512 * 1024
_TEXT_RESOURCE_MIME_TYPES = {
    "application/json",
    "application/javascript",
    "application/typescript",
    "application/xml",
    "application/x-yaml",
    "application/yaml",
    "application/toml",
    "application/sql",
}


def _resource_display_name(uri: str, name: str | None = None, title: str | None = None) -> str:
    """Human-readable attachment name for prompt context."""
    raw_name = (name or "").strip()
    raw_title = (title or "").strip()
    if raw_title and raw_name and raw_title != raw_name:
        return f"{raw_title} ({raw_name})"
    if raw_title or raw_name:
        return raw_title or raw_name
    parsed = urlparse(uri)
    candidate = parsed.path if parsed.scheme else uri
    return Path(unquote(candidate)).name or uri or "resource"


def _mime_main(mime_type: str | None) -> str:
    return (mime_type or "").split(";", 1)[0].strip().lower()


def _is_text_resource(mime_type: str | None) -> bool:
    mime = _mime_main(mime_type)
    return mime.startswith("text/") or mime in _TEXT_RESOURCE_MIME_TYPES


def _is_image_resource(mime_type: str | None) -> bool:
    return _mime_main(mime_type).startswith("image/")


_IMAGE_SUFFIX_MIME = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
    ".svg": "image/svg+xml",
}


def _path_from_file_uri(uri: str) -> Path | None:
    """Local file URI/path from an ACP client -> readable Path (None for non-file URIs).
    Windows drive forms (Zed via wsl.exe) become ``/mnt/<drive>/...``."""
    raw = (uri or "").strip()
    if not raw:
        return None

    # urlparse treats a bare Windows drive as a URI scheme.
    if len(raw) >= 3 and raw[0].isalpha() and raw[1] == ":" and raw[2] in "/\\":
        raw = "file:///" + raw.replace("\\", "/")
    parsed = urlparse(raw)
    if parsed.scheme and parsed.scheme != "file":
        return None

    if parsed.scheme == "file" and parsed.netloc and parsed.netloc not in {"", "localhost"}:
        return None
    path_text = unquote(parsed.path or "") if parsed.scheme == "file" else unquote(raw)

    # file:///C:/Users/... or C:\Users\...
    if len(path_text) >= 3 and path_text[0] == "/" and path_text[2] == ":" and path_text[1].isalpha():
        drive, rest = path_text[1], path_text[3:]
    elif len(path_text) >= 2 and path_text[1] == ":" and path_text[0].isalpha():
        drive, rest = path_text[0], path_text[2:]
    else:
        return Path(path_text)
    import os
    rest = rest.lstrip("/\\").replace("\\", "/")
    if os.name == "nt":
        return Path(f"{drive}:/{rest}")
    return Path("/mnt") / drive.lower() / rest


def _decode_text_bytes(data: bytes, mime_type: str | None) -> str | None:
    """Decode resource bytes if they are probably text; return None for binary."""
    if b"\x00" in data and not _is_text_resource(mime_type):
        return None
    for encoding in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    # Binary (ELF/Mach-O/PE), not a shell script: feeding its decoded bytes back into the guard tokenizes
    # machine code into bogus NUL-bearing paths and crashes the scanner (#77703). Mirror
    # lifecycle_guard._read_referenced_script and treat it as nothing to scan.
    return data.decode("utf-8", errors="replace")


def _format_resource_text(
    *, uri: str, body: str, name: str | None = None, title: str | None = None, note: str | None = None
) -> str:
    display = _resource_display_name(uri, name=name, title=title)
    header = f"[Attached file: {display}]"
    if note:
        header += f" ({note})"
    return f"{header}\nURI: {uri}\n\n{body}"


def _text_parts(**kwargs: Any) -> list[dict[str, Any]]:
    """Single OpenAI text part wrapping ``_format_resource_text(**kwargs)``."""
    return [{"type": "text", "text": _format_resource_text(**kwargs)}]


def _image_parts(uri: str, display: str, data: bytes, mime: str) -> list[dict[str, Any]]:
    """Text header + image_url data URL so vision models can see the attachment."""
    return [
        {"type": "text", "text": f"[Attached image: {display}]" + (f"\nURI: {uri}" if uri else "")},
        {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"}},
    ]


def _attr(obj: Any, name: str) -> str | None:
    """Stripped string attribute, ``None`` when missing/blank."""
    return str(getattr(obj, name, "") or "").strip() or None


async def _client_read_text_file(
    conn: Any,
    session_id: str | None,
    uri: str,
    can_use_client_fs: bool = False,
) -> str | None:
    """Fetch an attached file through the ACP client's fs/read_text_file.

    Used when the local disk read of a file URI failed — the path lives on
    the client's machine (e.g. the host, while the agent runs in a
    container). Per the ACP spec the reverse fs method is only available
    when the client advertises fs.readTextFile, so *can_use_client_fs* must
    be True (from the initialize() capability record). Returns the file
    text, or None when the client cannot help (capability off, no
    connection, method absent, request failed, empty result).
    """
    if conn is None or not session_id or not can_use_client_fs:
        return None
    read_fn = getattr(conn, "read_text_file", None)
    if read_fn is None:
        return None
    # Send the path back in the client's own spelling (its machine's path),
    # not the container-translated local form.
    raw_path = uri
    if raw_path.startswith("file://"):
        raw_path = unquote(urlparse(raw_path).path)
    try:
        resp = await read_fn(path=raw_path, session_id=session_id)
    except Exception as exc:
        logger.debug("ACP client fs/read_text_file failed for %s: %s", uri, exc)
        return None
    content = getattr(resp, "content", None)
    if content is None and isinstance(resp, dict):
        content = resp.get("content")
    if isinstance(content, str) and content:
        logger.info(
            "ACP attached file %s resolved via client fs/read_text_file (%d chars)",
            uri,
            len(content),
        )
        return content
    logger.debug("ACP client fs/read_text_file returned no content for %s", uri)
    return None


async def _resource_link_to_parts(
    block: ResourceContentBlock,
    conn: Any = None,
    session_id: str | None = None,
    can_use_client_fs: bool = False,
) -> list[dict[str, Any]]:
    """ACP resource_link -> OpenAI content parts: images become a text header + image_url,
    everything else a single text part with the inlined body (or a binary-omit note).

    When *conn*/*session_id* are given and the local disk read fails, the file
    is fetched through the client's fs/read_text_file — this is how an
    IDE-attached file on the host resolves when Hermes runs in a container."""
    uri = _attr(block, "uri")
    if not uri:
        return []

    name, title, mime_type = _attr(block, "name"), _attr(block, "title"), _attr(block, "mime_type")
    path = _path_from_file_uri(uri)
    ident = dict(uri=uri, name=name, title=title)

    if path is None:
        return _text_parts(
            **ident, body="[Resource link only; Hermes cannot read non-file ACP resource URIs directly.]"
        )

    image_mime = mime_type if _is_image_resource(mime_type) else _IMAGE_SUFFIX_MIME.get(path.suffix.lower())
    if image_mime and _is_image_resource(image_mime):
        try:
            size = path.stat().st_size
            if size > _MAX_ACP_RESOURCE_BYTES:
                return _text_parts(
                    **ident, body=f"[Image too large to inline: {size} bytes, cap={_MAX_ACP_RESOURCE_BYTES}]"
                )
            with path.open("rb") as fh:
                data = fh.read()
        except OSError as exc:
            logger.warning("ACP image resource read failed: %s", uri, exc_info=True)
            return _text_parts(**ident, body=f"[Could not read attached image: {exc}]")
        return _image_parts(uri, _resource_display_name(uri, name=name, title=title), data, image_mime)

    try:
        size = path.stat().st_size
        with path.open("rb") as fh:
            data = fh.read(min(size, _MAX_ACP_RESOURCE_BYTES))
        text = _decode_text_bytes(data, mime_type)
        if text is None:
            return _text_parts(**ident, body=f"[Binary file omitted: {size} bytes, mime={mime_type or 'unknown'}]")
        note = f"truncated to {_MAX_ACP_RESOURCE_BYTES} of {size} bytes" if size > _MAX_ACP_RESOURCE_BYTES else None
        return _text_parts(**ident, body=text, note=note)
    except OSError as exc:
        # The URI resolved to a path this agent process cannot read — most
        # commonly the IDE runs on the host while Hermes runs in a
        # container, so `file:///home/user/...` points at the host disk,
        # not ours. ACP's reverse fs/read_text_file lets the client (which
        # owns that path, including unsaved editor buffers) fetch it.
        client_text = await _client_read_text_file(
            conn, session_id, uri, can_use_client_fs=can_use_client_fs
        )
        if client_text is not None:
            return _text_parts(
                **ident, body=client_text, note="read via IDE file system (fs/read_text_file)"
            )
        logger.warning("ACP resource read failed: %s", uri, exc_info=True)
        return _text_parts(**ident, body=f"[Could not read attached file: {exc}]")


def _embedded_resource_to_parts(block: EmbeddedResourceContentBlock) -> list[dict[str, Any]]:
    resource = getattr(block, "resource", None)
    if resource is None:
        return []

    uri = _attr(resource, "uri") or ""
    mime_type = _attr(resource, "mime_type")

    if isinstance(resource, TextResourceContents):
        return _text_parts(uri=uri, body=resource.text)

    if isinstance(resource, BlobResourceContents):
        blob = resource.blob or ""
        try:
            data = base64.b64decode(blob, validate=True)
        except Exception:
            data = blob.encode("utf-8", errors="replace")

        if _is_image_resource(mime_type):
            if len(data) > _MAX_ACP_RESOURCE_BYTES:
                return _text_parts(
                    uri=uri,
                    body=f"[Embedded image too large to inline: {len(data)} bytes, cap={_MAX_ACP_RESOURCE_BYTES}]",
                )
            return _image_parts(uri, _resource_display_name(uri), data, mime_type or "image/png")

        body = _decode_text_bytes(data[:_MAX_ACP_RESOURCE_BYTES], mime_type)
        if body is None:
            body = f"[Binary embedded file omitted: {len(data)} bytes, mime={mime_type or 'unknown'}]"
        elif len(data) > _MAX_ACP_RESOURCE_BYTES:
            body += f"\n\n[Truncated to {_MAX_ACP_RESOURCE_BYTES} of {len(data)} bytes]"
        return _text_parts(uri=uri, body=body)

    text = getattr(resource, "text", None)
    if text:
        return _text_parts(uri=uri, body=str(text))
    return []


def _extract_text(prompt: list[PromptBlock]) -> str:
    """Extract plain text from ACP content blocks for display/commands."""
    return "\n".join(str(block.text) for block in prompt if hasattr(block, "text"))


def _image_block_to_openai_part(block: ImageContentBlock) -> dict[str, Any] | None:
    """Convert an ACP image content block to OpenAI-style multimodal content."""
    data, uri = _attr(block, "data"), _attr(block, "uri")
    mime_type = _attr(block, "mime_type") or "image/png"
    if data:
        url = data if data.startswith("data:") else f"data:{mime_type};base64,{data}"
    elif uri:
        url = uri
    else:
        return None
    return {"type": "image_url", "image_url": {"url": url}}


def _append_parts(parts: list, text_parts: list[str], new_parts: list[dict[str, Any]]) -> None:
    for part in new_parts:
        parts.append(part)
        if part.get("type") == "text":
            text_parts.append(part["text"])


async def _content_blocks_to_openai_user_content(
    prompt: list[PromptBlock],
    conn: Any = None,
    session_id: str | None = None,
    can_use_client_fs: bool = False,
) -> str | list[dict[str, Any]]:
    """Convert ACP prompt blocks into a Hermes/OpenAI-compatible user content payload."""
    parts: list[dict[str, Any]] = []
    text_parts: list[str] = []

    for block in prompt:
        if isinstance(block, TextContentBlock):
            if block.text:
                parts.append({"type": "text", "text": block.text})
                text_parts.append(block.text)
        elif isinstance(block, ImageContentBlock):
            image_part = _image_block_to_openai_part(block)
            if image_part is not None:
                parts.append(image_part)
        elif isinstance(block, ResourceContentBlock):
            resource_parts = await _resource_link_to_parts(
                block, conn=conn, session_id=session_id, can_use_client_fs=can_use_client_fs
            )
            _append_parts(parts, text_parts, resource_parts)
        elif isinstance(block, EmbeddedResourceContentBlock):
            _append_parts(parts, text_parts, _embedded_resource_to_parts(block))

    if not parts:
        return _extract_text(prompt)

    # Pure text stays a string (slash commands, text-only providers); structured only for media.
    if all(part.get("type") == "text" for part in parts):
        return "\n".join(text_parts)

    return parts
