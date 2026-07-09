"""Slack inbound document admission and bounded response reads."""

import html
import io
import re
from contextlib import asynccontextmanager

from hermes_cli.config import DEFAULT_CONFIG, load_config_readonly


_SETTING = "max_inbound_document_bytes"


def document_limit() -> int:
    """Resolve the adapter owner's gateway setting during adapter construction."""
    default = DEFAULT_CONFIG["gateway"][_SETTING]
    try:
        gateway = load_config_readonly().get("gateway")
    except OSError:
        return default
    raw = gateway.get(_SETTING) if isinstance(gateway, dict) else None
    if raw is None or isinstance(raw, bool) or (isinstance(raw, float) and not raw.is_integer()):
        return default
    try:
        return max(0, int(raw))
    except (TypeError, ValueError, OverflowError):
        return default


class DocumentTooLarge(ValueError):
    def notice(self, filename: str) -> str:
        label = re.sub(r"[\x00-\x1f\x7f]+", " ", filename).strip()[:200] or "document"
        label = html.escape(label.replace("`", "ˋ"), quote=False)
        size, limit = self.args
        return (f"Attachment `{label}` skipped: {size} bytes exceeds the inbound Slack document "
                f"limit ({limit} bytes). Ask the sender to shrink or split the file.")


def check_size(size, limit: int) -> None:
    try:
        size = int(size)
    except (TypeError, ValueError, OverflowError):
        return  # Missing/invalid metadata: the body is still checked as it arrives.
    if limit and size > limit:
        raise DocumentTooLarge(size, limit)


@asynccontextmanager
async def file_response(client, url, token, limit):
    headers = {"Authorization": f"Bearer {token}"}
    if limit is None:  # Existing image/audio/video path.
        yield await client.get(url, headers=headers)
    else:
        headers["Accept-Encoding"] = "identity"
        async with client.stream("GET", url, headers=headers) as response:
            yield response


async def read_file(response, html_label: str, limit: int | None) -> bytes:
    response.raise_for_status()
    ct = response.headers.get("content-type", "")
    if "text/html" in ct:
        raise ValueError(
            f"Slack returned HTML instead of {html_label} (content-type: {ct}); "
            "check bot token scopes and file permissions, or a cross-origin "
            "redirect dropped the token (Enterprise Grid files-origin)")
    if limit is None:
        return response.content
    if response.headers.get("content-encoding", "").strip().lower() not in {"", "identity"}:
        raise ValueError("Unexpected Slack Content-Encoding despite Accept-Encoding: identity")
    check_size(response.headers.get("content-length"), limit)
    buf = io.BytesIO()
    async for chunk in response.aiter_bytes():
        check_size(buf.tell() + len(chunk), limit)
        buf.write(chunk)
    return buf.getvalue()
