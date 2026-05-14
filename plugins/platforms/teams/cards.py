"""Teams card builders for file delivery flows.

These are pure dict factories. They emit the JSON shapes the Teams Bot
Framework recognizes; no SDK objects are involved so they work with any
Teams transport layer.

References:
- FileConsentCard: https://learn.microsoft.com/microsoftteams/platform/bots/how-to/bots-files
- FileInfoCard:    same doc, "Notify the user about the uploaded file"
- file.download.info attachment: channel / group upload references
"""
from __future__ import annotations

import uuid
from typing import Any, Dict, Optional

FILE_CONSENT_CONTENT_TYPE = "application/vnd.microsoft.teams.card.file.consent"
FILE_INFO_CONTENT_TYPE = "application/vnd.microsoft.teams.card.file.info"
FILE_DOWNLOAD_INFO_CONTENT_TYPE = "application/vnd.microsoft.teams.file.download.info"


def build_file_consent_card(
    filename: str,
    size_bytes: int,
    accept_context: Optional[Dict[str, Any]] = None,
    description: str = "",
) -> Dict[str, Any]:
    """Build the FileConsentCard attachment that kicks off an outbound DM upload.

    The user's accept/decline triggers a ``fileConsent/invoke`` activity;
    on accept the payload includes ``uploadInfo`` with a OneDrive URL the
    bot PUTs the bytes to. ``acceptContext`` is echoed back unchanged so
    the bot can correlate the invoke with the original upload request.

    A fresh ``upload_id`` is seeded into the context if the caller did not
    supply one — that's the natural correlation id for the upload.
    ``declineContext`` is a value-equal *copy* of ``acceptContext`` (not the
    same object): downstream handlers (e.g. the fileConsent/invoke handler)
    routinely mutate one branch's context to record status, and aliasing
    would silently leak those writes into the other branch.
    """
    ctx: Dict[str, Any] = dict(accept_context or {})
    ctx.setdefault("upload_id", str(uuid.uuid4()))
    return {
        "contentType": FILE_CONSENT_CONTENT_TYPE,
        "name": filename,
        "content": {
            "description": description,
            "sizeInBytes": int(size_bytes),
            "acceptContext": ctx,
            "declineContext": dict(ctx),
        },
    }


def build_file_info_card(
    filename: str,
    content_url: str,
    *,
    unique_id: Optional[str] = None,
    file_type: Optional[str] = None,
) -> Dict[str, Any]:
    """Build the FileInfoCard the bot sends *after* a successful FileConsent upload.

    Without this, the file does not render as a native attachment in the
    DM — the user just sees the consent card flip to 'uploaded'.

    ``content_url`` is the OneDrive ``contentUrl`` echoed back in the
    ``fileConsent/invoke`` ``upload_info``. ``unique_id`` is the OneDrive
    drive-item id from ``upload_info.unique_id`` — required by Teams
    clients to render the file as a native preview-able attachment;
    when omitted, a fresh uuid is generated as a fallback (the file
    still renders but the link may not resolve to a previewable item).
    ``file_type`` is auto-inferred from the filename extension when
    omitted (e.g. ``foo.pdf`` → ``"pdf"``).
    """
    return {
        "contentType": FILE_INFO_CONTENT_TYPE,
        "contentUrl": content_url,
        "name": filename,
        "content": {
            "uniqueId": unique_id or str(uuid.uuid4()),
            "fileType": file_type or _infer_file_type(filename),
        },
    }


def build_file_download_card(
    filename: str,
    content_url: str,
    *,
    unique_id: Optional[str] = None,
    file_type: Optional[str] = None,
) -> Dict[str, Any]:
    """Build a ``file.download.info`` attachment for channel / group uploads.

    Used after the bot has uploaded to SharePoint via Graph and has the
    drive item's webUrl. Teams clients render this as a downloadable
    attachment in channel posts.

    ``content_url`` is the SharePoint webUrl returned by
    :meth:`GraphClient.upload_to_sharepoint`. ``unique_id`` should be the
    Graph drive-item id when known (so subsequent interactions can resolve
    back to the file); when omitted, the card still renders but the bot
    cannot use it to look the file up later. ``file_type`` is auto-inferred
    from the filename extension when omitted (e.g. ``foo.pdf`` → ``"pdf"``).
    """
    content: Dict[str, Any] = {"downloadUrl": content_url}
    if unique_id:
        content["uniqueId"] = unique_id
    content["fileType"] = file_type or _infer_file_type(filename)
    return {
        "contentType": FILE_DOWNLOAD_INFO_CONTENT_TYPE,
        "contentUrl": content_url,
        "content": content,
        "name": filename,
    }


def _infer_file_type(filename: str) -> str:
    """Return the Teams-style file type token for *filename*.

    Teams accepts ``"png"`` / ``"jpg"`` / ``"pdf"`` / ... strings; the mapping
    lines up with common extension suffixes. Files without a dotted
    extension (``LICENSE``, ``Makefile``) fall back to ``"file"`` so Teams
    renders a generic document icon.
    """
    _head, sep, ext = filename.rpartition(".")
    if not sep:
        return "file"
    return ext.lower() or "file"


__all__ = [
    "FILE_CONSENT_CONTENT_TYPE",
    "FILE_INFO_CONTENT_TYPE",
    "FILE_DOWNLOAD_INFO_CONTENT_TYPE",
    "build_file_consent_card",
    "build_file_info_card",
    "build_file_download_card",
]
