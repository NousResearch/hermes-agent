"""The admission media rules the store applies when it releases settled admissions: the native references a payload holds and the retained documents a hosted prompt names; and the spillover archive names a transcript row points at, which a delete records as retirement candidates.

``hosted_document_references`` takes the profile's retained media root as ``media_root``. Standard library only.
"""

from __future__ import annotations

import re
from pathlib import Path


_API_IMAGE_NAME = re.compile(r'api_[0-9a-f]{32}\.(png|jpg|gif|webp)')
_SHA256_NAME = re.compile(r'[0-9a-f]{64}')


# A hosted room's non-image document is retained under ``native-inputs/<sha256>/<name>`` like any
# other input, but rides in the committed prompt as a path line (``session_hosted_attachments``),
# not as a structured reference. Same literal the hosted payload builders append.
HOSTED_DOCUMENT_LINE = '\n[Shared attachment] file: '
_HOSTED_DOCUMENT_RE = re.compile(re.escape(HOSTED_DOCUMENT_LINE) + r'([^\n]+)\n')


def hosted_document_references(request_id, payload, media_root):
    """Retained document references a hosted admission's prompt names (``[]`` for other rows).

    Only content-addressed names under this profile's retained root count, and never an API
    image name: those are held by ``api_turn_v1`` in every status, so a member's prompt text
    can name one without gaining deletion authority over it."""
    if not str(request_id).startswith('hosted:') or not isinstance(payload.get('text'), str):
        return []
    root = Path(media_root)
    references = []
    for value in _HOSTED_DOCUMENT_RE.findall(payload['text']):
        path = Path(value)
        if (path.is_absolute() and path.parent.parent == root and _SHA256_NAME.fullmatch(path.parent.name)
                and not _API_IMAGE_NAME.fullmatch(path.name)):
            references.append({'path': value, 'sha256': path.parent.name})
    return references


def admission_media_references(payload):
    """Native references eligible as deletion candidates after terminal settlement."""
    return list(payload.get('attachments_v1', {}).get('media', ())) + list(
        payload.get('native_text_v1', {}).get('media', ()))


# Every pointer form of a spill archive (host path, ``~``-relative, in-container mount) carries the
# ``spillover/<name>`` segment, with either separator (Windows paths are backslashed, JSON-escaped inside
# structured content); sandbox-fallback ``hermes-results/<name>`` pointers do not.
_SPILLOVER_NAME_RE = re.compile(r"spillover(?:/|\\+)([A-Za-z0-9_.-]+)")


def spillover_names_in_text(*texts) -> set:
    """Spillover archive basenames referenced by transcript message text (empty set on none)."""
    names: set = set()
    for text in texts:
        if isinstance(text, str) and "spillover" in text:
            names.update(_SPILLOVER_NAME_RE.findall(text))
    return names
