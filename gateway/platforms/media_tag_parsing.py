"""MEDIA: tag parsing and directive stripping helpers.

Extracted from ``gateway/platforms/base.py`` (god-file decomposition
campaign, wave 1 — shard s2, cluster c2, 21 move votes). Every function is
moved verbatim; ``base.py`` re-exports them so ``from
gateway.platforms.base import ...`` call sites are unchanged. The shared
regex constants (``MEDIA_TAG_CLEANUP_RE``, ``MEDIA_EXTENSIONLESS_TAG_RE``,
``MEDIA_DELIVERY_EXTS``), ``validate_media_delivery_path`` and the
``BasePlatformAdapter._mask_*`` class helpers stay in ``base.py`` (class
methods and tests still reference them there) and are imported here at the
bottom of this module — the same cycle-avoidance pattern documented in
``gateway/authz_mixin.py``.
"""

import re
from pathlib import Path
from typing import Optional, Tuple

def _match_extensionless_path(scan_text: str, match: "re.Match") -> Optional[Tuple[str, int]]:
    """Resolve an extensionless MEDIA tag match to a validated on-disk path.

    Tries the regex-captured path first. When that fails validation, the
    candidate is progressively extended forward across single spaces
    (validation-gated, bounded at 8 tokens, never past a newline or a
    subsequent ``MEDIA:`` keyword) so unknown-extension paths containing
    spaces deliver (#24032). Returns ``(safe_path, end_offset)`` where
    ``end_offset`` is the index in ``scan_text`` just past the matched path,
    or ``None`` when nothing validates.
    """
    raw = match.group("path")
    path = _normalize_media_tag_path(raw)
    if not path:
        return None
    safe = validate_media_delivery_path(path)
    if safe:
        return safe, match.end("path")
    start = match.start("path")
    nl = scan_text.find("\n", start)
    limit = nl if nl != -1 else len(scan_text)
    segment = scan_text[start:limit]
    nxt = segment.find("MEDIA:", 1)
    if nxt != -1:
        segment = segment[:nxt]
    pos = match.end("path") - start
    for _ in range(8):
        while pos < len(segment) and segment[pos] in " \t":
            pos += 1
        if pos >= len(segment):
            break
        tok_end = pos
        while tok_end < len(segment) and segment[tok_end] not in " \t":
            tok_end += 1
        candidate = _normalize_media_tag_path(segment[:tok_end])
        safe = validate_media_delivery_path(candidate)
        if safe:
            return safe, start + tok_end
        pos = tok_end
    return None


def _merge_spans(spans: list) -> list:
    """Merge overlapping/nested (start, end) spans so multi-pattern matches
    over the same tag never double-delete adjacent text."""
    merged: list = []
    for s, e in sorted(spans):
        if merged and s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    return merged


def _normalize_media_tag_path(raw: str) -> str:
    path = str(raw or "").strip()
    if len(path) >= 2 and path[0] == path[-1] and path[0] in "`\"'":
        path = path[1:-1].strip()
    return path.lstrip("`\"'").rstrip("`\"',.;:)}]")


def _path_lacks_deliverable_extension(path: str) -> bool:
    """True when MEDIA_TAG_CLEANUP_RE's extension alternation does not cover
    ``path`` — either the basename has no extension at all (Caddyfile,
    Makefile, …) or the extension is not in MEDIA_DELIVERY_EXTS (.py, .log,
    .weirdext, …). Such paths route through the validated delivery pass
    (``validate_media_delivery_path``) instead of the unconditional one, so
    every file type is deliverable (#36060) while nonexistent / denylisted
    paths stay visible in the text.
    """
    suffix = Path(path).suffix.lower()
    return not suffix or suffix not in MEDIA_DELIVERY_EXTS


def _resolve_extensionless_candidate(path: str) -> Optional[str]:
    """Validate a bare extensionless-branch path (no forward extension).

    Thin wrapper kept for call sites that only have the normalized path
    (no scan-text context for spaced-path recovery).
    """
    if not path:
        return None
    return validate_media_delivery_path(path)


def _strip_media_tag_directives(text: str) -> str:
    """Remove MEDIA: tags and [[audio_as_voice]] / [[as_document]] markers.

    Protected spans (fenced code blocks, inline code holding non-deliverable
    example tags, blockquotes, JSON string values) are used as a mask-locator
    only — tags inside them are neither stripped nor mangled, matching
    ``extract_media``'s treatment so display text and delivery agree (#16434).
    """
    if (
        "MEDIA:" not in text
        and "[[audio_as_voice]]" not in text
        and "[[as_document]]" not in text
    ):
        return text
    cleaned = text.replace("[[audio_as_voice]]", "").replace("[[as_document]]", "")

    # Locate real tag spans on a masked copy (offset-preserving), then delete
    # exactly those spans from the unmasked text — same pattern as
    # extract_media. Import-cycle-free: BasePlatformAdapter is defined later
    # in this module, so resolve it lazily at call time.
    masked = BasePlatformAdapter._mask_protected_spans(cleaned)
    masked = BasePlatformAdapter._mask_json_string_media(masked)

    spans: list = [m.span() for m in MEDIA_TAG_CLEANUP_RE.finditer(masked)]
    for match in MEDIA_EXTENSIONLESS_TAG_RE.finditer(masked):
        path = _normalize_media_tag_path(match.group("path"))
        if not path or not _path_lacks_deliverable_extension(path):
            continue
        resolved = _match_extensionless_path(masked, match)
        if resolved is not None:
            spans.append((match.start(), resolved[1]))

    if spans:
        chars = list(cleaned)
        for start, end in reversed(_merge_spans(spans)):
            del chars[start:end]
        cleaned = "".join(chars)
    return cleaned


def _strip_media_directives(text: str) -> str:
    """Strip internal delivery directives ([[audio_as_voice]], [[as_document]],
    MEDIA:<path>) so they never render as visible text.

    Backstop only: run ``extract_media`` first. MEDIA cleanup uses the shared
    ``MEDIA_TAG_CLEANUP_RE`` (only tags whose path has a known deliverable
    extension are removed; an unknown-extension tag is intentionally left so the
    bare-path detector downstream can still pick it up, per #34517). Validated
    extension-less tags (e.g. ``MEDIA:/output/Caddyfile``) are also removed.
    [[...]] is exact.
    """
    if not text:
        return text
    return _strip_media_tag_directives(text)


# Imported at the bottom so this module never triggers ``base.py``'s own
# import of this module mid-execution (cycle break). All names below are
# referenced only at call time by the moved functions.
from gateway.platforms.base import (  # noqa: E402
    MEDIA_DELIVERY_EXTS,
    MEDIA_EXTENSIONLESS_TAG_RE,
    MEDIA_TAG_CLEANUP_RE,
    BasePlatformAdapter,
    validate_media_delivery_path,
)
