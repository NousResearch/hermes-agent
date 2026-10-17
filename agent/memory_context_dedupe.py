"""Conservative, section-local deduplication of standalone recalled observations."""

import re
from datetime import datetime


_BULLET = re.compile(r"[-*+]\s+\S")
_STAMPED = re.compile(
    r"\[(?P<stamp>\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}"
    r"(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})?)\]\s+\S")
_FENCE = re.compile(r" {0,3}(`{3,}|~{3,})")
_BOUNDARY = re.compile(r"(?:#{1,6}\s|\*\*.+\*\*\s*$|[-*_]{3,}\s*$|`{3,}|~{3,})")


def _kind(text: str) -> str | None:
    if _BULLET.match(text):
        return "bullet"
    if stamp := _STAMPED.match(text):
        try:
            datetime.fromisoformat(stamp["stamp"])
        except ValueError:
            return None
        return "timestamp"
    return None


def _carries_continuation(lines: list[str], index: int, kind: str) -> bool:
    # A blank line does not detach indented provenance from its record.
    following = next((lines[i] for i in range(index + 1, len(lines)) if lines[i].strip()), "")
    if following and following[0].isspace():
        return True
    # Plain timestamp headlines can carry unindented prose. Ambiguous records are kept;
    # Markdown headings, separators and another observation provide an explicit boundary.
    return bool(kind == "timestamp" and following and _kind(following) is None
                and not _BOUNDARY.match(following))


def dedupe_recall_observations(text: str) -> str:
    """Suppress exact standalone repeats inside one section of one composed block.

    Recognized forms are top-level Markdown bullets and bracketed ISO date/time observations.
    The complete line, including its timestamp and provenance, is the key. No fuzzy matching,
    cross-section/peer suppression or cross-turn state is used. Providers own semantic and
    historical deduplication, plus other presentation formats.
    """
    lines = text.split("\n")
    seen: set[str] = set()
    kept: list[str] = []
    fence = None
    for index, line in enumerate(lines):
        marker = _FENCE.match(line)
        if fence is not None:
            kept.append(line)
            if (marker and marker[1][0] == fence[0] and len(marker[1]) >= len(fence)
                    and not line[marker.end():].strip()):
                fence = None
            continue
        if marker:
            fence = marker[1]
            seen.clear()
            kept.append(line)
            continue
        stripped = line.strip()
        if stripped and line[0].isspace():
            kept.append(line)
            continue
        kind = _kind(stripped)
        if stripped and kind is None:
            seen.clear()
        if kind and not _carries_continuation(lines, index, kind):
            if stripped in seen:
                continue
            seen.add(stripped)
        kept.append(line)
    return "\n".join(kept)
