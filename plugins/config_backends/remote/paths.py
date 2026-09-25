"""Dotted-path grammar of the config plane wire contract (contract.md §3).

A path is a list of mapping keys. On the wire it is one string: each segment escapes ``\\`` as
``\\\\`` and ``.`` as ``\\.``, and segments are joined with ``.``. Coverage (lock matching) is
segment-wise, never a string prefix (§3.4).
"""
from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple

Path = Tuple[str, ...]


class PathError(ValueError):
    """An encoded path that does not parse (§3.3)."""


def encode(segments: Sequence[str]) -> str:
    return ".".join(s.replace("\\", "\\\\").replace(".", "\\.") for s in segments)


def decode(encoded: str) -> Path:
    segments: List[str] = []
    current: List[str] = []
    i = 0
    while i < len(encoded):
        ch = encoded[i]
        if ch == "\\":
            if i + 1 >= len(encoded) or encoded[i + 1] not in "\\.":
                raise PathError(f"invalid escape in config path {encoded!r}")
            current.append(encoded[i + 1])
            i += 2
            continue
        if ch == ".":
            if not current:
                raise PathError(f"empty segment in config path {encoded!r}")
            segments.append("".join(current))
            current = []
        else:
            current.append(ch)
        i += 1
    if not current:
        raise PathError(f"empty segment in config path {encoded!r}")
    segments.append("".join(current))
    validate(segments, encoded)
    return tuple(segments)


MAX_SEGMENTS = 32
MAX_SEGMENT_BYTES = 256
MAX_ENCODED_BYTES = 2048


def validate(segments: Sequence[str], encoded: Optional[str] = None) -> None:
    """§3.1 segment and path limits (the plane rejects violations with ``config_path_invalid``)."""
    shown = encoded if encoded is not None else encode(segments)
    if not 1 <= len(segments) <= MAX_SEGMENTS:
        raise PathError(f"config path {shown!r} must have 1-{MAX_SEGMENTS} segments")
    if len((encoded if encoded is not None else encode(segments)).encode("utf-8")) > MAX_ENCODED_BYTES:
        raise PathError(f"config path is longer than {MAX_ENCODED_BYTES} bytes")
    for seg in segments:
        if not 1 <= len(seg.encode("utf-8")) <= MAX_SEGMENT_BYTES:
            raise PathError(f"config path {shown!r}: a segment must be 1-{MAX_SEGMENT_BYTES} bytes")
        if any(ord(c) < 0x20 or ord(c) == 0x7F for c in seg):
            raise PathError(f"config path {shown!r}: control character in a segment")
        if seg == "__proto__":
            raise PathError(f"reserved segment '__proto__' in config path {shown!r}")


def covers(lock: Sequence[str], path: Sequence[str]) -> bool:
    """``lock`` covers ``path`` when it is a segment-wise prefix of it (equal counts)."""
    return len(lock) <= len(path) and tuple(path[:len(lock)]) == tuple(lock)


def covering_lock(locks: Iterable[Tuple[Path, str]], path: Sequence[str]) -> Optional[Tuple[Path, str]]:
    """The first ``(lock, level)`` whose lock covers ``path``, or None."""
    for lock, level in locks:
        if covers(lock, path):
            return lock, level
    return None


def from_dotted(dotted: str) -> Path:
    """A dotted key as the CLI and in-tree callers spell it. Those callers never escape, so a key
    that does not parse as an encoded path is split on every ``.`` (today's behaviour)."""
    try:
        return decode(dotted)
    except PathError:
        return tuple(dotted.split("."))
