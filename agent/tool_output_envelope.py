"""Recoverable, capped envelopes for large tool outputs.

The agent transcript should not carry giant raw tool blobs.  When a string tool
result is too large, cache the full bytes under the active Hermes home and put a
bounded JSON receipt + excerpt in the conversation instead.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any, Iterable, Mapping

from hermes_constants import get_hermes_home

ENVELOPE_TYPE = "hermes_tool_output_envelope"
RECEIPT_TYPE = "hermes_tool_output_receipt"
DEFAULT_MAX_TRANSCRIPT_BYTES = 32 * 1024
_CACHE_DIR_NAME = "tool-output-cache"
_SAFE_STEM_RE = re.compile(r"[^A-Za-z0-9_.-]+")
_MAX_SAFE_STEM_CHARS = 80


def _raw_bytes(raw_content: str | bytes) -> bytes:
    if isinstance(raw_content, bytes):
        return raw_content
    return str(raw_content).encode("utf-8", errors="replace")


def _coerce_max_bytes(max_transcript_bytes: int | None) -> int:
    if max_transcript_bytes is None:
        return DEFAULT_MAX_TRANSCRIPT_BYTES
    try:
        value = int(max_transcript_bytes)
    except (TypeError, ValueError):
        return DEFAULT_MAX_TRANSCRIPT_BYTES
    return value if value > 0 else DEFAULT_MAX_TRANSCRIPT_BYTES


def _safe_stem(value: str, fallback: str) -> str:
    stem = _SAFE_STEM_RE.sub("_", str(value or fallback)).strip("._-")
    if not stem:
        stem = fallback
    return stem[:_MAX_SAFE_STEM_CHARS].rstrip("._-") or fallback


def _cache_root(cache_root: Path | None = None) -> Path:
    hermes_home = get_hermes_home().resolve()
    root = (Path(cache_root) if cache_root is not None else hermes_home / _CACHE_DIR_NAME).resolve()
    try:
        root.relative_to(hermes_home)
    except ValueError as exc:
        raise ValueError(f"cache_root must live under HERMES_HOME ({hermes_home})") from exc
    return root


def _cache_path(tool_name: str, tool_call_id: str, digest: str, cache_root: Path | None = None) -> Path:
    root = _cache_root(cache_root)
    tool_stem = _safe_stem(tool_name, "tool")
    call_stem = _safe_stem(tool_call_id, "call")
    return root / digest[:2] / f"{tool_stem}_{call_stem}_{digest[:16]}.txt"


def _decode_excerpt(raw: bytes, max_bytes: int) -> tuple[str, int]:
    if max_bytes <= 0:
        return "", 0
    clipped = raw[:max_bytes]
    text = clipped.decode("utf-8", errors="replace")
    encoded = text.encode("utf-8", errors="replace")
    while len(encoded) > max_bytes and text:
        text = text[:-1]
        encoded = text.encode("utf-8", errors="replace")
    return text, len(encoded)


def _looks_binaryish(text: str | bytes) -> bool:
    raw = text if isinstance(text, bytes) else str(text)
    if isinstance(raw, bytes):
        if b"\x00" in raw:
            return True
        sample = raw[:4096]
        if not sample:
            return False
        controls = sum(1 for b in sample if b < 32 and b not in (9, 10, 13))
        return controls / len(sample) > 0.05
    if "\x00" in raw:
        return True
    sample = raw[:4096]
    if not sample:
        return False
    controls = sum(1 for ch in sample if ord(ch) < 32 and ch not in "\t\n\r")
    return controls / len(sample) > 0.05


def _json_bytes(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8", errors="replace"
    )


def _dumps(payload: Mapping[str, Any]) -> str:
    return _json_bytes(payload).decode("utf-8")


def _payload_ref_is_valid(payload: Mapping[str, Any]) -> bool:
    full_ref = payload.get("full_ref")
    full_hash = payload.get("full_hash")
    if not isinstance(full_ref, str) or not isinstance(full_hash, str):
        return False
    if not re.fullmatch(r"[0-9a-f]{64}", full_hash):
        return False
    try:
        ref = Path(full_ref).resolve()
        cache_root = _cache_root()
        ref.relative_to(cache_root)
    except Exception:
        return False
    if not ref.is_file():
        return False
    try:
        return hashlib.sha256(ref.read_bytes()).hexdigest() == full_hash
    except Exception:
        return False


def _is_valid_envelope_payload(payload: Any) -> bool:
    return (
        isinstance(payload, dict)
        and payload.get("type") in {ENVELOPE_TYPE, RECEIPT_TYPE}
        and _payload_ref_is_valid(payload)
    )


def _loads_envelope_payload(content: str) -> dict[str, Any] | None:
    try:
        payload = json.loads(content)
    except Exception:
        payload = None
    if _is_valid_envelope_payload(payload):
        return payload

    marker_positions = [
        idx for marker in (f'"type":"{ENVELOPE_TYPE}"', f'"type":"{RECEIPT_TYPE}"')
        for idx in [content.find(marker)]
        if idx >= 0
    ]
    if not marker_positions:
        return None

    decoder = json.JSONDecoder()
    marker_at = min(marker_positions)
    for start in [idx for idx, char in enumerate(content[:marker_at]) if char == "{"]:
        try:
            candidate, _ = decoder.raw_decode(content[start:])
        except Exception:
            continue
        if _is_valid_envelope_payload(candidate):
            return candidate
    return None


def _referenced_cache_paths(contents: Iterable[str]) -> set[Path]:
    refs: set[Path] = set()
    for content in contents:
        if not isinstance(content, str):
            continue
        payload = _loads_envelope_payload(content)
        if payload is None:
            continue
        try:
            refs.add(Path(str(payload.get("full_ref") or "")).resolve())
        except Exception:
            continue
    return refs


def _safe_metadata(metadata: Mapping[str, Any] | None) -> dict[str, Any]:
    safe: dict[str, Any] = {}
    for key, value in dict(metadata or {}).items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            safe[str(key)] = value
        else:
            safe[str(key)] = repr(value)
    return safe


def _build_payload(
    *,
    tool_name: str,
    tool_call_id: str,
    full_ref: str,
    full_hash: str,
    raw_size: int,
    excerpt: str,
    excerpt_bytes: int,
    warnings: list[str],
    metadata: Mapping[str, Any] | None,
) -> dict[str, Any]:
    return {
        "type": ENVELOPE_TYPE,
        "tool_name": tool_name,
        "excerpt": excerpt,
        "full_ref": full_ref,
        "full_hash": full_hash,
        "raw_bytes": raw_size,
        "excerpt_bytes": excerpt_bytes,
        "was_truncated": True,
        "warnings": warnings,
        "metadata": {"tool_call_id": tool_call_id, **_safe_metadata(metadata)},
    }


def _fit_excerpt_to_cap(
    *,
    raw: bytes,
    base_payload: dict[str, Any],
    max_transcript_bytes: int,
) -> dict[str, Any]:
    payload = dict(base_payload)
    payload["excerpt"] = ""
    payload["excerpt_bytes"] = 0
    overhead = len(_json_bytes(payload))
    excerpt_limit = max(0, max_transcript_bytes - overhead - 16)

    while True:
        excerpt, excerpt_size = _decode_excerpt(raw, excerpt_limit)
        payload["excerpt"] = excerpt
        payload["excerpt_bytes"] = excerpt_size
        actual = len(_json_bytes(payload))
        if actual <= max_transcript_bytes or excerpt_limit <= 0:
            if actual > max_transcript_bytes and "envelope_receipt_exceeds_cap" not in payload["warnings"]:
                payload["warnings"] = [*payload["warnings"], "envelope_receipt_exceeds_cap"]
            return payload
        excerpt_limit = max(0, excerpt_limit - (actual - max_transcript_bytes) - 8)


def maybe_envelope_tool_output(
    tool_name: str,
    raw_content: str | bytes,
    *,
    tool_call_id: str = "",
    max_transcript_bytes: int | None = None,
    metadata: Mapping[str, Any] | None = None,
    cache_root: str | Path | None = None,
) -> str:
    """Return raw small content or a capped, recoverable JSON envelope.

    The cap applies to the serialized transcript payload, not just the excerpt,
    so tests can assert that the model-facing content stays bounded.  The full
    raw bytes are always written before the envelope is returned.
    """
    max_bytes = _coerce_max_bytes(max_transcript_bytes)
    raw = _raw_bytes(raw_content)
    if len(raw) <= max_bytes:
        return raw_content.decode("utf-8", errors="replace") if isinstance(raw_content, bytes) else raw_content

    digest = hashlib.sha256(raw).hexdigest()
    path = _cache_path(tool_name, tool_call_id, digest, Path(cache_root) if cache_root is not None else None)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)

    warnings = ["truncated_to_excerpt"]
    if _looks_binaryish(raw_content):
        warnings.append("binaryish_content")

    base_payload = _build_payload(
        tool_name=tool_name,
        tool_call_id=tool_call_id,
        full_ref=str(path),
        full_hash=digest,
        raw_size=len(raw),
        excerpt="",
        excerpt_bytes=0,
        warnings=warnings,
        metadata=metadata,
    )
    payload = _fit_excerpt_to_cap(raw=raw, base_payload=base_payload, max_transcript_bytes=max_bytes)
    return _dumps(payload)


def is_tool_output_envelope(content: Any) -> bool:
    if not isinstance(content, str):
        return False
    return _loads_envelope_payload(content) is not None


def compact_envelope_receipt(content: str) -> str:
    """Return a compact receipt that preserves recoverability but drops excerpt."""
    payload = _loads_envelope_payload(content)
    if payload is None:
        return content

    receipt = {
        "type": RECEIPT_TYPE,
        "tool_name": payload.get("tool_name", "unknown"),
        "full_ref": payload.get("full_ref", ""),
        "full_hash": payload.get("full_hash", ""),
        "raw_bytes": payload.get("raw_bytes", 0),
        "excerpt_bytes": payload.get("excerpt_bytes", 0),
        "was_truncated": bool(payload.get("was_truncated", True)),
        "warnings": list(payload.get("warnings") or []),
        "metadata": dict(payload.get("metadata") or {}),
    }
    return _dumps(receipt)


def prune_tool_output_cache(
    *,
    referenced_contents: Iterable[str] = (),
    max_age_seconds: int = 7 * 24 * 60 * 60,
    max_total_bytes: int | None = None,
    now: float | None = None,
    cache_root: str | Path | None = None,
) -> dict[str, int]:
    """Prune old unreferenced cached tool outputs.

    Referenced envelope/receipt files are always kept. Unreferenced files older
    than ``max_age_seconds`` are removed; an optional LRU pass then removes the
    oldest remaining unreferenced files until ``max_total_bytes`` is satisfied.
    """
    root = _cache_root(Path(cache_root) if cache_root is not None else None)
    referenced = _referenced_cache_paths(referenced_contents)
    current_time = time.time() if now is None else float(now)
    max_age = max(0, int(max_age_seconds))
    removed_count = 0
    removed_bytes = 0
    kept_referenced = 0
    candidates: list[tuple[float, Path, int]] = []

    if not root.exists():
        return {
            "removed_count": 0,
            "removed_bytes": 0,
            "kept_referenced_count": 0,
            "remaining_bytes": 0,
        }

    for path in root.rglob("*"):
        if not path.is_file():
            continue
        resolved = path.resolve()
        try:
            stat = path.stat()
        except OSError:
            continue
        size = int(stat.st_size)
        if resolved in referenced:
            kept_referenced += 1
            candidates.append((stat.st_mtime, path, size))
            continue
        if current_time - stat.st_mtime > max_age:
            try:
                path.unlink()
                removed_count += 1
                removed_bytes += size
            except OSError:
                pass
        else:
            candidates.append((stat.st_mtime, path, size))

    remaining_bytes = sum(size for _, path, size in candidates if path.exists())
    if max_total_bytes is not None:
        budget = max(0, int(max_total_bytes))
        for _mtime, path, size in sorted(candidates, key=lambda item: item[0]):
            if remaining_bytes <= budget:
                break
            resolved = path.resolve()
            if resolved in referenced or not path.exists():
                continue
            try:
                path.unlink()
                removed_count += 1
                removed_bytes += size
                remaining_bytes -= size
            except OSError:
                pass

    for directory in sorted((p for p in root.rglob("*") if p.is_dir()), reverse=True):
        try:
            directory.rmdir()
        except OSError:
            pass

    return {
        "removed_count": removed_count,
        "removed_bytes": removed_bytes,
        "kept_referenced_count": kept_referenced,
        "remaining_bytes": max(0, remaining_bytes),
    }
