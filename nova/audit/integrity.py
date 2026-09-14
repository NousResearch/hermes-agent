"""Keeping the audit log finite, and making tampering visible.

Two problems that share a file.

**Finite.** ``audit.jsonl`` grew without bound. A busy deployment writes on every
materialization, every knowledge search and every policy decision, so the log eventually
fills a disk — and the failure mode is an agent platform that can no longer record what it
did while still doing it.

**Tamper-evident.** The compiled policy hands each worker the log path and the enforcement
plugin appends to it, so the process being audited can write to the record. The file is
``0600``, but the worker runs as the same OS user, so the mode is no barrier. Today the
policy prevents an agent reaching a file-write tool at all — which makes the integrity of
the log depend on the control the log exists to evidence.

What is here is **evidence, not prevention**, and the distinction is the whole point.
NOVA cannot stop a process that shares its uid from rewriting a file. It can make the
rewrite detectable:

``nova audit seal``   records the size and digest of every segment at a moment in time.
``nova audit verify`` recomputes them and reports exactly where history stopped matching.

A seal's worth depends entirely on where it is kept. Written beside the log it proves
little — anyone who can rewrite one can rewrite the other. Written to a location the
runtime user cannot reach (an operator's machine, object storage with a write-once policy,
a different host) it is a real control. ``seal`` therefore takes a destination and the
documentation says why, rather than defaulting somewhere convenient and implying a
guarantee that location cannot provide.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

#: Rotate once the live log passes this. 32 MiB is roughly a million events — large enough
#: that a normal deployment rotates rarely, small enough to read with ordinary tools.
DEFAULT_MAX_BYTES = 32 * 1024 * 1024

#: Segments kept beside the live log. Older ones are removed by rotation, so retention is
#: bounded at ``max_bytes * (keep + 1)`` and an operator can size a volume from two numbers.
DEFAULT_KEEP = 9

SEAL_VERSION = 1


def segments(path: Path) -> list[Path]:
    """The live log and its rotated segments, oldest first.

    Oldest first because that is the order events happened in, and every consumer here —
    verification, reading, reporting — wants chronological order rather than filesystem
    order.
    """
    rotated = sorted(
        (p for p in path.parent.glob(f"{path.name}.*") if p.suffix.lstrip(".").isdigit()),
        key=lambda p: int(p.suffix.lstrip(".")),
        reverse=True,
    )
    return [*rotated, path] if path.exists() else rotated


def rotate(path: Path, *, max_bytes: int = DEFAULT_MAX_BYTES, keep: int = DEFAULT_KEEP) -> bool:
    """Roll the log over if it has grown past ``max_bytes``. True when it rotated.

    Renames rather than copies, so the operation is atomic and no event is duplicated.

    A worker holding the old path for the microseconds between ``rename`` and its next
    ``open`` would append to the rotated segment rather than the new live file. That event
    is still recorded, still sealed, and still read back by :func:`read_all` — the segments
    are one log in two files. Choosing rename over a lock is deliberate: a lock shared with
    every worker would make the audit log a coordination primitive, and a governance record
    that can block an agent is worse than one that occasionally lands a line in the previous
    segment.
    """
    try:
        if not path.exists() or path.stat().st_size < max_bytes:
            return False
    except OSError:
        return False

    try:
        # Shift the tail out first so .1 is free, dropping whatever falls past `keep`.
        for index in range(keep, 0, -1):
            source = path.with_suffix(path.suffix + f".{index}")
            if not source.exists():
                continue
            if index == keep:
                source.unlink(missing_ok=True)
            else:
                source.rename(path.with_suffix(path.suffix + f".{index + 1}"))
        path.rename(path.with_suffix(path.suffix + ".1"))
    except OSError:
        # Rotation is housekeeping. Failing it must never fail the write that triggered it,
        # or a full disk would also stop the record of why the disk filled.
        return False
    return True


@dataclass(frozen=True)
class SegmentSeal:
    """One segment's size and digest at seal time."""

    name: str
    size: int
    sha256: str
    lines: int

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "size": self.size, "sha256": self.sha256, "lines": self.lines}


@dataclass(frozen=True)
class Seal:
    """A point-in-time fingerprint of an audit log."""

    version: int
    tenant_id: str
    sealed_at: str
    segments: tuple[SegmentSeal, ...] = ()

    @property
    def lines(self) -> int:
        return sum(segment.lines for segment in self.segments)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "tenant_id": self.tenant_id,
            "sealed_at": self.sealed_at,
            "segments": [segment.to_dict() for segment in self.segments],
            "_comment": (
                "A NOVA audit seal. `nova audit verify --seal <this file>` detects any "
                "change to the events it covers. Keep it somewhere the runtime user "
                "cannot write, or it proves nothing."
            ),
        }

    @classmethod
    def from_dict(cls, data: Any) -> "Seal":
        if not isinstance(data, dict):
            raise ValueError("a seal must be a JSON object")
        return cls(
            version=int(data.get("version", 0)),
            tenant_id=str(data.get("tenant_id", "")),
            sealed_at=str(data.get("sealed_at", "")),
            segments=tuple(
                SegmentSeal(
                    name=str(entry.get("name", "")),
                    size=int(entry.get("size", 0)),
                    sha256=str(entry.get("sha256", "")),
                    lines=int(entry.get("lines", 0)),
                )
                for entry in (data.get("segments") or [])
                if isinstance(entry, dict)
            ),
        )


def digest_file(path: Path) -> tuple[str, int, int]:
    """``(sha256, bytes, lines)`` for one segment, read in chunks."""
    hasher = hashlib.sha256()
    size = 0
    lines = 0
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            hasher.update(chunk)
            size += len(chunk)
            lines += chunk.count(b"\n")
    return hasher.hexdigest(), size, lines


def seal(path: Path, *, tenant_id: str = "") -> Seal:
    """Fingerprint every segment of the log as it stands now."""
    return Seal(
        version=SEAL_VERSION,
        tenant_id=tenant_id,
        sealed_at=datetime.now(timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z"),
        segments=tuple(
            SegmentSeal(name=segment.name, sha256=d, size=s, lines=n)
            for segment in segments(path)
            for d, s, n in [digest_file(segment)]
        ),
    )


@dataclass
class VerifyResult:
    """What verification found."""

    ok: bool = True
    checked: int = 0
    findings: list[str] = field(default_factory=list)
    appended: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "segments_checked": self.checked,
            "events_appended_since_seal": self.appended,
            "findings": list(self.findings),
        }


def verify(path: Path, seal_doc: Seal) -> VerifyResult:
    """Compare the log against a seal.

    Growth is expected and is not a finding: the live segment gains events continuously, so
    a longer file whose sealed prefix still matches byte for byte is exactly what an
    untampered log looks like. What is reported is history *changing* — a segment that
    shrank, one whose sealed prefix no longer matches, or one that has gone.

    Checking the prefix rather than the whole file is what makes the seal usable on a
    running system. A seal that could only be verified against a stopped deployment would
    be verified approximately never.
    """
    result = VerifyResult()
    present = {segment.name: segment for segment in segments(path)}

    for sealed in seal_doc.segments:
        result.checked += 1
        current = present.get(sealed.name)
        if current is None:
            result.ok = False
            result.findings.append(f"{sealed.name}: sealed segment is missing")
            continue

        try:
            size = current.stat().st_size
        except OSError as exc:
            result.ok = False
            result.findings.append(f"{sealed.name}: cannot be read ({exc})")
            continue

        if size < sealed.size:
            result.ok = False
            result.findings.append(
                f"{sealed.name}: shrank from {sealed.size} to {size} bytes — "
                f"{sealed.size - size} bytes of history were removed"
            )
            continue

        prefix, lines = _digest_prefix(current, sealed.size)
        if prefix != sealed.sha256:
            result.ok = False
            result.findings.append(
                f"{sealed.name}: the first {sealed.size} bytes no longer match the seal — "
                "recorded history was modified"
            )
            continue
        result.appended += max(0, _count_lines(current) - lines)

    unsealed = sorted(set(present) - {s.name for s in seal_doc.segments})
    if unsealed:
        # Not a failure: a segment created after the seal is normal. Reported because "the
        # seal covers less than you think" is exactly what someone verifying needs to know.
        result.findings.append(
            f"not covered by this seal (created after it): {', '.join(unsealed)}"
        )
    return result


def _digest_prefix(path: Path, size: int) -> tuple[str, int]:
    hasher = hashlib.sha256()
    lines = 0
    remaining = size
    with open(path, "rb") as handle:
        while remaining > 0:
            chunk = handle.read(min(1024 * 1024, remaining))
            if not chunk:
                break
            hasher.update(chunk)
            lines += chunk.count(b"\n")
            remaining -= len(chunk)
    return hasher.hexdigest(), lines


def _count_lines(path: Path) -> int:
    count = 0
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                return count
            count += chunk.count(b"\n")


def write_seal(seal_doc: Seal, destination: Path) -> Path:
    """Write a seal, refusing to leave it 0644 where anyone can read the digests."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(seal_doc.to_dict(), indent=2, sort_keys=True) + "\n"
    handle = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        os.write(handle, payload.encode("utf-8"))
    finally:
        os.close(handle)
    return destination


def read_seal(path: Path) -> Seal:
    return Seal.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
