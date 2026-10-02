"""Read only host-granted local text references as bounded, versioned evidence."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import stat
from typing import Callable, Mapping

from .runner import Evidence, ToolBudgetExceeded, _positive_int


_TEXT_SUFFIXES = frozenset({".txt", ".md", ".markdown"})


@dataclass(frozen=True)
class LocalDocumentGrant:
    """A host-issued opaque reference's exact path, scope, and optional version pin."""

    root: Path
    path: Path
    start_line: int = 1
    end_line: int | None = None
    expected_version: str | None = None

    def __post_init__(self):
        if not isinstance(self.root, Path) or not isinstance(self.path, Path):
            raise TypeError("host must bind local paths")
        if not self.root.is_absolute() or not self.path.is_absolute():
            raise ValueError("host must bind absolute paths")
        if not _positive_int(self.start_line) or (
            self.end_line is not None and
            (not _positive_int(self.end_line) or self.end_line < self.start_line)
        ):
            raise ValueError("invalid approved line window")
        if self.expected_version is not None and (
            not isinstance(self.expected_version, str) or not self.expected_version
        ):
            raise ValueError("invalid expected version")


def _version(info: os.stat_result) -> str:
    # File identity plus metadata detects replacement and in-place rewrites.
    return "stat-v1:" + ":".join(str(value) for value in (
        info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns,
    ))


def _shared_file_state(info: os.stat_result) -> tuple[int, int, int, int]:
    # Windows can report different ctime_ns for a handle and its path.
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns


def _lexical(path: Path) -> Path:
    return Path(os.path.abspath(path))


def _checked_path(grant: LocalDocumentGrant) -> Path:
    root, path = _lexical(grant.root), _lexical(grant.path)
    if not path.is_relative_to(root):
        raise PermissionError("document path outside approved root")
    # Reject links/junctions in every component, including the root's ancestors.
    for component in (*reversed(path.parents), path):
        info = component.lstat()
        if stat.S_ISLNK(info.st_mode) or (
            getattr(info, "st_file_attributes", 0) & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
        ):
            raise PermissionError("document path redirects through a link")
        if component == path and not stat.S_ISREG(info.st_mode):
            raise ValueError("document is not a regular file")
    return path


def _decode_prefix(data: bytes, *, clipped: bool) -> bytes:
    try:
        data.decode("utf-8")
        return data
    except UnicodeDecodeError as exc:
        if clipped and exc.reason == "unexpected end of data" and exc.end == len(data):
            prefix = data[:exc.start]
            prefix.decode("utf-8")
            return prefix
        raise


class ScopedDocumentEvidenceReader:
    """The model supplies only an opaque key; the host owns every path and limit."""

    def __init__(self, *, grants: Mapping[str, LocalDocumentGrant],
                 max_bytes: int = 65536, max_lines: int = 256):
        if not isinstance(grants, Mapping) or not grants or not all(
            isinstance(ref, str) and ref and isinstance(grant, LocalDocumentGrant)
            for ref, grant in grants.items()
        ):
            raise ValueError("host-issued document grants required")
        if not _positive_int(max_bytes) or not _positive_int(max_lines):
            raise ValueError("finite positive document bounds required")
        self._grants = dict(grants)
        self.max_bytes, self.max_lines = max_bytes, max_lines

    @staticmethod
    def _evidence(ref: str, path: str, status: str, coverage: str,
                  payload: dict, digest_bytes: bytes | None = None) -> Evidence:
        if digest_bytes is None:
            digest_bytes = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
        return Evidence(
            source_kind="document", exact_reference=ref,
            digest="sha256:" + hashlib.sha256(digest_bytes).hexdigest(),
            location=path, read_at=datetime.now(timezone.utc).isoformat(),
            status=status, coverage=coverage, content=payload,
        )

    def read(self, reference: str, reserve_read: Callable[[], None]) -> Evidence:
        if not isinstance(reference, str) or reference not in self._grants:
            raise PermissionError("document reference outside host-bound scope")
        grant = self._grants[reference]
        # Charge the underlying read before stat, path traversal, or open.
        reserve_read()
        path = str(grant.path)
        if grant.path.suffix.lower() not in _TEXT_SUFFIXES:
            return self._evidence(reference, path, "error", "unsupported document type; no bytes read",
                                  {"error_type": "UnsupportedDocumentType"})
        try:
            checked = _checked_path(grant)
            flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
            fd = os.open(checked, flags)
            with os.fdopen(fd, "rb") as stream:
                before = os.fstat(stream.fileno())
                if not stat.S_ISREG(before.st_mode):
                    raise ValueError("document is not a regular file")
                identity = f"{before.st_dev}:{before.st_ino}"
                version = _version(before)
                if grant.expected_version is not None and grant.expected_version != version:
                    raise RuntimeError("document version differs from approved version")
                on_path_before = checked.lstat()
                if _shared_file_state(on_path_before) != _shared_file_state(before):
                    raise RuntimeError("document changed before read")

                chunks: list[bytes] = []
                first_byte = last_byte = 0
                first_line = last_line = 0
                scanned = 0
                clipped = False
                while scanned < self.max_lines and stream.tell() < before.st_size:
                    remaining = self.max_bytes - stream.tell()
                    if remaining <= 0:
                        break
                    offset = stream.tell()
                    line = stream.readline(remaining + 1)
                    scanned += 1
                    line_number = scanned
                    if len(line) > remaining:
                        line = line[:remaining]
                        clipped = True
                    if line_number >= grant.start_line:
                        if not chunks:
                            first_byte, first_line = offset, line_number
                        chunks.append(line)
                        last_byte, last_line = offset + len(line), line_number
                    if clipped or (grant.end_line is not None and line_number >= grant.end_line):
                        break

                has_more = stream.tell() < before.st_size
                raw = b"".join(chunks)
                raw = _decode_prefix(raw, clipped=clipped)
                last_byte = first_byte + len(raw) if chunks else 0
                after = os.fstat(stream.fileno())
                on_path = checked.lstat()
                if (_version(after) != version or
                    _version(on_path) != _version(on_path_before) or
                    _shared_file_state(on_path) != _shared_file_state(after)):
                    raise RuntimeError("document changed during read")
                # Recheck ancestors to catch a path redirected during the read.
                _checked_path(grant)

            full = grant.start_line == 1 and not has_more and not clipped
            status = "complete" if full else "truncated"
            scope = "full-file" if full else "window"
            location = f"{checked}:lines:{first_line}-{last_line}:bytes:{first_byte}-{last_byte}"
            coverage = (f"{scope}; lines {first_line}-{last_line}; bytes {first_byte}-{last_byte} "
                        f"of {before.st_size}; scanned {scanned} lines; "
                        f"approved lines {grant.start_line}-{grant.end_line or 'end'}")
            payload = {"text": raw.decode("utf-8"), "file_id": identity, "version": version,
                       "fingerprint_scope": scope, "line_start": first_line, "line_end": last_line,
                       "byte_start": first_byte, "byte_end": last_byte,
                       "file_bytes": before.st_size, "read_only_data": True}
            # A partial digest covers exactly the returned bytes and is never a full-file digest.
            digest_input = raw if full else (f"window:{identity}:{version}:{first_byte}:{last_byte}:".encode() + raw)
            return self._evidence(reference, location, status, coverage, payload, digest_input)
        except ToolBudgetExceeded:
            raise
        except FileNotFoundError:
            return self._evidence(reference, path, "missing", "approved local file absent; no content covered",
                                  {"error_type": "FileNotFoundError"})
        except (OSError, ValueError, RuntimeError) as exc:
            kind = ("VersionChanged" if isinstance(exc, RuntimeError) else
                    "PermissionDenied" if isinstance(exc, PermissionError) else
                    "UnicodeDecodeError" if isinstance(exc, UnicodeDecodeError) else
                    "InvalidDocument" if isinstance(exc, ValueError) else type(exc).__name__)
            return self._evidence(reference, path, "error", "approved local file unreadable; no content covered",
                                  {"error_type": kind})
