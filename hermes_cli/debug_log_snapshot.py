"""Capture consistent log views from one descriptor and one privacy projection."""

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Optional

from hermes_cli.debug_log_redaction import (
    _GENERIC_MESSAGE_MARKERS,
    _LEGACY_MESSAGE_PREVIEW_LOG_RE,
    _LEGACY_LOG_MESSAGE_FIELD_RE,
    _WHATSAPP_CONVERSATION_LOG_RE,
    _WHATSAPP_INBOUND_LOG_RE,
    _WHATSAPP_EXCEPTION_LOG_RE,
    _WhatsAppLogRedactionState,
    _generic_message_opener,
    _has_unescaped_quote,
    _is_safe_whatsapp_error_log_line,
    _looks_like_exception_continuation,
    _redact_log_text_with_state,
)

_WHATSAPP_STATE_SCAN_BYTES = 256 * 1024


class _LogChangedDuringSnapshot(RuntimeError):
    """The descriptor no longer contains the originally selected range."""


@dataclass
class LogSnapshot:
    """Single-read snapshot of a log file used by debug-share."""

    path: Optional[Path]
    tail_text: str
    full_text: Optional[str]


def _whatsapp_log_state_at(
    log_file: BinaryIO,
    byte_offset: int,
    *,
    content_hash: Optional[Any] = None,
) -> _WhatsAppLogRedactionState:
    """Recover redaction state from a bounded suffix of the discarded prefix.

    The exact discarded bytes still come from the same open descriptor and
    are fed into ``content_hash`` when requested, preserving the later
    append/overwrite/truncate verification.  Selector presence is detected
    with a bounded-memory byte scan rather than replaying every ordinary line
    through the Python state machine.  A selected continuation found inside
    the bounded replay window is preserved as redacted message state; an
    older selected record that cannot be replayed is marked unresolved and
    replaced with a safe fragment instead of risking a leak.
    """
    state = _WhatsAppLogRedactionState()
    if byte_offset <= 0:
        return state

    # Use the already-required complete-prefix pass to look for selectors
    # without decoding every ordinary diagnostic line in Python.  If no
    # selector occurs anywhere in
    # the discarded bytes, a large ordinary log is known not to contain an
    # older selected record and can retain its useful diagnostics.  A selector
    # does not prove that its record ended before the retained view, so keep
    # the fail-closed fragment in that case.
    selector_markers = (
        b" whatsapp",
        b"platform=whatsapp",
        b"[whatsapp]",
        b"[whatsapp_cloud]",
        b":whatsapp:",
        b":whatsapp_cloud:",
        b"for whatsapp",
        b"wamid",
        b"media_id",
        b"media id",
        b"processing queued message after agent completion:",
        b"processing pending message:",
        b"delivering leftover /steer as next turn:",
    ) + _GENERIC_MESSAGE_MARKERS
    selector_seen = False
    state_candidate_markers = (
        b"processing queued message after agent completion:",
        b"processing pending message:",
        b"delivering leftover /steer as next turn:",
        b"conversation turn:",
        b"inbound message:",
        b"[whatsapp]",
        b"[whatsapp_cloud]",
    ) + _GENERIC_MESSAGE_MARKERS
    state_opener_before_window = False
    last_candidate_line_start: Optional[int] = None

    def _classify_prefix_candidate(candidate_at: int, resume_at: int) -> None:
        """Classify one pre-window candidate without retaining offsets."""
        nonlocal state_opener_before_window, last_candidate_line_start
        try:
            line_start = _physical_line_start(log_file, candidate_at)
            if line_start == last_candidate_line_start:
                log_file.seek(resume_at)
                return
            last_candidate_line_start = line_start
            log_file.seek(line_start)
            candidate_line = log_file.readline(1_048_576)
        except Exception:
            state_opener_before_window = True
            log_file.seek(resume_at)
            return
        finally:
            # The caller's sequential scan must continue from the end of the
            # chunk even when candidate classification seeks elsewhere.
            if log_file.tell() != resume_at:
                log_file.seek(resume_at)

        if not candidate_line or (
            b"\n" not in candidate_line and len(candidate_line) >= 1_048_576
        ):
            # An unbounded candidate line cannot be classified safely.
            state_opener_before_window = True
            return
        candidate_text = candidate_line.decode("utf-8", errors="replace")
        if _LEGACY_MESSAGE_PREVIEW_LOG_RE.search(
            candidate_text
        ) or _generic_message_opener(candidate_text):
            state_opener_before_window = True
            return
        if _is_safe_whatsapp_error_log_line(candidate_text):
            # A current type/metadata-only warning and an older
            # logger.exception header can be byte-for-byte identical.  Read
            # the next physical line from this same descriptor before
            # declaring the prefix self-contained; a traceback continuation
            # means the retained suffix must remain fail-closed.
            next_at = line_start + len(candidate_line)
            log_file.seek(next_at)
            next_line = log_file.readline(1_048_576)
            next_text = next_line.decode("utf-8", errors="replace")
            if next_line and _looks_like_exception_continuation(next_text):
                state_opener_before_window = True
            log_file.seek(resume_at)
            return
        if _WHATSAPP_EXCEPTION_LOG_RE.search(
            candidate_text
        ) and not _is_safe_whatsapp_error_log_line(candidate_text):
            state_opener_before_window = True
            return
        if not (
            _WHATSAPP_CONVERSATION_LOG_RE.search(candidate_text)
            or _WHATSAPP_INBOUND_LOG_RE.search(candidate_text)
        ):
            return
        message_field = _LEGACY_LOG_MESSAGE_FIELD_RE.search(candidate_text)
        if not message_field:
            return
        message_value = message_field.group(1).rstrip("\r\n")
        if message_value[:1] in {"'", '"'}:
            if not _has_unescaped_quote(message_value[1:], message_value[0]):
                state_opener_before_window = True
        else:
            state_opener_before_window = True

    overlap = b""
    max_marker_len = max(map(len, selector_markers))
    remaining = byte_offset
    processed = 0
    scan_start = max(0, byte_offset - _WHATSAPP_STATE_SCAN_BYTES)
    log_file.seek(0)
    while remaining > 0:
        chunk = log_file.read(min(65536, remaining))
        if not chunk:
            break
        if content_hash is not None:
            content_hash.update(chunk)
        haystack = overlap + chunk.lower()
        if any(marker in haystack for marker in selector_markers):
            selector_seen = True
        for marker in state_candidate_markers:
            search_from = 0
            while True:
                marker_at = haystack.find(marker, search_from)
                if marker_at < 0:
                    break
                absolute_at = max(0, processed + marker_at - len(overlap))
                if absolute_at < scan_start:
                    _classify_prefix_candidate(absolute_at, processed + len(chunk))
                    if state_opener_before_window:
                        break
                search_from = marker_at + 1
            if state_opener_before_window:
                break
        overlap = haystack[-(max_marker_len - 1) :]
        remaining -= len(chunk)
        processed += len(chunk)

    if not selector_seen:
        return state

    # Replay only the final bounded window when a selector is present.  This
    # keeps the useful redacted-message view for the common case where the
    # selected record is recent, while avoiding the old O(file-size) Python
    # line replay for selector-free logs.
    scan_start = max(0, byte_offset - _WHATSAPP_STATE_SCAN_BYTES)
    log_file.seek(scan_start)
    scan_remaining = byte_offset - scan_start
    if scan_start > 0:
        log_file.seek(scan_start - 1)
        at_line_boundary = log_file.read(1) == b"\n"
        log_file.seek(scan_start)
        if not at_line_boundary:
            fragment = log_file.readline(scan_remaining)
            scan_remaining -= len(fragment)

    while scan_remaining > 0:
        line = log_file.readline(scan_remaining)
        if not line:
            break
        scan_remaining -= len(line)
        _unused, state = _redact_log_text_with_state(
            line.decode("utf-8", errors="replace"),
            state,
            redact_output=False,
        )
        if state.message_continuation or state.exception_continuation:
            return state

    if state_opener_before_window:
        # The selected marker predates the bounded replay, so no textual line
        # can prove that an attacker-controlled multiline record ended before
        # the retained view.
        state.prefix_unresolved = True
    return state


def _descriptor_digest(
    log_file: BinaryIO,
    byte_offset: int,
    byte_count: int,
) -> Optional[bytes]:
    """Hash an exact range from an already-open log descriptor."""
    content_hash = hashlib.sha256()
    remaining = byte_count
    log_file.seek(byte_offset)
    while remaining > 0:
        chunk = log_file.read(min(65536, remaining))
        if not chunk:
            return None
        remaining -= len(chunk)
        content_hash.update(chunk)
    return content_hash.digest()


def _physical_line_start(log_file: BinaryIO, byte_offset: int) -> int:
    """Find the start of the physical line containing ``byte_offset``."""
    cursor = max(0, byte_offset)
    while cursor:
        start = max(0, cursor - 65536)
        log_file.seek(start)
        block = log_file.read(cursor - start)
        newline = block.rfind(b"\n")
        if newline >= 0:
            return start + newline + 1
        cursor = start
    return 0


def _split_line_is_whatsapp(
    log_file: BinaryIO,
    byte_offset: int,
    file_size: int,
) -> bool:
    """Classify a retained no-newline suffix using its complete line.

    The retained suffix can begin in the middle of a physical line.  Looking
    only at that suffix would miss a selector split across the byte cap, while
    unconditionally replacing every such suffix destroys ordinary diagnostics.
    Scan the complete line on the already-open descriptor and fail closed when
    it contains WhatsApp-related markers.  The scan keeps bounded overlap and
    memory, and a non-WhatsApp line remains available for sharing.
    """
    line_start = _physical_line_start(log_file, byte_offset)
    remaining = max(0, file_size - line_start)
    log_file.seek(line_start)
    overlap = b""
    markers = (
        b" whatsapp",
        b"platform=whatsapp",
        b"[whatsapp]",
        b"[whatsapp_cloud]",
        b":whatsapp:",
        b":whatsapp_cloud:",
        b"whatsapp chat info",
        b"platform.whatsapp/",
        b"wamid ",
        b"wamid=",
        b"wamid:",
        b"for wamid",
        b"media_id=",
        b"media id=",
        b"media id ",
        # Historical watch-pattern notifications identify the platform with
        # ``for whatsapp[_cloud]`` rather than a ``platform=`` field.  Keep
        # this selector in the complete-line classifier so a byte-cap split
        # cannot expose the retained identity suffix.
        b"for whatsapp",
    ) + _GENERIC_MESSAGE_MARKERS
    while remaining:
        chunk = log_file.read(min(65536, remaining))
        if not chunk:
            break
        remaining -= len(chunk)
        haystack = overlap + chunk.lower()
        if any(marker in haystack for marker in markers):
            return True
        overlap = haystack[-128:]
    return False


def _decode_capped_utf8(data: bytes, max_bytes: int) -> str:
    """Decode a byte-capped view without invalid UTF-8 expansion.

    ``errors='replace'`` can turn an orphaned continuation byte at a suffix
    cut into a three-byte replacement character, making the returned string
    exceed ``max_bytes`` after re-encoding.  Decode the selected suffix while
    ignoring only incomplete/invalid leading bytes, then enforce the physical
    line boundary on the decoded text.  The result is valid UTF-8 and its
    encoded size is always at most the requested cap.
    """
    if max_bytes <= 0:
        return ""
    truncated = len(data) > max_bytes
    on_boundary = True
    selected = data
    if truncated:
        cut = len(data) - max_bytes
        on_boundary = cut > 0 and data[cut - 1 : cut] == b"\n"
        selected = data[cut:]

    text = selected.decode("utf-8", errors="ignore")
    if truncated and not on_boundary and "\n" in text:
        text = text.split("\n", 1)[1]

    # The ignore decode above normally makes this a no-op.  Keep the final
    # invariant explicit for callers handling unusual decoder input.
    while text and len(text.encode("utf-8")) > max_bytes:
        text = text[1:]
    return text


def capture_log_snapshot(
    log_path: Path,
    *,
    tail_lines: int,
    max_bytes: int,
    redact: bool = True,
) -> LogSnapshot:
    """Capture a log once and derive summary/full-log views from it.

    The report tail and standalone log upload must come from the same file
    snapshot. Otherwise a rotation/truncate between reads can make the report
    look newer than the uploaded ``agent.log`` paste.

    When ``redact`` is True (the default), both ``tail_text`` and
    ``full_text`` are run through ``_redact_log_text`` so the snapshot
    returned uses the same privacy projection. The on-disk log file is never modified.
    Pass ``redact=False`` to capture original log content (used by
    ``hermes debug share --no-redact``).
    """
    try:
        with open(log_path, "rb") as f:
            initial_stat = os.fstat(f.fileno())
            initial_fingerprint = (
                initial_stat.st_dev,
                initial_stat.st_ino,
                initial_stat.st_size,
                initial_stat.st_mtime_ns,
            )
            size = initial_stat.st_size
            if size == 0:
                # The file was truncated or replaced before the open completed.
                return LogSnapshot(
                    path=log_path,
                    tail_text="(file empty)",
                    full_text=None,
                )

            if size <= max_bytes:
                # Bind the view to the size observed at open time.  An
                # ordinary append must not leak into this point-in-time view.
                raw = f.read(size)
            else:
                # Read from the end until we have enough bytes for the
                # standalone upload and enough newline context to render the
                # summary tail from the same snapshot.
                chunk_size = 8192
                pos = size
                chunks: list[bytes] = []
                total = 0
                newline_count = 0

                while (
                    pos > 0
                    and (total < max_bytes or newline_count <= tail_lines + 1)
                    and total < max_bytes * 2
                ):
                    read_size = min(chunk_size, pos)
                    pos -= read_size
                    f.seek(pos)
                    chunk = f.read(read_size)
                    chunks.insert(0, chunk)
                    total += len(chunk)
                    newline_count += chunk.count(b"\n")
                    chunk_size = min(chunk_size * 2, 65536)

                raw = b"".join(chunks)

            raw_start = pos if size > max_bytes else 0
            split_physical_line = False
            if raw_start > 0 and raw:
                # Chunk reads begin at an arbitrary byte. Drop the incomplete
                # first physical line, then scan the discarded prefix through
                # this exact boundary so multiline state remains trustworthy.
                first_newline = raw.find(b"\n")
                if first_newline >= 0:
                    raw_start += first_newline + 1
                    raw = raw[first_newline + 1 :]
                else:
                    # No retained newline means the selected suffix may be
                    # the continuation of one physical record.  Its marker
                    # and platform selector can therefore be split across
                    # the discarded prefix and retained bytes.  Do not parse
                    # those fragments as independent records: redact the
                    # entire retained fragment below.
                    split_physical_line = _split_line_is_whatsapp(f, raw_start, size)

            initial_content_hash = hashlib.sha256()
            if redact and not split_physical_line:
                state = _whatsapp_log_state_at(
                    f,
                    raw_start,
                    content_hash=initial_content_hash,
                )
                if state.prefix_unresolved:
                    split_physical_line = True
            else:
                if redact:
                    # Preserve the exact initial descriptor range for the
                    # append-race check without reconstructing state from a
                    # partial physical line.
                    remaining = raw_start
                    f.seek(0)
                    while remaining > 0:
                        chunk = f.read(min(65536, remaining))
                        if not chunk:
                            break
                        remaining -= len(chunk)
                        initial_content_hash.update(chunk)
                state = _WhatsAppLogRedactionState()
            initial_content_hash.update(raw)
            initial_content_digest = initial_content_hash.digest()
            final_stat = os.fstat(f.fileno())
            final_fingerprint = (
                final_stat.st_dev,
                final_stat.st_ino,
                final_stat.st_size,
                final_stat.st_mtime_ns,
            )
            if final_fingerprint != initial_fingerprint:
                same_descriptor = (
                    final_stat.st_dev == initial_stat.st_dev
                    and final_stat.st_ino == initial_stat.st_ino
                )
                append_candidate = same_descriptor and final_stat.st_size > size
                verification_offset = 0 if redact else raw_start
                verification_size = size - verification_offset
                verified_digest = (
                    _descriptor_digest(
                        f,
                        verification_offset,
                        verification_size,
                    )
                    if append_candidate
                    else None
                )
                verified_stat = os.fstat(f.fileno())
                stable_initial_range = (
                    verified_digest == initial_content_digest
                    and verified_stat.st_dev == initial_stat.st_dev
                    and verified_stat.st_ino == initial_stat.st_ino
                    and verified_stat.st_size >= size
                )
                if not append_candidate or not stable_initial_range:
                    raise _LogChangedDuringSnapshot()

        full_raw = raw
        full_was_truncated = raw_start > 0 or len(full_raw) > max_bytes
        if len(full_raw) > max_bytes:
            cut = len(full_raw) - max_bytes
            # Check whether the cut lands exactly on a line boundary.  If the
            # byte just before the cut position is a newline the first retained
            # byte starts a complete line and we should keep it.  Only drop a
            # partial first line when we're genuinely mid-line.
            on_boundary = cut > 0 and full_raw[cut - 1 : cut] == b"\n"
            full_raw = full_raw[cut:]
            if not on_boundary and b"\n" in full_raw:
                full_raw = full_raw.split(b"\n", 1)[1]

        if redact:
            if split_physical_line:
                safe_text = "[REDACTED_LOG_FRAGMENT]\n"
            else:
                safe_text, _state = _redact_log_text_with_state(
                    raw.decode("utf-8", errors="replace"),
                    state,
                    finalize=True,
                )
            tail_text = "".join(
                safe_text.splitlines(keepends=True)[-tail_lines:]
            ).rstrip("\n")

            safe_full_raw = safe_text.encode("utf-8")
            if len(safe_full_raw) > max_bytes:
                # Redaction can expand a selected view (for example, a bare
                # seven-digit WhatsApp identity becomes ``12****67``).  The
                # same line-boundary cap below then omits one or more source
                # records, so preserve the existing truncation marker rather
                # than presenting the shortened view as complete.
                full_was_truncated = True
            full_text = _decode_capped_utf8(safe_full_raw, max_bytes)
        else:
            all_text = raw.decode("utf-8", errors="replace")
            tail_text = "".join(
                all_text.splitlines(keepends=True)[-tail_lines:]
            ).rstrip("\n")
            full_text = _decode_capped_utf8(full_raw, max_bytes)

        if full_was_truncated:
            full_text = (
                f"[... truncated — showing last ~{max_bytes // 1024}KB ...]\n"
                f"{full_text}"
            )

        return LogSnapshot(path=log_path, tail_text=tail_text, full_text=full_text)
    except Exception as exc:
        detail = (
            "log changed during snapshot capture"
            if isinstance(exc, _LogChangedDuringSnapshot)
            else type(exc).__name__
            if redact
            else str(exc)
        )
        return LogSnapshot(
            path=log_path, tail_text=f"(error reading: {detail})", full_text=None
        )
