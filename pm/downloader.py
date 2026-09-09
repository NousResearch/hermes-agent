"""Resumable, hash-verified, multi-connection downloads.

Every large fetch in Hermes goes through this one downloader: pm
packages (pinned sha256 from lock.json) and local models (deliberately
unverified — catalog sizes may lag an upstream re-upload, so sha256 is
optional per source).

Partial state is the downloader's own. Files land in a managed
partials area keyed by sha256(url). One process owns each key while it
transfers or publishes. Complete bytes are copied to destination-local
staging before atomic replacement; a failed copy leaves the old destination
intact. An interrupted download keeps its durable ranges bound to the remote
length, strong ETag and optional pinned hash. Unidentified data restarts
instead of mixing bytes from different upstream versions.

The progress callback reports the whole job AND the per-dest bitmap:
``progress(overall_done, overall_total, ranges)`` where ``ranges`` maps
``dest.name`` to half-open [start, end) runs. ``done_bytes`` is the sum;
``ranges`` is the shape — one datum, two resolutions.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import logging
import threading
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Sequence

# GitHub's release-asset CDN (release-assets.githubusercontent.com, which
# TUR's pool 302s to) 403s unknown tool UAs from CI runner IP ranges --
# their docs require a real User-Agent. A browser-shaped one is the
# least-privileged string every asset CDN accepts.
_UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) hermes-pm/1.0", "Accept-Encoding": "identity"}
_LOOPBACK = ("http://127.0.0.1:", "http://localhost:", "http://[::1]:")
_CHUNK = 1 << 20  # read/write block, also the minimum range size


class _HttpsRedirectHandler(urllib.request.HTTPRedirectHandler):
    """The https-only gate must hold across redirects, not just the first
    hop — an https URL could otherwise bounce to http mid-download and
    carry the payload in the clear."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if not (newurl.startswith("https://") or newurl.startswith(_LOOPBACK)):
            raise DownloadError(f"refusing redirect to non-https url: {newurl}")
        # urllib preserves our User-Agent and range headers. Do not re-add
        # arbitrary headers after its redirect policy has processed them.
        return super().redirect_request(req, fp, code, msg, headers, newurl)


_OPENER = urllib.request.build_opener(_HttpsRedirectHandler())


class DownloadError(RuntimeError):
    """Base class for downloader failures."""


def _validate_range(response, start: int, end: int, total: int | None = None) -> int:
    """A range body is useful only for the exact requested interval."""
    match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", response.headers.get("Content-Range", ""))
    if response.status != 206 or match is None:
        raise _RangeError("server did not honor the requested byte range")
    first, last, size = map(int, match.groups())
    if (first, last) != (start, end - 1) or size < end or (total is not None and size != total):
        raise _RangeError("server returned a different byte range or representation size")
    length = response.headers.get("Content-Length")
    if length is not None and (not length.isdecimal() or int(length) != end - start):
        raise _RangeError("range Content-Length does not match its bounds")
    if response.headers.get("Content-Encoding", "identity").lower() != "identity":
        raise _RangeError("encoded response cannot be written into an identity byte range")
    return size


class _RangeError(DownloadError):
    """Range or representation identity changed; partial bytes cannot be reused."""


@dataclass(frozen=True)
class _Remote:
    total: int
    ranged: bool
    etag: str = ""


def _strong_etag(response) -> str:
    value = response.headers.get("ETag", "")
    return value if value.startswith('"') and value.endswith('"') else ""


class HashError(DownloadError):
    """The downloaded bytes did not match the pinned sha256."""


class DownloadPaused(DownloadError):
    """pause() was called mid-download; partials were left intact."""


@dataclass(frozen=True)
class Source:
    url: str
    dest: Path
    sha256: str = ""  # "" = no integrity check (model catalog policy)


_Ranges = list[tuple[int, int]]
ProgressFn = Callable[[int, int, dict[str, _Ranges]], None]


def _coalesce(ranges: _Ranges) -> _Ranges:
    """Merge half-open [start, end) ranges into sorted, disjoint runs."""
    runs = sorted((a, b) for a, b in ranges if b > a)
    if not runs:
        return []
    out: _Ranges = []
    a, b = runs[0]
    for x, y in runs[1:]:
        if x <= b:
            b = max(b, y)
        else:
            out.append((a, b))
            a, b = x, y
    out.append((a, b))
    return out


def _missing(total: int, covered: _Ranges) -> _Ranges:
    """The gaps in [0, total) not covered by the coalesced bitmap."""
    missing: _Ranges = []
    cursor = 0
    for a, b in _coalesce(covered):
        if a > cursor:
            missing.append((cursor, a))
        cursor = max(cursor, b)
    if cursor < total:
        missing.append((cursor, total))
    return missing


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(_CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def _existing_dest_ok(source: "Source") -> bool:
    """Pinned destinations are rehashed; unpinned model files are accepted as-is.

    Catalog policy does not supply their expected hash or stable length.
    Only this downloader's publication path guarantees complete new files.
    """
    if not source.dest.exists():
        return False
    if not source.sha256:
        return True
    try:
        return _sha256_file(source.dest) == source.sha256
    except OSError:
        return False


class Download:
    """One resumable download job: a plan of sources, run in parallel.

    Per source: probe Range support, preallocate a flat .part in the
    managed partials area, split into at most ``connections`` byte
    ranges, and stream each with a ``Range:`` header from a worker
    thread. A lock-protected bitmap records which ranges are actually
    durable; on resume only the missing ranges are re-fetched.

    ``Source.dest`` is atomically replaced only after successful transfer.
    ``pause()`` stops between chunks or while waiting for a partial's owner;
    ``run()`` raises :class:`DownloadPaused` and preserves resumable bytes.
    """

    CONNECTIONS = 8

    def __init__(
        self,
        sources: Sequence[Source],
        *,
        resume: bool = True,
        connections: int = CONNECTIONS,
        partials_dir: Optional[Path] = None,
    ):
        self.sources = [Source(s.url, Path(s.dest), s.sha256) for s in sources]
        self.resume = resume
        self.connections = max(1, int(connections))
        if partials_dir:
            self.partials_dir = Path(partials_dir)
        else:
            from pm import paths

            self.partials_dir = paths.partials_root()
        self._paused = threading.Event()

    def pause(self) -> None:
        """Request a stop between chunks; run() raises DownloadPaused."""
        self._paused.set()

    def run(self, progress: Optional[ProgressFn] = None) -> list[Path]:
        """Fetch every source; return the moved destination paths."""
        self._paused.clear()
        self.partials_dir.mkdir(parents=True, exist_ok=True)
        for source in self.sources:
            if not (source.url.startswith("https://")
                    or source.url.startswith(_LOOPBACK)):
                raise ValueError(f"refusing non-https url: {source.url}")

        # Probe every source up front so progress covers the whole plan.
        probed = [(source, _Remote(source.dest.stat().st_size, False) if _existing_dest_ok(source)
                   else self._probe(source.url)) for source in self.sources]
        overall_total = sum(remote.total for _, remote in probed)
        completed: dict[str, _Ranges] = {}
        done_base = 0
        from pm.download_state import partial_lock

        for source, initial in probed:
            with partial_lock(self.partials_dir, self._key(source.url), cancelled=self._paused.is_set) as acquired:
                if not acquired or self._paused.is_set():
                    raise DownloadPaused(source.url)
                if _existing_dest_ok(source):
                    size = source.dest.stat().st_size
                else:
                    # Remote data can change while another process owns this URL.
                    remote = self._probe(source.url)
                    overall_total += remote.total - initial.total

                    def tick(covered: _Ranges) -> None:
                        if progress is not None:
                            progress(done_base + sum(end - start for start, end in covered),
                                     overall_total, {**completed, source.dest.name: covered})

                    if remote.total and remote.ranged and (source.sha256 or remote.etag):
                        size = self._fetch_ranged(source, remote, tick)
                    else:
                        size = self._fetch_single(source, remote, tick)
                    overall_total += size - remote.total
                completed[source.dest.name] = [(0, size)]
                done_base += size
                if progress is not None:
                    progress(done_base, overall_total, dict(completed))
        return [source.dest for source in self.sources]

    # ── internals ─────────────────────────────────────────────

    @staticmethod
    def _probe(url: str) -> _Remote:
        req = urllib.request.Request(url, headers={**_UA, "Range": "bytes=0-0"})
        try:
            with _OPENER.open(req, timeout=60) as response:
                etag = _strong_etag(response)
                if response.status == 206:
                    total = _validate_range(response, 0, 1)
                    if len(response.read(2)) != 1:
                        raise _RangeError("range probe returned the wrong byte count")
                    return _Remote(total, True, etag)
                if response.status != 200:
                    raise DownloadError(f"unexpected download probe status: {response.status}")
                return _Remote(int(response.headers.get("Content-Length") or 0), False, etag)
        except urllib.error.HTTPError as exc:
            if exc.code in (401, 403):
                raise DownloadError("the host refused the download; check source access and URL") from exc
            raise
        except DownloadError:
            raise
        except (OSError, ValueError):
            return _Remote(0, False)

    def _key(self, url: str) -> str:
        return hashlib.sha256(url.encode("utf-8")).hexdigest()

    def _load_sidecar(self, side: Path, part: Path, remote: _Remote, sha256: str) -> _Ranges:
        if not self.resume or not (sha256 or remote.etag):
            return []
        try:
            data = json.loads(side.read_text(encoding="utf-8"))
            if (data["total"], data["etag"], data["sha256"]) != (remote.total, remote.etag, sha256):
                return []
            size = part.stat().st_size
            ranges = data["ranges"]
            if not isinstance(ranges, list) or any(
                not isinstance(row, list) or len(row) != 2
                or any(type(value) is not int for value in row)
                or not 0 <= row[0] < row[1] <= min(size, remote.total)
                for row in ranges
            ):
                return []
            return _coalesce([tuple(row) for row in ranges])
        except (OSError, ValueError, KeyError, TypeError):
            return []

    @staticmethod
    def _write_sidecar(side: Path, part: Path, covered: _Ranges, remote: _Remote, sha256: str) -> None:
        import os
        from hermes_cli.runtime_state import _atomic_bytes

        # All writers have closed before coverage is persisted. A sidecar can
        # describe durable bytes, never data still buffered in a worker.
        with part.open("r+b") as stream:
            os.fsync(stream.fileno())
        record = {"total": remote.total, "etag": remote.etag, "sha256": sha256, "ranges": covered}
        _atomic_bytes(side, json.dumps(record).encode("utf-8"))

    def _fetch_ranged(self, source: Source, remote: _Remote, tick) -> int:
        key = self._key(source.url)
        part = self.partials_dir / f"{key}.part"
        side = self.partials_dir / f"{key}.ranges"
        covered = self._load_sidecar(side, part, remote, source.sha256)
        with part.open("r+b" if covered else "w+b") as stream:
            stream.truncate(remote.total)
        gap_count = len(_missing(remote.total, covered)) if covered else (remote.total + _CHUNK - 1) // _CHUNK
        connections = max(1, min(self.connections, gap_count))
        attempts = (connections, 1) if connections > 1 else (1,)
        for connections in attempts:
            covered, errors = self._ranged_attempt(source, remote, part, covered, connections, tick)
            protocol_error = next((error for error in errors if isinstance(error, _RangeError)), None)
            if protocol_error is not None:
                part.unlink(missing_ok=True)
                side.unlink(missing_ok=True)
                raise protocol_error
            self._write_sidecar(side, part, covered, remote, source.sha256)
            if self._paused.is_set():
                raise DownloadPaused(source.url)
            if errors:
                if connections == 1:
                    raise errors[0]
                logging.getLogger(__name__).warning(
                    "parallel range fetch failed for %s; retrying with one connection", source.url,
                )
                continue
            written = sum(end - start for start, end in covered)
            if written != remote.total:
                raise DownloadError(f"download incomplete ({written} of {remote.total} bytes)")
            self._finalize(source, part, side)
            return written
        raise DownloadError("ranged download exhausted its retry")

    def _ranged_attempt(self, source: Source, remote: _Remote, part: Path,
                        covered: _Ranges, connections: int, tick) -> tuple[_Ranges, list[Exception]]:
        from concurrent.futures import ThreadPoolExecutor

        ranges = _missing(remote.total, covered)
        if not covered:
            count = max(1, min(connections, (remote.total + _CHUNK - 1) // _CHUNK))
            ranges = [(index * remote.total // count, (index + 1) * remote.total // count)
                      for index in range(count)]
        lock = threading.Lock()
        errors: list[Exception] = []
        stop = threading.Event()
        covered = list(covered)

        def worker(start: int, end: int) -> None:
            if self._paused.is_set() or stop.is_set():
                return
            try:
                headers = {**_UA, "Range": f"bytes={start}-{end - 1}"}
                if remote.etag:
                    headers["If-Range"] = remote.etag
                request = urllib.request.Request(source.url, headers=headers)
                with _OPENER.open(request, timeout=120) as response, part.open("r+b") as stream:
                    _validate_range(response, start, end, remote.total)
                    if remote.etag and _strong_etag(response) != remote.etag:
                        raise _RangeError("remote representation changed during download")
                    stream.seek(start)
                    position = start
                    while position < end:
                        if self._paused.is_set() or stop.is_set():
                            return
                        chunk = response.read(min(_CHUNK, end - position))
                        if not chunk:
                            raise DownloadError("range body ended before its declared bounds")
                        stream.write(chunk)
                        position += len(chunk)
                        with lock:
                            covered[:] = _coalesce(covered + [(start, position)])
                            snapshot = list(covered)
                        tick(snapshot)
                    if response.read(1):
                        raise _RangeError("range body exceeds its declared bounds")
            except Exception as exc:
                with lock:
                    errors.append(exc)
                stop.set()

        with ThreadPoolExecutor(max_workers=connections, thread_name_prefix="hermes-download") as pool:
            futures = [pool.submit(worker, start, end) for start, end in ranges]
            for future in futures:
                future.result()
        return covered, errors

    def _fetch_single(self, source: Source, remote: _Remote, tick) -> int:
        # A stream with no strong validator cannot safely reuse earlier bytes.
        key = self._key(source.url)
        part = self.partials_dir / f"{key}.part"
        side = self.partials_dir / f"{key}.ranges"
        covered: _Ranges = []
        request = urllib.request.Request(source.url, headers=_UA)
        try:
            with _OPENER.open(request, timeout=120) as response, part.open("wb") as stream:
                if response.status != 200:
                    raise DownloadError(f"unexpected download status: {response.status}")
                if remote.etag and _strong_etag(response) != remote.etag:
                    raise _RangeError("remote representation changed during download")
                if response.headers.get("Content-Encoding", "identity").lower() != "identity":
                    raise DownloadError("encoded response cannot be used as an identity download")
                declared = int(response.headers.get("Content-Length") or 0)
                position = 0
                while True:
                    if self._paused.is_set():
                        raise DownloadPaused(source.url)
                    chunk = response.read(_CHUNK)
                    if not chunk:
                        break
                    stream.write(chunk)
                    position += len(chunk)
                    covered = [(0, position)]
                    tick(list(covered))
                if self._paused.is_set():
                    raise DownloadPaused(source.url)
                if (declared and position != declared) or (remote.total and position != remote.total):
                    raise DownloadError(f"download incomplete ({position} bytes, expected {declared or remote.total})")
        except BaseException:
            if part.exists():
                self._write_sidecar(side, part, covered, remote, source.sha256)
            raise
        self._write_sidecar(side, part, covered, remote, source.sha256)
        self._finalize(source, part, side)
        return position

    def _finalize(self, source: Source, part: Path, side: Path) -> None:
        if source.sha256:
            actual = _sha256_file(part)
            if actual != source.sha256:
                part.unlink(missing_ok=True)
                side.unlink(missing_ok=True)
                raise HashError(
                    f"sha256 mismatch for {source.url}: pinned "
                    f"{source.sha256}, got {actual}")
        import os
        import tempfile

        source.dest.parent.mkdir(parents=True, exist_ok=True)
        fd, name = tempfile.mkstemp(prefix=f".{source.dest.name}-", suffix=".download", dir=source.dest.parent)
        staged = Path(name)
        try:
            with os.fdopen(fd, "wb") as target, part.open("rb") as incoming:
                shutil.copyfileobj(incoming, target, _CHUNK)
                target.flush()
                os.fsync(target.fileno())
            if staged.stat().st_size != part.stat().st_size:
                raise DownloadError("destination copy did not preserve the complete download")
            os.replace(staged, source.dest)
        finally:
            staged.unlink(missing_ok=True)
        part.unlink(missing_ok=True)
        side.unlink(missing_ok=True)
