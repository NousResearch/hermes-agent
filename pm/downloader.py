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
the destination path to half-open [start, end) runs. ``done_bytes`` is the sum;
``ranges`` is the shape — one datum, two resolutions.
"""

from __future__ import annotations

import hashlib
import http.client
import json
import os
import re
import shutil
import logging
import socket
import threading
import time
import urllib.error
import urllib.request
from contextlib import contextmanager, suppress
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Sequence

from pm.network import is_transient, retry_network

# GitHub's release-asset CDN (release-assets.githubusercontent.com, which
# TUR's pool 302s to) 403s unknown tool UAs from CI runner IP ranges --
# their docs require a real User-Agent. A browser-shaped one is the
# least-privileged string every asset CDN accepts.
_UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) hermes-pm/1.0", "Accept-Encoding": "identity"}
_LOOPBACK = ("http://127.0.0.1:", "http://localhost:", "http://[::1]:")
_CHUNK = 1 << 20  # read/write block, also the minimum range size
_READ_QUANTUM = 64 << 10
_NETWORK_BUDGET = 60.0
_REQUEST_GUARD: ContextVar["_RequestGuard | None"] = ContextVar("download_request_guard", default=None)


class _TransferStopped(Exception):
    """Another range failed; this worker must close without replacing its cause."""


class _RequestFailure(Exception):
    """An error raised by opening or reading the remote response."""

    def __init__(self, cause):
        self.cause = cause
        super().__init__(str(cause))


class _LocalFailure(Exception):
    """Keep local I/O and observer errors outside the network retry policy."""

    def __init__(self, cause):
        self.cause = cause
        super().__init__(str(cause))


def _retry_request(operation, url, wait):
    def request_only():
        try:
            return operation()
        except _RequestFailure as exc:
            raise exc.cause
        except BaseException as exc:
            # A local TimeoutError is still a local error. Only exceptions
            # originating at the open/read boundary enter retry_network.
            raise _LocalFailure(exc) from exc

    try:
        return retry_network(request_only, wait=wait)
    except _LocalFailure as exc:
        raise exc.cause
    except (OSError, http.client.HTTPException) as exc:
        raise DownloadTransportError(url, exc) from exc


class _RequestGuard:
    """Own the socket and a joined watchdog, never a detached download worker.

    A socket timeout alone resets with every received byte. Bound the whole
    header exchange and each useful body quantum instead: even a trickling
    host must release its source rung, while an active multi-GB transfer has
    no overall duration limit. Local writes and progress observers do not
    consume the network budget.
    """

    def __init__(self, url, paused, stopped=None):
        self.url, self.paused, self.stopped = url, paused, stopped
        self.connection = None
        self.socket = None
        self.deadline = None
        self.failure = None
        self.lock = threading.Lock()
        self.finished = threading.Event()
        self.watcher = threading.Thread(target=self._watch, name="hermes-request", daemon=True)

    def _watch(self):
        while not self.finished.wait(.05):
            with self.lock:
                if self.paused.is_set():
                    self.failure = DownloadPaused(self.url)
                elif self.stopped is not None and self.stopped.is_set():
                    self.failure = _TransferStopped()
                elif self.deadline is not None and time.monotonic() >= self.deadline:
                    self.failure = DownloadTransportError(self.url, TimeoutError(
                        "host stalled beyond the bounded network read budget"))
                if self.failure is None:
                    continue
                active = self.connection.sock if self.connection is not None else None
                active = active if active is not None else self.socket
                if active is not None:
                    # shutdown wakes a socket makefile read; close alone leaves
                    # its file-reference alive until the blocked reader exits.
                    with suppress(OSError):
                        active.shutdown(socket.SHUT_RDWR)

    def check(self):
        if self.paused.is_set():
            raise DownloadPaused(self.url)
        if self.stopped is not None and self.stopped.is_set():
            raise _TransferStopped()
        with self.lock:
            failure = self.failure
            if failure is None and self.deadline is not None and time.monotonic() >= self.deadline:
                failure = self.failure = DownloadTransportError(self.url, TimeoutError(
                    "host stalled beyond the bounded network read budget"))
        if failure is not None:
            raise failure

    def arm(self):
        self.check()
        with self.lock:
            self.deadline = time.monotonic() + _NETWORK_BUDGET

    def disarm(self):
        with self.lock:
            self.deadline = None

    def read(self, response, size):
        chunks = []
        remaining = size
        while remaining:
            self.arm()
            try:
                try:
                    chunk = response.read(min(_READ_QUANTUM, remaining))
                except (OSError, http.client.HTTPException) as exc:
                    self.check()
                    raise _RequestFailure(exc) from exc
                self.check()
            finally:
                self.disarm()
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)


class _GuardedResponse(http.client.HTTPResponse):
    def __init__(self, sock, *args, **kwargs):
        guard = _REQUEST_GUARD.get()
        if guard is not None:
            with guard.lock:
                guard.socket = sock
        super().__init__(sock, *args, **kwargs)


class _GuardedHandlerMixin:
    def do_open(self, http_class, req, **kwargs):
        def connection(*args, **options):
            instance = http_class(*args, **options)
            instance.response_class = _GuardedResponse
            guard = _REQUEST_GUARD.get()
            if guard is not None:
                with guard.lock:
                    guard.connection, guard.socket = instance, None
            return instance

        return super().do_open(connection, req, **kwargs)


class _GuardedHTTPHandler(_GuardedHandlerMixin, urllib.request.HTTPHandler):
    pass


class _GuardedHTTPSConnection(http.client.HTTPSConnection):
    def connect(self):
        # The ordinary wrap_socket call detaches the raw fd and blocks inside
        # the handshake before assigning self.sock. Publish the verified TLS
        # socket first so pause and a failed peer range can wake that handshake.
        # HTTPConnection.connect preserves proxy CONNECT and source-address use.
        http.client.HTTPConnection.connect(self)
        server_hostname = self._tunnel_host or self.host
        self.sock = self._context.wrap_socket(
            self.sock, server_hostname=server_hostname, do_handshake_on_connect=False)
        self.sock.do_handshake()


class _GuardedHTTPSHandler(_GuardedHandlerMixin, urllib.request.HTTPSHandler):
    # HTTPSHandler supplies the ordinary certificate-verifying SSL context;
    # only ownership during its handshake changes.
    def do_open(self, http_class, req, **kwargs):
        return super().do_open(_GuardedHTTPSConnection, req, **kwargs)


class _HttpsRedirectHandler(urllib.request.HTTPRedirectHandler):
    """The https-only gate must hold across redirects, not just the first
    hop — an https URL could otherwise bounce to http mid-download and
    carry the payload in the clear."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        # Plain-http loopback exists for test servers, and only a download that
        # started on loopback may stay there: an https origin bouncing to a local
        # listener would hand an unpinned model file to whatever is bound there.
        loopback = req.full_url.startswith(_LOOPBACK) and newurl.startswith(_LOOPBACK)
        if not (newurl.startswith("https://") or loopback):
            raise DownloadError(f"refusing redirect to non-https url: {newurl}")
        # urllib preserves our User-Agent and range headers. Do not re-add
        # arbitrary headers after its redirect policy has processed them.
        return super().redirect_request(req, fp, code, msg, headers, newurl)


_OPENER = urllib.request.build_opener(_HttpsRedirectHandler(), _GuardedHTTPHandler(), _GuardedHTTPSHandler())


@contextmanager
def _open_request(request, paused, stopped=None):
    guard = _RequestGuard(request.full_url, paused, stopped)
    token = _REQUEST_GUARD.set(guard)
    try:
        guard.arm()
        guard.watcher.start()
        try:
            response = _OPENER.open(request, timeout=_NETWORK_BUDGET)
        except Exception as exc:
            if isinstance(exc, urllib.error.HTTPError):
                exc.close()
            guard.check()
            if isinstance(exc, (OSError, http.client.HTTPException)):
                raise _RequestFailure(exc) from exc
            raise
        guard.disarm()
        with response:
            guard.check()
            yield response, guard
    finally:
        guard.finished.set()
        if guard.watcher.ident is not None:
            guard.watcher.join()
        _REQUEST_GUARD.reset(token)


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


class DownloadTransportError(DownloadError):
    """An exhausted network request, with its original status and URL."""

    def __init__(self, url: str, cause: Exception):
        self.url = url
        self.status = cause.code if isinstance(cause, urllib.error.HTTPError) else None
        self.fallback_allowed = (
            self.status in (401, 403, 404, 410) or is_transient(cause)
        )
        reason = f"{cause}; the host refused access" if self.status in (401, 403) else str(cause)
        super().__init__(f"download failed from {url}: {reason}")


@dataclass(frozen=True)
class Source:
    url: str
    dest: Path
    sha256: str = ""  # "" = no integrity check (model catalog policy)
    fallbacks: tuple[str, ...] = ()


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


# How long a finished download may wait for another process to let go of it before the
# publication is reported as failed.
_RELEASE_WAIT_SECONDS = 60.0


def replace_when_released(tmp: Path, dest: Path, *, timeout: float = _RELEASE_WAIT_SECONDS) -> None:
    """Rename a finished download into place, waiting out a transient hold on the file.

    On Windows a multi-gigabyte file whose last write handle just closed is often still open
    to an antivirus or indexing scan, and renaming it fails with a permission error until the
    scan lets go — which for a 22 GB model can take many seconds. ``os.replace`` is retried
    through that window; it never falls back to copying (``shutil.move`` does, which duplicates
    the whole file and then reports the leftover's failed delete as the download's failure).
    A hold that outlasts the window raises a plain-language error.
    """
    deadline = time.monotonic() + timeout
    delay = 0.1
    while True:
        try:
            os.replace(tmp, dest)
            return
        except PermissionError as exc:
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    f"The download finished, but another program (usually an antivirus scan) kept "
                    f"{tmp.name} open and it could not be renamed into place. Please try again.") from exc
            time.sleep(delay)
            delay = min(delay * 2, 2.0)


def _existing_dest_ok(source: "Source") -> bool:
    """Pinned destinations are rehashed; unpinned model files are accepted as-is.

    Catalog policy does not supply their expected hash or stable length.
    Only this downloader's publication path guarantees complete new files.
    """
    if not source.dest.exists():
        return False
    if not source.dest.is_file():
        raise IsADirectoryError(f"download destination is not a file: {source.dest}")
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
        pause_event: Optional[threading.Event] = None,
    ):
        self.sources = [Source(s.url, Path(s.dest), s.sha256, tuple(s.fallbacks)) for s in sources]
        self.resume = resume
        self.connections = max(1, int(connections))
        if partials_dir:
            self.partials_dir = Path(partials_dir)
        else:
            from pm import paths

            self.partials_dir = paths.partials_root()
        self._owns_pause_event = pause_event is None
        self._paused = pause_event if pause_event is not None else threading.Event()

    def pause(self) -> None:
        """Request a stop between chunks; run() raises DownloadPaused."""
        self._paused.set()

    def run(self, progress: Optional[ProgressFn] = None) -> list[Path]:
        """Fetch every source; return the moved destination paths."""
        if self._owns_pause_event:
            self._paused.clear()
        self._check_pause()
        self.partials_dir.mkdir(parents=True, exist_ok=True)
        for source in self.sources:
            if source.fallbacks and not re.fullmatch(r"[a-f0-9]{64}", source.sha256):
                raise ValueError("mirror fallback requires a full lowercase SHA256")
            for url in (source.url, *source.fallbacks):
                if not (url.startswith("https://") or url.startswith(_LOOPBACK)):
                    raise ValueError(f"refusing non-https url: {url}")

        # Probe the whole plan before reporting a denominator. A missing
        # Content-Length keeps the bar indeterminate until that file ends.
        from pm.download_state import partial_lock

        totals: dict[str, int] = {}
        unknown: set[str] = set()
        coverage: dict[str, _Ranges] = {}
        selected: list[Source] = []
        failures: dict[str, list[DownloadTransportError]] = {}
        for source in self.sources:
            self._check_pause()
            key = str(source.dest)
            failures[key] = []
            if _existing_dest_ok(source):
                remote = _Remote(source.dest.stat().st_size, False)
                covered = [(0, remote.total)]
            else:
                source, remote = self._try_sources(source, lambda candidate: self._probe(candidate.url), failures[key])
                covered = []
                with partial_lock(self.partials_dir, self._key(source.url), wait=False) as acquired:
                    if acquired and remote.ranged:
                        covered = self._partial_ranges(source, remote)
                if not remote.total:
                    unknown.add(key)
            selected.append(source)
            totals[key] = remote.total
            coverage[key] = covered

        def report(key: str, written: _Ranges, total: int, *, complete: bool = False) -> None:
            totals[key] = total
            if total or complete:
                unknown.discard(key)
            else:
                unknown.add(key)
            # The last key identifies the source reporting this tick.
            coverage.pop(key, None)
            coverage[key] = list(written)
            if progress is not None:
                done = sum(b - a for rows in coverage.values() for a, b in rows)
                progress(done, 0 if unknown else sum(totals.values()), dict(coverage))

        for source in selected:
            key = str(source.dest)

            def tick(written: _Ranges, total: int) -> None:
                report(key, written, total)

            _, size = self._try_sources(source, lambda candidate: self._transfer(candidate, tick), failures[key])
            report(key, [(0, size)], size, complete=True)
        return [source.dest for source in self.sources]

    def _try_sources(self, source: Source, operation, failures: list[DownloadTransportError]):
        urls = tuple(dict.fromkeys((source.url, *source.fallbacks)))
        for index, url in enumerate(urls):
            self._check_pause()
            candidate = Source(url, source.dest, source.sha256, urls[index + 1:])
            try:
                return candidate, operation(candidate)
            except DownloadTransportError as exc:
                failures.append(exc)
                if not exc.fallback_allowed or index == len(urls) - 1:
                    if len(failures) == 1:
                        raise
                    raise DownloadError("\n".join(str(error) for error in failures)) from exc
                logging.getLogger(__name__).debug("%s; trying pinned mirror %s", exc, urls[index + 1])

    def _transfer(self, source: Source, tick) -> int:
        from pm.download_state import partial_lock

        with partial_lock(self.partials_dir, self._key(source.url), cancelled=self._paused.is_set) as acquired:
            if not acquired:
                raise DownloadPaused(source.url)
            self._check_pause()
            if _existing_dest_ok(source):
                return source.dest.stat().st_size
            # Recheck identity after waiting for another process's partial.
            remote = self._probe(source.url)
            tick(self._partial_ranges(source, remote) if remote.ranged else [], remote.total)

            def report(written: _Ranges) -> None:
                tick(written, remote.total)

            def fetch() -> int:
                self._check_pause()
                if remote.total and remote.ranged and (source.sha256 or remote.etag):
                    return self._fetch_ranged(source, remote, report)
                return self._fetch_single(source, remote, report)

            configured_connections = self.connections
            try:
                size = _retry_request(fetch, source.url, self._wait_retry)
            finally:
                self.connections = configured_connections
            partial_key = self._key(source.url)
            self._finalize(source, self.partials_dir / f"{partial_key}.part",
                           self.partials_dir / f"{partial_key}.ranges")
            return size

    # ── internals ─────────────────────────────────────────────

    def _check_pause(self) -> None:
        if self._paused.is_set():
            raise DownloadPaused("download paused")

    def _wait_retry(self, delay: float) -> None:
        if self._paused.wait(delay):
            raise DownloadPaused("download paused during retry backoff")

    def _probe(self, url: str) -> _Remote:
        def request():
            self._check_pause()
            req = urllib.request.Request(url, headers={**_UA, "Range": "bytes=0-0"})
            with _open_request(req, self._paused) as (response, guard):
                etag = _strong_etag(response)
                if response.status == 206:
                    total = _validate_range(response, 0, 1)
                    body = guard.read(response, 2)
                    if not body:
                        raise _RequestFailure(http.client.IncompleteRead(b"", 1))
                    if len(body) != 1:
                        raise _RangeError("range probe returned the wrong byte count")
                    return _Remote(total, True, etag)
                if response.status != 200:
                    raise DownloadError(f"unexpected download probe status: {response.status}")
                return _Remote(int(response.headers.get("Content-Length") or 0), False, etag)
        try:
            return _retry_request(request, url, self._wait_retry)
        except ValueError:
            return _Remote(0, False)

    def _key(self, url: str) -> str:
        return hashlib.sha256(url.encode("utf-8")).hexdigest()

    def _load_sidecar(self, side: Path, part: Path, remote: _Remote, sha256: str) -> _Ranges:
        if not self.resume or not (sha256 or remote.etag):
            return []
        try:
            data = json.loads(side.read_text(encoding="utf-8-sig"))
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

    def _partial_ranges(self, source: Source, remote: _Remote) -> _Ranges:
        key = self._key(source.url)
        return self._load_sidecar(self.partials_dir / f"{key}.ranges",
                                  self.partials_dir / f"{key}.part", remote, source.sha256)

    @staticmethod
    def _write_sidecar(side: Path, part: Path, covered: _Ranges, remote: _Remote, sha256: str) -> None:
        from pm.filesystem import durable_write_bytes

        # All writers have closed before coverage is persisted. A sidecar can
        # describe durable bytes, never data still buffered in a worker.
        with part.open("r+b") as stream:
            os.fsync(stream.fileno())
        record = {"total": remote.total, "etag": remote.etag, "sha256": sha256, "ranges": covered}
        durable_write_bytes(side, json.dumps(record).encode("utf-8"))

    def _fetch_ranged(self, source: Source, remote: _Remote, tick) -> int:
        key = self._key(source.url)
        part = self.partials_dir / f"{key}.part"
        side = self.partials_dir / f"{key}.ranges"
        covered = self._load_sidecar(side, part, remote, source.sha256)
        with part.open("r+b" if covered else "w+b") as stream:
            stream.truncate(remote.total)
        while True:
            gap_count = len(_missing(remote.total, covered)) if covered else (remote.total + _CHUNK - 1) // _CHUNK
            connections = max(1, min(self.connections, gap_count))
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
                if self.connections > 1 and all(
                    isinstance(error, _RequestFailure)
                    and isinstance(error.cause, urllib.error.HTTPError)
                    and error.cause.code in (403, 404)
                    for error in errors
                ):
                    logging.getLogger(__name__).warning(
                        "parallel range fetch refused for %s; retrying with one connection", source.url,
                    )
                    self.connections = 1
                    continue
                raise next((error for error in errors
                            if not isinstance(error, _RequestFailure) or not is_transient(error.cause)), errors[0])
            written = sum(end - start for start, end in covered)
            if written != remote.total:
                raise _RequestFailure(http.client.IncompleteRead(b"", remote.total - written))
            return written

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
        updated = threading.Event()
        pending = len(ranges)
        covered = list(covered)
        reported = list(covered)

        def worker(start: int, end: int) -> None:
            nonlocal pending
            try:
                if self._paused.is_set() or stop.is_set():
                    return
                headers = {**_UA, "Range": f"bytes={start}-{end - 1}"}
                if remote.etag:
                    headers["If-Range"] = remote.etag
                request = urllib.request.Request(source.url, headers=headers)
                with _open_request(request, self._paused, stop) as (response, guard), part.open("r+b") as stream:
                    _validate_range(response, start, end, remote.total)
                    if remote.etag and _strong_etag(response) != remote.etag:
                        raise _RangeError("remote representation changed during download")
                    stream.seek(start)
                    position = start
                    while position < end:
                        if self._paused.is_set() or stop.is_set():
                            return
                        chunk = guard.read(response, min(_CHUNK, end - position))
                        if not chunk:
                            raise _RequestFailure(http.client.IncompleteRead(b"", end - position))
                        stream.write(chunk)
                        position += len(chunk)
                        with lock:
                            covered[:] = _coalesce(covered + [(start, position)])
                            updated.set()
                    if guard.read(response, 1):
                        raise _RangeError("range body exceeds its declared bounds")
            except _TransferStopped:
                return
            except Exception as exc:
                if isinstance(exc, urllib.error.HTTPError):
                    exc.close()
                with lock:
                    errors.append(exc)
                stop.set()
            finally:
                with lock:
                    pending -= 1
                    updated.set()

        with ThreadPoolExecutor(max_workers=connections, thread_name_prefix="hermes-download") as pool:
            futures = [pool.submit(worker, start, end) for start, end in ranges]
            observer_failed = False
            finished = not ranges
            while not finished:
                updated.wait()
                with lock:
                    snapshot = list(covered)
                    finished = pending == 0
                    updated.clear()
                # Only the coordinator observes progress. Slow UI/worker IPC
                # coalesces intermediate snapshots, never holds up range writes.
                if not observer_failed and snapshot != reported:
                    try:
                        tick(snapshot)
                    except Exception as exc:
                        with lock:
                            errors.append(exc)
                        observer_failed = True
                        stop.set()
                    except BaseException:
                        stop.set()
                        raise
                    reported = snapshot

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
            with _open_request(request, self._paused) as (response, guard), part.open("wb") as stream:
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
                    chunk = guard.read(response, _CHUNK)
                    if not chunk:
                        break
                    stream.write(chunk)
                    position += len(chunk)
                    covered = [(0, position)]
                    tick(list(covered))
                if self._paused.is_set():
                    raise DownloadPaused(source.url)
                if position < (declared or remote.total):
                    raise _RequestFailure(http.client.IncompleteRead(b"", (declared or remote.total) - position))
                if (declared and position != declared) or (remote.total and position != remote.total):
                    raise DownloadError(f"download incomplete ({position} bytes, expected {declared or remote.total})")
        except BaseException:
            if part.exists():
                self._write_sidecar(side, part, covered, remote, source.sha256)
            raise
        self._write_sidecar(side, part, covered, remote, source.sha256)
        return position

    def _finalize(self, source: Source, part: Path, side: Path) -> None:
        self._check_pause()
        if source.sha256:
            actual = _sha256_file(part)
            if actual != source.sha256:
                # Best effort: a leftover that cannot be removed must not mask the hash error.
                with suppress(OSError):
                    part.unlink(missing_ok=True)
                    side.unlink(missing_ok=True)
                raise HashError(
                    f"sha256 mismatch for {source.url}: pinned "
                    f"{source.sha256}, got {actual}")
        import tempfile

        self._check_pause()
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
            self._check_pause()
            replace_when_released(staged, source.dest)
        finally:
            # Best effort: a leftover that cannot be removed must not hide the error that left it.
            with suppress(OSError):
                staged.unlink(missing_ok=True)
        # The published file is what the caller asked for; a partial the OS still holds must not
        # turn a completed download into a failure.
        with suppress(OSError):
            part.unlink(missing_ok=True)
            side.unlink(missing_ok=True)
