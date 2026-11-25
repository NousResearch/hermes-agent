"""Resumable, hash-verified, multi-connection downloads.

Every large fetch in Hermes goes through this one downloader: pm
packages (pinned sha256 from lock.json) and local models (deliberately
unverified — catalog sizes may lag an upstream re-upload, so sha256 is
optional per source).

Transport is https, plus plain http on loopback (test servers) and on the
user's own npm registry when it is http (``Source.http_mirror``, #150032).
Bytes from that registry are pinned by sha256 and bounded in size before the
hash check; anything it refuses or cannot serve falls back to the next source.

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
import ssl
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Sequence

from pm.network import edge_block, is_transient, retry_network

# GitHub's release-asset CDN (release-assets.githubusercontent.com, which
# TUR's pool 302s to) 403s unknown tool UAs from CI runner IP ranges --
# their docs require a real User-Agent. A browser-shaped one is the
# least-privileged string every asset CDN accepts.
_UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) hermes-pm/1.0", "Accept-Encoding": "identity"}
_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})
_CHUNK = 1 << 20  # read/write block, also the minimum range size
_CHECKPOINT_SECONDS = 5.0
# The largest npm-hosted artifact in pm/lock.json is ~25 MB (a pnpm executable).
# Lock rows pin no size, so this ceiling is what keeps a plain-http mirror (or
# whoever sits on its path) from filling the disk before the hash check runs.
_HTTP_MIRROR_MAX_BYTES = 256 << 20
_PINNED = re.compile(r"[a-f0-9]{64}")


def _pinned(sha256: str) -> bool:
    return _PINNED.fullmatch(sha256) is not None


def _host(url: str) -> str | None:
    return urllib.parse.urlsplit(url).hostname


def _loopback(url: str) -> bool:
    """Plain http to a local test server. Parsed, not prefix-matched:
    ``http://127.0.0.1:@evil.com/`` names evil.com as its host."""
    try:
        parts = urllib.parse.urlsplit(url)
        host = parts.hostname
    except ValueError:  # malformed bracketed host
        return False
    return parts.scheme == "http" and "@" not in parts.netloc and host in _LOOPBACK_HOSTS


class _HttpsRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Redirects keep the transport rule: an https hop never lands on http.

    Only a request already on plain http (loopback, or the user's http npm
    registry) may be redirected to http, and only to the same host: registries
    such as Nexus or Artifactory 30x tarballs for path normalisation or to a blob
    port. Another http host is refused, so an injected 302 cannot aim a request
    at arbitrary LAN or loopback services; the pinned hash still decides the bytes.
    """

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        # An https origin bouncing to a local listener would hand an unpinned
        # model file to whatever is bound there.
        loopback = _loopback(req.full_url) and _loopback(newurl)
        same_host_http = (req.full_url.startswith("http://") and newurl.startswith("http://")
                          and _host(req.full_url) == _host(newurl))
        if not (newurl.startswith("https://") or loopback or same_host_http):
            raise MirrorRefused(req.full_url, f"refusing redirect to non-https url: {newurl}")
        # urllib preserves our User-Agent and range headers. Do not re-add
        # arbitrary headers after its redirect policy has processed them.
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _https_context() -> ssl.SSLContext:
    # Trust a CA the store holds even when its own issuer is absent: Windows can
    # cache Let's Encrypt "Root YR" without ISRG Root X1. Python 3.13+ does this
    # by default; the older interpreter that starts a cold update does not.
    context = ssl.create_default_context()
    context.verify_flags |= ssl.VERIFY_X509_PARTIAL_CHAIN
    return context


_OPENER = urllib.request.build_opener(
    _HttpsRedirectHandler(), urllib.request.HTTPSHandler(context=_https_context())
)


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
        # A host that does not resolve is not worth a retry, but the pinned
        # mirror is a different host and the hash still guards the bytes.
        unresolved = isinstance(getattr(cause, "reason", cause), socket.gaierror)
        # Any 5xx is the upstream host failing, not its bytes changing: a CDN's
        # origin-down codes (520-527) are what a dead third-party host returns.
        self.fallback_allowed = (
            self.status in (401, 403, 404, 410) or (self.status or 0) >= 500
            or unresolved or is_transient(cause)
        )
        refused = edge_block(cause) or ("the host refused access" if self.status in (401, 403) else None)
        reason = f"{cause}; {refused}" if refused else str(cause)
        super().__init__(f"download failed from {url}: {reason}")


class MirrorRefused(DownloadTransportError):
    """A source the transport policy will not use (a refused redirect, an oversized
    plain-http body); the pinned fallback is tried, as for an unreachable host."""

    def __init__(self, url: str, reason: str):
        self.url, self.status, self.fallback_allowed = url, None, True
        DownloadError.__init__(self, f"download failed from {url}: {reason}")


@dataclass(frozen=True)
class Source:
    url: str
    dest: Path
    sha256: str = ""  # "" = no integrity check (model catalog policy)
    fallbacks: tuple[str, ...] = ()
    # The user's plain-http npm registry (pinned_source sets it). Only ``url``
    # under this prefix may use http; fallbacks and other pins stay https.
    http_mirror: str = ""

    def plain_http(self) -> bool:
        return bool(self.http_mirror) and self.url.startswith(self.http_mirror) and _pinned(self.sha256)

    def byte_limit(self) -> int | None:
        return _HTTP_MIRROR_MAX_BYTES if self.plain_http() else None


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


def _existing_dest_ok(source: Source) -> bool:
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
        self.sources = [Source(s.url, Path(s.dest), s.sha256, tuple(s.fallbacks), s.http_mirror) for s in sources]
        self.resume = resume
        self.connections = max(1, int(connections))
        if partials_dir:
            self.partials_dir = Path(partials_dir)
        else:
            from pm import paths

            self.partials_dir = paths.partials_root()
        self._owns_pause_event = pause_event is None
        self._paused = pause_event if pause_event is not None else threading.Event()
        self._durable_done = 0

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
            if source.fallbacks and not _pinned(source.sha256):
                raise ValueError("mirror fallback requires a full lowercase SHA256")
            # A pinned SHA256 is verified before anything is published, so the user's
            # http npm registry (Verdaccio, Nexus) cannot change what installs (#150032).
            for url in (source.url, *source.fallbacks):
                plain = url == source.url and source.plain_http()
                if not (url.startswith("https://") or _loopback(url) or plain):
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
                source, remote = self._try_sources(source, lambda candidate: self._probe(candidate.url, candidate.byte_limit()), failures[key])
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
            candidate = Source(url, source.dest, source.sha256, urls[index + 1:], source.http_mirror)
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
            remote = self._probe(source.url, source.byte_limit())
            covered = self._partial_ranges(source, remote) if remote.ranged else []
            self._durable_done = sum(end - start for start, end in covered)
            tick(covered, remote.total)
            resumable = bool(remote.total and remote.ranged and (source.sha256 or remote.etag))

            def report(written: _Ranges) -> None:
                tick(written, remote.total)

            def fetch() -> int:
                self._check_pause()
                if resumable:
                    return self._fetch_ranged(source, remote, report)
                return self._fetch_single(source, remote, report)

            configured_connections = self.connections
            try:
                size = retry_network(fetch, wait=self._wait_retry,
                                     progress=(lambda: self._durable_done) if self.resume and resumable else None,
                                     fallback_available=bool(source.fallbacks))
            except (OSError, http.client.HTTPException) as exc:
                if isinstance(exc, urllib.error.URLError) or is_transient(exc):
                    raise DownloadTransportError(source.url, exc) from exc
                raise
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

    def _probe(self, url: str, limit: int | None = None) -> _Remote:
        def request():
            self._check_pause()
            req = urllib.request.Request(url, headers={**_UA, "Range": "bytes=0-0"})
            with _OPENER.open(req, timeout=60) as response:
                etag = _strong_etag(response)
                if response.status == 206:
                    total = _validate_range(response, 0, 1)
                    body = response.read(2)
                    if not body:
                        raise http.client.IncompleteRead(b"", 1)
                    if len(body) != 1:
                        raise _RangeError("range probe returned the wrong byte count")
                    return _Remote(total, True, etag)
                if response.status != 200:
                    raise DownloadError(f"unexpected download probe status: {response.status}")
                return _Remote(int(response.headers.get("Content-Length") or 0), False, etag)
        try:
            remote = retry_network(request, wait=self._wait_retry)
        except (OSError, http.client.HTTPException) as exc:
            raise DownloadTransportError(url, exc) from exc
        except DownloadError:
            raise
        except ValueError:
            return _Remote(0, False)
        if limit is not None and remote.total > limit:
            raise MirrorRefused(url, f"declares {remote.total} bytes, over the {limit}-byte plain-http limit")
        return remote

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

    def _partial_ranges(self, source: Source, remote: _Remote) -> _Ranges:
        key = self._key(source.url)
        return self._load_sidecar(self.partials_dir / f"{key}.ranges",
                                  self.partials_dir / f"{key}.part", remote, source.sha256)

    @staticmethod
    def _write_sidecar(side: Path, part: Path, covered: _Ranges, remote: _Remote, sha256: str) -> None:
        from pm.filesystem import durable_write_bytes, retry_held

        # Workers flush before announcing coverage. Snapshot those ranges,
        # fsync the data, then atomically publish only that durable snapshot.
        with part.open("r+b") as stream:
            os.fsync(stream.fileno())
        record = {"total": remote.total, "etag": remote.etag, "sha256": sha256, "ranges": covered}
        retry_held(lambda: durable_write_bytes(side, json.dumps(record).encode("utf-8")))

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
            self._durable_done = sum(end - start for start, end in covered)
            if self._paused.is_set():
                raise DownloadPaused(source.url)
            if errors:
                if self.connections > 1 and all(
                    isinstance(error, urllib.error.HTTPError) and error.code in (403, 404)
                    for error in errors
                ):
                    logging.getLogger(__name__).warning(
                        "parallel range fetch refused for %s; retrying with one connection", source.url,
                    )
                    self.connections = 1
                    continue
                raise next((error for error in errors if not is_transient(error)), errors[0])
            written = self._durable_done
            if written != remote.total:
                raise http.client.IncompleteRead(b"", remote.total - written)
            return written

    def _read_range(self, source: Source, remote: _Remote, part: Path,
                    start: int, end: int, stop: threading.Event, record) -> None:
        if self._paused.is_set() or stop.is_set():
            return
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
                    raise http.client.IncompleteRead(b"", end - position)
                stream.write(chunk)
                stream.flush()
                position += len(chunk)
                # A complete checkpoint must not bypass the body-bounds check
                # if the process dies between the last write and EOF validation.
                if position == end and response.read(1):
                    raise _RangeError("range body exceeds its declared bounds")
                record(start, position)

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
        checkpointed = list(covered)
        next_checkpoint = time.monotonic() + _CHECKPOINT_SECONDS

        def record(start: int, position: int) -> None:
            with lock:
                covered[:] = _coalesce(covered + [(start, position)])
                updated.set()

        def worker(start: int, end: int) -> None:
            nonlocal pending
            try:
                self._read_range(source, remote, part, start, end, stop, record)
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
                updated.wait(max(0.0, next_checkpoint - time.monotonic()))
                with lock:
                    snapshot = list(covered)
                    finished = pending == 0
                    updated.clear()
                now = time.monotonic()
                if now >= next_checkpoint:
                    if snapshot != checkpointed:
                        try:
                            self._write_sidecar(part.with_suffix(".ranges"), part, snapshot, remote, source.sha256)
                        except OSError as exc:
                            logging.getLogger(__name__).warning(
                                "could not save download progress for %s; continuing: %s", source.dest.name, exc,
                            )
                        except BaseException:
                            stop.set()
                            raise
                        else:
                            self._durable_done = sum(end - start for start, end in snapshot)
                            checkpointed = snapshot
                    next_checkpoint = time.monotonic() + _CHECKPOINT_SECONDS
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
            with _OPENER.open(request, timeout=120) as response, part.open("wb") as stream:
                if response.status != 200:
                    raise DownloadError(f"unexpected download status: {response.status}")
                if remote.etag and _strong_etag(response) != remote.etag:
                    raise _RangeError("remote representation changed during download")
                if response.headers.get("Content-Encoding", "identity").lower() != "identity":
                    raise DownloadError("encoded response cannot be used as an identity download")
                declared = int(response.headers.get("Content-Length") or 0)
                limit = source.byte_limit()
                if limit is not None and declared > limit:
                    raise MirrorRefused(source.url, f"declares {declared} bytes, over the {limit}-byte plain-http limit")
                position = 0
                while True:
                    if self._paused.is_set():
                        raise DownloadPaused(source.url)
                    chunk = response.read(_CHUNK if limit is None else min(_CHUNK, limit - position + 1))
                    if not chunk:
                        break
                    if limit is not None and position + len(chunk) > limit:
                        raise MirrorRefused(source.url, f"sent more than the {limit}-byte plain-http limit")
                    stream.write(chunk)
                    position += len(chunk)
                    covered = [(0, position)]
                    tick(list(covered))
                if self._paused.is_set():
                    raise DownloadPaused(source.url)
                if position < (declared or remote.total):
                    raise http.client.IncompleteRead(b"", (declared or remote.total) - position)
                if (declared and position != declared) or (remote.total and position != remote.total):
                    raise DownloadError(f"download incomplete ({position} bytes, expected {declared or remote.total})")
        except MirrorRefused:
            part.unlink(missing_ok=True)
            side.unlink(missing_ok=True)
            raise
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
