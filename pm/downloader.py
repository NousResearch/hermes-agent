"""Resumable, hash-verified, multi-connection downloads.

Every large fetch in Hermes goes through this one downloader: pm
packages (pinned sha256 from lock.json) and local models (deliberately
unverified — catalog sizes may lag an upstream re-upload, so sha256 is
optional per source).

Partial state is the downloader's own. Files land in a managed
partials area keyed by sha256(url), and a finished file is MOVED into
its real destination, so a caller only ever sees a complete file or
nothing. An interrupted download leaves its partial + a .ranges sidecar
(the durable byte-range bitmap) behind, and the next run re-fetches
exactly the ranges the bitmap says are missing.

The progress callback reports the whole job AND the per-dest bitmap:
``progress(overall_done, overall_total, ranges)`` where ``ranges`` maps
the destination path to half-open [start, end) runs. ``done_bytes`` is the sum;
``ranges`` is the shape — one datum, two resolutions.
"""

from __future__ import annotations

import hashlib
import json
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
_UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) hermes-pm/1.0"}
_LOOPBACK = ("http://127.0.0.1:", "http://localhost:", "http://[::1]:")
_CHUNK = 1 << 20  # read/write block, also the minimum range size


class _HttpsRedirectHandler(urllib.request.HTTPRedirectHandler):
    """The https-only gate must hold across redirects, not just the first
    hop — an https URL could otherwise bounce to http mid-download and
    carry the payload in the clear."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if not (newurl.startswith("https://") or newurl.startswith(_LOOPBACK)):
            raise DownloadError(f"refusing redirect to non-https url: {newurl}")
        forwarded = super().redirect_request(req, fp, code, msg, headers, newurl)
        if forwarded is not None:
            # urllib's default redirect DROPS custom headers (rebuilds the
            # request from the URL alone). Release-asset CDNs (GitHub's,
            # TUR's) 403 requests without a real User-Agent, so the pin
            # fetch died on the redirect hop. Carry our headers forward.
            carried = dict(req.headers)
            carried.pop("Host", None)
            for k, v in carried.items():
                forwarded.add_header(k, v)
        return forwarded


_OPENER = urllib.request.build_opener(_HttpsRedirectHandler())


class DownloadError(RuntimeError):
    """Base class for downloader failures."""


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
    """A dest already on disk counts as done only when it is a genuine
    complete download. No pinned hash (model-catalog policy): a present
    file is accepted — size is the tripwire, same as a fresh fetch. With a
    pinned sha256 the existing file must match, or it is re-fetched."""
    if not source.dest.exists():
        return False
    if not source.sha256:
        return True
    try:
        return _sha256_file(source.dest) == source.sha256
    except OSError:
        return False


# `pm gc` must not delete partials an in-flight download is still writing.
# A sidecar's mtime is the heartbeat: it is (re)written whenever a download
# pauses or errors, and the .part grows on every chunk. Anything touched
# within the grace window is live; everything older is fair game.
GC_GRACE_SECONDS = 6 * 60 * 60


def gc_protected_names(
    partials_dir: Path, *, grace_seconds: float = GC_GRACE_SECONDS
) -> set[str]:
    """Entry names under ``partials_dir`` an in-flight (or recently
    interrupted) download still owns — ``<key>.part`` / ``<key>.ranges``
    pairs whose newest mtime is within ``grace_seconds`` of now.

    gc sweeps call this BEFORE rmtree-ing store entries: partials live in
    the store root, and the old name-prefix skip (``startswith(".")``)
    does not cover the "partials" directory. A caller deletes only the
    names absent from the returned set.
    """
    import time

    partials_dir = Path(partials_dir)
    now = time.time()
    protected: set[str] = set()
    if not partials_dir.is_dir():
        return protected
    for entry in partials_dir.iterdir():
        try:
            # Strict < : grace_seconds=0 protects nothing (the test
            # contract for "sweep everything"). mtime can tick a hair
            # NEWER than `now` on some filesystems, which <= would
            # wrongly treat as inside a zero window.
            if grace_seconds > 0 and now - entry.stat().st_mtime < grace_seconds:
                protected.add(entry.name)
        except OSError:
            protected.add(entry.name)  # unreadable → assume live
    for name in tuple(protected):
        path = Path(name)
        if path.suffix in (".part", ".ranges"):
            protected.update({path.with_suffix(".part").name, path.with_suffix(".ranges").name})
    return protected


class Download:
    """One resumable download job: a plan of sources, run in parallel.

    Per source: probe Range support, preallocate a flat .part in the
    managed partials area, split into at most ``connections`` byte
    ranges, and stream each with a ``Range:`` header from a worker
    thread. A lock-protected bitmap records which ranges are actually
    durable; on resume only the missing ranges are re-fetched.

    A finished source is MOVED into ``Source.dest``. ``pause()`` stops
    between chunks and ``run()`` raises :class:`DownloadPaused`,
    leaving every partial + sidecar intact for a later resume.
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
        self.sources = [Source(s.url, Path(s.dest), s.sha256) for s in sources]
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
            if not (source.url.startswith("https://")
                    or source.url.startswith(_LOOPBACK)):
                raise ValueError(f"refusing non-https url: {source.url}")

        # Probe the whole plan before reporting a denominator. A missing
        # Content-Length keeps the bar indeterminate until that file ends.
        probed = []
        totals: dict[str, int] = {}
        unknown: set[str] = set()
        coverage: dict[str, _Ranges] = {}
        for source in self.sources:
            self._check_pause()
            key = str(source.dest)
            cached = _existing_dest_ok(source)
            if cached:
                total, supported = source.dest.stat().st_size, True
                covered = [(0, total)]
            else:
                if source.dest.exists():
                    source.dest.unlink(missing_ok=True)
                total, supported = self._probe(source.url)
                covered = self._partial_ranges(source.url, total) if supported else []
                if not total:
                    unknown.add(key)
            totals[key] = total
            coverage[key] = _coalesce(covered)
            probed.append((source, total, supported, cached))

        progress_lock = threading.Lock()

        def report(key: str, written: _Ranges) -> None:
            with progress_lock:
                # The last key identifies the source reporting this tick.
                coverage.pop(key, None)
                coverage[key] = list(written)
                if progress is not None:
                    done = sum(b - a for rows in coverage.values() for a, b in rows)
                    progress(done, 0 if unknown else sum(totals.values()), dict(coverage))

        moved: list[Path] = []
        for source, total, supported, cached in probed:
            self._check_pause()
            key = str(source.dest)
            report(key, coverage[key])
            if not cached:
                def tick(written: _Ranges) -> None:
                    report(key, written)

                if total and supported:
                    self._fetch_ranged(source, total, tick)
                else:
                    self._fetch_single(source, total, tick)
            size = source.dest.stat().st_size
            totals[key] = size
            unknown.discard(key)
            report(key, [(0, size)])
            moved.append(source.dest)
        return moved

    # ── internals ─────────────────────────────────────────────

    def _check_pause(self) -> None:
        if self._paused.is_set():
            raise DownloadPaused("download paused")

    @staticmethod
    def _probe(url: str) -> tuple[int, bool]:
        """(total bytes, range_supported). A server that ignores Range
        reports its Content-Length instead; 0 means unknown (the
        single-stream fallback judges completeness by the body)."""
        req = urllib.request.Request(url, headers={**_UA, "Range": "bytes=0-0"})
        try:
            with _OPENER.open(req, timeout=60) as r:
                if r.status == 206:
                    content_range = r.headers.get("Content-Range", "")
                    if "/" in content_range:
                        return int(content_range.rsplit("/", 1)[1]), True
                    return 0, False
                length = int(r.headers.get("Content-Length") or 0)
                return length, False
        except urllib.error.HTTPError as exc:
            if exc.code in (401, 403):
                raise DownloadError(
                    "the host refused the download (gated or moved). "
                    "This is a catalog problem, not yours — please report it."
                ) from exc
            raise
        except Exception:  # noqa: BLE001 - unknown length, fall back
            return 0, False

    def _key(self, url: str) -> str:
        return hashlib.sha256(url.encode("utf-8")).hexdigest()

    def _load_sidecar(self, side: Path) -> _Ranges:
        if not self.resume or not side.is_file():
            return []
        try:
            return [tuple(r) for r in json.loads(side.read_text(encoding="utf-8"))]
        except (OSError, ValueError):
            return []

    def _partial_ranges(self, url: str, total: int) -> _Ranges:
        key = self._key(url)
        part = self.partials_dir / f"{key}.part"
        if not part.is_file():
            return []
        size = part.stat().st_size
        covered = self._load_sidecar(self.partials_dir / f"{key}.ranges")
        if any(a < 0 or b < a or b > min(size, total) for a, b in covered):
            return []
        return _coalesce(covered)

    @staticmethod
    def _write_sidecar(side: Path, covered: _Ranges) -> None:
        side.parent.mkdir(parents=True, exist_ok=True)
        side.write_text(json.dumps(covered), encoding="utf-8")

    def _fetch_ranged(self, source: Source, total: int, tick) -> None:
        key = self._key(source.url)
        part = self.partials_dir / f"{key}.part"
        side = self.partials_dir / f"{key}.ranges"
        covered = self._partial_ranges(source.url, total)
        # Create the file WITHOUT truncating an existing partial (resume),
        # then size it to total (extends with zeros / shrinks stale tails).
        open(part, "ab").close()
        with open(part, "r+b") as f:
            f.truncate(total)
        if covered:
            ranges = _missing(total, covered)
        else:
            n = max(1, min(self.connections, (total + _CHUNK - 1) // _CHUNK))
            ranges = [(i * total // n, (i + 1) * total // n) for i in range(n)]
            # NOT coalesced: the fresh split is the parallel fan-out, and
            # adjacent chunks must stay separate workers. (_missing already
            # returns disjoint gaps, so resume needs no merge either.)
            ranges = [r for r in ranges if r[1] > r[0]]

        lock = threading.Lock()
        errors: list[Exception] = []
        stop = threading.Event()
        written = [sum(b - a for a, b in covered)]

        def worker(start: int, end: int) -> None:
            try:
                req = urllib.request.Request(
                    source.url,
                    headers={**_UA, "Range": f"bytes={start}-{end - 1}"})
                with _OPENER.open(req, timeout=120) as r, \
                        open(part, "r+b") as f:
                    f.seek(start)
                    pos = start
                    while pos < end:
                        if self._paused.is_set() or stop.is_set():
                            return
                        chunk = r.read(min(_CHUNK, end - pos))
                        if not chunk:
                            break
                        f.write(chunk)
                        pos += len(chunk)
                        with lock:
                            covered[:] = _coalesce(covered + [(start, pos)])
                            written[0] += len(chunk)
                            tick(list(covered))
            except Exception as exc:  # noqa: BLE001
                with lock:
                    errors.append(exc)
                stop.set()

        threads = [
            threading.Thread(target=worker, args=b, daemon=True,
                             name=f"dl-{key[:8]}-{i}")
            for i, b in enumerate(ranges)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        if self._paused.is_set():
            self._write_sidecar(side, covered)
            raise DownloadPaused(source.url)
        if errors:
            self._write_sidecar(side, covered)
            # Parallel ranged GETs trip edge/CDN rate limits that a single
            # connection does not (observed live: every single-request probe
            # green while the 4-thread fetch 404ed, same runner, same URL).
            # Retry once single-stream; the digest check still proves the
            # bytes. A partial from the failed attempt is kept (covered is
            # persisted above), so the retry only fetches the gaps.
            if self.connections > 1:
                logging.getLogger(__name__).warning(
                    "parallel ranged fetch failed for %s (%s); retrying single-stream",
                    source.url, errors[0],
                )
                saved = self.connections
                try:
                    self.connections = 1
                    self._fetch_ranged(source, total, tick)
                finally:
                    self.connections = saved
                return
            raise errors[0]
        if written[0] != total:
            self._write_sidecar(side, covered)
            raise DownloadError(
                f"download incomplete ({written[0]} of {total} bytes)")
        self._write_sidecar(side, covered)
        self._finalize(source, part, side)

    def _fetch_single(self, source: Source, total: int, tick) -> None:
        """No-Range fallback: one stream. A server without Range support
        cannot resume, so each run restarts the file."""
        key = self._key(source.url)
        part = self.partials_dir / f"{key}.part"
        side = self.partials_dir / f"{key}.ranges"
        covered: _Ranges = []
        req = urllib.request.Request(source.url, headers=_UA)
        try:
            with _OPENER.open(req, timeout=120) as r, open(part, "wb") as f:
                declared = int(r.headers.get("Content-Length") or 0)
                pos = 0
                while True:
                    if self._paused.is_set():
                        self._write_sidecar(side, covered)
                        raise DownloadPaused(source.url)
                    chunk = r.read(_CHUNK)
                    if not chunk:
                        break
                    f.write(chunk)
                    pos += len(chunk)
                    covered = [(0, pos)]
                    tick(list(covered))
                if self._paused.is_set():
                    self._write_sidecar(side, covered)
                    raise DownloadPaused(source.url)
                if declared and pos != declared:
                    raise DownloadError(
                        f"download ended at {pos:,} bytes but the server "
                        f"said {declared:,} — connection dropped?")
                if total and pos != total:
                    raise DownloadError(
                        f"download incomplete ({pos} of {total} bytes)")
        except DownloadError:
            self._write_sidecar(side, covered)
            raise
        except Exception:
            self._write_sidecar(side, covered)
            raise
        self._write_sidecar(side, covered)
        self._finalize(source, part, side)

    def _finalize(self, source: Source, part: Path, side: Path) -> None:
        self._check_pause()
        if source.sha256:
            actual = _sha256_file(part)
            if actual != source.sha256:
                part.unlink(missing_ok=True)
                side.unlink(missing_ok=True)
                raise HashError(
                    f"sha256 mismatch for {source.url}: pinned "
                    f"{source.sha256}, got {actual}")
        self._check_pause()
        source.dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(part), str(source.dest))
        side.unlink(missing_ok=True)
