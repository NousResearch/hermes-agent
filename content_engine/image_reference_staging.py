"""Safe, provider-free staging for explicitly supplied image references.

The stager never performs network I/O itself. Its caller must inject a bounded
fetcher, making this module testable and preventing an accidental network path
from becoming available before the explicit execution lane is approved.
"""
from __future__ import annotations

import hashlib
import ipaddress
import json
import re
import shutil
import socket
from dataclasses import asdict, dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import Callable, Iterable, Mapping
from urllib.error import HTTPError
from urllib.parse import urljoin, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

from image_jobs import ImageRequestError, ReferenceRequest, _validate_reference_url


MAX_REFERENCE_BYTES = 10 * 1024 * 1024
MAX_REDIRECTS = 3
MAX_TEXT_CHARS = 20_000
_ALLOWED_CONTENT_TYPES = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
    "text/html": ".html",
    "text/plain": ".txt",
}
_JOB_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}\Z")
Resolver = Callable[[str], Iterable[str]]
Fetcher = Callable[[str], "FetchedReference"]


class ReferenceStagingError(RuntimeError):
    """A requested source cannot safely become a generation reference."""


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[override]
        return None


@dataclass(frozen=True)
class RawFetchResponse:
    """A single no-follow HTTP response returned by a bounded transport."""

    status: int
    url: str
    headers: Mapping[str, str]
    body: bytes


RawGetter = Callable[[str, int], RawFetchResponse]


@dataclass(frozen=True)
class FetchedReference:
    """A bounded fetch result supplied by a caller-controlled transport.

    ``final_url`` must be the URL after redirect handling. The stager validates
    it again before any bytes are staged. A future live transport must disable
    automatic redirects or apply this same validation to every hop.
    """

    final_url: str
    content_type: str
    body: bytes
    redirect_count: int


@dataclass(frozen=True)
class StagedReference:
    source_url: str
    final_url: str
    content_type: str
    sha256: str
    relative_path: str
    requested_roles: frozenset[str]
    realised_roles: frozenset[str]
    text_excerpt: str


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._parts: list[str] = []
        self._ignored_depth = 0

    def handle_starttag(self, tag: str, attrs) -> None:  # type: ignore[override]
        if tag.lower() in {"script", "style", "noscript"}:
            self._ignored_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() in {"script", "style", "noscript"} and self._ignored_depth:
            self._ignored_depth -= 1

    def handle_data(self, data: str) -> None:
        if not self._ignored_depth:
            self._parts.append(data)

    def text(self) -> str:
        return " ".join(" ".join(self._parts).split())[:MAX_TEXT_CHARS]


def default_resolver(hostname: str) -> tuple[str, ...]:
    try:
        records = socket.getaddrinfo(hostname, None, type=socket.SOCK_STREAM)
    except OSError as exc:
        raise ReferenceStagingError(f"unable to resolve reference host: {exc}") from exc
    return tuple(sorted({str(record[4][0]) for record in records if record[4]}))


def _is_public_address(value: str) -> bool:
    try:
        return ipaddress.ip_address(value).is_global
    except ValueError:
        return False


def _validate_staging_url(url: str) -> str:
    try:
        return _validate_reference_url(url)
    except ImageRequestError as exc:
        raise ReferenceStagingError(f"reference host must be public: {exc}") from exc


def _host_from_url(url: str) -> str:
    parsed = urlsplit(_validate_staging_url(url))
    hostname = parsed.hostname
    if not hostname:
        raise ReferenceStagingError("reference URL must include a public hostname")
    return hostname


def _require_public_resolution(url: str, resolver: Resolver) -> None:
    hostname = _host_from_url(url)
    addresses = tuple(resolver(hostname))
    if not addresses or any(not _is_public_address(address) for address in addresses):
        raise ReferenceStagingError("reference host must resolve only to public addresses")


def _content_type(raw: str) -> str:
    return raw.split(";", 1)[0].strip().lower()


def _image_bytes_match_content_type(content_type: str, body: bytes) -> bool:
    if content_type == "image/png":
        return body.startswith(b"\x89PNG\r\n\x1a\n")
    if content_type == "image/jpeg":
        return body.startswith(b"\xff\xd8\xff")
    if content_type == "image/webp":
        return len(body) >= 12 and body[:4] == b"RIFF" and body[8:12] == b"WEBP"
    return True


def _header(headers: Mapping[str, str], name: str) -> str | None:
    expected = name.lower()
    for key, value in headers.items():
        if key.lower() == expected:
            return value
    return None


def _urllib_raw_get(url: str, max_bytes: int) -> RawFetchResponse:
    request = Request(url, headers={"Accept": "image/*,text/html,text/plain;q=0.9"}, method="GET")
    opener = build_opener(_NoRedirect())
    try:
        with opener.open(request, timeout=10) as response:  # noqa: S310 -- URL was resolver-validated per hop
            return RawFetchResponse(
                status=int(response.status),
                url=str(response.geturl()),
                headers={str(key): str(value) for key, value in response.headers.items()},
                body=response.read(max_bytes + 1),
            )
    except HTTPError as exc:
        return RawFetchResponse(
            status=exc.code,
            url=url,
            headers={str(key): str(value) for key, value in exc.headers.items()},
            body=exc.read(max_bytes + 1),
        )
    except OSError as exc:
        raise ReferenceStagingError(f"reference transport failed: {exc}") from exc


class SafeReferenceFetcher:
    """No-follow bounded HTTP fetcher that revalidates each requested hop."""

    def __init__(
        self,
        *,
        resolver: Resolver = default_resolver,
        raw_get: RawGetter = _urllib_raw_get,
        max_bytes: int = MAX_REFERENCE_BYTES,
        max_redirects: int = MAX_REDIRECTS,
    ) -> None:
        if max_bytes <= 0:
            raise ValueError("max_bytes must be positive")
        if max_redirects < 0:
            raise ValueError("max_redirects must not be negative")
        self._resolver = resolver
        self._raw_get = raw_get
        self._max_bytes = max_bytes
        self._max_redirects = max_redirects

    def __call__(self, url: str) -> FetchedReference:
        current = _validate_staging_url(url)
        for redirect_count in range(self._max_redirects + 1):
            _require_public_resolution(current, self._resolver)
            response = self._raw_get(current, self._max_bytes)
            if len(response.body) > self._max_bytes:
                raise ReferenceStagingError("reference response exceeds the byte limit")
            response_url = _validate_staging_url(response.url)
            if response_url != current:
                raise ReferenceStagingError("reference transport must not follow redirects automatically")
            if 300 <= response.status < 400:
                location = _header(response.headers, "Location")
                if not location:
                    raise ReferenceStagingError("reference redirect did not include a location")
                if redirect_count >= self._max_redirects:
                    raise ReferenceStagingError("reference redirect count exceeds the hard limit")
                current = _validate_staging_url(urljoin(current, location))
                continue
            if not 200 <= response.status < 300:
                raise ReferenceStagingError(f"reference transport returned HTTP {response.status}")
            content_type = _header(response.headers, "Content-Type")
            if not content_type:
                raise ReferenceStagingError("reference response did not include a content type")
            return FetchedReference(
                final_url=current,
                content_type=content_type,
                body=response.body,
                redirect_count=redirect_count,
            )
        raise ReferenceStagingError("reference redirect count exceeds the hard limit")


def _extract_text(content_type: str, body: bytes) -> str:
    if content_type == "text/plain":
        return body.decode("utf-8", errors="replace")[:MAX_TEXT_CHARS]
    if content_type == "text/html":
        parser = _TextExtractor()
        parser.feed(body.decode("utf-8", errors="replace"))
        parser.close()
        return parser.text()
    return ""


class ReferenceStager:
    """Stage bounded, hash-addressed source material below one trusted root."""

    def __init__(
        self,
        staging_root: Path,
        *,
        resolver: Resolver = default_resolver,
        fetcher: Fetcher,
        max_bytes: int = MAX_REFERENCE_BYTES,
        max_redirects: int = MAX_REDIRECTS,
    ) -> None:
        if max_bytes <= 0:
            raise ValueError("max_bytes must be positive")
        if max_redirects < 0:
            raise ValueError("max_redirects must not be negative")
        self._root = Path(staging_root)
        self._resolver = resolver
        self._fetcher = fetcher
        self._max_bytes = max_bytes
        self._max_redirects = max_redirects

    def _root_path(self) -> Path:
        if self._root.is_symlink():
            raise ReferenceStagingError("staging root must not be a symlink")
        self._root.mkdir(mode=0o700, parents=True, exist_ok=True)
        root = self._root.resolve(strict=True)
        if not root.is_dir():
            raise ReferenceStagingError("staging root must be a directory")
        return root

    @staticmethod
    def _contained(candidate: Path, root: Path) -> Path:
        resolved = candidate.resolve(strict=True)
        try:
            resolved.relative_to(root)
        except ValueError as exc:
            raise ReferenceStagingError("staged path escaped the trusted root") from exc
        return resolved

    def _new_job_directory(self, root: Path, job_id: str) -> Path:
        if not _JOB_ID_RE.fullmatch(job_id):
            raise ReferenceStagingError("job id must contain only letters, digits, underscores or hyphens")
        job_dir = root / job_id
        try:
            job_dir.mkdir(mode=0o700)
        except FileExistsError as exc:
            raise ReferenceStagingError("job directory already exists or is unsafe") from exc
        return self._contained(job_dir, root)

    def _stage_one(self, source: ReferenceRequest, job_dir: Path, index: int, root: Path) -> StagedReference:
        source_url = _validate_staging_url(source.url)
        _require_public_resolution(source_url, self._resolver)
        fetched = self._fetcher(source_url)
        if fetched.redirect_count < 0 or fetched.redirect_count > self._max_redirects:
            raise ReferenceStagingError("reference redirect count exceeds the hard limit")
        final_url = _validate_staging_url(fetched.final_url)
        _require_public_resolution(final_url, self._resolver)
        content_type = _content_type(fetched.content_type)
        extension = _ALLOWED_CONTENT_TYPES.get(content_type)
        if extension is None:
            raise ReferenceStagingError(f"unsupported reference content type: {content_type or 'missing'}")
        if len(fetched.body) > self._max_bytes:
            raise ReferenceStagingError("reference response exceeds the byte limit")
        if not _image_bytes_match_content_type(content_type, fetched.body):
            raise ReferenceStagingError("reference image bytes do not match the declared content type")

        digest = hashlib.sha256(fetched.body).hexdigest()
        filename = f"reference-{index:03d}-{digest[:16]}{extension}"
        output = job_dir / filename
        output.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        with output.open("xb") as handle:
            handle.write(fetched.body)
        output.chmod(0o600)
        output = self._contained(output, root)

        realised_roles = (
            frozenset({"visual_reference"})
            if content_type.startswith("image/")
            else frozenset({"written_inspiration"})
        )
        return StagedReference(
            source_url=source_url,
            final_url=final_url,
            content_type=content_type,
            sha256=digest,
            relative_path=str(output.relative_to(root)),
            requested_roles=source.requested_roles,
            realised_roles=realised_roles,
            text_excerpt=_extract_text(content_type, fetched.body),
        )

    def _write_manifest(self, job_dir: Path, references: tuple[StagedReference, ...], root: Path) -> None:
        manifest = job_dir / "reference-manifest.json"
        payload = {
            "schema_version": 1,
            "references": [
                {
                    **asdict(reference),
                    "requested_roles": sorted(reference.requested_roles),
                    "realised_roles": sorted(reference.realised_roles),
                }
                for reference in references
            ],
        }
        with manifest.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        manifest.chmod(0o600)
        self._contained(manifest, root)

    def stage(self, sources: Iterable[ReferenceRequest], *, job_id: str) -> tuple[StagedReference, ...]:
        """Safely stage references or clean the isolated job directory on error."""
        root = self._root_path()
        job_dir = self._new_job_directory(root, job_id)
        try:
            staged = tuple(
                self._stage_one(source, job_dir, index, root)
                for index, source in enumerate(sources, start=1)
            )
            self._write_manifest(job_dir, staged, root)
            return staged
        except Exception:
            # Only remove the directory we created after rechecking containment.
            contained = self._contained(job_dir, root)
            shutil.rmtree(contained)
            raise
