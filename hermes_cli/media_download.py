"""Safe public media downloader used by the /download slash command.

The downloader intentionally avoids platform circumvention. It only downloads a
URL when the final response is a directly accessible media/download object, or
when a public HTML page contains directly linked media URLs in normal markup.
It does not use cookies, private APIs, hidden endpoints, login-wall bypasses, or
manifest reconstruction.
"""

from __future__ import annotations

import html.parser
import json
import mimetypes
import os
import re
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from collections.abc import Iterable, Mapping
from urllib.parse import unquote, urljoin, urlparse

import requests

from hermes_constants import get_hermes_home


ACCESS_RESTRICTED = "Access restricted: direct download unavailable without authorized access."

_DEFAULT_TIMEOUT = (10, 60)
_MAX_HTML_BYTES = 2_000_000
_CHUNK_SIZE = 1024 * 1024

# Conservative allowlist: media plus common direct downloadable objects whose
# containers we can cheaply validate from magic bytes.
_ALLOWED_PRIMARY_TYPES = {"video", "audio", "image"}
_ALLOWED_EXACT_TYPES = {
    "application/pdf",
    "application/zip",
    "application/x-zip-compressed",
    "application/gzip",
    "application/x-gzip",
    "application/octet-stream",  # accepted only when URL/headers/sniffing prove a known type
}
_KNOWN_EXTENSIONS = {
    ".mp4", ".m4v", ".mov", ".webm", ".mkv", ".avi", ".mpeg", ".mpg", ".ts",
    ".mp3", ".m4a", ".aac", ".wav", ".ogg", ".oga", ".opus", ".flac",
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".pdf", ".zip", ".gz",
}
_EXTENSION_BY_TYPE = {
    "video/mp4": ".mp4",
    "video/quicktime": ".mov",
    "video/webm": ".webm",
    "video/x-matroska": ".mkv",
    "video/mpeg": ".mpeg",
    "video/mp2t": ".ts",
    "audio/mpeg": ".mp3",
    "audio/mp4": ".m4a",
    "audio/aac": ".aac",
    "audio/wav": ".wav",
    "audio/x-wav": ".wav",
    "audio/ogg": ".ogg",
    "audio/opus": ".opus",
    "audio/flac": ".flac",
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/gif": ".gif",
    "image/webp": ".webp",
    "application/pdf": ".pdf",
    "application/zip": ".zip",
    "application/x-zip-compressed": ".zip",
    "application/gzip": ".gz",
    "application/x-gzip": ".gz",
}

_LOGIN_HINTS = (
    "log in", "login", "sign in", "signin", "create account", "forgot password",
    "access denied", "not available", "private", "subscribe", "paywall",
)


class DownloadError(Exception):
    """Raised for user-facing /download failures."""


@dataclass
class DownloadResult:
    saved_path: str
    media_type: str
    final_url: str
    content_type: str
    metadata: dict[str, str | int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "saved_path": self.saved_path,
            "media_type": self.media_type,
            "final_url": self.final_url,
            "content_type": self.content_type,
            "metadata": self.metadata,
        }

    def render_text(self) -> str:
        lines = [
            "✅ Download complete",
            f"Saved file: `{self.saved_path}`",
            f"Detected media type: `{self.media_type}`",
            f"Final URL: `{self.final_url}`",
            f"Content-Type: `{self.content_type}`",
        ]
        useful = {k: v for k, v in self.metadata.items() if v not in (None, "")}
        if useful:
            lines.append("Metadata:")
            for key in sorted(useful):
                lines.append(f"- {key}: `{useful[key]}`")
        return "\n".join(lines)


class _MediaLinkParser(html.parser.HTMLParser):
    """Collect directly exposed URL attributes from public HTML markup."""

    ATTRS = {"src", "href", "poster"}
    TAGS = {"a", "source", "video", "audio", "img", "track"}

    def __init__(self, base_url: str):
        super().__init__(convert_charrefs=True)
        self.base_url = base_url
        self.links: list[str] = []
        self.text_chunks: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() not in self.TAGS:
            return
        for name, value in attrs:
            if name and name.lower() in self.ATTRS and value:
                absolute = urljoin(self.base_url, value.strip())
                if absolute.startswith(("http://", "https://")):
                    self.links.append(absolute)

    def handle_data(self, data: str) -> None:
        if data:
            self.text_chunks.append(data[:500])


def download_public_media(url: str, output_dir: str | os.PathLike[str] | None = None) -> DownloadResult:
    """Download a public direct media/file URL or a direct media link in HTML.

    Args:
        url: Single HTTP(S) URL supplied by the user.
        output_dir: Optional destination directory; defaults to
            ``$HERMES_HOME/downloads``.

    Raises:
        DownloadError: User-facing failure message.
    """

    normalized = _normalize_url(url)
    out_dir = Path(output_dir) if output_dir else Path(get_hermes_home()) / "downloads"
    out_dir.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.headers.update({
        "User-Agent": "HermesAgent/1.0 (+https://hermes-agent.nousresearch.com)",
        "Accept": "*/*",
    })

    candidate = _resolve_candidate(session, normalized)
    return _stream_download(session, candidate, out_dir)


def _normalize_url(raw_url: str) -> str:
    if not isinstance(raw_url, str) or not raw_url.strip():
        raise DownloadError("Usage: /download <url>")
    parts = raw_url.strip().split()
    if len(parts) != 1:
        raise DownloadError("Usage: /download <single-url>")
    parsed = urlparse(parts[0])
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise DownloadError("Usage: /download <http-or-https-url>")
    return parts[0]


def _resolve_candidate(session: requests.Session, url: str) -> requests.Response:
    """Return a response that points at a downloadable object, not a web page."""

    head = _safe_request(session, "HEAD", url, stream=False)
    if head is not None and _is_access_restricted(head):
        raise DownloadError(ACCESS_RESTRICTED)
    if head is not None and _is_downloadable_response(head):
        return head

    # Some servers do not implement HEAD or omit useful headers. Use GET but do
    # not download the body unless the response is directly downloadable.
    get = _safe_request(session, "GET", url, stream=True)
    if get is None:
        raise DownloadError("Unable to fetch URL.")
    if _is_access_restricted(get):
        _close_quietly(get)
        raise DownloadError(ACCESS_RESTRICTED)
    if _is_downloadable_response(get):
        return get

    content_type = _content_type(get)
    if not _is_html(content_type):
        _close_quietly(get)
        raise DownloadError(f"Unsupported content type: {content_type or 'unknown'}")

    html = _read_limited_text(get, _MAX_HTML_BYTES)
    _close_quietly(get)
    parser = _MediaLinkParser(get.url)
    parser.feed(html)
    if _looks_like_access_wall(html, parser.text_chunks):
        raise DownloadError(ACCESS_RESTRICTED)

    for link in _dedupe(parser.links):
        if not _url_has_known_extension(link):
            continue
        linked_head = _safe_request(session, "HEAD", link, stream=False)
        if linked_head is not None and _is_access_restricted(linked_head):
            continue
        if linked_head is not None and _is_downloadable_response(linked_head):
            return linked_head
        linked_get = _safe_request(session, "GET", link, stream=True)
        if linked_get is not None and not _is_access_restricted(linked_get) and _is_downloadable_response(linked_get):
            return linked_get
        _close_quietly(linked_get)

    raise DownloadError(ACCESS_RESTRICTED)


def _safe_request(session: requests.Session, method: str, url: str, *, stream: bool) -> requests.Response | None:
    try:
        resp = session.request(method, url, allow_redirects=True, timeout=_DEFAULT_TIMEOUT, stream=stream)
    except requests.RequestException:
        return None
    # Do not raise for 401/403: those are access-state signals.
    if resp.status_code >= 400 and resp.status_code not in {401, 403, 404, 405}:
        _close_quietly(resp)
        return None
    if method == "HEAD" and resp.status_code == 405:
        _close_quietly(resp)
        return None
    return resp


def _is_access_restricted(resp: requests.Response) -> bool:
    if resp.status_code in {401, 403}:
        return True
    final = (resp.url or "").lower()
    if any(token in final for token in ("/login", "/signin", "auth", "checkpoint", "oauth")):
        return True
    ctype = _content_type(resp)
    if _is_html(ctype):
        # A final HTML page is not a downloadable success. We inspect public
        # markup later; common login-wall URLs can stop immediately here.
        return any(token in final for token in ("login", "signin", "checkpoint"))
    return False


def _is_downloadable_response(resp: requests.Response) -> bool:
    ctype = _content_type(resp)
    if not ctype or _is_html(ctype):
        return False
    primary = ctype.split("/", 1)[0]
    if primary in _ALLOWED_PRIMARY_TYPES:
        return True
    if ctype in _ALLOWED_EXACT_TYPES:
        return _url_has_known_extension(resp.url) or _content_disposition_filename(resp.headers).lower().endswith(tuple(_KNOWN_EXTENSIONS))
    if _url_has_known_extension(resp.url):
        guessed = mimetypes.guess_type(urlparse(resp.url).path)[0] or ""
        return guessed.split("/", 1)[0] in _ALLOWED_PRIMARY_TYPES or guessed in _ALLOWED_EXACT_TYPES
    return False


def _stream_download(session: requests.Session, response: requests.Response, output_dir: Path) -> DownloadResult:
    if getattr(response, "request", None) is not None and response.request.method == "HEAD":
        _close_quietly(response)
        response = session.get(response.url, allow_redirects=True, timeout=_DEFAULT_TIMEOUT, stream=True)
    try:
        if _is_access_restricted(response) or not _is_downloadable_response(response):
            raise DownloadError(ACCESS_RESTRICTED)

        content_type = _content_type(response)
        metadata = _extract_metadata(response)
        filename = _choose_filename(response.url, response.headers, content_type)
        tmp_fd, tmp_name = tempfile.mkstemp(prefix=f".{filename}.", suffix=".part", dir=str(output_dir))
        try:
            with os.fdopen(tmp_fd, "wb") as fh:
                for chunk in response.iter_content(chunk_size=_CHUNK_SIZE):
                    if chunk:
                        fh.write(chunk)
            tmp_path = Path(tmp_name)
            if not tmp_path.exists() or tmp_path.stat().st_size <= 0:
                raise DownloadError("Downloaded file is empty.")

            sniffed = _sniff_file_type(tmp_path)
            declared = _declared_media_type(content_type, filename)
            if not _container_matches(declared, sniffed, filename):
                raise DownloadError(
                    f"Downloaded content did not match declared media type "
                    f"({content_type or 'unknown'}; sniffed {sniffed or 'unknown'})."
                )
            final_ext = _extension_for(sniffed or content_type, filename)
            final_name = Path(filename).with_suffix(final_ext).name if final_ext else filename
            final_path = _unique_path(output_dir / final_name)
            shutil.move(str(tmp_path), final_path)
            return DownloadResult(
                saved_path=str(final_path),
                media_type=sniffed or declared or content_type or "application/octet-stream",
                final_url=response.url,
                content_type=content_type or "application/octet-stream",
                metadata=metadata,
            )
        except Exception:
            try:
                os.unlink(tmp_name)
            except OSError:
                pass
            raise
    finally:
        _close_quietly(response)


def _content_type(resp: requests.Response) -> str:
    return (resp.headers.get("Content-Type", "").split(";", 1)[0].strip().lower())


def _is_html(content_type: str) -> bool:
    return content_type in {"text/html", "application/xhtml+xml"} or content_type.endswith("+html")


def _read_limited_text(resp: requests.Response, limit: int) -> str:
    data = bytearray()
    for chunk in resp.iter_content(chunk_size=65536):
        if not chunk:
            continue
        data.extend(chunk)
        if len(data) > limit:
            break
    encoding = resp.encoding or "utf-8"
    return bytes(data[:limit]).decode(encoding, errors="replace")


def _looks_like_access_wall(html_text: str, text_chunks: Iterable[str]) -> bool:
    haystack = " ".join(text_chunks).lower() or html_text[:5000].lower()
    return sum(1 for hint in _LOGIN_HINTS if hint in haystack) >= 2


def _dedupe(items: Iterable[str]) -> list[str]:
    seen = set()
    out = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def _url_has_known_extension(url: str) -> bool:
    ext = Path(unquote(urlparse(url).path)).suffix.lower()
    return ext in _KNOWN_EXTENSIONS


def _content_disposition_filename(headers: Mapping[str, str]) -> str:
    cd = headers.get("Content-Disposition", "")
    match = re.search(r"filename\*=UTF-8''([^;]+)", cd, flags=re.I)
    if match:
        return unquote(match.group(1).strip().strip('"'))
    match = re.search(r'filename="?([^";]+)"?', cd, flags=re.I)
    return unquote(match.group(1).strip()) if match else ""


def _choose_filename(url: str, headers: Mapping[str, str], content_type: str) -> str:
    name = _content_disposition_filename(headers)
    if not name:
        name = Path(unquote(urlparse(url).path)).name
    if not name or name in {"/", ".", ".."}:
        name = "download"
    name = re.sub(r"[^A-Za-z0-9._ -]+", "_", name).strip(" .") or "download"
    ext = Path(name).suffix.lower()
    desired = _EXTENSION_BY_TYPE.get(content_type) or mimetypes.guess_extension(content_type or "")
    if (not ext or ext not in _KNOWN_EXTENSIONS) and desired:
        name = f"{name}{desired}"
    return name


def _extract_metadata(resp: requests.Response) -> dict[str, str | int]:
    metadata: dict[str, str | int] = {}
    for header, key in (
        ("Content-Length", "content_length"),
        ("Last-Modified", "last_modified"),
        ("ETag", "etag"),
        ("Accept-Ranges", "accept_ranges"),
    ):
        value = resp.headers.get(header)
        if value:
            if key == "content_length":
                try:
                    metadata[key] = int(value)
                    continue
                except ValueError:
                    pass
            metadata[key] = value
    return metadata


def _declared_media_type(content_type: str, filename: str) -> str:
    if content_type and content_type != "application/octet-stream":
        return content_type
    guessed = mimetypes.guess_type(filename)[0]
    return guessed or content_type or "application/octet-stream"


def _sniff_file_type(path: Path) -> str:
    with path.open("rb") as fh:
        head = fh.read(64)
    if len(head) >= 12 and head[4:8] == b"ftyp":
        brand = head[8:12].lower()
        return "video/quicktime" if brand == b"qt  " else "video/mp4"
    if head.startswith(b"\x1a\x45\xdf\xa3"):
        # EBML: WebM and Matroska share the same magic. Extension/declared type
        # disambiguates later.
        return "video/webm"
    if head.startswith(b"ID3") or (len(head) >= 2 and head[0] == 0xFF and (head[1] & 0xE0) == 0xE0):
        return "audio/mpeg"
    if head.startswith(b"RIFF") and head[8:12] == b"WAVE":
        return "audio/wav"
    if head.startswith(b"OggS"):
        return "audio/ogg"
    if head.startswith(b"fLaC"):
        return "audio/flac"
    if head.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if head.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if head.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    if head.startswith(b"RIFF") and head[8:12] == b"WEBP":
        return "image/webp"
    if head.startswith(b"%PDF-"):
        return "application/pdf"
    if head.startswith(b"PK\x03\x04") or head.startswith(b"PK\x05\x06") or head.startswith(b"PK\x07\x08"):
        return "application/zip"
    if head.startswith(b"\x1f\x8b"):
        return "application/gzip"
    if len(head) >= 188 and head[0] == 0x47 and head[188 - 1] == 0x47:
        return "video/mp2t"
    return ""


def _container_matches(declared: str, sniffed: str, filename: str) -> bool:
    if not sniffed:
        return False
    declared_primary = declared.split("/", 1)[0] if declared else ""
    sniffed_primary = sniffed.split("/", 1)[0]
    ext = Path(filename).suffix.lower()
    if declared in {"application/octet-stream", ""}:
        return ext in _KNOWN_EXTENSIONS
    if declared == sniffed:
        return True
    if declared_primary in {"video", "audio", "image"}:
        return declared_primary == sniffed_primary
    if declared in _ALLOWED_EXACT_TYPES:
        return sniffed in _ALLOWED_EXACT_TYPES or sniffed_primary in _ALLOWED_PRIMARY_TYPES
    return False


def _extension_for(media_type: str, filename: str) -> str:
    ext = _EXTENSION_BY_TYPE.get(media_type) or mimetypes.guess_extension(media_type or "")
    if media_type == "video/webm" and Path(filename).suffix.lower() == ".mkv":
        return ".mkv"
    return ext or Path(filename).suffix


def _unique_path(path: Path) -> Path:
    if not path.exists():
        return path
    stem, suffix = path.stem, path.suffix
    for i in range(1, 10_000):
        candidate = path.with_name(f"{stem}-{i}{suffix}")
        if not candidate.exists():
            return candidate
    raise DownloadError("Could not choose a unique output filename.")


def _close_quietly(resp: requests.Response | None) -> None:
    if resp is not None:
        try:
            resp.close()
        except Exception:
            pass


def main(argv: list[str] | None = None) -> int:
    """Small CLI entry point for manual testing: python -m hermes_cli.media_download URL."""
    import sys

    args = list(sys.argv[1:] if argv is None else argv)
    try:
        result = download_public_media(args[0] if args else "")
    except DownloadError as exc:
        print(str(exc))
        return 2
    print(json.dumps(result.to_dict(), indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
