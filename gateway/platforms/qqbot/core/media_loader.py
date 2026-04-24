# -*- coding: utf-8 -*-
"""QQ Bot media loading, upload, and chunked-upload helpers.

Classes
-------
- :class:`MediaLoader`       — load media from URL or local path into API format
- :class:`MediaUploader`     — dispatches to URL upload or local chunked upload
- :class:`ChunkedUploader`   — large-file upload via prepare / part / complete flow

Exceptions
----------
- :class:`UploadDailyLimitExceededError`  — error code 40093002
- :class:`UploadFileTooLargeError`        — file exceeds platform per-file size limit

Upload strategy (decided by :class:`MediaUploader`):
    HTTP(S) URL  → URL upload  (server fetches the URL directly)
    Local file   → chunked upload (prepare → PUT parts → complete)
                   Server returns 1 part for small files, N parts for large files.

Zero hermes dependencies.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import logging
import mimetypes
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
from urllib.parse import urlparse

from .api_client import QQApiClient
from .constants import MEDIA_TYPE_FILE
from .dto import (
    CompleteUploadResponse,
    RichMediaMessage,
    UploadPart,
    UploadPrepareResponse,
)

logger = logging.getLogger(__name__)

# Error codes from the QQ Bot API
_BIZ_CODE_DAILY_LIMIT = 40093002    # upload_prepare: daily cumulative limit exceeded
_BIZ_CODE_PART_RETRYABLE = 40093001 # upload_part_finish: transient, retry until timeout

# Chunked upload concurrency defaults (mirrors TS implementation)
_DEFAULT_CONCURRENT_PARTS = 1
_MAX_CONCURRENT_PARTS = 10

# Per-part upload settings
_PART_UPLOAD_TIMEOUT = 300.0        # 5 minutes per part
_PART_UPLOAD_MAX_RETRIES = 2
_PART_FINISH_RETRY_INTERVAL = 1.0   # seconds between retryable part_finish retries
_PART_FINISH_DEFAULT_TIMEOUT = 120.0  # default timeout for retryable part_finish
_PART_FINISH_MAX_TIMEOUT = 600.0    # cap even if server returns larger retry_timeout

# complete_upload retry settings
_COMPLETE_UPLOAD_MAX_RETRIES = 2
_COMPLETE_UPLOAD_BASE_DELAY = 2.0

# MD5_10M: first 10,002,432 bytes used for the md5_10m hash (matches QQ API spec)
_MD5_10M_SIZE = 10_002_432


# ── Exceptions ────────────────────────────────────────────────────────

class UploadDailyLimitExceededError(Exception):
    """Raised when upload_prepare returns biz_code 40093002.

    The daily cumulative upload quota for this bot has been reached.
    Callers should surface :attr:`file_name` and :attr:`file_size` so the
    model can compose a helpful reply to the user.

    :param file_name: Original filename.
    :param file_size: File size in bytes.
    :param message: Raw API error message.
    """

    def __init__(self, file_name: str, file_size: int, message: str = "") -> None:
        self.file_name = file_name
        self.file_size = file_size
        super().__init__(message or f"Daily upload limit exceeded for {file_name!r}")

    @property
    def file_size_human(self) -> str:
        """Human-readable file size string (e.g. ``'12.3 MB'``)."""
        return _format_size(self.file_size)


class UploadFileTooLargeError(Exception):
    """Raised when a file exceeds the platform per-file size limit.

    :param file_name: Original filename.
    :param file_size: Actual file size in bytes.
    :param limit_bytes: Platform limit in bytes (0 = unknown).
    """

    def __init__(
        self,
        file_name: str,
        file_size: int,
        limit_bytes: int = 0,
        message: str = "",
    ) -> None:
        self.file_name = file_name
        self.file_size = file_size
        self.limit_bytes = limit_bytes
        limit_str = f" ({_format_size(limit_bytes)})" if limit_bytes else ""
        super().__init__(
            message
            or f"File {file_name!r} ({_format_size(file_size)}) exceeds platform limit{limit_str}"
        )

    @property
    def file_size_human(self) -> str:
        return _format_size(self.file_size)

    @property
    def limit_human(self) -> str:
        return _format_size(self.limit_bytes) if self.limit_bytes else "unknown"


# ── MediaLoader ───────────────────────────────────────────────────────

class MediaLoader:
    """Load media from a URL or local file path.

    Returns the data representation, content-type, and resolved filename
    expected by the QQ Bot upload API.
    """

    @staticmethod
    def load(
        source: str,
        file_name: Optional[str] = None,
    ) -> Tuple[str, str, str]:
        """Load media and return ``(data_or_url, content_type, filename)``.

        For HTTP(S) URLs the URL itself is returned as ``data_or_url``
        (the API accepts URL-based uploads directly).  For local paths the
        raw bytes are base64-encoded.

        :param source: HTTP(S) URL or local file path.
        :param file_name: Override the resolved filename.
        :returns: ``(data_or_url, content_type, filename)`` triple.
        :raises ValueError: If *source* is empty or looks like a placeholder.
        :raises FileNotFoundError: If a local path does not exist.
        """
        source = str(source).strip()
        if not source:
            raise ValueError("Media source is required")

        parsed = urlparse(source)
        if parsed.scheme in ("http", "https"):
            return MediaLoader._load_url(source, file_name, parsed)
        return MediaLoader._load_local(source, file_name)

    @staticmethod
    def _load_url(
        source: str,
        file_name: Optional[str],
        parsed: Any,
    ) -> Tuple[str, str, str]:
        content_type = mimetypes.guess_type(source)[0] or "application/octet-stream"
        resolved_name = file_name or Path(parsed.path).name or "media"
        return source, content_type, resolved_name

    @staticmethod
    def _load_local(source: str, file_name: Optional[str]) -> Tuple[str, str, str]:
        local_path = Path(source).expanduser()
        if not local_path.is_absolute():
            local_path = (Path.cwd() / local_path).resolve()

        if not local_path.exists() or not local_path.is_file():
            if source.startswith("<") or len(source) < 3:
                raise ValueError(
                    f"Invalid media source (looks like a placeholder): {source!r}"
                )
            raise FileNotFoundError(f"Media file not found: {local_path}")

        raw = local_path.read_bytes()
        resolved_name = file_name or local_path.name
        content_type = (
            mimetypes.guess_type(str(local_path))[0] or "application/octet-stream"
        )
        b64 = base64.b64encode(raw).decode("ascii")
        return b64, content_type, resolved_name

    @staticmethod
    def is_url(source: str) -> bool:
        """Return ``True`` if *source* is an HTTP(S) URL."""
        return urlparse(str(source)).scheme in ("http", "https")


# ── MediaUploader ─────────────────────────────────────────────────────

class MediaUploader:
    """Upload media to the QQ Bot API and return the ``file_info`` token.

    Upload strategy:

    - **Local files** → chunked upload (prepare / PUT parts / complete).
      The server returns 1 part for small files, multiple parts for large files.
      No client-side size threshold needed.
    - **HTTP(S) URLs** → simple upload (``RichMediaMessage.url``).
      The server fetches the URL directly; no local data transfer required.

    :param api_client: Authenticated :class:`~api_client.QQApiClient`.
    :param http_client: ``httpx.AsyncClient`` for COS PUT requests (local files only).
    :param log_tag: Log prefix.
    """

    _URL_UPLOAD_MAX_ATTEMPTS = 3
    _URL_UPLOAD_RETRY_DELAY_BASE = 1.5
    _URL_UPLOAD_FATAL_KEYWORDS = ("400", "401", "Invalid", "timeout", "Timeout")

    def __init__(
        self,
        api_client: QQApiClient,
        http_client: Any = None,
        log_tag: str = "QQBot",
    ) -> None:
        self._api = api_client
        self._http_client = http_client
        self._log_tag = log_tag

    def update_http_client(self, http_client: Any) -> None:
        """Replace the HTTP client (called after connect())."""
        self._http_client = http_client

    async def upload(
        self,
        chat_type: str,
        chat_id: str,
        source: str,
        file_type: int,
        file_name: Optional[str] = None,
    ) -> str:
        """Upload media and return the ``file_info`` token.

        :param chat_type: ``'c2c'`` or ``'group'``.
        :param chat_id: Target user or group openid.
        :param source: HTTP(S) URL or local file path.
        :param file_type: QQ media type constant (see ``constants.MEDIA_TYPE_*``).
        :param file_name: Optional filename override.
        :returns: ``file_info`` string from the upload response.
        :raises UploadDailyLimitExceededError: On biz_code 40093002 (local files only).
        :raises UploadFileTooLargeError: On file too large (local files only).
        :raises ValueError: If *chat_type* is unsupported.
        :raises RuntimeError: On other upload failures.
        """
        if chat_type not in ("c2c", "group"):
            raise ValueError(f"Unsupported chat_type for media upload: {chat_type!r}")

        if MediaLoader.is_url(source):
            return await self._url_upload(chat_type, chat_id, source, file_type, file_name)

        resolved_name = file_name or Path(source).name
        return await self._local_upload(chat_type, chat_id, source, file_type, resolved_name)

    # ------------------------------------------------------------------
    # URL upload (server fetches the URL directly)
    # ------------------------------------------------------------------

    async def _url_upload(
        self,
        chat_type: str,
        chat_id: str,
        url: str,
        file_type: int,
        file_name: Optional[str],
    ) -> str:
        """Upload via URL: server pulls the content, no local data transfer."""
        __, __, resolved_name = MediaLoader.load(url, file_name)
        upload_msg = RichMediaMessage(
            file_type=file_type,
            url=url,
            srv_send_msg=False,
            file_name=resolved_name if file_type == MEDIA_TYPE_FILE else "",
        )
        response = await self._url_upload_with_retry(chat_type, chat_id, upload_msg)
        file_info = response.get("file_info")
        if not file_info:
            raise RuntimeError(f"URL upload returned no file_info: {response}")
        return str(file_info)

    async def _url_upload_with_retry(
        self,
        chat_type: str,
        chat_id: str,
        msg: RichMediaMessage,
    ) -> dict:
        last_exc: Optional[Exception] = None

        for attempt in range(self._URL_UPLOAD_MAX_ATTEMPTS):
            try:
                if chat_type == "c2c":
                    return await self._api.upload_c2c_file(chat_id, msg)
                return await self._api.upload_group_file(chat_id, msg)
            except RuntimeError as exc:
                last_exc = exc
                if self._is_fatal_url_error(str(exc)):
                    raise
                if attempt < self._URL_UPLOAD_MAX_ATTEMPTS - 1:
                    delay = self._URL_UPLOAD_RETRY_DELAY_BASE * (attempt + 1)
                    logger.warning(
                        "[%s] URL upload retry %d/%d after %.1fs: %s",
                        self._log_tag, attempt + 1, self._URL_UPLOAD_MAX_ATTEMPTS, delay, exc,
                    )
                    await asyncio.sleep(delay)

        raise last_exc  # type: ignore[misc]

    def _is_fatal_url_error(self, error_msg: str) -> bool:
        return any(kw in error_msg for kw in self._URL_UPLOAD_FATAL_KEYWORDS)

    # ------------------------------------------------------------------
    # Local file upload (chunked: prepare → PUT parts → complete)
    # ------------------------------------------------------------------

    async def _local_upload(
        self,
        chat_type: str,
        chat_id: str,
        file_path: str,
        file_type: int,
        file_name: str,
    ) -> str:
        """Upload a local file via the chunked upload flow.

        The server returns 1 part for small files and multiple parts for
        large files — no client-side size threshold is needed.
        """
        chunked = ChunkedUploader(
            api=self._api,
            http_client=self._http_client,
            log_tag=self._log_tag,
        )
        result = await chunked.upload(
            chat_type=chat_type,
            target_id=chat_id,
            file_path=file_path,
            file_type=file_type,
            file_name=file_name,
        )
        file_info = result.token
        if not file_info:
            raise RuntimeError(f"Chunked upload returned no file_info: {result}")
        return file_info


# ── ChunkedUploader ───────────────────────────────────────────────────

@dataclass
class _UploadProgress:
    completed_parts: int = 0
    total_parts: int = 0
    uploaded_bytes: int = 0
    total_bytes: int = 0


class ChunkedUploader:
    """Three-step large-file upload.

    Flow (mirrors TS ``chunked-upload.ts``)::

        1. upload_prepare → upload_id, block_size, presigned part URLs
        2. PUT each part to its COS presigned URL (with retry)
           then call upload_part_finish (with retryable error handling)
        3. complete_upload → file_info

    Parts are uploaded with bounded concurrency (server-controlled).

    :param api: Authenticated :class:`~api_client.QQApiClient`.
    :param http_client: ``httpx.AsyncClient`` for COS PUT requests.
    :param log_tag: Log prefix.
    :param on_progress: Optional progress callback receiving a progress dict.
    """

    def __init__(
        self,
        api: QQApiClient,
        http_client: Any,
        log_tag: str = "QQBot",
        on_progress: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> None:
        self._api = api
        self._http = http_client
        self._log_tag = log_tag
        self._on_progress = on_progress

    async def upload(
        self,
        chat_type: str,
        target_id: str,
        file_path: str,
        file_type: int,
        file_name: str,
    ) -> CompleteUploadResponse:
        """Run the full chunked upload and return the complete_upload response.

        :raises UploadDailyLimitExceededError: On biz_code 40093002.
        :raises UploadFileTooLargeError: When the file exceeds the platform limit.
        :raises RuntimeError: On other API or I/O failures.
        """
        path = Path(file_path)
        file_size = path.stat().st_size

        logger.info(
            "[%s] Chunked upload start: file=%s size=%s type=%d",
            self._log_tag, file_name, _format_size(file_size), file_type,
        )

        # Step 1: compute hashes (blocking I/O → executor)
        hashes = await asyncio.get_running_loop().run_in_executor(
            None, _compute_file_hashes, file_path, file_size
        )
        logger.debug(
            "[%s] File hashes: md5=%s sha1=%s md5_10m=%s",
            self._log_tag, hashes["md5"], hashes["sha1"], hashes["md5_10m"],
        )

        # Step 2: upload_prepare
        prepare = await self._prepare(
            chat_type, target_id, file_type, file_name, file_size, hashes
        )
        upload_id = prepare.upload_id
        block_size = prepare.block_size
        parts = prepare.parts
        max_concurrent = min(prepare.concurrency, _MAX_CONCURRENT_PARTS)
        retry_timeout = min(
            prepare.retry_timeout if prepare.retry_timeout > 0 else _PART_FINISH_DEFAULT_TIMEOUT,
            _PART_FINISH_MAX_TIMEOUT,
        )

        logger.info(
            "[%s] Prepared: upload_id=%s block_size=%s parts=%d concurrency=%d",
            self._log_tag, upload_id, _format_size(block_size), len(parts), max_concurrent,
        )

        progress = _UploadProgress(total_parts=len(parts), total_bytes=file_size)

        # Step 3: upload all parts with concurrency control
        # Note: parts run concurrently via asyncio.gather, but Python's GIL ensures
        # progress counter increments are effectively atomic for int operations.
        tasks: List[Callable[[], Any]] = [
            lambda p=part: self._upload_part(
                chat_type, target_id, file_path, file_size,
                upload_id, block_size, p, retry_timeout, progress,
            )
            for part in parts
        ]
        await _run_with_concurrency(tasks, max_concurrent)

        logger.info(
            "[%s] All %d parts uploaded, completing...", self._log_tag, len(parts)
        )

        # Step 4: complete_upload (with retry)
        return await self._complete(chat_type, target_id, upload_id)

    async def _prepare(
        self,
        chat_type: str,
        target_id: str,
        file_type: int,
        file_name: str,
        file_size: int,
        hashes: Dict[str, str],
    ) -> UploadPrepareResponse:
        try:
            return await self._api.upload_prepare(
                chat_type, target_id, file_type, file_name, file_size,
                hashes["md5"], hashes["sha1"], hashes["md5_10m"],
            )
        except RuntimeError as exc:
            err_msg = str(exc)
            if f"{_BIZ_CODE_DAILY_LIMIT}" in err_msg:
                raise UploadDailyLimitExceededError(file_name, file_size, err_msg) from exc
            raise

    async def _upload_part(
        self,
        chat_type: str,
        target_id: str,
        file_path: str,
        file_size: int,
        upload_id: str,
        block_size: int,
        part: UploadPart,
        retry_timeout: float,
        progress: _UploadProgress,
    ) -> None:
        part_index = part.index             # 1-based
        presigned_url = part.presigned_url
        # Per-part block_size takes priority; fall back to the Rsp-level block_size.
        actual_block_size = part.block_size if part.block_size > 0 else block_size
        offset = (part_index - 1) * block_size  # offset always uses Rsp block_size
        length = min(actual_block_size, file_size - offset)

        # Read chunk (blocking I/O → executor)
        data = await asyncio.get_running_loop().run_in_executor(
            None, _read_file_chunk, file_path, offset, length
        )
        md5_hex = hashlib.md5(data).hexdigest()

        logger.debug(
            "[%s] Part %d/%d: uploading %s (offset=%d md5=%s)",
            self._log_tag, part_index, progress.total_parts,
            _format_size(length), offset, md5_hex,
        )

        # PUT to presigned URL (with retry)
        await self._put_to_presigned_url(presigned_url, data, part_index, progress.total_parts)

        # Notify platform (with retryable error handling)
        await self._part_finish_with_retry(
            chat_type, target_id, upload_id,
            part_index, length, md5_hex, retry_timeout,
        )

        progress.completed_parts += 1
        progress.uploaded_bytes += length
        logger.debug(
            "[%s] Part %d/%d done (%d/%d completed)",
            self._log_tag, part_index, progress.total_parts,
            progress.completed_parts, progress.total_parts,
        )

        if self._on_progress:
            self._on_progress({
                "completed_parts": progress.completed_parts,
                "total_parts": progress.total_parts,
                "uploaded_bytes": progress.uploaded_bytes,
                "total_bytes": progress.total_bytes,
            })

    async def _put_to_presigned_url(
        self,
        url: str,
        data: bytes,
        part_index: int,
        total_parts: int,
    ) -> None:
        """PUT part data to COS presigned URL with retry."""
        last_exc: Optional[Exception] = None

        for attempt in range(_PART_UPLOAD_MAX_RETRIES + 1):
            try:
                resp = await asyncio.wait_for(
                    self._http.put(
                        url,
                        content=data,
                        headers={"Content-Length": str(len(data))},
                    ),
                    timeout=_PART_UPLOAD_TIMEOUT,
                )
                resp.raise_for_status()
                logger.debug(
                    "[%s] PUT Part %d/%d: %d OK ETag=%s",
                    self._log_tag, part_index, total_parts,
                    resp.status_code, resp.headers.get("ETag", "-"),
                )
                return
            except Exception as exc:
                last_exc = exc
                if attempt < _PART_UPLOAD_MAX_RETRIES:
                    delay = 1.0 * (2 ** attempt)
                    logger.warning(
                        "[%s] PUT Part %d/%d attempt %d failed, retry in %.1fs: %s",
                        self._log_tag, part_index, total_parts, attempt + 1, delay, exc,
                    )
                    await asyncio.sleep(delay)

        raise RuntimeError(
            f"Part {part_index}/{total_parts} upload failed after "
            f"{_PART_UPLOAD_MAX_RETRIES + 1} attempts: {last_exc}"
        )

    async def _part_finish_with_retry(
        self,
        chat_type: str,
        target_id: str,
        upload_id: str,
        part_index: int,
        block_size: int,
        md5: str,
        retry_timeout: float,
    ) -> None:
        """Call upload_part_finish, retrying on biz_code 40093001."""
        loop = asyncio.get_running_loop()
        start = loop.time()
        attempt = 0

        while True:
            try:
                await self._api.upload_part_finish(
                    chat_type, target_id, upload_id, part_index, block_size, md5
                )
                return
            except RuntimeError as exc:
                err_msg = str(exc)
                if f"{_BIZ_CODE_PART_RETRYABLE}" not in err_msg:
                    raise

                elapsed = loop.time() - start
                if elapsed >= retry_timeout:
                    raise RuntimeError(
                        f"upload_part_finish persistent retry timed out "
                        f"after {retry_timeout:.0f}s ({attempt} retries): {exc}"
                    ) from exc

                attempt += 1
                logger.debug(
                    "[%s] part_finish retryable error, attempt %d, elapsed=%.1fs: %s",
                    self._log_tag, attempt, elapsed, exc,
                )
                await asyncio.sleep(_PART_FINISH_RETRY_INTERVAL)

    async def _complete(
        self,
        chat_type: str,
        target_id: str,
        upload_id: str,
    ) -> CompleteUploadResponse:
        """Call complete_upload with retry (unconditional, mirrors TS)."""
        last_exc: Optional[Exception] = None

        for attempt in range(_COMPLETE_UPLOAD_MAX_RETRIES + 1):
            try:
                return await self._api.complete_upload(chat_type, target_id, upload_id)
            except Exception as exc:
                last_exc = exc
                if attempt < _COMPLETE_UPLOAD_MAX_RETRIES:
                    delay = _COMPLETE_UPLOAD_BASE_DELAY * (2 ** attempt)
                    logger.warning(
                        "[%s] complete_upload attempt %d failed, retry in %.1fs: %s",
                        self._log_tag, attempt + 1, delay, exc,
                    )
                    await asyncio.sleep(delay)

        raise RuntimeError(
            f"complete_upload failed after {_COMPLETE_UPLOAD_MAX_RETRIES + 1} "
            f"attempts: {last_exc}"
        )


# ── Helpers ───────────────────────────────────────────────────────────

def _format_size(size_bytes: int) -> str:
    """Return a human-readable file size string (e.g. ``'12.3 MB'``)."""
    size = float(size_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024.0:
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} TB"


def _read_file_chunk(file_path: str, offset: int, length: int) -> bytes:
    """Read *length* bytes from *file_path* starting at *offset*."""
    with open(file_path, "rb") as fh:
        fh.seek(offset)
        return fh.read(length)


def _compute_file_hashes(file_path: str, file_size: int) -> Dict[str, str]:
    """Compute md5, sha1, and md5_10m in a single pass (sync, run in executor)."""
    md5 = hashlib.md5()
    sha1 = hashlib.sha1()
    md5_10m = hashlib.md5()

    need_10m = file_size > _MD5_10M_SIZE
    bytes_read = 0

    with open(file_path, "rb") as fh:
        while True:
            chunk = fh.read(65536)
            if not chunk:
                break
            md5.update(chunk)
            sha1.update(chunk)
            if need_10m:
                remaining = _MD5_10M_SIZE - bytes_read
                if remaining > 0:
                    md5_10m.update(chunk[:remaining])
            bytes_read += len(chunk)

    full_md5 = md5.hexdigest()
    return {
        "md5": full_md5,
        "sha1": sha1.hexdigest(),
        "md5_10m": md5_10m.hexdigest() if need_10m else full_md5,
    }


async def _run_with_concurrency(
    tasks: List[Callable[[], Any]],
    max_concurrent: int,
) -> None:
    """Run async tasks in batches of *max_concurrent* (mirrors TS runWithConcurrency)."""
    for i in range(0, len(tasks), max_concurrent):
        batch = tasks[i: i + max_concurrent]
        await asyncio.gather(*[t() for t in batch])
