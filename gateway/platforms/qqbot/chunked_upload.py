"""QQ Bot chunked upload flow.

The QQ v2 API caps inline base64 uploads (``file_data`` / ``url``) at ~10 MB.
For files between 10 MB and ~100 MB we have to use the three-step chunked
upload flow::

    1. POST /v2/{users|groups}/{id}/upload_prepare
       → returns upload_id, block_size, and an array of pre-signed COS part URLs.
    2. For each part:
         PUT the part bytes to its pre-signed COS URL,
         then POST /v2/{users|groups}/{id}/upload_part_finish to acknowledge.
    3. POST /v2/{users|groups}/{id}/files with {"upload_id": ...}
       → returns the ``file_info`` token the caller uses in a RichMedia
       message.

Error-code semantics (from the QQ Bot v2 API spec):

- ``40093001`` — ``upload_part_finish`` retryable. Retry until the server-provided
  ``retry_timeout`` elapses (or a local cap).
- ``40093002`` — daily cumulative upload quota exceeded. Not retryable; surface
  as :class:`UploadDailyLimitExceededError` so the caller can build a
  user-friendly reply.

Exceptions:

- :class:`UploadDailyLimitExceededError` — daily quota hit (non-retryable).
- :class:`UploadFileTooLargeError` — file exceeds the platform per-file limit.
- :class:`RuntimeError` — generic upload failure (network, part PUT, complete).

Ported from WideLee's qqbot-agent-sdk v1.2.2 (``media_loader.py::ChunkedUploader``)
so the heavy-upload path stays in-tree. Authorship preserved via Co-authored-by.
"""

from __future__ import annotations

import asyncio
import functools
import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional

from gateway.platforms.qqbot.constants import FILE_UPLOAD_TIMEOUT

logger = logging.getLogger(__name__)


# ── Error codes ──────────────────────────────────────────────────────
_BIZ_CODE_DAILY_LIMIT = 40093002     # upload_prepare: daily cumulative limit
_BIZ_CODE_PART_RETRYABLE = 40093001  # upload_part_finish: transient

# ── Part upload tuning ───────────────────────────────────────────────
_DEFAULT_CONCURRENT_PARTS = 1
_MAX_CONCURRENT_PARTS = 10

_PART_UPLOAD_TIMEOUT = 300.0        # 5 minutes per COS PUT
_PART_UPLOAD_MAX_RETRIES = 2
_PART_FINISH_RETRY_INTERVAL = 1.0
_PART_FINISH_DEFAULT_TIMEOUT = 120.0
_PART_FINISH_MAX_TIMEOUT = 600.0

_COMPLETE_UPLOAD_MAX_RETRIES = 2
_COMPLETE_UPLOAD_BASE_DELAY = 2.0

# First 10,002,432 bytes used for the ``md5_10m`` hash (per QQ API spec).
_MD5_10M_SIZE = 10_002_432


# ── Exceptions ───────────────────────────────────────────────────────

class UploadDailyLimitExceededError(Exception):
    """Raised when ``upload_prepare`` returns biz_code 40093002.

    The daily cumulative upload quota for this bot has been reached. Callers
    should surface :attr:`file_name` + :attr:`file_size_human` so the model
    can compose a helpful reply.
    """

    def __init__(self, file_name: str, file_size: int, message: str = "") -> None:
        self.file_name = file_name
        self.file_size = file_size
        super().__init__(
            message or f"Daily upload limit exceeded for {file_name!r}"
        )

    @property
    def file_size_human(self) -> str:
        return format_size(self.file_size)


class UploadFileTooLargeError(Exception):
    """Raised when a file exceeds the platform per-file size limit."""

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
        limit_str = f" ({format_size(limit_bytes)})" if limit_bytes else ""
        super().__init__(
            message
            or (
                f"File {file_name!r} ({format_size(file_size)}) "
                f"exceeds platform limit{limit_str}"
            )
        )

    @property
    def file_size_human(self) -> str:
        return format_size(self.file_size)

    @property
    def limit_human(self) -> str:
        return format_size(self.limit_bytes) if self.limit_bytes else "unknown"


# ── Progress tracking ────────────────────────────────────────────────

@dataclass
class _UploadProgress:
    total_parts: int = 0
    total_bytes: int = 0
    completed_parts: int = 0
    uploaded_bytes: int = 0


# ── Prepare-response shape ───────────────────────────────────────────

@dataclass
class _PreparePart:
    index: int
    presigned_url: str
    block_size: int = 0


@dataclass
class _PrepareResult:
    upload_id: str
    block_size: int
    parts: List[_PreparePart]
    concurrency: int = _DEFAULT_CONCURRENT_PARTS
    retry_timeout: float = 0.0


def _parse_prepare_response(raw: Dict[str, Any]) -> _PrepareResult:
    """Parse the upload_prepare API response into a normalized shape.

    The API may return the response directly or wrapped in ``data``.
    """
    src = raw.get("data") if isinstance(raw.get("data"), dict) else raw
    upload_id = str(src.get("upload_id", ""))
    if not upload_id:
        raise ValueError(
            f"upload_prepare response missing upload_id: {str(raw)[:200]}"
        )
    block_size = int(src.get("block_size", 0))
    raw_parts = src.get("parts") or src.get("part_list") or []
    if not isinstance(raw_parts, list) or not raw_parts:
        raise ValueError(
            f"upload_prepare response missing parts: {str(raw)[:200]}"
        )
    parts: List[_PreparePart] = []
    for p in raw_parts:
        if not isinstance(p, dict):
            continue
        parts.append(
            _PreparePart(
                index=int(p.get("part_index") or p.get("index") or 0),
                presigned_url=str(
                    p.get("presigned_url") or p.get("url") or ""
                ),
                block_size=int(p.get("block_size", 0)),
            )
        )
    return _PrepareResult(
        upload_id=upload_id,
        block_size=block_size,
        parts=parts,
        concurrency=int(src.get("concurrency", _DEFAULT_CONCURRENT_PARTS)) or _DEFAULT_CONCURRENT_PARTS,
        retry_timeout=float(src.get("retry_timeout", 0.0) or 0.0),
    )


# ── Chunked upload driver ────────────────────────────────────────────

ApiRequestFn = Callable[..., Awaitable[Dict[str, Any]]]
"""Signature of the adapter's ``_api_request`` callable.

We pass the bound method in rather than importing the adapter, to avoid
circular imports and keep this module testable in isolation.
"""


class ChunkedUploader:
    """Run the prepare → PUT parts → complete sequence.

    :param api_request: Bound ``_api_request(method, path, body=..., timeout=...)``
        coroutine from the adapter. Must raise ``RuntimeError`` with the biz_code
        embedded in the message on API errors.
    :param http_put: Coroutine ``(url, data, headers, timeout) -> response`` for
        COS part uploads. Typically wraps ``httpx.AsyncClient.put``.
    :param log_tag: Log prefix.
    """

    def __init__(
        self,
        api_request: ApiRequestFn,
        http_put: Callable[..., Awaitable[Any]],
        log_tag: str = "QQBot",
    ) -> None:
        self._api_request = api_request
        self._http_put = http_put
        self._log_tag = log_tag

    async def upload(
        self,
        chat_type: str,
        target_id: str,
        file_path: str,
        file_type: int,
        file_name: str,
    ) -> Dict[str, Any]:
        """Run the full chunked upload and return the ``complete_upload`` response.

        :param chat_type: ``'c2c'`` or ``'group'``.
        :param target_id: User or group openid.
        :param file_path: Absolute path to a local file.
        :param file_type: ``MEDIA_TYPE_*`` constant.
        :param file_name: Original filename (for upload_prepare).
        :returns: The raw response dict from ``complete_upload`` — contains
            ``file_info`` that the caller uses in a RichMedia message body.
        :raises UploadDailyLimitExceededError: On biz_code 40093002.
        :raises UploadFileTooLargeError: When the file exceeds the platform limit.
        :raises RuntimeError: On other API or I/O failures.
        """
        if chat_type not in ("c2c", "group"):
            raise ValueError(
                f"ChunkedUploader: unsupported chat_type {chat_type!r}"
            )

        path = Path(file_path)
        file_size = path.stat().st_size

        logger.info(
            "[%s] Chunked upload start: file=%s size=%s type=%d",
            self._log_tag, file_name, format_size(file_size), file_type,
        )

        # Step 1: compute hashes (blocking I/O → executor).
        hashes = await asyncio.get_running_loop().run_in_executor(
            None, _compute_file_hashes, file_path, file_size
        )

        # Step 2: upload_prepare.
        prepare = await self._prepare(
            chat_type, target_id, file_type, file_name, file_size, hashes
        )
        max_concurrent = min(prepare.concurrency, _MAX_CONCURRENT_PARTS)
        retry_timeout = min(
            prepare.retry_timeout if prepare.retry_timeout > 0 else _PART_FINISH_DEFAULT_TIMEOUT,
            _PART_FINISH_MAX_TIMEOUT,
        )
        logger.info(
            "[%s] Prepared: upload_id=%s block_size=%s parts=%d concurrency=%d",
            self._log_tag, prepare.upload_id, format_size(prepare.block_size),
            len(prepare.parts), max_concurrent,
        )

        progress = _UploadProgress(
            total_parts=len(prepare.parts),
            total_bytes=file_size,
        )

        # Step 3: PUT each part + notify.
        tasks: List[Callable[[], Awaitable[None]]] = [
            functools.partial(
                self._upload_one_part,
                chat_type=chat_type,
                target_id=target_id,
                file_path=file_path,
                file_size=file_size,
                upload_id=prepare.upload_id,
                rsp_block_size=prepare.block_size,
                part=part,
                retry_timeout=retry_timeout,
                progress=progress,
            )
            for part in prepare.parts
        ]
        await _run_with_concurrency(tasks, max_concurrent)

        logger.info(
            "[%s] All %d parts uploaded, completing…",
            self._log_tag, len(prepare.parts),
        )

        # Step 4: complete_upload (retry on transient errors).
        return await self._complete(chat_type, target_id, prepare.upload_id)

    # ──────────────────────────────────────────────────────────────────
    # Step 1 — upload_prepare
    # ──────────────────────────────────────────────────────────────────

    async def _prepare(
        self,
        chat_type: str,
        target_id: str,
        file_type: int,
        file_name: str,
        file_size: int,
        hashes: Dict[str, str],
    ) -> _PrepareResult:
        base = "/v2/users" if chat_type == "c2c" else "/v2/groups"
        path = f"{base}/{target_id}/upload_prepare"
        body = {
            "file_type": file_type,
            "file_name": file_name,
            "file_size": file_size,
            "md5": hashes["md5"],
            "sha1": hashes["sha1"],
            "md5_10m": hashes["md5_10m"],
        }
        try:
            raw = await self._api_request(
                "POST", path, body=body, timeout=FILE_UPLOAD_TIMEOUT
            )
        except RuntimeError as exc:
            err_msg = str(exc)
            if f"{_BIZ_CODE_DAILY_LIMIT}" in err_msg:
                raise UploadDailyLimitExceededError(
                    file_name, file_size, err_msg
                ) from exc
            raise
        return _parse_prepare_response(raw)

    # ──────────────────────────────────────────────────────────────────
    # Step 2 — PUT one part + part_finish
    # ──────────────────────────────────────────────────────────────────

    async def _upload_one_part(
        self,
        chat_type: str,
        target_id: str,
        file_path: str,
        file_size: int,
        upload_id: str,
        rsp_block_size: int,
        part: _PreparePart,
        retry_timeout: float,
        progress: _UploadProgress,
    ) -> None:
        """PUT one part to COS, then call ``upload_part_finish``."""
        part_index = part.index
        # Per-part block_size wins; fall back to the response-level value.
        actual_block_size = part.block_size if part.block_size > 0 else rsp_block_size
        offset = (part_index - 1) * rsp_block_size
        length = min(actual_block_size, file_size - offset)

        # Read this slice of the file (blocking → executor).
        data = await asyncio.get_running_loop().run_in_executor(
            None, _read_file_chunk, file_path, offset, length
        )
        md5_hex = hashlib.md5(data).hexdigest()

        logger.debug(
            "[%s] Part %d/%d: uploading %s (offset=%d md5=%s)",
            self._log_tag, part_index, progress.total_parts,
            format_size(length), offset, md5_hex,
        )

        await self._put_to_presigned_url(
            part.presigned_url, data, part_index, progress.total_parts
        )
        await self._part_finish_with_retry(
            chat_type, target_id, upload_id,
            part_index, length, md5_hex, retry_timeout,
        )

        progress.completed_parts += 1
        progress.uploaded_bytes += length
        logger.debug(
            "[%s] Part %d/%d done (%d/%d total)",
            self._log_tag, part_index, progress.total_parts,
            progress.completed_parts, progress.total_parts,
        )

    async def _put_to_presigned_url(
        self,
        url: str,
        data: bytes,
        part_index: int,
        total_parts: int,
    ) -> None:
        """PUT part data to a pre-signed COS URL with retry."""
        last_exc: Optional[Exception] = None
        for attempt in range(_PART_UPLOAD_MAX_RETRIES + 1):
            try:
                resp = await asyncio.wait_for(
                    self._http_put(
                        url,
                        data=data,
                        headers={"Content-Length": str(len(data))},
                    ),
                    timeout=_PART_UPLOAD_TIMEOUT,
                )
                # Caller's http_put is expected to return an httpx-like response.
                status = getattr(resp, "status_code", 0)
                if 200 <= status < 300:
                    logger.debug(
                        "[%s] PUT part %d/%d: %d OK",
                        self._log_tag, part_index, total_parts, status,
                    )
                    return
                body_preview = ""
                try:
                    body_preview = getattr(resp, "text", "")[:200]
                except Exception:  # pragma: no cover — defensive
                    pass
                raise RuntimeError(
                    f"COS PUT returned {status}: {body_preview}"
                )
            except Exception as exc:
                last_exc = exc
                if attempt < _PART_UPLOAD_MAX_RETRIES:
                    delay = 1.0 * (2 ** attempt)
                    logger.warning(
                        "[%s] PUT part %d/%d attempt %d failed, retry in %.1fs: %s",
                        self._log_tag, part_index, total_parts,
                        attempt + 1, delay, exc,
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
        """Call ``upload_part_finish``, retrying on biz_code 40093001."""
        base = "/v2/users" if chat_type == "c2c" else "/v2/groups"
        path = f"{base}/{target_id}/upload_part_finish"
        body = {
            "upload_id": upload_id,
            "part_index": part_index,
            "block_size": block_size,
            "md5": md5,
        }

        loop = asyncio.get_running_loop()
        start = loop.time()
        attempt = 0
        while True:
            try:
                await self._api_request(
                    "POST", path, body=body, timeout=FILE_UPLOAD_TIMEOUT
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
                    "[%s] part_finish retryable error, attempt %d, "
                    "elapsed=%.1fs: %s",
                    self._log_tag, attempt, elapsed, exc,
                )
                await asyncio.sleep(_PART_FINISH_RETRY_INTERVAL)

    # ──────────────────────────────────────────────────────────────────
    # Step 3 — complete_upload
    # ──────────────────────────────────────────────────────────────────

    async def _complete(
        self,
        chat_type: str,
        target_id: str,
        upload_id: str,
    ) -> Dict[str, Any]:
        """Call ``complete_upload`` with retry.

        This reuses the ``/files`` endpoint (same as the simple URL-based upload)
        but signals the chunked-completion path by sending only ``upload_id``.
        """
        base = "/v2/users" if chat_type == "c2c" else "/v2/groups"
        path = f"{base}/{target_id}/files"
        body = {"upload_id": upload_id}

        last_exc: Optional[Exception] = None
        for attempt in range(_COMPLETE_UPLOAD_MAX_RETRIES + 1):
            try:
                return await self._api_request(
                    "POST", path, body=body, timeout=FILE_UPLOAD_TIMEOUT
                )
            except Exception as exc:
                last_exc = exc
                if attempt < _COMPLETE_UPLOAD_MAX_RETRIES:
                    delay = _COMPLETE_UPLOAD_BASE_DELAY * (2 ** attempt)
                    logger.warning(
                        "[%s] complete_upload attempt %d failed, "
                        "retry in %.1fs: %s",
                        self._log_tag, attempt + 1, delay, exc,
                    )
                    await asyncio.sleep(delay)
        raise RuntimeError(
            f"complete_upload failed after "
            f"{_COMPLETE_UPLOAD_MAX_RETRIES + 1} attempts: {last_exc}"
        )


# ── Helpers (module-level for testability) ───────────────────────────

def format_size(size_bytes: int) -> str:
    """Return a human-readable file size string (e.g. ``'12.3 MB'``)."""
    size = float(size_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024.0:
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} TB"


def _read_file_chunk(file_path: str, offset: int, length: int) -> bytes:
    """Read *length* bytes from *file_path* starting at *offset*.

    :raises IOError: If fewer bytes were read than expected (truncated file).
    """
    with open(file_path, "rb") as fh:
        fh.seek(offset)
        data = fh.read(length)
        if len(data) != length:
            raise IOError(
                f"Short read from {file_path}: expected {length} bytes at "
                f"offset {offset}, got {len(data)} (file may be truncated)"
            )
        return data


def _compute_file_hashes(file_path: str, file_size: int) -> Dict[str, str]:
    """Compute md5, sha1, and md5_10m in a single pass."""
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
        # For small files the "10m" hash is just the full md5.
        "md5_10m": md5_10m.hexdigest() if need_10m else full_md5,
    }


async def _run_with_concurrency(
    tasks: List[Callable[[], Awaitable[None]]],
    concurrency: int,
) -> None:
    """Run a list of thunks with a bounded number in flight at once."""
    if concurrency < 1:
        concurrency = 1
    sem = asyncio.Semaphore(concurrency)

    async def _wrap(thunk: Callable[[], Awaitable[None]]) -> None:
        async with sem:
            await thunk()

    await asyncio.gather(*(_wrap(t) for t in tasks))
