"""QQ Bot chunked upload flow (files 10 MB..~100 MB; inline uploads cap at ~10 MB):
1. POST .../upload_prepare → upload_id, block_size, pre-signed COS part URLs;
2. per part: PUT bytes to COS, then POST .../upload_part_finish;
3. POST .../files with {"upload_id"} → ``file_info`` for a RichMedia message.
biz_code 40093001 = part_finish retryable until ``retry_timeout``; 40093002 = daily
quota (UploadDailyLimitExceededError); other API/I/O failures raise RuntimeError.
Ported from WideLee's qqbot-agent-sdk v1.2.2 (authorship via Co-authored-by).
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List

from gateway.platforms.qqbot.constants import FILE_UPLOAD_TIMEOUT

logger = logging.getLogger(__name__)

_BIZ_CODE_DAILY_LIMIT = 40093002  # upload_prepare: daily cumulative limit
_BIZ_CODE_PART_RETRYABLE = 40093001  # upload_part_finish: transient
_DEFAULT_CONCURRENT_PARTS = 1
_MAX_CONCURRENT_PARTS = 10
_PART_UPLOAD_TIMEOUT = 300.0  # 5 minutes per COS PUT
_PART_UPLOAD_MAX_RETRIES = 2
_PART_FINISH_RETRY_INTERVAL = 1.0
_PART_FINISH_DEFAULT_TIMEOUT = 120.0
_PART_FINISH_MAX_TIMEOUT = 600.0
_COMPLETE_UPLOAD_MAX_RETRIES = 2
_COMPLETE_UPLOAD_BASE_DELAY = 2.0
_MD5_10M_SIZE = 10_002_432  # first N bytes used for the ``md5_10m`` hash (per QQ API spec)

class _UploadError(Exception):
    def __init__(self, file_name: str, file_size: int, message: str) -> None:
        self.file_name, self.file_size = file_name, file_size
        super().__init__(message)

    @property
    def file_size_human(self) -> str:
        return format_size(self.file_size)


class UploadDailyLimitExceededError(_UploadError):
    """Raised when ``upload_prepare`` returns biz_code 40093002 (daily quota hit)."""

    def __init__(self, file_name: str, file_size: int, message: str = "") -> None:
        super().__init__(file_name, file_size, message or f"Daily upload limit exceeded for {file_name!r}")


class UploadFileTooLargeError(_UploadError):
    """Raised when a file exceeds the platform per-file size limit."""

    def __init__(self, file_name: str, file_size: int, limit_bytes: int = 0, message: str = "") -> None:
        self.limit_bytes = limit_bytes
        limit_str = f" ({format_size(limit_bytes)})" if limit_bytes else ""
        super().__init__(file_name, file_size,
                         message or f"File {file_name!r} ({format_size(file_size)}) exceeds platform limit{limit_str}")

    @property
    def limit_human(self) -> str:
        return format_size(self.limit_bytes) if self.limit_bytes else "unknown"


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
    """Parse upload_prepare response (either bare or wrapped in ``data``)."""
    src = raw.get("data") if isinstance(raw.get("data"), dict) else raw
    if not (upload_id := str(src.get("upload_id", ""))):
        raise ValueError(f"upload_prepare response missing upload_id: {str(raw)[:200]}")
    block_size = int(src.get("block_size", 0))
    raw_parts = src.get("parts") or src.get("part_list") or []
    if not isinstance(raw_parts, list) or not raw_parts:
        raise ValueError(f"upload_prepare response missing parts: {str(raw)[:200]}")
    parts = [_PreparePart(index=int(p.get("part_index") or p.get("index") or 0),
                          presigned_url=str(p.get("presigned_url") or p.get("url") or ""),
                          block_size=int(p.get("block_size", 0)))
             for p in raw_parts if isinstance(p, dict)]
    return _PrepareResult(
        upload_id=upload_id, block_size=block_size, parts=parts,
        concurrency=int(src.get("concurrency", _DEFAULT_CONCURRENT_PARTS)) or _DEFAULT_CONCURRENT_PARTS,
        retry_timeout=float(src.get("retry_timeout", 0.0) or 0.0))


def _api_path(chat_type: str, target_id: str, endpoint: str) -> str:
    return f"{'/v2/users' if chat_type == 'c2c' else '/v2/groups'}/{target_id}/{endpoint}"


@dataclass
class _Job:
    """Per-upload state shared by the part workers."""
    chat_type: str
    target_id: str
    file_path: str
    file_size: int
    upload_id: str = ""
    block_size: int = 0
    total_parts: int = 0
    retry_timeout: float = 0.0
    completed: int = 0

    def path(self, endpoint: str) -> str:
        return _api_path(self.chat_type, self.target_id, endpoint)


class ChunkedUploader:
    """Run the prepare → PUT parts → complete sequence. ``api_request`` is the adapter's
    bound ``_api_request(method, path, body=..., timeout=...)`` (injected to avoid a
    circular import; must raise RuntimeError with the biz_code in the message);
    ``http_put`` is ``(url, data, headers) -> httpx-like response`` for COS PUTs."""

    def __init__(self, api_request: Callable[..., Awaitable[Dict[str, Any]]],
                 http_put: Callable[..., Awaitable[Any]], log_tag: str = "QQBot") -> None:
        self._api_request = api_request
        self._http_put = http_put
        self._log_tag = log_tag

    async def _post(self, job: _Job, endpoint: str, body: Dict[str, Any]) -> Dict[str, Any]:
        return await self._api_request("POST", job.path(endpoint), body=body, timeout=FILE_UPLOAD_TIMEOUT)

    async def upload(self, chat_type: str, target_id: str, file_path: str, file_type: int, file_name: str) -> Dict[str, Any]:
        """Run the full chunked upload (``chat_type`` 'c2c'|'group', ``file_type`` MEDIA_TYPE_*)
        and return the raw ``complete_upload`` response (contains ``file_info``).
        Raises UploadDailyLimitExceededError (40093002), UploadFileTooLargeError, RuntimeError."""
        if chat_type not in {"c2c", "group"}:
            raise ValueError(f"ChunkedUploader: unsupported chat_type {chat_type!r}")
        job = _Job(chat_type, target_id, file_path, Path(file_path).stat().st_size)
        logger.info("[%s] Chunked upload start: file=%s size=%s type=%d", self._log_tag, file_name,
                    format_size(job.file_size), file_type)
        # Hashing is blocking I/O → executor.
        hashes = await asyncio.get_running_loop().run_in_executor(None, _compute_file_hashes, file_path, job.file_size)
        prepare = await self._prepare(job, file_type, file_name, hashes)
        max_concurrent = min(prepare.concurrency, _MAX_CONCURRENT_PARTS)
        job.upload_id, job.block_size, job.total_parts = prepare.upload_id, prepare.block_size, len(prepare.parts)
        job.retry_timeout = min(prepare.retry_timeout if prepare.retry_timeout > 0 else _PART_FINISH_DEFAULT_TIMEOUT,
                                _PART_FINISH_MAX_TIMEOUT)
        logger.info("[%s] Prepared: upload_id=%s block_size=%s parts=%d concurrency=%d", self._log_tag,
                    job.upload_id, format_size(job.block_size), job.total_parts, max_concurrent)
        sem = asyncio.Semaphore(max(max_concurrent, 1))

        async def _run(part: _PreparePart) -> None:
            async with sem:
                await self._upload_one_part(job, part)

        await asyncio.gather(*(_run(p) for p in prepare.parts))
        logger.info("[%s] All %d parts uploaded, completing…", self._log_tag, job.total_parts)
        return await self._complete(job)

    async def _prepare(self, job: _Job, file_type: int, file_name: str, hashes: Dict[str, str]) -> _PrepareResult:
        body = {"file_type": file_type, "file_name": file_name, "file_size": job.file_size,
                "md5": hashes["md5"], "sha1": hashes["sha1"], "md5_10m": hashes["md5_10m"]}
        try:
            raw = await self._post(job, "upload_prepare", body)
        except RuntimeError as exc:
            if f"{_BIZ_CODE_DAILY_LIMIT}" in str(exc):
                raise UploadDailyLimitExceededError(file_name, job.file_size, str(exc)) from exc
            raise
        return _parse_prepare_response(raw)

    async def _upload_one_part(self, job: _Job, part: _PreparePart) -> None:
        """PUT one part to COS, then call ``upload_part_finish``."""
        part_index, total_parts = part.index, job.total_parts
        offset = (part_index - 1) * job.block_size
        # Per-part block_size wins; fall back to the response-level value.
        length = min(part.block_size if part.block_size > 0 else job.block_size, job.file_size - offset)
        data = await asyncio.get_running_loop().run_in_executor(None, _read_file_chunk, job.file_path, offset, length)
        md5_hex = hashlib.md5(data).hexdigest()
        logger.debug("[%s] Part %d/%d: uploading %s (offset=%d md5=%s)", self._log_tag, part_index, total_parts,
                     format_size(length), offset, md5_hex)
        await self._put_to_presigned_url(part.presigned_url, data, part_index, total_parts)
        await self._part_finish_with_retry(job, part_index, length, md5_hex)
        job.completed += 1
        logger.debug("[%s] Part %d/%d done (%d/%d total)", self._log_tag, part_index, total_parts, job.completed,
                     total_parts)

    async def _with_retries(self, attempt_fn: Callable[[], Awaitable[Any]], *, max_retries: int, base_delay: float,
                            label: str, failure_label: str) -> Any:
        """Run *attempt_fn* up to ``max_retries + 1`` times with exponential backoff."""
        last_exc: Exception | None = None
        for attempt in range(max_retries + 1):
            try:
                return await attempt_fn()
            except Exception as exc:
                last_exc = exc
                if attempt < max_retries:
                    delay = base_delay * (2 ** attempt)
                    logger.warning("[%s] %s attempt %d failed, retry in %.1fs: %s", self._log_tag, label, attempt + 1,
                                   delay, exc)
                    await asyncio.sleep(delay)
        raise RuntimeError(f"{failure_label} failed after {max_retries + 1} attempts: {last_exc}")

    async def _put_to_presigned_url(self, url: str, data: bytes, part_index: int, total_parts: int) -> None:
        """PUT part data to a pre-signed COS URL with retry."""

        async def _attempt() -> None:
            resp = await asyncio.wait_for(self._http_put(url, data=data, headers={"Content-Length": str(len(data))}),
                                          timeout=_PART_UPLOAD_TIMEOUT)
            status = getattr(resp, "status_code", 0)
            if 200 <= status < 300:
                logger.debug("[%s] PUT part %d/%d: %d OK", self._log_tag, part_index, total_parts, status)
                return
            try:
                body_preview = getattr(resp, "text", "")[:200]
            except Exception:  # pragma: no cover — defensive
                body_preview = ""
            raise RuntimeError(f"COS PUT returned {status}: {body_preview}")

        await self._with_retries(_attempt, max_retries=_PART_UPLOAD_MAX_RETRIES, base_delay=1.0,
                                 label=f"PUT part {part_index}/{total_parts}",
                                 failure_label=f"Part {part_index}/{total_parts} upload")

    async def _part_finish_with_retry(self, job: _Job, part_index: int, block_size: int, md5: str) -> None:
        """Call ``upload_part_finish``, retrying on biz_code 40093001 until ``job.retry_timeout``."""
        body = {"upload_id": job.upload_id, "part_index": part_index, "block_size": block_size, "md5": md5}
        loop = asyncio.get_running_loop()
        start, attempt = loop.time(), 0
        while True:
            try:
                await self._post(job, "upload_part_finish", body)
                return
            except RuntimeError as exc:
                if f"{_BIZ_CODE_PART_RETRYABLE}" not in str(exc):
                    raise
                elapsed = loop.time() - start
                if elapsed >= job.retry_timeout:
                    raise RuntimeError(f"upload_part_finish persistent retry timed out after "
                                       f"{job.retry_timeout:.0f}s ({attempt} retries): {exc}") from exc
                attempt += 1
                logger.debug("[%s] part_finish retryable error, attempt %d, elapsed=%.1fs: %s", self._log_tag, attempt,
                             elapsed, exc)
                await asyncio.sleep(_PART_FINISH_RETRY_INTERVAL)

    async def _complete(self, job: _Job) -> Dict[str, Any]:
        """Call ``complete_upload`` with retry — the ``/files`` endpoint (same as the simple URL upload)
        selects the chunked-completion path when only ``upload_id`` is sent."""
        return await self._with_retries(
            lambda: self._post(job, "files", {"upload_id": job.upload_id}), max_retries=_COMPLETE_UPLOAD_MAX_RETRIES,
            base_delay=_COMPLETE_UPLOAD_BASE_DELAY, label="complete_upload", failure_label="complete_upload")


def format_size(size_bytes: int) -> str:
    """Return a human-readable file size string (e.g. ``'12.3 MB'``)."""
    size = float(size_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024.0:
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} TB"


def _read_file_chunk(file_path: str, offset: int, length: int) -> bytes:
    """Read *length* bytes at *offset*; raises IOError on a short read (truncated file)."""
    with open(file_path, "rb") as fh:
        fh.seek(offset)
        data = fh.read(length)
    if len(data) != length:
        raise IOError(f"Short read from {file_path}: expected {length} bytes at offset {offset}, got {len(data)} "
                      f"(file may be truncated)")
    return data


def _compute_file_hashes(file_path: str, file_size: int) -> Dict[str, str]:
    """Compute md5, sha1, and md5_10m in a single pass (for small files md5_10m is just the full md5)."""
    md5, sha1, md5_10m = hashlib.md5(), hashlib.sha1(), hashlib.md5()
    need_10m = file_size > _MD5_10M_SIZE
    bytes_read = 0
    with open(file_path, "rb") as fh:
        while chunk := fh.read(65536):
            md5.update(chunk)
            sha1.update(chunk)
            if need_10m and (remaining := _MD5_10M_SIZE - bytes_read) > 0:
                md5_10m.update(chunk[:remaining])
            bytes_read += len(chunk)
    full_md5 = md5.hexdigest()
    return {"md5": full_md5, "sha1": sha1.hexdigest(), "md5_10m": md5_10m.hexdigest() if need_10m else full_md5}


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.
from typing import Optional  # noqa: F401,E402
import functools  # noqa: F401,E402

ApiRequestFn = Callable[..., Awaitable[Dict[str, Any]]]
# ---- END PLUGIN-COMPAT ----
