from __future__ import annotations

import asyncio
import sys
import time
import warnings
from pathlib import Path
from typing import BinaryIO

import httpx

from .exceptions import (
    AuthError,
    DeepParserError,
    JobNotFoundError,
    ParseFailedError,
    ParseTimeoutError,
    RateLimitError,
)
from .models import AskResult, Citation, KeyInfo, ParseJob, ParseResult

# Sample contract PDF used by client.demo() — pinned to SDK v1.0
_DEMO_URL = (
    "https://raw.githubusercontent.com/ysh145/hermes-agent/main"
    "/deepparser/assets/sample_contract_v1.pdf"
)
_DEMO_QUESTION = "What are the payment terms and total contract value?"

_DEFAULT_BASE_URL = "http://localhost:8000"
_POLL_INTERVAL_INIT = 2.0
_POLL_INTERVAL_MAX = 10.0
_POLL_TOTAL_SECS = 300.0


class DeepParserClient:
    """
    Async HTTP client for the DeepParser Developer API.

    Usage::

        async with DeepParserClient(api_key="dp_live_...") as client:
            result = await client.parse_and_ask("report.pdf", "What is the total revenue?")
            print(result.answer)
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = _DEFAULT_BASE_URL,
        *,
        timeout: float = 300.0,
        debug: bool = False,
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._debug = debug
        self._http: httpx.AsyncClient | None = None

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    async def __aenter__(self) -> "DeepParserClient":
        self._http = httpx.AsyncClient(
            base_url=self._base_url,
            headers={"X-API-Key": self._api_key},
            timeout=self._timeout,
        )
        return self

    async def __aexit__(self, *_: object) -> None:
        if self._http:
            await self._http.aclose()
            self._http = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _client(self) -> httpx.AsyncClient:
        if self._http is None:
            raise RuntimeError(
                "DeepParserClient must be used as an async context manager: "
                "`async with DeepParserClient(...) as client:`"
            )
        return self._http

    def _log(self, method: str, url: str, status: int, elapsed: float) -> None:
        if self._debug:
            print(
                f"[deepparser] {method} {url} → {status} ({elapsed*1000:.0f}ms)",
                file=sys.stderr,
            )

    def _check_deprecation(self, response: httpx.Response) -> None:
        header = response.headers.get("Deepparser-Deprecation")
        if header:
            warnings.warn(
                f"DeepParser API deprecation notice: {header}",
                DeprecationWarning,
                stacklevel=4,
            )

    def _raise_for_status(self, response: httpx.Response) -> None:
        self._check_deprecation(response)
        if response.is_success:
            return
        try:
            body = response.json()
            code = body.get("detail", {}).get("code") or body.get("code", "UNKNOWN")
            message = (
                body.get("detail", {}).get("message")
                or body.get("message")
                or response.text
            )
        except Exception:
            code = "UNKNOWN"
            message = response.text

        status = response.status_code
        if status in (401, 403):
            raise AuthError(message, code=code, status_code=status)
        if status == 429:
            raise RateLimitError(message, code=code, status_code=status)
        if status == 404:
            raise JobNotFoundError(message, code=code, status_code=status)
        raise DeepParserError(message, code=code, status_code=status)

    async def _get(self, path: str) -> httpx.Response:
        client = self._client()
        t0 = time.monotonic()
        r = await client.get(path)
        self._log("GET", path, r.status_code, time.monotonic() - t0)
        self._raise_for_status(r)
        return r

    async def _post_json(self, path: str, body: dict) -> httpx.Response:
        client = self._client()
        t0 = time.monotonic()
        r = await client.post(path, json=body)
        self._log("POST", path, r.status_code, time.monotonic() - t0)
        self._raise_for_status(r)
        return r

    async def _post_file(self, path: str, file_bytes: bytes, filename: str) -> httpx.Response:
        client = self._client()
        t0 = time.monotonic()
        r = await client.post(
            path,
            files={"file": (filename, file_bytes, "application/octet-stream")},
        )
        self._log("POST", path, r.status_code, time.monotonic() - t0)
        self._raise_for_status(r)
        return r

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def parse(
        self,
        file: str | Path | bytes | BinaryIO,
        *,
        filename: str | None = None,
        sync: bool = False,
    ) -> ParseJob:
        """
        Upload a file for parsing.  Returns immediately with status QUEUED
        (or READY when sync=True and the file is small enough).
        """
        if isinstance(file, (str, Path)):
            p = Path(file)
            file_bytes = p.read_bytes()
            filename = filename or p.name
        elif isinstance(file, bytes):
            file_bytes = file
            filename = filename or "upload.bin"
        else:
            file_bytes = file.read()
            filename = filename or getattr(file, "name", "upload.bin")

        url = "/parse" + ("?mode=sync" if sync else "")
        r = await self._post_file(url, file_bytes, filename)
        data = r.json()
        result = None
        if data.get("result"):
            result = ParseResult(**data["result"])
        return ParseJob(
            job_id=data["job_id"],
            status=data["status"],
            created_at=data.get("created_at", ""),
            completed_at=data.get("completed_at"),
            result=result,
            error_detail=data.get("error_detail"),
        )

    async def get_status(self, job_id: str) -> ParseJob:
        """Poll the status of a parse job."""
        r = await self._get(f"/parse/{job_id}")
        data = r.json()
        result = None
        if data.get("result"):
            result = ParseResult(**data["result"])
        return ParseJob(
            job_id=data["job_id"],
            status=data["status"],
            created_at=data["created_at"],
            completed_at=data.get("completed_at"),
            result=result,
            error_detail=data.get("error_detail"),
        )

    async def ask(self, job_id: str, question: str) -> AskResult:
        """Ask a question against a READY parse job."""
        r = await self._post_json("/ask", {"job_id": job_id, "question": question})
        data = r.json()
        return AskResult(
            job_id=data["job_id"],
            answer=data["answer"],
            citations=[Citation(**c) for c in data.get("citations", [])],
        )

    async def wait_until_ready(self, job_id: str) -> ParseJob:
        """
        Poll GET /parse/{job_id} with exponential back-off until READY, PARSE_FAILED, or TIMEOUT.
        Raises ParseFailedError / ParseTimeoutError on terminal failure states.
        """
        interval = _POLL_INTERVAL_INIT
        deadline = time.monotonic() + _POLL_TOTAL_SECS
        while True:
            job = await self.get_status(job_id)
            if job.status == "READY":
                return job
            if job.status == "PARSE_FAILED":
                raise ParseFailedError(
                    f"Parse failed for job {job_id}",
                    detail=job.error_detail,
                )
            if job.status == "TIMEOUT":
                raise ParseTimeoutError(job_id)
            if time.monotonic() > deadline:
                raise ParseTimeoutError(job_id)
            await asyncio.sleep(interval)
            interval = min(interval * 1.5, _POLL_INTERVAL_MAX)

    async def parse_and_ask(
        self,
        file: str | Path | bytes | BinaryIO,
        question: str,
        *,
        filename: str | None = None,
    ) -> AskResult:
        """
        Upload, parse, and ask in one call.  Handles the polling loop internally.

        Example::

            async with DeepParserClient(api_key="dp_live_...") as client:
                result = await client.parse_and_ask("invoice.pdf", "What is the total amount due?")
                print(result.answer)
                for c in result.citations:
                    print(f"  → {c.filename} p.{c.page}")
        """
        job = await self.parse(file, filename=filename, sync=True)
        if job.status != "READY":
            job = await self.wait_until_ready(job.job_id)
        return await self.ask(job.job_id, question)

    async def demo(self) -> AskResult:
        """
        Run against a hosted sample contract PDF and ask a preset question.
        Requires internet access to download the sample file.

        Example::

            import asyncio
            from deepparser import DeepParserClient

            async def main():
                async with DeepParserClient(api_key="dp_live_...") as client:
                    result = await client.demo()
                    print(result.answer)

            asyncio.run(main())
        """
        async with httpx.AsyncClient(timeout=30) as http:
            r = await http.get(_DEMO_URL)
            r.raise_for_status()
            sample_bytes = r.content

        return await self.parse_and_ask(
            sample_bytes,
            _DEMO_QUESTION,
            filename="sample_contract.pdf",
        )

    # ------------------------------------------------------------------
    # Key management (convenience)
    # ------------------------------------------------------------------

    @staticmethod
    async def register_key(
        email: str,
        *,
        intended_use: str | None = None,
        base_url: str = _DEFAULT_BASE_URL,
    ) -> KeyInfo:
        """
        Register a new API key — no existing key required.

        Example::

            info = await DeepParserClient.register_key("you@company.com")
            print(info.api_key)  # dp_live_...
        """
        async with httpx.AsyncClient(base_url=base_url.rstrip("/"), timeout=30) as http:
            r = await http.post("/keys", json={"email": email, "intended_use": intended_use})
            r.raise_for_status()
            data = r.json()
        return KeyInfo(api_key=data["api_key"], created_at=data["created_at"])
