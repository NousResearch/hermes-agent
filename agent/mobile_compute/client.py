"""HTTP client for the Mobile Compute Node (iPhone 16e).

Communicates with the iPhone-side server using:
- aiohttp for async HTTP
- mTLS when configured
- exponential backoff retry
- JSON request/response envelope

No LLM calls, no Core ML — purely transport-level.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, Optional

import httpx

from agent.mobile_compute.config import MobileComputeClientConfig

logger = logging.getLogger("agent.mobile_compute.client")


@dataclass
class ComputeResult:
    """Result from a compute request to the iPhone."""
    success: bool
    task_id: str
    status: str  # "queued" | "running" | "completed" | "failed"
    result: Optional[Any] = None
    error: Optional[str] = None
    latency_ms: float = 0.0


class MobileComputeClient:
    """Async HTTP client that talks to the iPhone-side Mobile Compute server.

    Parameters
    ----------
    config : MobileComputeClientConfig
        Client configuration (endpoint, timeout, TLS, retry).
    """

    def __init__(self, config: MobileComputeClientConfig) -> None:
        self._config = config
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Lazily create the shared httpx client."""
        if self._client is None or self._client.is_closed:
            tls = self._config.tls
            transport_kwargs: Dict[str, Any] = {}

            if tls.enabled and tls.ca_path:
                transport_kwargs["verify"] = tls.ca_path
            elif tls.enabled and tls.cert_path and tls.key_path:
                transport_kwargs["cert"] = (tls.cert_path, tls.key_path)
                if tls.ca_path:
                    transport_kwargs["verify"] = tls.ca_path

            self._client = httpx.AsyncClient(
                base_url=self._config.default_endpoint,
                timeout=self._config.timeout_seconds,
                **transport_kwargs,
            )
        return self._client

    async def _request_with_retry(
        self,
        method: str,
        path: str,
        **kwargs: Any,
    ) -> httpx.Response:
        """Execute an HTTP request with exponential backoff retry."""
        client = await self._get_client()
        attempt = 0
        backoff = self._config.retry_backoff_base
        last_exc: Optional[Exception] = None

        while attempt < self._config.retry_attempts:
            try:
                response = await client.request(method, path, **kwargs)
                response.raise_for_status()
                return response
            except (httpx.HTTPError, httpx.TimeoutException) as exc:
                last_exc = exc
                logger.warning(
                    "HTTP %s %s attempt %d/%d failed: %s",
                    method, path, attempt + 1, self._config.retry_attempts, exc,
                )
                attempt += 1
                if attempt < self._config.retry_attempts:
                    await asyncio.sleep(backoff)
                    backoff *= self._config.retry_backoff_base

        raise RuntimeError(
            f"All {self._config.retry_attempts} retry attempts failed for "
            f"{method} {path}: {last_exc}"
        ) from last_exc

    async def health(self) -> Dict[str, Any]:
        """Check iPhone health. Returns parsed JSON from /health."""
        response = await self._request_with_retry("GET", "/health")
        return response.json()

    async def capabilities(self) -> Dict[str, Any]:
        """Get iPhone capabilities. Returns parsed JSON from /capabilities."""
        response = await self._request_with_retry("GET", "/capabilities")
        return response.json()

    async def submit_task(
        self,
        task_type: str,
        payload: Dict[str, Any],
        task_id: Optional[str] = None,
    ) -> ComputeResult:
        """Submit a compute task to the iPhone.

        Parameters
        ----------
        task_type : str
            Identifier for the computation (e.g. "image_classify", "audio_transcribe").
        payload : dict
            Task-specific parameters.
        task_id : str, optional
            Client-side task identifier. Auto-generated if omitted.

        Returns
        -------
        ComputeResult
            Status of the submitted task.
        """
        if task_id is None:
            task_id = str(uuid.uuid4())[:8]

        body = {
            "task_id": task_id,
            "task_type": task_type,
            "payload": payload,
            "timestamp": time.time(),
        }

        start = time.monotonic()
        try:
            response = await self._request_with_retry("POST", "/compute", json=body)
            data = response.json()
            elapsed = (time.monotonic() - start) * 1000
            return ComputeResult(
                success=data.get("success", True),
                task_id=data.get("task_id", task_id),
                status=data.get("status", "queued"),
                result=data.get("result"),
                error=data.get("error"),
                latency_ms=elapsed,
            )
        except Exception as exc:
            elapsed = (time.monotonic() - start) * 1000
            logger.error("Task submission failed: %s", exc)
            return ComputeResult(
                success=False, task_id=task_id, status="failed",
                error=str(exc), latency_ms=elapsed,
            )

    async def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """Check status of a previously submitted task."""
        response = await self._request_with_retry("GET", f"/tasks/{task_id}")
        return response.json()

    async def list_tasks(self) -> Dict[str, Any]:
        """List all tasks."""
        response = await self._request_with_retry("GET", "/tasks")
        return response.json()

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def __aenter__(self) -> "MobileComputeClient":
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.close()


async def test_client_connection(config: MobileComputeClientConfig) -> bool:
    """Test if the iPhone server is reachable. Returns True on success."""
    client = MobileComputeClient(config)
    try:
        async with client:
            health = await client.health()
            logger.info("iPhone health check OK: %s", health)
            return True
    except Exception as exc:
        logger.warning("iPhone health check failed: %s", exc)
        return False