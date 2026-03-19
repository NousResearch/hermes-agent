"""
Bankr API Client.

Async client for the Bankr Agent API with job polling pattern.
"""

import os
import asyncio
import logging
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from datetime import datetime

try:
    import httpx
except ImportError:
    httpx = None

from .config import get_bankr_config, BankrConfig

logger = logging.getLogger(__name__)


class BankrError(Exception):
    """Base exception for Bankr API errors."""

    def __init__(self, message: str, status_code: int = None, response: Dict = None):
        super().__init__(message)
        self.status_code = status_code
        self.response = response or {}


class BankrAuthError(BankrError):
    """Authentication error."""
    pass


class BankrRateLimitError(BankrError):
    """Rate limit exceeded."""
    pass


class BankrJobError(BankrError):
    """Job execution error."""
    pass


@dataclass
class JobResult:
    """Result from a completed Bankr job."""
    job_id: str
    status: str
    result: Dict[str, Any]
    transactions: List[Dict] = None
    responses: List[str] = None
    created_at: datetime = None
    completed_at: datetime = None

    @classmethod
    def from_response(cls, data: Dict) -> "JobResult":
        """Create from API response."""
        return cls(
            job_id=data.get("jobId", ""),
            status=data.get("status", "unknown"),
            result=data.get("result", {}),
            transactions=data.get("transactions", []),
            responses=data.get("responses", []),
        )


class BankrClient:
    """
    Async client for Bankr Agent API.

    Uses the job-based async pattern:
    1. Submit prompt -> get job ID
    2. Poll job status until complete
    3. Return results
    """

    def __init__(
        self,
        api_key: str = None,
        config: BankrConfig = None
    ):
        if httpx is None:
            raise ImportError("httpx is required: pip install httpx")

        self.config = config or get_bankr_config()
        self.api_key = api_key or self.config.api_key

        if not self.api_key:
            logger.warning("Bankr API key not configured")

        self._session: Optional[httpx.AsyncClient] = None

    async def _get_session(self) -> httpx.AsyncClient:
        """Get or create the HTTP session."""
        if self._session is None or self._session.is_closed:
            self._session = httpx.AsyncClient(
                base_url=self.config.base_url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                timeout=self.config.request_timeout,
            )
        return self._session

    async def close(self):
        """Close the HTTP session."""
        if self._session:
            await self._session.aclose()
            self._session = None

    async def submit_prompt(self, prompt: str) -> str:
        """
        Submit a natural language prompt to Bankr.

        Args:
            prompt: The command or query in natural language

        Returns:
            Job ID for polling
        """
        session = await self._get_session()

        try:
            response = await session.post(
                "/agent/prompt",
                json={"prompt": prompt}
            )

            if response.status_code == 401:
                raise BankrAuthError("Invalid API key", 401)
            elif response.status_code == 429:
                raise BankrRateLimitError("Rate limit exceeded", 429)

            response.raise_for_status()
            data = response.json()

            job_id = data.get("jobId")
            if not job_id:
                raise BankrError("No job ID in response", response=data)

            logger.debug(f"Submitted prompt, job ID: {job_id}")
            return job_id

        except httpx.HTTPStatusError as e:
            raise BankrError(
                f"HTTP error: {e.response.status_code}",
                status_code=e.response.status_code
            )

    async def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """Get the current status of a job."""
        session = await self._get_session()

        response = await session.get(f"/agent/job/{job_id}")
        response.raise_for_status()

        return response.json()

    async def poll_job(
        self,
        job_id: str,
        timeout: float = None
    ) -> JobResult:
        """
        Poll a job until completion or timeout.

        Args:
            job_id: The job ID to poll
            timeout: Maximum time to wait (seconds)

        Returns:
            JobResult with the completed data
        """
        timeout = timeout or self.config.job_timeout
        interval = self.config.job_poll_interval
        start_time = asyncio.get_event_loop().time()

        while True:
            elapsed = asyncio.get_event_loop().time() - start_time
            if elapsed >= timeout:
                raise TimeoutError(f"Job {job_id} timed out after {timeout}s")

            data = await self.get_job_status(job_id)
            status = data.get("status", "unknown")

            if status == "completed":
                logger.debug(f"Job {job_id} completed")
                return JobResult.from_response(data)

            elif status == "failed":
                error_msg = data.get("error", "Job failed")
                raise BankrJobError(error_msg, response=data)

            elif status in ("pending", "running"):
                await asyncio.sleep(interval)

            else:
                logger.warning(f"Unknown job status: {status}")
                await asyncio.sleep(interval)

    async def cancel_job(self, job_id: str) -> bool:
        """Cancel a running job."""
        session = await self._get_session()

        try:
            response = await session.post(f"/agent/job/{job_id}/cancel")
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Failed to cancel job {job_id}: {e}")
            return False

    async def execute(self, prompt: str, timeout: float = None) -> Dict[str, Any]:
        """
        Submit a prompt and wait for the result.

        This is the main convenience method that combines submit + poll.

        Args:
            prompt: Natural language command
            timeout: Max time to wait

        Returns:
            The result dict from the completed job
        """
        job_id = await self.submit_prompt(prompt)
        result = await self.poll_job(job_id, timeout)
        return result.result

    async def execute_trade(self, prompt: str) -> JobResult:
        """
        Execute a trading command with full result tracking.

        Args:
            prompt: Trading command (e.g., "Buy $50 of ETH on Base")

        Returns:
            Full JobResult including transaction details
        """
        job_id = await self.submit_prompt(prompt)
        return await self.poll_job(job_id)

    # Convenience methods for common operations

    async def get_price(self, token: str) -> float:
        """Get current price of a token."""
        result = await self.execute(f"What is the current price of {token}?")
        return result.get("price", 0.0)

    async def get_balance(self, chain: str = None) -> Dict[str, Any]:
        """Get wallet balance."""
        chain = chain or self.config.default_chain
        return await self.execute(f"Show my wallet balance on {chain}")

    async def health_check(self) -> bool:
        """Check if the API is reachable."""
        try:
            session = await self._get_session()
            response = await session.get("/health")
            return response.status_code == 200
        except Exception:
            return False
