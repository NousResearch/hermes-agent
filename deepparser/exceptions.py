from __future__ import annotations


class DeepParserError(Exception):
    """Base class for all DeepParser SDK errors."""

    def __init__(self, message: str, code: str | None = None, status_code: int | None = None):
        super().__init__(message)
        self.code = code
        self.status_code = status_code


class AuthError(DeepParserError):
    """API key missing, invalid, or revoked (HTTP 401/403)."""


class RateLimitError(DeepParserError):
    """Too many requests from this IP (HTTP 429)."""


class ParseFailedError(DeepParserError):
    """The dp_cli subprocess failed to parse the document."""

    def __init__(self, message: str, detail: str | None = None):
        super().__init__(message, code="PARSE_FAILED")
        self.detail = detail


class ParseTimeoutError(DeepParserError):
    """Parse job exceeded the server-side timeout (120 s by default)."""

    def __init__(self, job_id: str):
        super().__init__(
            f"Parse job {job_id} timed out. "
            "Try splitting the document or running the Docker image locally.",
            code="TIMEOUT",
        )
        self.job_id = job_id


class JobNotFoundError(DeepParserError):
    """Job ID not found or does not belong to this API key."""
