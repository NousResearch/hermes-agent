"""WordPress plugin exceptions."""

from __future__ import annotations


class WordPressError(Exception):
    """Base WordPress plugin error."""


class WordPressConfigError(WordPressError):
    """Raised when required WordPress configuration is missing."""


class WordPressAPIError(WordPressError):
    """Raised when the WordPress REST API returns an error response."""

    def __init__(self, message: str, *, status_code: int | None = None, payload=None):
        super().__init__(message)
        self.status_code = status_code
        self.payload = payload
