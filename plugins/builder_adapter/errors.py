"""Stable typed errors for ``hermes.builder_dispatch.v1``."""

from __future__ import annotations


class AdapterError(RuntimeError):
    """A safe error that can cross the provider boundary."""

    def __init__(self, code: str, message: str, *, retryable: bool = False):
        super().__init__(message)
        self.code = code
        self.safe_message = message
        self.retryable = retryable

    def as_dict(self) -> dict:
        return {
            "code": self.code,
            "message": self.safe_message,
            "retryable": self.retryable,
        }
