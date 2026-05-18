"""Minimal ACP exception types for lean test environments."""

from __future__ import annotations

from typing import Any


class RequestError(Exception):
    def __init__(self, code: int, message: str, data: Any = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.data = data

    @classmethod
    def method_not_found(cls, method: str) -> "RequestError":
        return cls(-32601, "Method not found", {"method": method})

    @classmethod
    def invalid_params(cls, data: Any = None) -> "RequestError":
        return cls(-32602, "Invalid params", data)
