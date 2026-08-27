"""Stable typed failures for Ares installed-runtime operations."""

from __future__ import annotations


class AresRuntimeError(RuntimeError):
    """A fail-closed installed-runtime error with a machine-readable code."""

    def __init__(self, code: str, detail: str = "") -> None:
        self.code = code
        self.detail = detail
        super().__init__(f"{code}: {detail}" if detail else code)
