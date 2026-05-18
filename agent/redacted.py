"""Explicit wrapper for values that must not leak through logs or repr output."""

from __future__ import annotations

from typing import Any, Generic, TypeVar
from weakref import WeakKeyDictionary

A = TypeVar("A")
_REDACTED = "<redacted>"
_registry: WeakKeyDictionary[Any, object] = WeakKeyDictionary()


class Redacted(Generic[A]):
    """Container whose string/representation forms never reveal its value.

    The wrapped value is stored in a weak side registry rather than on the
    instance, so accidental ``repr()``, ``str()``, logging, or formatting of the
    wrapper produces only ``<redacted>``. Code that genuinely needs the secret
    must call ``Redacted.value(...)`` explicitly at the use boundary.
    """

    __slots__ = ("__weakref__",)

    def __repr__(self) -> str:
        return _REDACTED

    def __str__(self) -> str:
        return _REDACTED

    def __format__(self, format_spec: str) -> str:
        return format(str(self), format_spec)

    @classmethod
    def make(cls, value: A) -> "Redacted[A]":
        redacted = cls()
        _registry[redacted] = value
        return redacted

    @staticmethod
    def value(self: "Redacted[A]") -> A:
        try:
            return _registry[self]  # type: ignore[return-value]
        except (KeyError, TypeError) as exc:
            raise ValueError("Redacted value was not in registry") from exc
