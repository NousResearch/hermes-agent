from __future__ import annotations

import contextlib
import dataclasses
import sys
import typing

__all__ = ["ExceptionGroup"]


def __dir__() -> list[str]:
    return __all__


if sys.version_info >= (3, 11):  # pragma: no cover
    from builtins import ExceptionGroup
else:  # pragma: no cover

    class ExceptionGroup(Exception):
        """A minimal implementation of :external:exc:`ExceptionGroup` from Python 3.11.

        If :external:exc:`ExceptionGroup` is already defined by Python itself,
        that version is used instead.
        """

        message: str
        exceptions: list[Exception]

        def __init__(self, message: str, exceptions: list[Exception]) -> None:
            self.message = message
            self.exceptions = exceptions

        def __repr__(self) -> str:
            return f"{self.__class__.__name__}({self.message!r}, {self.exceptions!r})"


@dataclasses.dataclass
class _ErrorCollector:
    """
    Collect errors into ExceptionGroups.

    Used like this:

        collector = _ErrorCollector()
        # Add a single exception
        collector.error(ValueError("one"))

        # Supports nesting, including combining ExceptionGroups
        with collector.collect():
            raise ValueError("two")
        collector.finalize("Found some errors")

    Since making a collector and then calling finalize later is a common pattern,
    a convenience method ``on_exit`` is provided.
    """

    errors: list[Exception] = dataclasses.field(default_factory=list, init=False)

    def finalize(self, msg: str) -> None:
        """Raise a group exception if there are any errors."""
        if self.errors:
            raise ExceptionGroup(msg, self.errors)

    @contextlib.contextmanager
    def on_exit(self, msg: str) -> typing.Generator[_ErrorCollector, None, None]:
        """
        Calls finalize if no uncollected errors were present.

        Uncollected errors are raised normally.
        """
        yield self
        self.finalize(msg)

    @contextlib.contextmanager
    def collect(self, *err_cls: type[Exception]) -> typing.Generator[None, None, None]:
        """
        Context manager to collect errors into the error list.

        Must be inside loops, as only one error can be collected at a time.
        """
        error_classes = err_cls or (Exception,)
        try:
            yield
        except ExceptionGroup as error:
            self.errors.extend(error.exceptions)
        except error_classes as error:
            self.errors.append(error)

    def error(
        self,
        error: Exception,
    ) -> None:
        """Add an error to the list."""
        self.errors.append(error)
