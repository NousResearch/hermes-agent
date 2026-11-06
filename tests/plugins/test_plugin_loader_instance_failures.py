"""A category plugin that raises while loading must be loud, not a silent "no instance".

``instance_from_module`` used to log ``register()`` failures at DEBUG and swallow a
subclass constructor's exception entirely, so the only trace of a broken engine was the
generic "loaded but no engine instance found" warning. A corrupt database in the
constructor then left the built-in fallback running for days with nothing in the log
that named the real error.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Any, Optional

from plugins.plugin_loader import instance_from_module

LOGGER_NAME = "tests.plugin_loader_instance_failures"


class _Base:
    """Stand-in for ``ContextEngine`` / ``CronScheduler`` / etc."""


class _Collector:
    """Stand-in for the fake register() context a category passes in."""

    def __init__(self) -> None:
        self.instance: Optional[Any] = None

    def register_instance(self, instance: Any) -> None:
        self.instance = instance


def _load(module: Any) -> Any:
    return instance_from_module(
        module,
        collector=_Collector(),
        collected_attr="instance",
        base_cls=_Base,
        name="acme",
        logger=logging.getLogger(LOGGER_NAME),
    )


def _warnings(caplog) -> list[logging.LogRecord]:
    return [record for record in caplog.records if record.levelno >= logging.WARNING]


def test_register_failure_is_logged_with_its_traceback(caplog):
    """A register(ctx) that raises must be a WARNING naming the plugin and the exception."""

    def register(_ctx):
        raise RuntimeError("guard broken")

    caplog.set_level(logging.DEBUG, logger=LOGGER_NAME)

    assert _load(SimpleNamespace(register=register)) is None

    warned = _warnings(caplog)
    assert len(warned) == 1, [record.getMessage() for record in warned]
    message = warned[0].getMessage()
    assert "acme" in message
    assert "RuntimeError" in message
    assert "guard broken" in message
    assert warned[0].exc_info is not None, "the traceback must reach the log record"


def test_constructor_failure_is_logged_with_its_traceback(caplog):
    """A raising subclass constructor must be a WARNING, not a suppressed exception."""

    class Exploding(_Base):
        def __init__(self) -> None:
            raise RuntimeError("database disk image is malformed")

    caplog.set_level(logging.DEBUG, logger=LOGGER_NAME)

    assert _load(SimpleNamespace(Exploding=Exploding)) is None

    warned = _warnings(caplog)
    assert len(warned) == 1, [record.getMessage() for record in warned]
    message = warned[0].getMessage()
    assert "acme" in message
    assert "Exploding" in message
    assert "database disk image is malformed" in message
    assert warned[0].exc_info is not None, "the traceback must reach the log record"


def test_a_failing_subclass_does_not_hide_a_working_sibling(caplog):
    """The scan keeps trying after a constructor raises; the failure is still reported."""

    class AlphaBroken(_Base):
        def __init__(self) -> None:
            raise RuntimeError("no database at /nowhere")

    class BetaWorking(_Base):
        def __init__(self) -> None:
            self.name = "beta"

    caplog.set_level(logging.DEBUG, logger=LOGGER_NAME)

    instance = _load(SimpleNamespace(AlphaBroken=AlphaBroken, BetaWorking=BetaWorking))

    assert isinstance(instance, BetaWorking)
    assert instance.name == "beta"
    warned = _warnings(caplog)
    assert len(warned) == 1, [record.getMessage() for record in warned]
    assert "AlphaBroken" in warned[0].getMessage()


def test_successful_register_is_unchanged(caplog):
    """The register() success path keeps returning its instance without any warning."""
    instance = _Base()

    def register(ctx):
        ctx.register_instance(instance)

    caplog.set_level(logging.DEBUG, logger=LOGGER_NAME)

    assert _load(SimpleNamespace(register=register)) is instance
    assert _warnings(caplog) == []


def test_successful_subclass_is_unchanged(caplog):
    """The subclass success path keeps constructing, and stays quiet."""

    class Working(_Base):
        def __init__(self) -> None:
            self.name = "working"

    caplog.set_level(logging.DEBUG, logger=LOGGER_NAME)

    instance = _load(SimpleNamespace(Working=Working))

    assert isinstance(instance, Working)
    assert _warnings(caplog) == []
