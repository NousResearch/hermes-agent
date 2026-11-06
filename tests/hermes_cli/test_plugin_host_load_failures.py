"""The plugin-host lane must report a raising ``register()``/constructor, not swallow it.

``HostRuntime.op_load_instance`` runs category plugins (memory provider, context engine,
cron scheduler) that a profile isolates in the plugin host. It had the same two blind
spots as the in-process loader: a ``register()`` that raised with nothing captured was a
DEBUG line, and a subclass constructor exception was a bare ``continue``. An operator
running ``plugins.isolation: host`` got no more than "no instance" for either.
"""

from __future__ import annotations

import logging
import sys
import types
from pathlib import Path

from hermes_cli.plugin_host_child import HostRuntime

LOGGER_NAME = "hermes_cli.plugin_host_child"
_MODULE_PREFIX = "_hermes_test_host_load_failures"
_BASE_MODULE = "_hermes_test_host_load_base"


class _Base:
    pass


def _fake_runtime() -> HostRuntime:
    """A HostRuntime with the protocol channel and reference tables stubbed out."""
    runtime = HostRuntime.__new__(HostRuntime)
    runtime.modules = {}
    runtime.instance_modules = {}
    runtime._run = lambda result: result
    runtime._describe_object = lambda plugin_key, obj: {"plugin_key": plugin_key, "obj": obj}
    return runtime


def _install_base() -> None:
    module = types.ModuleType(_BASE_MODULE)
    module._Base = _Base  # type: ignore[attr-defined]
    sys.modules[_BASE_MODULE] = module


def _cleanup() -> None:
    for name in tuple(sys.modules):
        if name == _BASE_MODULE or name.startswith(f"{_MODULE_PREFIX}."):
            sys.modules.pop(name, None)


def _write_plugin(tmp_path: Path, source: str) -> Path:
    plugin = tmp_path / "acme"
    plugin.mkdir()
    (plugin / "__init__.py").write_text(source, encoding="utf-8")
    return plugin


def _params(plugin: Path) -> dict:
    return {
        "plugin_key": "acme",
        "path": str(plugin),
        "module_name": f"{_MODULE_PREFIX}.acme",
        "base": f"{_BASE_MODULE}:_Base",
        "capture": "register_engine",
        "name": "acme",
        "forward": False,
    }


def _warnings(caplog) -> list[logging.LogRecord]:
    return [record for record in caplog.records if record.levelno >= logging.WARNING]


def test_register_failure_is_logged_with_its_traceback(tmp_path, caplog):
    """A host-side register() that raises names the plugin and carries the traceback."""
    plugin = _write_plugin(tmp_path, "def register(ctx):\n"
                                    "    raise RuntimeError('provider db is locked')\n")
    _install_base()
    caplog.set_level(logging.DEBUG, logger=LOGGER_NAME)

    try:
        assert _fake_runtime().op_load_instance(_params(plugin)) is None
    finally:
        _cleanup()
        sys.modules.pop(_BASE_MODULE, None)

    warned = _warnings(caplog)
    assert len(warned) == 1, [record.getMessage() for record in warned]
    message = warned[0].getMessage()
    assert "acme" in message
    assert "RuntimeError" in message
    assert "provider db is locked" in message
    assert warned[0].exc_info is not None


def test_constructor_failure_is_logged_with_its_traceback(tmp_path, caplog):
    """A host-side subclass constructor that raises is reported, not silently skipped."""
    plugin = _write_plugin(tmp_path, f"from {_BASE_MODULE} import _Base\n"
                                    "\n"
                                    "\n"
                                    "class Exploding(_Base):\n"
                                    "    def __init__(self):\n"
                                    "        raise RuntimeError('engine db is malformed')\n")
    _install_base()
    caplog.set_level(logging.DEBUG, logger=LOGGER_NAME)

    try:
        assert _fake_runtime().op_load_instance(_params(plugin)) is None
    finally:
        _cleanup()
        sys.modules.pop(_BASE_MODULE, None)

    warned = _warnings(caplog)
    assert len(warned) == 1, [record.getMessage() for record in warned]
    message = warned[0].getMessage()
    assert "acme" in message
    assert "Exploding" in message
    assert "engine db is malformed" in message
    assert warned[0].exc_info is not None


def test_a_working_plugin_stays_quiet(tmp_path, caplog):
    """The success path still returns the instance and logs no warning."""
    plugin = _write_plugin(tmp_path, f"from {_BASE_MODULE} import _Base\n"
                                    "\n"
                                    "\n"
                                    "class Working(_Base):\n"
                                    "    pass\n")
    _install_base()
    caplog.set_level(logging.DEBUG, logger=LOGGER_NAME)

    try:
        described = _fake_runtime().op_load_instance(_params(plugin))
    finally:
        _cleanup()
        sys.modules.pop(_BASE_MODULE, None)

    assert described is not None
    assert described["obj"].__class__.__name__ == "Working"
    assert _warnings(caplog) == []
