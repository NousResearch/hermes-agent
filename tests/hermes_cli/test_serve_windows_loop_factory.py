"""Windows serve must run uvicorn on a SelectorEventLoop.

Regression for #120164. uvicorn 0.41's ``asyncio_loop_factory`` returns
ProactorEventLoop on win32. Driving uvicorn's socket stack on the proactor loop
binds-but-never-accepts: READY prints, then ``Accept failed on a socket`` +
WinError 10014, exit 1, desktop ECONNREFUSED (regression of #50641, whose fix
trusted ``config.get_loop_factory()`` back when that still meant selector).
"""

import asyncio

import pytest

pytestmark = pytest.mark.windows_only


class _ProactorConfig:
    def get_loop_factory(self):
        return asyncio.ProactorEventLoop


class _SelectorConfig:
    def get_loop_factory(self):
        return asyncio.SelectorEventLoop


def _capture_runner(monkeypatch):
    captured = {}

    def _fake_runner(main, **kwargs):
        captured.update(kwargs)
        main.close()
        return None

    monkeypatch.setattr("uvicorn._compat.asyncio_run", _fake_runner)
    return captured


async def _noop():
    return None


def test_proactor_factory_is_overridden_to_selector(monkeypatch):
    from hermes_cli import web_server

    captured = _capture_runner(monkeypatch)
    web_server._run_serve(_noop, _ProactorConfig(), "127.0.0.1", 0)
    factory = captured.get("loop_factory")
    assert factory is not None, "expected an explicit loop_factory on win32"
    loop = factory()
    try:
        assert isinstance(loop, asyncio.SelectorEventLoop), (
            f"serve must not run on {type(loop).__name__} (#120164)"
        )
    finally:
        loop.close()


def test_selector_factory_passes_through(monkeypatch):
    from hermes_cli import web_server

    captured = _capture_runner(monkeypatch)
    web_server._run_serve(_noop, _SelectorConfig(), "127.0.0.1", 0)
    assert captured.get("loop_factory") is asyncio.SelectorEventLoop
