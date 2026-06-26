from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import gateway.run as gateway_run


class _ExitCalled(Exception):
    def __init__(self, code: int):
        super().__init__(code)
        self.code = code


def _raise_exit(code: int) -> None:
    raise _ExitCalled(code)


def test_main_force_exits_zero_after_clean_shutdown(monkeypatch):
    async def fake_start_gateway(config=None):
        return True

    stdout = SimpleNamespace(flush=Mock())
    stderr = SimpleNamespace(flush=Mock())

    monkeypatch.setattr(gateway_run, "start_gateway", fake_start_gateway)
    monkeypatch.setattr(gateway_run.os, "_exit", _raise_exit)
    monkeypatch.setattr(gateway_run.sys, "argv", ["gateway.run"])
    monkeypatch.setattr(gateway_run.sys, "stdout", stdout)
    monkeypatch.setattr(gateway_run.sys, "stderr", stderr)

    with pytest.raises(_ExitCalled) as exc_info:
        gateway_run.main()

    assert exc_info.value.code == 0
    stdout.flush.assert_called_once_with()
    stderr.flush.assert_called_once_with()


def test_main_force_exits_one_after_failed_shutdown(monkeypatch):
    async def fake_start_gateway(config=None):
        return False

    stdout = SimpleNamespace(flush=Mock())
    stderr = SimpleNamespace(flush=Mock())

    monkeypatch.setattr(gateway_run, "start_gateway", fake_start_gateway)
    monkeypatch.setattr(gateway_run.os, "_exit", _raise_exit)
    monkeypatch.setattr(gateway_run.sys, "argv", ["gateway.run"])
    monkeypatch.setattr(gateway_run.sys, "stdout", stdout)
    monkeypatch.setattr(gateway_run.sys, "stderr", stderr)

    with pytest.raises(_ExitCalled) as exc_info:
        gateway_run.main()

    assert exc_info.value.code == 1
    stdout.flush.assert_called_once_with()
    stderr.flush.assert_called_once_with()
