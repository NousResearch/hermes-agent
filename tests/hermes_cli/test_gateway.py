import sys
import types

import pytest

from hermes_cli import gateway as gateway_cli


def test_run_gateway_exits_success_when_already_running(monkeypatch, capsys):
    start_calls = []

    async def _start_gateway():
        start_calls.append(True)
        return True

    monkeypatch.setitem(sys.modules, "gateway.run", types.SimpleNamespace(start_gateway=_start_gateway))
    monkeypatch.setitem(sys.modules, "gateway.status", types.SimpleNamespace(is_gateway_running=lambda: True))

    gateway_cli.run_gateway()

    out = capsys.readouterr().out
    assert "Gateway is already running" in out
    assert start_calls == []


def test_run_gateway_exits_nonzero_when_start_fails(monkeypatch):
    async def _start_gateway():
        return False

    monkeypatch.setitem(sys.modules, "gateway.run", types.SimpleNamespace(start_gateway=_start_gateway))
    monkeypatch.setitem(sys.modules, "gateway.status", types.SimpleNamespace(is_gateway_running=lambda: False))

    with pytest.raises(SystemExit) as exc:
        gateway_cli.run_gateway()

    assert exc.value.code == 1
