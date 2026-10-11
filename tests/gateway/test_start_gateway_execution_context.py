"""Gateway execution establishes its context without poisoning incidental imports."""

import asyncio
import os

import pytest


def test_direct_start_gateway_marks_context_before_startup(monkeypatch):
    from gateway.run import start_gateway
    from hermes_cli import resource_limits

    monkeypatch.delenv("_HERMES_GATEWAY", raising=False)
    monkeypatch.delenv("HERMES_EXEC_ASK", raising=False)

    class StartupObserved(Exception):
        pass

    def observe_context():
        assert os.environ.get("_HERMES_GATEWAY") == "1"
        assert os.environ.get("HERMES_EXEC_ASK") == "1"
        raise StartupObserved

    monkeypatch.setattr(resource_limits, "apply_nofile_soft_limit", observe_context)

    with pytest.raises(StartupObserved):
        asyncio.run(start_gateway())
