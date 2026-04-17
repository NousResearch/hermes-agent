"""Firecracker provider rejects exec until wired."""

import asyncio

import pytest

from sandbox.errors import SandboxNotSupportedError
from sandbox.registry import get_provider


def test_firecracker_exec_raises():
    p = get_provider("firecracker", {})
    with pytest.raises(SandboxNotSupportedError):
        asyncio.run(p.exec_cmd(["true"], "/", {}, 1.0))
