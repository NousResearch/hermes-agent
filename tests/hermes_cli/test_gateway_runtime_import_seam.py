"""Import-order and evaluated-annotation contract for the gateway runtime shard."""

import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize("first", ["hermes_cli.gateway_runtime", "hermes_cli.gateway"])
def test_gateway_runtime_imports_in_either_order_with_facade_type(first, tmp_path):
    other = "hermes_cli.gateway" if first.endswith("_runtime") else "hermes_cli.gateway_runtime"
    script = (
        "import importlib\n"
        f"importlib.import_module({first!r})\n"
        f"importlib.import_module({other!r})\n"
        "from hermes_cli import gateway, gateway_runtime\n"
        "assert gateway.get_gateway_runtime_snapshot is gateway_runtime.get_gateway_runtime_snapshot\n"
        "assert gateway_runtime.get_gateway_runtime_snapshot.__annotations__['return'] "
        "is gateway.GatewayRuntimeSnapshot\n"
    )
    env = os.environ.copy()
    env["HERMES_HOME"] = str(tmp_path)
    env["HOME"] = str(tmp_path)
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[2],
        env=env,
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stderr
