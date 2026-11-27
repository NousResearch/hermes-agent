"""The cua-driver probe env must not import the application graph.

``CuaDriver._probe_env`` used to call ``tools.computer_use.cua_backend``'s
``cua_driver_child_env()``. That import pulls ``hermes_cli.config``, which re-runs
provider plugin discovery at import time, so a bundled provider plugin missing a
dependency the PM runtime deliberately does not carry (``solstice`` -> ``httpx``,
see ``pm/pyproject.toml``) printed
``Failed to load bundled provider plugin solstice: No module named 'httpx'``
out of ``hermes pm doctor`` / ``hermes update``. PM can act on none of that —
the provider registry is application state, not package state.

``load_config`` is unusable for the same reason: it imports the registry too.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from pm import get_package
from pm.packages import CuaDriver

REPO = Path(__file__).resolve().parents[2]
TELEMETRY_ENV_VAR = "CUA_DRIVER_RS_TELEMETRY_ENABLED"

# Modules that exist only for the application: importing any of them from a PM
# probe is exactly the defect this file pins.
_APPLICATION_MODULES = ("hermes_cli.config", "providers")

_PROBE = """
import sys
sys.path.insert(0, {repo!r})
from pm import get_package

env = get_package("cua-driver")._probe_env()
print("TELEMETRY=" + str(env.get({var!r})))
print("REACHED=" + ",".join(name for name in {modules!r} if name in sys.modules))
"""


def test_probe_env_disables_telemetry():
    env = get_package("cua-driver")._probe_env()
    assert env[TELEMETRY_ENV_VAR] == "0"


def test_probe_env_never_imports_the_application_graph():
    """Regression: the probe must not reach ``hermes_cli.config`` (provider discovery).

    A fresh interpreter, because this test process has already imported the
    application for other tests — a preloaded module would hide the defect.
    """
    script = _PROBE.format(repo=str(REPO), var=TELEMETRY_ENV_VAR,
                           modules=list(_APPLICATION_MODULES))
    result = subprocess.run([sys.executable, "-c", script], capture_output=True,
                            text=True, timeout=60, cwd=str(REPO))

    assert result.returncode == 0, result.stdout + result.stderr
    assert "TELEMETRY=0" in result.stdout, result.stdout
    assert "REACHED=" in result.stdout, result.stdout
    reached = result.stdout.split("REACHED=", 1)[1].strip()
    assert reached == "", f"the probe imported application modules: {reached}"


def test_probe_env_survives_a_broken_bot_desktop(monkeypatch):
    """A Bot Desktop failure is not fatal: telemetry off still has to be returned."""
    import tools.bot_desktop.runtime as bd_runtime

    def boom(_env=None):
        raise RuntimeError("no screen")

    monkeypatch.setattr(bd_runtime, "desktop_env", boom)
    env = get_package("cua-driver")._probe_env()
    assert env[TELEMETRY_ENV_VAR] == "0"


def test_probe_env_constant_matches_the_driver_gate():
    """``CuaDriver`` carries the env var name so the probe needs no app import for it."""
    assert CuaDriver.CUA_DRIVER_TELEMETRY_ENV_VAR == TELEMETRY_ENV_VAR
