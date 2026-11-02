"""Startup guards for ``import run_agent``, which every CLI, TUI, desktop-backend and gateway
process pays before it can do anything.

Built-in tool modules are indexed at import and imported on first need
(``tools.registry.defer_builtin_tools``), and the scheduler and the messaging-gateway runtime load
only where they are used. The contracts below are deterministic and catch a return to eager
loading; the wall-clock and peak-RSS ceilings are coarse backstops against gross bloat (a blocking
probe or a heavy SDK at import), sized for a loaded CI runner. For scale, on an M-series Mac with
warm bytecode the import measured ~0.47 s / ~103 MB after deferral and ~0.61 s / ~117 MB before.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

# Each run is a fresh interpreter; the first one only warms bytecode and the discovery cache.
_MEASURED_RUNS = 3
_IMPORT_SECONDS_CEILING = 3.0
_PEAK_RSS_MB_CEILING = 150.0

# Modules that only the scheduler and the messaging gateway's runtime need.
_GATEWAY_RUNTIME = ("gateway.run", "gateway.platforms.base", "gateway.session", "gateway.delivery")

_PROBE = """
import json, sys, time
start = time.perf_counter()
import run_agent  # noqa: F401
seconds = time.perf_counter() - start
try:
    import resource
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1 << 20 if sys.platform == "darwin" else 1 << 10)
except ImportError:  # no resource module on Windows
    peak = None
from tools.registry import registry
print(json.dumps({"seconds": seconds, "peak_rss_mb": peak, "modules": sorted(sys.modules),
                  "deferred": sorted(registry._pending)}))
"""


@pytest.fixture(scope="module")
def import_runs(tmp_path_factory) -> list[dict]:
    home = tmp_path_factory.mktemp("import-budget-home")
    env = {**os.environ, "HERMES_HOME": str(home), "PYTHONPATH": str(REPO_ROOT)}
    env.pop("HERMES_PROFILE", None)
    # Measure the import users get after their first run, whatever this shell says about bytecode.
    env.pop("PYTHONDONTWRITEBYTECODE", None)
    runs = []
    for _ in range(_MEASURED_RUNS + 1):
        probe = subprocess.run([sys.executable, "-c", _PROBE], cwd=REPO_ROOT, env=env, stdin=subprocess.DEVNULL,
                               capture_output=True, text=True, timeout=120, check=False)
        assert probe.returncode == 0, probe.stderr[-4000:]
        runs.append(json.loads(probe.stdout.strip().splitlines()[-1]))
    return runs[1:]


def test_import_leaves_builtin_tool_discovery_deferred(import_runs):
    run = import_runs[-1]
    assert run["deferred"], "import run_agent imported every built-in tool module (eager discovery is back)"
    imported = sorted(set(run["deferred"]) & set(run["modules"]))
    assert not imported, f"deferred tool modules imported during import run_agent: {imported}"


def test_import_loads_neither_the_scheduler_nor_the_gateway_runtime(import_runs):
    modules = set(import_runs[-1]["modules"])
    leaked = sorted(m for m in modules if m == "cron" or m.startswith("cron.")) + sorted(
        m for m in _GATEWAY_RUNTIME if m in modules)
    assert not leaked, f"import run_agent loaded {leaked}; import them at their use sites"


def test_import_time_ceiling(import_runs):
    fastest = min(run["seconds"] for run in import_runs)
    assert fastest <= _IMPORT_SECONDS_CEILING, f"import run_agent took {fastest:.2f}s (best of {len(import_runs)})"


@pytest.mark.platforms("posix")
def test_import_peak_rss_ceiling(import_runs):
    lowest = min(run["peak_rss_mb"] for run in import_runs)
    assert lowest <= _PEAK_RSS_MB_CEILING, f"import run_agent peaked at {lowest:.0f} MB RSS"
