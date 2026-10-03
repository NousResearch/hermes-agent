"""mem0ai reads MEM0_TELEMETRY once, at its first import; the bundled provider must default it off by then."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]

# Records the flag mem0 would see, then fails like a missing extra; shadows any installed mem0ai.
_RECORDING_MEM0 = """
import json, os, pathlib
with open(pathlib.Path(__file__).parent.parent / "seen.json", "a") as f:
    f.write(json.dumps(os.environ.get("MEM0_TELEMETRY")) + "\\n")
raise ImportError("recording stub")
"""

_LOAD = """
import plugins.memory as pm
pm.load_memory_provider("mem0")
"""


@pytest.mark.parametrize(("user_value", "seen"), [(None, "false"), ("true", "true")])
def test_first_mem0_import_sees_telemetry_off_unless_user_opted_in(tmp_path, user_value, seen):
    stub = tmp_path / "stub" / "mem0"
    stub.mkdir(parents=True)
    (stub / "__init__.py").write_text(_RECORDING_MEM0)
    env = {k: v for k, v in os.environ.items() if k != "MEM0_TELEMETRY"}
    env["PYTHONPATH"] = os.pathsep.join([str(stub.parent), str(REPO_ROOT)])
    if user_value is not None:
        env["MEM0_TELEMETRY"] = user_value

    subprocess.run([sys.executable, "-c", _LOAD], cwd=REPO_ROOT, env=env, check=True, timeout=120)

    recorded = [json.loads(line) for line in (tmp_path / "stub" / "seen.json").read_text().splitlines()]
    assert recorded and recorded[0] == seen
