"""Run the local auth fixture and verify the readiness response is closed."""

import subprocess
import sys
from pathlib import Path


_PROBE = r"""
import runpy
import sys
import urllib.request

observed = []
urlopen = urllib.request.urlopen

def tracked_urlopen(*args, **kwargs):
    response = urlopen(*args, **kwargs)
    observed.append(response)
    return response

urllib.request.urlopen = tracked_urlopen
script, root = sys.argv[1:]
sys.argv = [script, root]
runpy.run_path(script, run_name="__main__")
assert observed, "fixture never became ready"
assert observed[0].closed, "readiness response was left open"
"""


def test_live_readiness_closes_response():
    root = Path(__file__).resolve().parents[2]
    script = root / "evals/dashboard_auth/refresh_singleflight_live_e2e.py"
    result = subprocess.run([sys.executable, "-c", _PROBE, str(script), str(root)],
                            cwd=root, capture_output=True, text=True, timeout=90)

    assert result.returncode == 0, result.stdout + result.stderr
