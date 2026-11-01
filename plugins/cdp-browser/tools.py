"""Handlers for cdp-browser — run the bundled composed CDP driver."""
import json
import subprocess
import sys
from pathlib import Path

# Driver is bundled inside the plugin (self-contained install). Fall back to
# the workspace copy if present (pre-standalone layout).
_PLUGIN_DIR = Path(__file__).resolve().parent
DRIVER = _PLUGIN_DIR / "driver" / "cdp_browser.py"
_FALLBACK = Path(
    r"C:\HERMES WORKSPACE\YOUTUBE CHANNELS\collector\img2vid\helpers\cdp_browser.py"
)


def _check_deps():
    """Return an error string if a runtime dependency is missing, else None."""
    try:
        import websocket  # noqa: F401
        return None
    except ImportError:
        return (
            "missing dependency: websocket-client. Install with: "
            "pip install websocket-client"
        )


def _run_driver(args, timeout=120):
    """Run the driver script, return JSON-string result for the LLM."""
    dep_err = _check_deps()
    if dep_err:
        return json.dumps({"error": dep_err})

    driver = DRIVER if DRIVER.exists() else _FALLBACK
    if not driver.exists():
        return json.dumps(
            {"error": f"cdp_browser.py driver not found at {DRIVER} or {_FALLBACK}"}
        )
    try:
        r = subprocess.run(
            [sys.executable, str(driver)] + args,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return json.dumps({"error": f"driver timed out after {timeout}s"})
    if r.returncode != 0:
        return json.dumps(
            {"error": f"driver exited {r.returncode}", "stderr": r.stderr[-2000:]}
        )
    out = r.stdout.strip()
    try:
        # Driver prints JSON — pass through parsed so the LLM sees structured data.
        return json.dumps(json.loads(out), ensure_ascii=False)
    except Exception:
        return out[-4000:] or json.dumps({"error": "empty driver output"})


def cdp_list(port: int = 9333) -> str:
    return _run_driver(["list", "--port", str(port)])


def cdp_run(steps: str, tab: str = "auto", port: int = 9333) -> str:
    # Accept either a JSON string (from the LLM) or a Python list.
    if isinstance(steps, (list, dict)):
        steps = json.dumps(steps)
    return _run_driver(["run", steps, "--tab", tab, "--port", str(port)])


def cdp_spaces(spaces: str, port: int = 9333) -> str:
    if isinstance(spaces, (list, dict)):
        spaces = json.dumps(spaces)
    return _run_driver(["spaces", spaces, "--port", str(port)])
