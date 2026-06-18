#!/usr/bin/env python3
import json
import os
import subprocess
import sys
from pathlib import Path


def _desktop_dir() -> Path:
    if os.name == "nt":
        try:
            import winreg

            key_path = r"Software\Microsoft\Windows\CurrentVersion\Explorer\User Shell Folders"
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, key_path) as key:
                raw, _ = winreg.QueryValueEx(key, "Desktop")
            return Path(os.path.expandvars(str(raw)))
        except Exception:
            pass
    return Path.home() / "Desktop"


ROOT = _desktop_dir() / "ShinkaEvolve-OSINT-main" / "ShinkaEvolve-OSINT-main"
EXAMPLE = "milspec_security_jp"
EXAMPLE_DIR = ROOT / "examples" / EXAMPLE

RUNNER = Path(__file__).with_name("_shinka_isolated_runner_snippet.py")
# inline
CODE = """
import importlib.util, json, os, sys
from pathlib import Path
payload = json.loads(sys.stdin.read())
root = Path(os.environ["SHINKA_OSINT_ROOT"]).resolve()
example = (payload.get("arguments") or {}).get("example", "")
example_dir = root / "examples" / example
sys.path[:0] = [str(example_dir), str(root)]
os.chdir(str(example_dir))
spec = importlib.util.spec_from_file_location("shinka_mcp_server", root / "shinka_mcp_server.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
result = module.TOOL_HANDLERS[payload["tool"]](payload.get("arguments") or {})
print(json.dumps(result, ensure_ascii=False, default=str))
"""

def call(tool, arguments):
    env = os.environ.copy()
    env["SHINKA_OSINT_ROOT"] = str(ROOT)
    env["SHINKA_DISABLE_GEMINI_EMBEDDING"] = "1"
    env["PYTHONPATH"] = os.pathsep.join([str(EXAMPLE_DIR), str(ROOT)])
    proc = subprocess.run(
        [sys.executable, "-c", CODE],
        input=json.dumps({"tool": tool, "arguments": arguments}),
        cwd=str(EXAMPLE_DIR),
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr[-2000:])
    return json.loads(proc.stdout)

if __name__ == "__main__":
    data = call("shinka_list_scenarios", {"example": EXAMPLE})
    for s in data.get("scenarios", []):
        print(s.get("scenario_id"), "|", s.get("domain"), "|", (s.get("query") or "")[:70])
