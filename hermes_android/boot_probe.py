import json
import os
import platform
import sys
from pathlib import Path


def boot_probe() -> str:
    """Return a small JSON payload proving the embedded Python runtime can import Hermes code."""
    payload = {
        "status": "ok",
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "executable": sys.executable,
        "cwd": os.getcwd(),
        "hermes_android_file": str(Path(__file__).resolve()),
    }
    return json.dumps(payload, sort_keys=True)
