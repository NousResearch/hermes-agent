"""config.yaml top-level scalars must never set a launch-only env key (#135200, #144685).

The gateway bridges every top-level config.yaml scalar into ``os.environ`` at import; a
user-writable ``HERMES_MANAGED_DIR:`` there would repoint the administrator's managed scope.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_gateway_import_ignores_launch_only_keys_in_config_yaml(tmp_path: Path) -> None:
    home = tmp_path / ".hermes"
    home.mkdir()
    planted = tmp_path / "planted-managed"
    planted.mkdir()
    (home / "config.yaml").write_text(
        f"HERMES_MANAGED_DIR: {planted}\nhermes_managed_dir: {planted}\n"
        "TELEGRAM_HOME_CHANNEL: '5550001111'\n", encoding="utf-8")
    script = textwrap.dedent(f"""
        import json, os, sys
        sys.path.insert(0, {str(PROJECT_ROOT)!r})
        from gateway import run  # noqa: F401  -- import runs the bridge
        print(json.dumps({{k: os.environ.get(k) for k in
                          ("HERMES_MANAGED_DIR", "hermes_managed_dir", "TELEGRAM_HOME_CHANNEL")}}))
    """)
    env = {k: v for k, v in os.environ.items()
           if k in ("PATH", "PYTHONPATH", "VIRTUAL_ENV", "HOME", "USERPROFILE", "LOCALAPPDATA",
                    "APPDATA", "SYSTEMROOT", "TEMP", "TMP")}
    env["HERMES_HOME"] = str(home)
    result = subprocess.run([sys.executable, "-c", script], env=env, capture_output=True,
                            text=True, timeout=60, check=False)
    assert result.returncode == 0, result.stderr
    seen = json.loads(result.stdout.strip().splitlines()[-1])
    assert seen["TELEGRAM_HOME_CHANNEL"] == "5550001111"  # ordinary scalars still bridge
    assert seen["HERMES_MANAGED_DIR"] is None
    assert seen["hermes_managed_dir"] is None
    assert "launch-only key HERMES_MANAGED_DIR" in result.stderr
