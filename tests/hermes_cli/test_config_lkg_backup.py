"""A fresh process recovers the last successfully parsed config.yaml when the file is broken.

The in-process last-known-good (#31188 port) only helps a long-running gateway. A CLI restart or
``hermes config get`` against broken YAML used to run on ``DEFAULT_CONFIG`` — dropping every
override, including ``approvals.deny`` (#102945). Successful loads now leave a ``good`` copy in
``backups/config/`` and the fallback reads it.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

from hermes_cli.config_backups import list_config_backups

REPO = Path(__file__).resolve().parents[2]
GOOD = (
    "model:\n  default: test/secure\n"
    "approvals:\n  deny:\n    - 'curl*evil*'\n"
    "custom_providers:\n  - name: p\n    base_url: https://x.invalid/v1\n    api_key: ${LKG_TOKEN}\n"
)
BROKEN = "approvals:\n  deny: [unclosed\n"


def _fresh_load(home: Path) -> tuple[dict, str]:
    env = {**os.environ, "HERMES_HOME": str(home), "PYTHONPATH": str(REPO), "LKG_TOKEN": "expanded-secret"}
    proc = subprocess.run(
        [sys.executable, "-c", "import json; from hermes_cli.config import load_config; print(json.dumps(load_config()))"],
        cwd=REPO, env=env, text=True, capture_output=True, check=True, stdin=subprocess.DEVNULL,
    )
    return json.loads(proc.stdout), proc.stderr


def test_fresh_process_recovers_last_good_config_and_leaves_broken_file_alone(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(GOOD, encoding="utf-8")
    first, _ = _fresh_load(tmp_path)
    assert first["approvals"]["deny"] == ["curl*evil*"]

    config_path.write_text(BROKEN, encoding="utf-8")
    recovered, stderr = _fresh_load(tmp_path)

    assert recovered["approvals"]["deny"] == ["curl*evil*"]
    assert recovered["model"]["default"] == "test/secure"
    assert recovered["custom_providers"][0]["api_key"] == "expanded-secret"
    assert "LAST KNOWN GOOD" in stderr
    assert config_path.read_text(encoding="utf-8") == BROKEN


def test_good_backup_keeps_env_templates_and_dedupes_repeat_loads(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(GOOD, encoding="utf-8")
    _fresh_load(tmp_path)
    _fresh_load(tmp_path)

    good = list_config_backups(config_path, "good")
    assert len(good) == 1
    text = good[0].read_text(encoding="utf-8")
    assert "${LKG_TOKEN}" in text and "expanded-secret" not in text
