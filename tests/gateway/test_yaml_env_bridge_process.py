"""Real interpreter boundaries must preserve bridge ownership without defeating dotenv."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


_BOOT = """
import json, os, sys
from pathlib import Path
sys.path[:] = json.loads(sys.argv[2])
from hermes_cli.env_loader import load_hermes_dotenv
from gateway.platforms import _shared
from utils import load_yaml_file_readonly
home = Path(sys.argv[1])
load_hermes_dotenv(hermes_home=home, load_external_secrets=False)
config = load_yaml_file_readonly(home / 'config.yaml')
_shared.yaml_env_setter()('TELEGRAM_REACTIONS', str(config['telegram']['reactions']).lower())
print(json.dumps({'value': os.environ.get('TELEGRAM_REACTIONS'),
                  'marker': os.environ.get('HERMES_BRIDGED_TELEGRAM_REACTIONS'),
                  'module': _shared.__file__}))
"""


def _boot(home, env):
    result = subprocess.run(
        [sys.executable, "-c", _BOOT, str(home), json.dumps(sys.path)],
        env=env, capture_output=True, text=True, timeout=60, check=False,
    )
    assert result.returncode == 0, result.stderr
    observed = json.loads(result.stdout.splitlines()[-1])
    assert Path(observed["module"]).resolve() == (
        Path(__file__).resolve().parents[2] / "gateway/platforms/_shared.py"
    )
    return observed


@pytest.mark.parametrize("source", ["bridge", "shell", "dotenv"])
def test_fresh_process_refreshes_only_bridge_owned_values(tmp_path, source):
    """Two fresh interpreters read actual YAML and dotenv; no ownership record is mocked."""
    home = tmp_path / "profile"
    home.mkdir()
    config = home / "config.yaml"
    config.write_text("telegram:\n  reactions: false\n", encoding="utf-8")
    env = {key: value for key, value in os.environ.items()
           if not key.startswith(("HERMES_", "TELEGRAM_", "DISCORD_"))}
    env.update({key: str(home) for key in ("HERMES_HOME", "HOME", "USERPROFILE", "TMP", "TEMP", "TMPDIR")})
    first = _boot(home, env)
    assert first["value"] == "false"
    env["TELEGRAM_REACTIONS"] = first["value"]
    if source != "shell" and first["marker"] is not None:
        env["HERMES_BRIDGED_TELEGRAM_REACTIONS"] = first["marker"]
    if source == "dotenv":
        (home / ".env").write_text("TELEGRAM_REACTIONS=false\n", encoding="utf-8")
    config.write_text("telegram:\n  reactions: true\n", encoding="utf-8")

    second = _boot(home, env)

    assert second["value"] == ("true" if source == "bridge" else "false")
    if source == "bridge":
        assert second["marker"] == "true"
