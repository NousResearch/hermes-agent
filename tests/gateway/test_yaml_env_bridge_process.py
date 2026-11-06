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
from gateway.config_loader import load_yaml_layer
home = Path(sys.argv[1])
load_hermes_dotenv(hermes_home=home, load_external_secrets=False)
load_yaml_layer(home, {})
print(json.dumps({'value': os.environ.get('TELEGRAM_REACTIONS'),
                  'marker': os.environ.get('HERMES_BRIDGED_TELEGRAM_REACTIONS'),
                  'module': _shared.__file__}))
"""


_CHAIN = """
import json, os, subprocess, sys
from pathlib import Path
sys.path[:] = json.loads(sys.argv[2])
from hermes_cli.env_loader import load_hermes_dotenv
from gateway.config_loader import load_yaml_layer
from tools.environments.local import served_profile_child_env
home = Path(sys.argv[1])
load_hermes_dotenv(hermes_home=home, load_external_secrets=False)
load_yaml_layer(home, {})
first = {key: os.environ.get(key) for key in ('TELEGRAM_REACTIONS', 'HERMES_BRIDGED_TELEGRAM_REACTIONS')}
(home / 'config.yaml').write_text(sys.argv[3], encoding='utf-8')
if sys.argv[4] == 'dotenv':
    (home / '.env').write_text('TELEGRAM_REACTIONS=false\\n', encoding='utf-8')
elif sys.argv[4] == 'shell':
    os.environ.pop('HERMES_BRIDGED_TELEGRAM_REACTIONS', None)
# Match spawn_unmanaged_gateway's builder arguments, including the target secret overlay.
child_env = served_profile_child_env(base=dict(os.environ), target_home=home, inherit_credentials=True)
child = subprocess.run([sys.executable, '-c', sys.argv[5], str(home), json.dumps(sys.path)],
                       env=child_env, capture_output=True, text=True, timeout=60, check=False)
assert child.returncode == 0, child.stderr
result = json.loads(child.stdout.splitlines()[-1])
result['first'] = first
result['builder_module'] = sys.modules[served_profile_child_env.__module__].__file__
print(json.dumps(result))
"""


def _boot(home, env, *, replacement=None, source="bridge"):
    command = [sys.executable, "-c", _BOOT, str(home), json.dumps(sys.path)]
    if replacement is not None:
        command = [sys.executable, "-c", _CHAIN, str(home), json.dumps(sys.path), replacement, source, _BOOT]
    result = subprocess.run(
        command,
        env=env, capture_output=True, text=True, timeout=120, check=False,
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


@pytest.mark.parametrize("replacement", ["{}\n", "telegram:\n  allow_from: ['111']\n"])
def test_fresh_process_retracts_inherited_value_when_yaml_key_is_removed(tmp_path, replacement):
    home = tmp_path / "profile"
    home.mkdir()
    config = home / "config.yaml"
    config.write_text("telegram:\n  reactions: false\n", encoding="utf-8")
    env = {key: value for key, value in os.environ.items()
           if not key.startswith(("HERMES_", "TELEGRAM_", "DISCORD_"))}
    env.update({key: str(home) for key in ("HERMES_HOME", "HOME", "USERPROFILE", "TMP", "TEMP", "TMPDIR")})
    first = _boot(home, env)
    assert first["value"] == "false"
    assert first["marker"] == "false"
    env["TELEGRAM_REACTIONS"] = first["value"]
    env["HERMES_BRIDGED_TELEGRAM_REACTIONS"] = first["marker"]
    config.write_text(replacement, encoding="utf-8")

    second = _boot(home, env)

    assert second["value"] is None
    assert second["marker"] is None


@pytest.mark.parametrize("source", ["bridge", "shell", "dotenv"])
@pytest.mark.parametrize("replacement", ["telegram:\n  reactions: true\n", "{}\n"])
def test_real_child_env_builder_preserves_refresh_retraction_and_operator_precedence(tmp_path, source, replacement):
    home = tmp_path / "profile"
    home.mkdir()
    (home / "config.yaml").write_text("telegram:\n  reactions: false\n", encoding="utf-8")
    env = {key: value for key, value in os.environ.items()
           if not key.startswith(("HERMES_", "TELEGRAM_", "DISCORD_"))}
    env.update({key: str(home) for key in ("HERMES_HOME", "HOME", "USERPROFILE", "TMP", "TEMP", "TMPDIR")})

    observed = _boot(home, env, replacement=replacement, source=source)

    assert observed["first"] == {"TELEGRAM_REACTIONS": "false", "HERMES_BRIDGED_TELEGRAM_REACTIONS": "false"}
    assert Path(observed["builder_module"]).resolve() == (
        Path(__file__).resolve().parents[2] / "tools/environments/local.py"
    )
    expected = "false" if source != "bridge" else (None if replacement == "{}\n" else "true")
    assert observed["value"] == expected
    if source == "bridge":
        assert observed["marker"] == expected
