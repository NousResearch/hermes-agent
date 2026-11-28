"""A platform that left core, configured only under the ``gateway.<platform>:`` shorthand, starts with that block.

The live gateway merges config.yaml's platform sections before plugin discovery, when ``Platform(<platform>)``
does not resolve yet for a platform that left core (its plugin has not registered it). The merge counts the
shorthand for every LEFT_CORE platform by name, as ``config_loader.platform_blocks`` does, so a fresh process
whose plugin registers the platform during discovery loads the saved settings (#159651 4726083361).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

from hermes_cli.left_core_migration import LEFT_CORE

_PLATFORMS = sorted({f.platform for f in LEFT_CORE if f.platform})

_SCRIPT = r'''
import json, sys
from gateway.config import Platform, load_gateway_config
config = load_gateway_config()
platform = config.platforms.get(Platform(sys.argv[1]))
print(json.dumps(None if platform is None else {"enabled": platform.enabled, "extra": platform.extra}))
'''


@pytest.mark.parametrize("name", _PLATFORMS)
def test_gateway_shorthand_of_a_moved_platform_is_loaded_in_a_fresh_process(tmp_path, name):
    home = tmp_path / "home"
    plugin = home / "plugins" / name
    plugin.mkdir(parents=True)
    (plugin / "plugin.yaml").write_text(f"name: {name}\nversion: 0.1.0\ndescription: fixture\nkind: platform\n")
    (plugin / "__init__.py").write_text(
        "def register(ctx):\n"
        f"    ctx.register_platform(name={name!r}, label='Fixture', adapter_factory=lambda cfg: None,\n"
        "                          check_fn=lambda: True)\n")
    (home / "config.yaml").write_text(
        f"plugins:\n  enabled: [{name}]\ngateway:\n  {name}:\n    enabled: true\n    extra:\n      marker: saved\n")
    env = {k: v for k, v in os.environ.items() if not k.startswith(name.upper())}
    env.update({"HERMES_HOME": str(home), "PYTHONPATH": os.getcwd()})
    completed = subprocess.run([sys.executable, "-c", _SCRIPT, name], cwd=os.getcwd(), env=env, text=True,
                               capture_output=True, check=True)
    loaded = json.loads(completed.stdout.strip().splitlines()[-1])
    assert loaded is not None and loaded["enabled"] is True
    assert loaded["extra"].get("marker") == "saved"
