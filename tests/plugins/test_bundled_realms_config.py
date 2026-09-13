"""Real canonical settings in explicit profiles."""
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
PLUGIN = ROOT / "plugins/hermes-realms"


@pytest.mark.linux_only
def test_explicit_profile_settings_are_canonical_and_read_only(tmp_path):
    home = tmp_path / "requested"
    home.mkdir()
    config = home / "config.yaml"
    config.write_text(
        'plugins:\n  realms:\n    size: "${REALMS_TEST_SIZE}"\n'
        '    renderer: pixman\n    overlay: false\n    idle_ttl: 42\n',
        encoding="utf-8",
    )
    original = config.read_bytes()
    code = r'''
import os, runpy, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from hermes_cli.config import load_config_readonly
root, home = map(Path, sys.argv[1:])
load = runpy.run_path(str(root / 'realms/_binding.py'))['load_runtime']
Config = load('config').Config
before_env, before_path = dict(os.environ), list(sys.path)
ambient = Path(os.environ['HERMES_HOME'])
before_ambient = sorted(str(p.relative_to(ambient)) for p in ambient.rglob('*'))
missing = home.parent / 'missing'
with ThreadPoolExecutor(max_workers=2) as executor:
    configured, defaults = list(executor.map(Config.load, [home, missing]))
assert configured.size == '384x256', configured
assert configured.renderer == 'pixman' and configured.overlay is False
assert configured.idle_ttl == 42
assert defaults == Config()
assert not missing.exists(), 'reading defaults must not create a profile'
assert sorted(p.name for p in home.iterdir()) == ['config.yaml']
assert dict(os.environ) == before_env
assert sys.path == before_path, 'configuration must not change import routing'
assert sorted(str(p.relative_to(ambient)) for p in ambient.rglob('*')) == before_ambient
print('explicit profiles, canonical expansion, read-only defaults and unchanged routing')
'''
    # A project cannot provide the canonical owner or the generic plugin package.
    (tmp_path / "hermes_cli.py").write_text("raise AssertionError('shadow core')", encoding="utf-8")
    (tmp_path / "realms.py").write_text("raise AssertionError('shadow plugin')", encoding="utf-8")
    env = {key: os.environ[key] for key in ("PATH", "LANG", "TZ") if key in os.environ}
    env.update(HOME=str(tmp_path), HERMES_HOME=str(tmp_path / "ambient"),
               PYTHONPATH=str(ROOT),
               PYTHONDONTWRITEBYTECODE="1", REALMS_TEST_SIZE="384x256")
    result = subprocess.run([sys.executable, "-P", "-c", code, str(PLUGIN), str(home)],
                            cwd=tmp_path, env=env, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    assert config.read_bytes() == original
