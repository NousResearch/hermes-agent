"""The image's non-root runtime uses its staged tools and generated assets."""
from __future__ import annotations

import subprocess


def test_python_uses_pm_interpreter_as_runtime_user(built_image: str) -> None:
    probe = """
from pathlib import Path
import sys
from pm.lock import Facts
from pm.registry import get_package
from pm.store import current_target

store = Path('/opt/hermes/tools')
assert not list(store.glob('fetch-*')), 'completed download archives must not ship'
fact = Facts(store / 'facts.json').get('python')
expected = get_package('python').binary(store / fact['entry'], current_target())
assert Path(sys._base_executable).resolve() == expected.resolve()
import hermes_yaml as yaml
print('PM interpreter and application dependencies load as hermes')
"""
    result = subprocess.run(
        ["docker", "run", "--rm", "--network", "none", "--user", "hermes",
         "--entrypoint", "/opt/hermes/.venv/bin/python", built_image, "-c", probe],
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_runtime_source_shipped_and_cli_starts(built_image: str) -> None:
    """Check the real .dockerignore-filtered image, not just setuptools discovery."""
    probe = """
from pathlib import Path
import sys

root = Path('/opt/hermes/runtime')
required = {
    '__init__.py', 'desktop_identity.py', 'git_subprocess.py', 'processes.py',
    'process_identity.py', 'resource_limits.py', 'sqlite_runtime.py',
    'stdio.py', 'subprocess_compat.py',
}
missing = sorted(name for name in required if not (root / name).is_file())
assert not missing, f'Docker build omitted runtime source: {missing}'
from hermes_cli.main import main
sys.argv = ['hermes', '--version']
raise SystemExit(main())
"""
    result = subprocess.run(
        ["docker", "run", "--rm", "--network", "none", "--user", "hermes",
         "--entrypoint", "/opt/hermes/.venv/bin/python", built_image, "-c", probe],
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Hermes" in result.stdout


def test_dashboard_ships_generated_icon_without_build_environment(built_image: str) -> None:
    probe = """
from pathlib import Path
from PIL import Image

with Image.open('/opt/hermes/hermes_cli/web_dist/favicon.ico') as image:
    image.load()
    assert image.width > 0 and image.height > 0
assert not Path('/opt/hermes/node_modules/vite').exists()
assert not Path('/opt/hermes/node_modules/esbuild').exists()
assert Path('/opt/hermes/node_modules/typescript/bin/tsc').is_file()
"""
    result = subprocess.run(
        ["docker", "run", "--rm", "--network", "none", "--user", "hermes",
         "--entrypoint", "/opt/hermes/.venv/bin/python", built_image, "-c", probe],
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
