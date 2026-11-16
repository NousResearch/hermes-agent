"""Real installer/admission -> fresh-process bare command -> uninstall.

The core dependency graph and user bin are deliberately isolated. The package
is an offline synthetic wheel, not a mock installer or a live user's profile.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

import pytest
from tests.pm.test_plugin_survival_contract import admission_env, _uv_available


_BACKEND = '''from pathlib import Path
import zipfile

def build_wheel(wheel_directory, config_settings=None, metadata_directory=None):
    filename = "fixture_warden-1.0-py3-none-any.whl"
    files = {
        "fixture_warden.py": "import json,os,sys\\ndef main():\\n print(json.dumps({'args':sys.argv[1:],'home':os.environ.get('HERMES_HOME')}))\\n return 0\\n",
        "fixture_warden-1.0.dist-info/METADATA": "Metadata-Version: 2.1\\nName: fixture-warden\\nVersion: 1.0\\n",
        "fixture_warden-1.0.dist-info/WHEEL": "Wheel-Version: 1.0\\nGenerator: fixture\\nRoot-Is-Purelib: true\\nTag: py3-none-any\\n",
        "fixture_warden-1.0.dist-info/entry_points.txt": "[console_scripts]\\nwarden = fixture_warden:main\\n",
    }
    files["fixture_warden-1.0.dist-info/RECORD"] = "".join(name + ",,\\n" for name in files) + "fixture_warden-1.0.dist-info/RECORD,,\\n"
    with zipfile.ZipFile(Path(wheel_directory) / filename, "w") as archive:
        for name, text in files.items(): archive.writestr(name, text)
    return filename

build_editable = build_wheel
'''


@pytest.mark.skipif(not _uv_available(), reason='requires the real uv dependency transaction')
def test_install_bare_command_profile_binding_disable_enable_and_remove(admission_env, monkeypatch, capsys):
    root, home = admission_env
    from hermes_cli import plugins_cmd
    from hermes_cli import plugin_shell_commands as shell
    source = root / 'source'
    source.mkdir()
    (source / 'plugin.yaml').write_text('name: fixture-warden\nshell_commands: [warden]\n', encoding='utf-8')
    (source / '__init__.py').write_text('def register(ctx):\n    pass\n', encoding='utf-8')
    (source / 'after-install.md').write_text('Run `warden setup`.\n', encoding='utf-8')
    (source / 'backend.py').write_text(_BACKEND, encoding='utf-8')
    (source / 'pyproject.toml').write_text('''[build-system]
requires = []
build-backend = "backend"
backend-path = ["."]
[project]
name = "fixture-warden"
version = "1.0"
[project.scripts]
warden = "fixture_warden:main"
''', encoding='utf-8')
    for args in [('init', '-q'), ('add', '.'), ('-c', 'user.name=fixture', '-c', 'user.email=test@example.invalid', 'commit', '-qm', 'fixture')]:
        subprocess.run(['git', *args], cwd=source, check=True, capture_output=True, timeout=30)
    out = root / 'bin'
    out.mkdir(mode=0o700)
    core_source = Path(__file__).resolve().parents[2]
    # Reuse existing core test dependencies read-only; the plugin itself MUST
    # resolve from the generation created by the actual install below.
    dependencies = [p for p in sys.path if p and Path(p).name in {'site-packages', 'dist-packages'}]
    driver = root / 'driver.py'
    driver.write_text(
        'import argparse,sys,site\nfrom pathlib import Path\n'
        f'sys.path[:0] = {[str(core_source), *dependencies]!r}\n'
        'from pm.environments import selected_venv,site_packages\n'
        f'site.addsitedir(str(site_packages(selected_venv(Path({str(root / "core")!r})))))\n'
        'from hermes_cli.subcommands.plugins import build_plugins_parser\n'
        'from hermes_cli.plugins_cmd import plugins_command\n'
        'parser=argparse.ArgumentParser(prog="hermes")\n'
        'build_plugins_parser(parser.add_subparsers(dest="command"),cmd_plugins=plugins_command)\n'
        'raise SystemExit(plugins_command(parser.parse_args(sys.argv[1:])))\n', encoding='utf-8')
    if os.name == 'nt':
        hermes = shell._windows_launcher(out, 'hermes', f'import runpy\nrunpy.run_path({str(driver)!r},run_name="__main__")\n', Path(sys.executable))
    else:
        hermes = out / 'hermes'
        hermes.write_text('#!/bin/sh\nexec ' + shlex.join([sys.executable, '-I', str(driver)]) + ' "$@"\n', encoding='utf-8')
        hermes.chmod(0o755)
    monkeypatch.setattr(shell, 'command_location', lambda: (out, hermes))
    monkeypatch.setattr(shell, 'default_command_directory', lambda: out)
    monkeypatch.setenv('PATH', str(out) + os.pathsep + os.environ.get('PATH', ''))
    plugins_cmd.cmd_install(source.as_uri(), allow_removed=True, enable=True, yes_deps=True)
    assert 'Terminal command ready: warden' in capsys.readouterr().out
    target = out / ('warden.exe' if os.name == 'nt' else 'warden')
    assert target.is_file()
    env = dict(os.environ, HOME=str(root), USERPROFILE=str(root), LOCALAPPDATA=str(root / 'local'), HERMES_HOME='/wrong-profile')
    env.pop('PYTHONPATH', None)
    result = subprocess.run(['warden', 'setup', 'a&b', 'space value'], env=env, cwd=root, text=True, capture_output=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    value = json.loads(result.stdout)
    assert value == {'args': ['setup', 'a&b', 'space value'], 'home': str(home)}
    plugins_cmd.cmd_disable('fixture-warden')
    assert not target.exists()
    plugins_cmd.cmd_enable('fixture-warden')
    assert target.exists()
    plugins_cmd.cmd_remove('fixture-warden')
    assert not target.exists()
    assert hermes.is_file()
