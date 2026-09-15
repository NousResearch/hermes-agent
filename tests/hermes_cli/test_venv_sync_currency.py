"""Checkout freshness follows the selected PM generation, not a second stamp."""
import importlib
import json
import os
from pathlib import Path
import subprocess

import hermes_yaml as yaml

from hermes_cli import venv_sync
from hermes_cli.runtime_paths import install_state_dir, selected_venv
from pm import paths
from pm.lock import Lockfile
from tests.pm.test_plugin_survival_contract import admission_env  # noqa: F401


def test_check_uses_real_pm_selection_and_keeps_invalid_evidence(admission_env, monkeypatch, capsys):
    monkeypatch.setattr("pm.client.is_runtime", lambda: True)
    root, home = admission_env
    core = root / 'core'
    ensure = importlib.import_module('pm.ensure')
    pin_path = root / 'pins.json'
    pins = Lockfile(pin_path)
    pins.set_pin('python', '1.0', {'any': {'url': 'https://example.invalid/python', 'sha256': 'a' * 64}})
    pins.save()
    monkeypatch.setattr(paths, 'lockfile_path', lambda: pin_path)
    # A historical foreign-root stamp must not certify a PM environment.
    old_stamp = core / ".hermes-runtime" / "cache" / "venv-sync.json"
    old_stamp.parent.mkdir(parents=True)
    old_stamp.write_text('{"lockDigest": "old-bootstrap-stamp"}')
    cached = old_stamp.read_bytes()

    def check(expected, code=0):
        capsys.readouterr()
        result = venv_sync.main(['--project-root', str(core), '--check', '--json'])
        output = json.loads(capsys.readouterr().out)
        assert result == code, output
        assert output['state'] == expected, output
        assert old_stamp.read_bytes() == cached
        return output

    check('would-sync')
    ensure.sync_venv(explicit=True)
    facts_path = paths.runtime_facts_path()
    pristine = facts_path.read_bytes()
    selected = selected_venv(core)
    assert selected.is_relative_to(install_state_dir(core) / 'environments')
    check('current')
    assert not any(problem.startswith('venv:') for problem in ensure.check())

    config = home / 'config.yaml'
    old_config = config.read_bytes()
    member = home / 'plugins' / 'extra-member'
    member.mkdir(parents=True)
    (member / 'plugin.yaml').write_text('name: extra-member\npython_dependencies: ["example-dep==1"]\n', encoding='utf-8')
    config.write_text(yaml.safe_dump({'plugins': {'enabled': ['extra-member']}}), encoding='utf-8')
    check('would-sync')
    assert any(problem.startswith('venv:') for problem in ensure.check())
    config.write_bytes(old_config)
    check('current')

    altered = json.loads(pristine)
    altered['packages']['venv']['extras'] = ['changed-extra']
    facts_path.write_text(json.dumps(altered), encoding='utf-8')
    check('would-sync')
    assert any(problem.startswith('venv:') for problem in ensure.check())
    facts_path.write_bytes(pristine)

    pins.set_pin('python', '1.0', {'any': {'url': 'https://example.invalid/python', 'sha256': 'b' * 64}})
    pins.save()
    check('would-sync')
    assert any(problem.startswith('venv:') for problem in ensure.check())
    pins.set_pin('python', '1.0', {'any': {'url': 'https://example.invalid/python', 'sha256': 'a' * 64}})
    pins.save()

    marker = selected / 'pyvenv.cfg'
    marker_bytes = marker.read_bytes()
    marker.unlink()
    check('would-sync')
    assert any(problem.startswith('venv:') for problem in ensure.check())
    marker.write_bytes(marker_bytes)
    check('current')

    outside = root / 'foreign-environment'
    outside.mkdir()
    (outside / 'pyvenv.cfg').write_bytes(marker_bytes)
    altered = json.loads(pristine)
    altered['packages']['venv']['environment'] = str(outside)
    facts_path.write_text(json.dumps(altered), encoding='utf-8')
    check('would-sync')
    assert any(problem.startswith('venv:') for problem in ensure.check())

    for invalid in (b'not JSON', b'{"schema":1,"packages":{"venv":[]}}'):
        facts_path.write_bytes(invalid)
        output = check('failed', 1)
        assert output.get('detail')
        assert any(problem.startswith('venv:') for problem in ensure.check())
        assert facts_path.read_bytes() == invalid
        assert not facts_path.with_suffix('.corrupt').exists()
    facts_path.unlink()
    check('would-sync')
    assert not facts_path.exists()


def test_own_tree_sync_reuses_pm_without_writing_an_extra_stamp(admission_env, monkeypatch):
    # Keep the public client seam; run its in-process path against admission's real engine.
    monkeypatch.setattr("pm.client.is_runtime", lambda: True)
    root, home = admission_env
    core = root / 'core'
    assert not (core / ".hermes-runtime" / "cache" / "venv-sync.json").exists()
    assert venv_sync.sync(core) == {'state': 'synced', 'ok': True}
    environment = selected_venv(core)
    saved = paths.runtime_facts_path().read_bytes()
    assert not (core / ".hermes-runtime" / "cache" / "venv-sync.json").exists()
    assert venv_sync.sync(core) == {'state': 'current', 'ok': True}
    assert paths.runtime_facts_path().read_bytes() == saved
    python = environment / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
    result = subprocess.run([str(python), '-I', '-c', 'import sys; print(sys.prefix)'],
                            cwd=home, capture_output=True, text=True, check=True, timeout=30)
    assert Path(result.stdout.strip()).resolve() == environment.resolve()
    (environment / 'pyvenv.cfg').unlink()
    assert venv_sync.sync(core) == {'state': 'synced', 'ok': True}
    replacement = selected_venv(core)
    assert replacement != environment
    assert (replacement / 'pyvenv.cfg').is_file()
    assert venv_sync.sync(core) == {'state': 'current', 'ok': True}
    assert not (core / ".hermes-runtime" / "cache" / "venv-sync.json").exists()
