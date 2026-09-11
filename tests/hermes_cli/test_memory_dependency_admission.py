"""Memory setup admits a complete candidate before saving its selection."""
import importlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest
import hermes_yaml as yaml

from hermes_cli import memory_setup
from hermes_cli.runtime_paths import selected_venv
from pm import paths
from tests.pm.test_workspace_build_inputs import _wheel


@pytest.mark.parametrize('picker', [False, True])
def test_setup_requires_dependencies_and_keeps_the_existing_union(tmp_path, monkeypatch, picker):
    uv = shutil.which('uv')
    assert uv, 'uv is required for the real admission contract'
    core, home, wheels = (tmp_path / name for name in ('core', 'home', 'wheels'))
    for directory in (core, home, wheels):
        directory.mkdir()
    for name in ('existing_dep', 'provider_dep', 'modern_dep', 'other_dep'):
        _wheel(wheels, name, '1.0')
    (core / 'pyproject.toml').write_text(
        '[project]\nname="core"\nversion="1"\nrequires-python=">=3.14"\n'
        '[tool.uv]\npackage=false\nno-index=true\n'
        f'find-links=[{json.dumps(wheels.as_posix())}]\n', encoding='utf-8')
    env = {**os.environ, 'UV_OFFLINE': '1', 'UV_PYTHON_DOWNLOADS': 'never',
           'UV_CACHE_DIR': str(tmp_path / 'cache')}
    subprocess.run([uv, 'lock', '--python', sys.executable], cwd=core, env=env,
                   capture_output=True, check=True, timeout=60)
    plugins = home / 'plugins'
    incumbent, candidate = plugins / 'incumbent', plugins / 'candidate'
    for plugin in (incumbent, candidate):
        plugin.mkdir(parents=True)
        (plugin / '__init__.py').write_text('# MemoryProvider fixture\n', encoding='utf-8')
    (incumbent / 'plugin.yaml').write_text(
        'name: incumbent\npython_dependencies: ["existing_dep==1.0"]\n', encoding='utf-8')
    candidate_manifest = candidate / 'plugin.yaml'
    candidate_manifest.write_text(
        'name: candidate\npip_dependencies: ["provider_dep==1.0", "provider_dep==2.0"]\n',
        encoding='utf-8')
    config = home / 'config.yaml'
    config.write_text(yaml.safe_dump({'plugins': {'enabled': ['incumbent']},
                                     'memory': {'provider': 'incumbent', 'setting': 'keep'}}), encoding='utf-8')
    other_home = home / 'profiles' / 'other'
    other_plugin = other_home / 'plugins' / 'other-plugin'
    other_plugin.mkdir(parents=True)
    (other_plugin / 'plugin.yaml').write_text(
        'name: other-plugin\npython_dependencies: ["other_dep==1.0"]\n', encoding='utf-8')
    other_config = other_home / 'config.yaml'
    other_config.write_text('plugins:\n  enabled: [other-plugin]\n', encoding='utf-8')
    monkeypatch.setenv('HERMES_HOME', str(home))
    monkeypatch.setenv('HERMES_RUNTIME_DIR', str(tmp_path / 'tools'))
    monkeypatch.setattr(paths, 'repo_root', lambda: core)
    ensure = importlib.import_module('pm.ensure')
    monkeypatch.setattr(ensure, 'lazy_installs_allowed', lambda: True)
    monkeypatch.setattr("pm._uv._toolchain", lambda **kwargs: (Path(uv), Path(sys.executable)))
    # Keep the public client and real engine; only tool acquisition is injected.
    monkeypatch.setattr("pm.client.is_runtime", lambda: True)
    ensure.sync_venv(explicit=True)
    original_environment = selected_venv(core)
    before = {file: file.read_bytes() for file in (config, other_config, paths.runtime_facts_path())}
    post_calls = []

    class Provider:
        def post_setup(self, actual_home, proposal):
            assert Path(actual_home) == home
            post_calls.append(proposal)

    monkeypatch.setattr(memory_setup, '_get_available_providers', lambda: [('candidate', 'local', Provider())])
    monkeypatch.setattr(memory_setup, '_curses_select', lambda *args, **kwargs: 0)

    def setup():
        from hermes_cli.main_agent_cmds import cmd_memory

        cmd_memory(SimpleNamespace(memory_command='setup', provider=None if picker else 'candidate'))

    with pytest.raises(SystemExit) as refused:
        setup()
    assert refused.value.code == 1
    assert not post_calls
    assert {file: file.read_bytes() for file in before} == before
    assert selected_venv(core) == original_environment

    for malformed in ('name: [broken', 'name: candidate\npython_dependencies: 9\n'):
        candidate_manifest.write_text(malformed, encoding='utf-8')
        with pytest.raises(SystemExit) as refused:
            setup()
        assert refused.value.code == 1
        assert {file: file.read_bytes() for file in before} == before
        assert not post_calls

    candidate_manifest.write_text('name: candidate\npip_dependencies: ["provider_dep==1.0"]\n', encoding='utf-8')
    # Generic setup saves the selection only after the real sync completes.
    monkeypatch.setattr(memory_setup, '_get_available_providers', lambda: [('candidate', 'local', SimpleNamespace())])
    setup()
    assert yaml.safe_load(config.read_text(encoding='utf-8'))['memory']['provider'] == 'candidate'
    selected = selected_venv(core)
    python = selected / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
    result = subprocess.run([str(python), '-I', '-c', 'import existing_dep, provider_dep, other_dep; print("both")'],
                            env=env, cwd=tmp_path, capture_output=True, text=True, check=True, timeout=30)
    assert result.stdout.strip() == 'both'
    assert not (candidate / 'pyproject.toml').exists()
    assert other_config.read_bytes() == before[other_config]

    modern = plugins / 'modern'
    modern.mkdir()
    (modern / '__init__.py').write_text('# MemoryProvider fixture\n', encoding='utf-8')
    (modern / 'plugin.yaml').write_text('name: modern\n', encoding='utf-8')
    (modern / 'pyproject.toml').write_text(
        '[project]\nname="modern-provider"\nversion="1"\nrequires-python=">=3.14"\n'
        'dependencies=["modern_dep==1.0"]\n[tool.uv]\npackage=false\n', encoding='utf-8')
    monkeypatch.setattr(memory_setup, '_get_available_providers', lambda: [('modern', 'local', SimpleNamespace())])
    memory_setup.cmd_setup_provider('modern')
    selected = selected_venv(core)
    python = selected / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
    result = subprocess.run([str(python), '-I', '-c', 'import existing_dep, provider_dep, modern_dep, other_dep; print("all")'],
                            env=env, cwd=tmp_path, capture_output=True, text=True, check=True, timeout=30)
    assert result.stdout.strip() == 'all'
    assert other_config.read_bytes() == before[other_config]
