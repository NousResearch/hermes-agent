"""Enabling a plugin at the install prompt loads it where it can go live, like ``--enable`` does."""
import os
import subprocess

import hermes_yaml as yaml
import pytest

from hermes_cli import plugins_activation, plugins_cmd
from tests.pm.test_plugin_survival_contract import _uv_available, admission_env


@pytest.mark.skipif(not _uv_available(), reason="uv not on PATH")
def test_yes_at_the_enable_prompt_activates_the_plugin_in_running_processes(admission_env, monkeypatch):  # health: allow F811 -- pytest injects the imported admission_env fixture by parameter name
    """A "y" at "Enable now?" enables the plugin AND nudges the running gateway / Desktop backend
    (report_activation), exactly like ``--enable``. Without the nudge a gateway started before the
    install never loads the plugin, so its MCP servers are missing from every session it serves."""
    root, home = admission_env
    source = root / 'plugin-source'
    source.mkdir()
    (source / 'plugin.yaml').write_text(yaml.safe_dump({'name': 'prompt-enabled-plugin'}), encoding='utf-8')
    (source / '__init__.py').write_text('def register(ctx):\n    pass\n', encoding='utf-8')
    env = {k: v for k, v in os.environ.items() if k not in {'GIT_DIR', 'GIT_WORK_TREE', 'GIT_INDEX_FILE'}}
    for args in [('init', '-q'), ('add', '.'), ('-c', 'user.name=test', '-c', 'user.email=test@example.invalid', 'commit', '-qm', 'fixture')]:
        subprocess.run(['git', *args], cwd=source, env=env, check=True, capture_output=True, timeout=30)
    monkeypatch.setattr(plugins_cmd, '_is_tty', lambda: True)
    monkeypatch.setattr(plugins_cmd, '_ask_yes', lambda *a, **k: True)
    activated = []
    monkeypatch.setattr(plugins_activation, 'report_activation', lambda console, name: activated.append(name))

    plugins_cmd.cmd_install(source.as_uri(), allow_removed=True)

    enabled = yaml.safe_load((home / 'config.yaml').read_text(encoding='utf-8'))['plugins']['enabled']
    assert 'prompt-enabled-plugin' in enabled
    assert activated == ['prompt-enabled-plugin']
