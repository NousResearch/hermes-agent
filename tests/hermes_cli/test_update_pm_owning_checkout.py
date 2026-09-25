"""PM workspace updates return to their recorded checkout, never a guessed install."""
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_cli import update_owning_install as owning
from pm.environments import install_key, record_activation_inputs


@pytest.fixture
def pm_workspace(tmp_path, monkeypatch):
    checkout = tmp_path / 'checkout'
    (checkout / 'hermes_cli').mkdir(parents=True)
    (checkout / 'hermes_cli' / 'main.py').touch()
    (checkout / '.git').mkdir()
    state = tmp_path / 'installs' / install_key(checkout)
    generation = state / 'environments' / 'generation'
    workspace = generation / 'workspace'
    workspace.mkdir(parents=True)
    venv = generation / 'venv'
    venv.mkdir()
    record_activation_inputs(state / 'inputs', {}, checkout, test_environment=False)
    monkeypatch.setattr(sys, 'prefix', str(venv))
    monkeypatch.setattr(sys, 'base_prefix', str(tmp_path / 'python'))
    monkeypatch.delenv('PYTHONPATH', raising=False)
    return checkout, workspace, state


@pytest.mark.parametrize('check', [True, False])
def test_update_dispatch_retargets_recorded_pm_checkout(pm_workspace, monkeypatch, check):
    from hermes_cli import main

    checkout, workspace, _ = pm_workspace
    monkeypatch.setattr(main, 'PROJECT_ROOT', workspace)
    argv = ['hermes', 'update', *(['--check'] if check else [])]
    monkeypatch.setattr(sys, 'argv', argv)
    calls = []

    def run_child(command, *, cwd, env):
        calls.append((command, cwd, env))
        return 17

    monkeypatch.setattr(owning.subprocess, 'call', run_child)
    def wrong_preflight(args):
        pytest.fail('update reached workspace preflight instead of owning checkout')
    monkeypatch.setattr(main, '_update_preflight_handled', wrong_preflight)
    with pytest.raises(SystemExit) as result:
        main.cmd_update(SimpleNamespace(check=check))
    assert result.value.code == 17
    assert len(calls) == 1
    command, cwd, env = calls[0]
    assert cwd == checkout
    assert command == [sys.executable, '-m', 'hermes_cli.main', *argv[1:]]
    assert env['PYTHONPATH'] == str(checkout)
    # The re-entered owner must not recursively retarget itself.
    monkeypatch.setenv('PYTHONPATH', str(checkout))
    assert owning.owning_install_root(checkout) is None


@pytest.mark.parametrize('invalid', ['missing', 'empty', 'relative', 'wrong_owner', 'no_git', 'no_cli'])
def test_unverifiable_pm_owner_is_not_retargeted(pm_workspace, invalid):
    checkout, workspace, state = pm_workspace
    marker = state / 'inputs' / '.project-root'
    if invalid == 'missing':
        marker.unlink()
    elif invalid == 'empty':
        marker.write_text('', encoding='utf8')
    elif invalid == 'relative':
        marker.write_text('checkout', encoding='utf8')
    elif invalid == 'wrong_owner':
        other = checkout.parent / 'other'
        (other / 'hermes_cli').mkdir(parents=True)
        (other / 'hermes_cli/main.py').touch()
        (other / '.git').mkdir()
        marker.write_text(str(other), encoding='utf8')
    elif invalid == 'no_git':
        (checkout / '.git').rmdir()
    else:
        (checkout / 'hermes_cli/main.py').unlink()
    assert owning.owning_install_root(workspace) is None


def test_explicit_workspace_selection_is_preserved(pm_workspace, monkeypatch):
    _, workspace, _ = pm_workspace
    monkeypatch.setenv('PYTHONPATH', str(workspace))
    assert owning.owning_install_root(workspace) is None


def test_other_checkout_on_pm_interpreter_is_not_retargeted(pm_workspace):
    checkout, _, _ = pm_workspace
    assert owning.owning_install_root(checkout.parent / 'developer-tree') is None


def test_unreadable_owner_record_is_not_retargeted(pm_workspace, monkeypatch):
    _, workspace, state = pm_workspace
    original = Path.read_text

    def read(path, *args, **kwargs):
        if path == state / 'inputs' / '.project-root':
            raise PermissionError('fixture denies marker')
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, 'read_text', read)
    assert owning.owning_install_root(workspace) is None


@pytest.mark.live_system_guard_bypass  # Child imports only the throwaway fixture CLI below.
def test_retarget_runs_recorded_checkout_in_real_child(pm_workspace, monkeypatch):
    checkout, workspace, _ = pm_workspace
    (checkout / 'hermes_cli/__init__.py').touch()
    (checkout / 'hermes_cli/main.py').write_text(
        "import os, sys\n"
        "from pathlib import Path\n"
        "assert Path.cwd() == Path(__file__).resolve().parent.parent\n"
        "assert os.environ['PYTHONPATH'] == str(Path.cwd())\n"
        "assert sys.argv[1:] == ['update', '--check']\n"
        "raise SystemExit(23)\n", encoding='utf8')
    monkeypatch.setattr(sys, 'argv', ['hermes', 'update', '--check'])
    with pytest.raises(SystemExit) as result:
        owning.retarget_to_owning_install(workspace)
    assert result.value.code == 23


def test_in_tree_venv_routing_is_preserved(tmp_path, monkeypatch):
    checkout = tmp_path / 'checkout'
    (checkout / 'hermes_cli').mkdir(parents=True)
    (checkout / 'hermes_cli/main.py').touch()
    monkeypatch.setattr(sys, 'prefix', str(checkout / 'venv'))
    monkeypatch.setattr(sys, 'base_prefix', str(tmp_path / 'python'))
    monkeypatch.delenv('PYTHONPATH', raising=False)
    assert owning.owning_install_root(tmp_path / 'dev') == checkout
