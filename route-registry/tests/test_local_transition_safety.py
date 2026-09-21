"""Regression gates at the real transition/CLI boundaries; private tmp files only."""
import copy
import importlib.util
import json
from pathlib import Path
import sys

import pytest
import yaml

HERE = Path(__file__).parents[1]
sys.path.insert(0, str(HERE / 'route_registry'))
import local_transition as lt


@pytest.fixture
def setup(tmp_path):
    root = tmp_path / 'home'
    root.mkdir()
    (root / 'profiles/a').mkdir(parents=True)
    raw = b'fallback_providers:\n- provider: custom:turbohaul-local\n  model: qwen3.8-27b\n'
    for path in (root / 'config.yaml', root / 'profiles/a/config.yaml'):
        path.write_bytes(raw)
    spec = yaml.safe_load((HERE / 'registry/route-slots.yaml').read_text())['local_transition']
    return root, spec, raw


def test_cli_plan_is_read_only_and_does_not_fall_through(setup, tmp_path):
    root, spec, raw = setup
    module_spec = importlib.util.spec_from_file_location('local_cli', HERE / 'local_transition_cli.py')
    cli = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(cli)
    planfile = tmp_path / 'plan.json'
    assert cli.main(['plan', '--target-root', str(root), '--plan', str(planfile)]) == 0
    assert (root / 'config.yaml').read_bytes() == raw
    assert len(json.loads(planfile.read_text())['entries']) == 2
    wrong_root = tmp_path / 'different-home'
    assert cli.main(['apply', '--target-root', str(wrong_root), '--plan', str(planfile),
                     '--backup-root', str(tmp_path / 'backup')]) == 2
    assert (root / 'config.yaml').read_bytes() == raw


def test_late_after_hash_error_changes_nothing(setup, tmp_path):
    root, spec, raw = setup
    plan = lt.build_local_plan(root, spec)
    plan['entries'][-1]['after_sha256'] = '0' * 64
    with pytest.raises(lt.LocalTransitionError):
        lt.apply_local_plan(plan, spec, tmp_path / 'backup')
    assert all(p.read_bytes() == raw for p in lt.config_paths(root))


def test_late_corrupt_backup_restores_nothing(setup, tmp_path):
    root, spec, raw = setup
    plan = lt.build_local_plan(root, spec)
    backup = tmp_path / 'backup'
    lt.apply_local_plan(plan, spec, backup)
    after = {p: p.read_bytes() for p in lt.config_paths(root)}
    (backup / 'profiles/a/config.yaml').write_bytes(b'corrupt')
    with pytest.raises(lt.LocalTransitionError):
        lt.rollback_local_plan(plan, backup)
    assert all(p.read_bytes() == data for p, data in after.items())


def test_partial_apply_is_recoverable(setup, tmp_path):
    root, spec, raw = setup
    plan = lt.build_local_plan(root, spec)
    backup = tmp_path / 'backup'
    lt.apply_local_plan(plan, spec, backup)
    # A previous interrupted recovery has restored only the last file.
    (root / 'profiles/a/config.yaml').write_bytes(raw)
    lt.rollback_local_plan(plan, backup)
    assert all(p.read_bytes() == raw for p in lt.config_paths(root))


def test_symlink_escape_refused(setup, tmp_path):
    root, spec, raw = setup
    other = tmp_path / 'outside.yaml'
    other.write_bytes(raw)
    (root / 'profiles/a/config.yaml').unlink()
    (root / 'profiles/a/config.yaml').symlink_to(other)
    with pytest.raises(lt.LocalTransitionError):
        lt.build_local_plan(root, spec)
    assert other.read_bytes() == raw


def test_duplicate_entries_refused(setup, tmp_path):
    root, spec, raw = setup
    plan = lt.build_local_plan(root, spec)
    plan['entries'].append(copy.deepcopy(plan['entries'][0]))
    with pytest.raises(lt.LocalTransitionError):
        lt.apply_local_plan(plan, spec, tmp_path / 'backup')
    assert all(p.read_bytes() == raw for p in lt.config_paths(root))
