"""Reconciliation invariants for profile-owned child credentials."""
import os
from pathlib import Path
from contextvars import copy_context
from threading import Event, Thread
from unittest.mock import patch

import pytest

@pytest.fixture
def homes(tmp_path):
    source, target, managed = [tmp_path / n for n in ('source', 'target', 'managed')]
    for home in (source, target, managed):
        home.mkdir()
        (home / 'config.yaml').write_text('{}\n', encoding='utf-8')
    seed = {'PATH': '/usr/bin:/bin', 'HOME': str(tmp_path), 'USER': 'synthetic-audit',
            'HERMES_HOME': str(source), 'HERMES_RUNTIME_DIR': str(Path.cwd()),
            'LANG': 'C.UTF-8', 'LC_ALL': 'C.UTF-8', 'TZ': 'UTC',
            'PYTEST_CURRENT_TEST': 'reconciliation'}
    with patch.dict(os.environ, seed, clear=True), patch.object(Path, 'home', return_value=tmp_path):
        from agent import secret_scope
        from hermes_constants import pin_process_hermes_home
        from hermes_cli import env_loader, managed_scope
        assert Path(secret_scope.__file__).resolve().is_relative_to(Path.cwd())
        env_loader.reset_secret_source_cache()
        managed_scope.invalidate_managed_cache()
        try:
            yield source, target, managed
        finally:
            secret_scope.set_multiplex_active(False)
            pin_process_hermes_home(None)
            env_loader.reset_secret_source_cache()
            managed_scope.invalidate_managed_cache()


def test_scope_composes_administrator_managed_values_last(homes):
    _, target, managed = homes
    (target / '.env').write_text('SHARED_KEY=fake-user\nTARGET_ONLY=fake-target\n', encoding='utf-8')
    (managed / '.env').write_text('SHARED_KEY=fake-org\nMANAGED_ONLY=fake-policy\n', encoding='utf-8')
    os.environ['HERMES_MANAGED_DIR'] = str(managed)
    from hermes_cli.managed_scope import invalidate_managed_cache
    invalidate_managed_cache()
    from agent.secret_scope import build_profile_secret_scope
    values = build_profile_secret_scope(target)
    observed = {name: values.get(name) for name in ['SHARED_KEY', 'TARGET_ONLY', 'MANAGED_ONLY']}
    assert observed == {'SHARED_KEY': 'fake-org', 'TARGET_ONLY': 'fake-target', 'MANAGED_ONLY': 'fake-policy'}


def test_refresh_replaces_owner_and_revokes_values_without_mutating_old_scope(homes):
    source, target, managed = homes
    from agent import secret_scope as ss
    from hermes_constants import set_hermes_home_override, reset_hermes_home_override
    (target / '.env').write_text('REVOKED_LOGIN=fake-old\n', encoding='utf-8')
    old = ss.build_profile_secret_scope(target)
    sibling = ss.build_profile_secret_scope(source)
    ht = set_hermes_home_override(target)
    token = ss.set_secret_scope(old, profile_home=str(target))
    updated = Event()
    observed = []
    def sibling_reader():
        assert updated.wait(timeout=10)
        observed.append(ss.current_secret_scope())
    thread = Thread(target=copy_context().run, args=(sibling_reader,))
    thread.start()
    try:
        (target / '.env').write_text('NEW_LOGIN=fake-new\n', encoding='utf-8')
        assert ss.refresh_installed_secret_scope(target)
        replacement = ss.current_secret_scope()
        assert replacement is not old
        assert replacement.profile_home == target.resolve()
        assert replacement.generation != old.generation
        assert dict(replacement) == {'NEW_LOGIN': 'fake-new'}
        assert old['REVOKED_LOGIN'] == 'fake-old'
        assert 'REVOKED_LOGIN' in ss.get_profile_owned_secret_names(target)
        with pytest.raises(RuntimeError, match='home'):
            ss.refresh_installed_secret_scope(source)
        assert ss.build_profile_secret_scope(source).generation == sibling.generation
    finally:
        updated.set()
        thread.join(timeout=10)
        ss.reset_secret_scope(token)
        reset_hermes_home_override(ht)
    assert not thread.is_alive()
    assert len(observed) == 1 and observed[0] is old


@pytest.mark.parametrize('replacement, expected', [
    ('MANAGED_LOGIN=fake-rotated\n', 'fake-rotated'),
    ('MANAGED_LOGIN=fake-old\n# policy edit\n', 'fake-old'),
])
def test_managed_rotation_invalidates_installed_authority(homes, replacement, expected):
    source, target, managed = homes
    from agent import secret_scope as ss
    from hermes_cli.managed_scope import invalidate_managed_cache
    from hermes_constants import set_hermes_home_override, reset_hermes_home_override
    os.environ['HERMES_MANAGED_DIR'] = str(managed)
    (managed / '.env').write_text('MANAGED_LOGIN=fake-old\n', encoding='utf-8')
    invalidate_managed_cache()
    old = ss.build_profile_secret_scope(target)
    ht = set_hermes_home_override(target)
    token = ss.set_secret_scope(old, profile_home=str(target))
    try:
        (managed / '.env').write_text(replacement, encoding='utf-8')
        invalidate_managed_cache()
        with pytest.raises(RuntimeError, match='stale'):
            ss.build_profile_env_boundary(source, target)
        assert ss.refresh_installed_secret_scope(target)
        boundary = ss.build_profile_env_boundary(source, target)
        assert boundary.target_values['MANAGED_LOGIN'] == expected
    finally:
        ss.reset_secret_scope(token)
        reset_hermes_home_override(ht)


@pytest.mark.parametrize('loader', ['private', 'process'])
@pytest.mark.parametrize('config', ['{}\n', 'secrets:\n  bitwarden:\n    enabled: false\n'])
def test_unchanged_empty_hydration_does_not_revoke_concurrent_scope(homes, loader, config):
    """Independent requests may hydrate unchanged absence without revoking each other."""
    source, target, _ = homes
    from agent import secret_scope as ss
    from hermes_cli import env_loader
    (target / 'config.yaml').write_text(config, encoding='utf-8')
    load = env_loader.hydrate_profile_secret_sources if loader == 'private' else env_loader._apply_external_secret_sources
    load(target)
    old = ss.build_profile_secret_scope(target)
    sibling = ss.build_profile_secret_scope(source)
    token = ss.set_secret_scope(old, profile_home=str(target))
    try:
        copy_context().run(load, target)
        assert env_loader.get_external_secret_snapshot(target).generation == old.external_generation
        assert ss.build_profile_env_boundary(source, target).target_generation == old.generation

        env_loader.reset_secret_source_cache(target)
        with pytest.raises(RuntimeError, match='stale'):
            ss.build_profile_env_boundary(source, target)
        assert ss.refresh_installed_secret_scope(target)
        assert ss.current_secret_scope().generation != old.generation
        assert ss.build_profile_secret_scope(source).generation == sibling.generation

        before_change = ss.current_secret_scope()
        (target / 'config.yaml').write_text(config + '# policy changed\n', encoding='utf-8')
        with pytest.raises(RuntimeError, match='stale'):
            ss.build_profile_env_boundary(source, target)
        assert ss.refresh_installed_secret_scope(target)
        assert ss.current_secret_scope().generation != before_change.generation
    finally:
        ss.reset_secret_scope(token)
