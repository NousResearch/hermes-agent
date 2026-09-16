"""Bounded mention-only policy loader contract; synthetic scopes only."""
from pathlib import Path
import pytest
import yaml
from plugins.platforms.buzz import settings


def save(home, value):
    home.mkdir(parents=True, exist_ok=True)
    target = home / 'config.yaml'
    temporary = home / 'next.yaml'
    temporary.write_text(yaml.safe_dump(value))
    temporary.replace(target)


def policy(**values):
    return {'gateway': {'platforms': {'buzz': {'extra': values}}}}


def test_live_deletion_empty_and_invalid(tmp_path, monkeypatch):
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    save(tmp_path, policy(require_mention=False, thread_require_mention=False))
    assert settings.effective_runtime_policy()['require_mention'] is False
    save(tmp_path, policy(thread_require_mention=False))
    assert settings.effective_runtime_policy() == dict(require_mention=True, thread_require_mention=False)
    save(tmp_path, policy(require_mention=''))
    assert all(settings.effective_runtime_policy().values())
    (tmp_path / 'config.yaml').unlink()
    assert all(settings.effective_runtime_policy().values())


@pytest.mark.parametrize('bad', ['gateway: [', '[]', 'false', 'gateway:\n  platforms:\n    buzz: []'])
def test_malformed_cold_and_warm(tmp_path, monkeypatch, bad):
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    loader = settings.RuntimePolicyLoader()
    (tmp_path / 'config.yaml').write_text(bad)
    assert all(loader.load().values())
    save(tmp_path, policy(require_mention=False))
    assert loader.load()['require_mention'] is False
    (tmp_path / 'config.yaml').write_text(bad)
    assert loader.load()['require_mention'] is False  # exact-scope last good


def test_profile_and_same_name_different_home_cache(tmp_path, monkeypatch):
    loader = settings.RuntimePolicyLoader()
    a, b = tmp_path / 'a', tmp_path / 'b'
    save(a / 'profiles' / 'work', policy(require_mention=False))
    save(b / 'profiles' / 'work', policy())
    monkeypatch.setenv('HERMES_HOME', str(a))
    assert loader.load('work')['require_mention'] is False
    monkeypatch.setenv('HERMES_HOME', str(b))
    (b / 'profiles' / 'work' / 'config.yaml').write_text('gateway: [')
    assert all(loader.load('work').values())


def test_managed_malformed_and_deletion(tmp_path, monkeypatch):
    home, managed = tmp_path / 'home', tmp_path / 'managed'
    monkeypatch.setenv('HERMES_HOME', str(home))
    monkeypatch.setenv('HERMES_MANAGED_DIR', str(managed))
    save(home, {'buzz': {'require_mention': False}})
    save(managed, policy(require_mention=True))
    loader = settings.RuntimePolicyLoader()
    assert loader.load()['require_mention'] is True  # managed outranks local legacy aliases
    (managed / 'config.yaml').write_text('gateway: [')
    assert loader.load()['require_mention'] is True
    assert settings.RuntimePolicyLoader().load()['require_mention'] is True
    (managed / 'config.yaml').unlink()
    assert loader.load()['require_mention'] is False


def test_alias_precedence():
    value = policy(require_mention=True)
    value['platforms'] = {'buzz': {'require_mention': False}}
    value['gateway']['buzz'] = {'require_mention': True, 'extra': {'require_mention': False}}
    value['buzz'] = {'require_mention': True, 'extra': {'thread_require_mention': False}}
    assert settings.policy_from_config(value) == dict(require_mention=True, thread_require_mention=False)


def test_environment_absent_empty_managed_and_live_rotation(tmp_path, monkeypatch):
    home, managed = tmp_path / 'home', tmp_path / 'managed'
    save(home, policy(require_mention=False))
    managed.mkdir()
    monkeypatch.setenv('HERMES_HOME', str(home))
    monkeypatch.setenv('HERMES_MANAGED_DIR', str(managed))
    assert settings.effective_runtime_policy()['require_mention'] is False
    (home / '.env').write_text('BUZZ_REQUIRE_MENTION=true\n')
    assert settings.effective_runtime_policy()['require_mention'] is True
    (managed / '.env').write_text('BUZZ_REQUIRE_MENTION=false\n')
    assert settings.effective_runtime_policy()['require_mention'] is False
    (managed / '.env').write_text('BUZZ_REQUIRE_MENTION=\n')
    assert settings.effective_runtime_policy()['require_mention'] is True
    (managed / '.env').write_text('BUZZ_REQUIRE_MENTION="unterminated\n')
    assert settings.effective_runtime_policy()['require_mention'] is True
    (managed / '.env').unlink()
    (home / '.env').unlink()
    assert settings.effective_runtime_policy()['require_mention'] is False


def test_named_profile_no_process_borrow_and_expansion(tmp_path, monkeypatch):
    home = tmp_path / 'profiles' / 'work'
    save(home, policy(require_mention='${SYNTHETIC_MENTION}'))
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    monkeypatch.setenv('SYNTHETIC_MENTION', 'false')
    monkeypatch.setenv('BUZZ_REQUIRE_MENTION', 'false')
    assert settings.effective_runtime_policy('work')['require_mention'] is True
    (home / '.env').write_text('SYNTHETIC_MENTION=false\n')
    assert settings.effective_runtime_policy('work')['require_mention'] is False
    (home / '.env').write_text('SYNTHETIC_MENTION=true\n')
    assert settings.effective_runtime_policy('work')['require_mention'] is True


def test_multiplex_pinned_owner_never_borrows_routed_or_process_values(tmp_path, monkeypatch):
    from agent import secret_scope
    owner = tmp_path / 'owner'
    save(owner, policy(require_mention='${SYNTHETIC_MENTION}'))
    monkeypatch.setenv('BUZZ_REQUIRE_MENTION', 'false')
    monkeypatch.setattr(secret_scope, '_MULTIPLEX_ACTIVE', True)
    token = secret_scope.set_secret_scope({'BUZZ_REQUIRE_MENTION': 'false', 'SYNTHETIC_MENTION': 'false'})
    try:
        assert settings.effective_runtime_policy(home=owner)['require_mention'] is True
        (owner / '.env').write_text('SYNTHETIC_MENTION=false\n')
        assert settings.effective_runtime_policy(home=owner)['require_mention'] is False
        (owner / '.env').write_text('SYNTHETIC_MENTION=true\n')
        assert settings.effective_runtime_policy(home=owner)['require_mention'] is True
    finally:
        secret_scope.reset_secret_scope(token)
