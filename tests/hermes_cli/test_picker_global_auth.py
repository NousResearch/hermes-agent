"""Picker admission must agree with runtime auth across named profiles."""

import json

from agent.secret_scope import set_multiplex_active
from hermes_constants import set_hermes_home_override, reset_hermes_home_override
from hermes_cli.auth import resolve_codex_runtime_credentials
from hermes_cli.model_switch_providers import _auth_store_has_provider


def test_picker_sees_runtime_global_login_across_profiles(tmp_path, monkeypatch):
    root = tmp_path / 'hermes'
    homes = [root / 'profiles' / name for name in ('a', 'b')]
    for home in homes:
        home.mkdir(parents=True)
        (home / 'config.yaml').write_text('model: {}\n')
    (root / 'auth.json').write_text(json.dumps({'providers': {
        'openai-codex': {'tokens': {
            'access_token': 'synthetic-access', 'refresh_token': 'synthetic-refresh',
        }},
    }}))
    (homes[0] / 'auth.json').write_text(json.dumps({'providers': {'profile-only': {}}}))
    monkeypatch.setenv('HERMES_HOME', str(root))
    before = (root / 'auth.json').read_bytes()
    set_multiplex_active(True)
    try:
        for home in (homes[0], homes[1], homes[0]):
            token = set_hermes_home_override(home)
            try:
                assert resolve_codex_runtime_credentials(read_only=True)['api_key']
                assert _auth_store_has_provider('openai-codex')
                assert _auth_store_has_provider('profile-only') == (home == homes[0])
                assert not _auth_store_has_provider('absent-provider')
            finally:
                reset_hermes_home_override(token)
    finally:
        set_multiplex_active(False)
    assert (root / 'auth.json').read_bytes() == before
    assert not (homes[1] / 'auth.json').exists()


def test_absent_or_malformed_global_login_does_not_admit_provider(tmp_path, monkeypatch):
    root = tmp_path / 'hermes'
    home = root / 'profiles' / 'a'
    home.mkdir(parents=True)
    (home / 'config.yaml').write_text('model: {}\n')
    monkeypatch.setenv('HERMES_HOME', str(home))
    assert not _auth_store_has_provider('openai-codex')
    (root / 'auth.json').write_text('{malformed')
    assert not _auth_store_has_provider('openai-codex')
