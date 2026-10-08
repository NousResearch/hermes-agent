"""Synthetic CLI-backed recovery; no production vault or browser access."""
import json
from unittest.mock import Mock

import pytest

from agent.vault_backends import dashlane as module
from agent.vault_backends.base import UnlockRequired
from agent.vault_store import VaultError
from tests.agent.test_dashlane_vault_backend import enrolled, calls, ITEM_ID, PASSWORD
from tests.agent.test_dashlane_search import searchable, CANARIES

pytestmark = pytest.mark.platforms("posix")


def bare(searchable):
    backend, state, initial, home, config = searchable
    initial['records'][0]['url'] = 'service.test'
    state.write_text(json.dumps(initial))
    return backend, state, initial, home, config


def bind(searchable):
    backend, state, initial, home, config = bare(searchable)
    backend.cfg['items'] = [dict(id=ITEM_ID, label='Selected synthetic login',
        identifier='person@example.test', identifier_type='email',
        origin='https://sso.example.test', source_host='service.test')]
    (home / 'config.yaml').write_text(json.dumps(config))
    return backend, state, initial, home, config


def test_unfillable_projection_no_handle_or_secret(searchable, caplog, capsys):
    from tools import browser_vault_tool as tool
    backend, state, _, _, _ = bare(searchable)
    result = tool.browser_vault_list()
    parsed = json.loads(result)
    assert parsed['items'] == [] and 'hint' not in parsed
    candidate, = parsed['unfillable_candidates']
    assert candidate == dict(backend='dashlane', source_id=ITEM_ID, website_host='service.test',
        identifier='person@example.test', identifier_type='email', available=False,
        fillable=False, stage='search_projection', reason='invalid_origin',
        status='explicit_origin_binding_required')
    assert not module._DISCOVERED
    with pytest.raises(VaultError):
        backend.resolve_password('dl:' + candidate['source_id'])
    assert not any(c['argv'][0] == 'read' for c in calls(state))
    for secret in CANARIES:
        assert secret not in result + repr(module._DISCOVERED) + caplog.text + str(capsys.readouterr())


def test_bad_bare_candidate_does_not_hide_good_same_host(searchable):
    backend, state, initial, _, _ = bare(searchable)
    initial['records'].append(dict(initial['records'][0],
        id='{ABCDEF02-2345-4567-89AB-0123456789AB}', url='https://service.test'))
    state.write_text(json.dumps(initial))
    assert len(backend.list_items()) == 1
    assert len(backend.unfillable_candidates) == 1


def test_unrelated_schemeless_match_excluded(searchable):
    backend, state, initial, _, _ = bare(searchable)
    initial['records'].append({'url': 'evil-service.test', 'id': 'CANARY-invalid'})
    state.write_text(json.dumps(initial))
    assert backend.list_items() == []
    assert len(backend.unfillable_candidates) == 1


def test_multiple_unfillable_records_never_choose(searchable):
    from tools import browser_vault_tool as tool
    _, state, initial, _, _ = bare(searchable)
    initial['records'].append(dict(initial['records'][0],
        id='{ABCDEF02-2345-4567-89AB-0123456789AB}', email='second@example.test'))
    state.write_text(json.dumps(initial))
    result = json.loads(tool.browser_vault_list())
    assert len(result['unfillable_candidates']) == 2 and not result['items']
    assert 'multiple records require user selection' in result['binding_required']
    assert not module._DISCOVERED


def test_explicit_binding_uses_existing_fill_only(searchable, monkeypatch):
    from tools import browser_vault_tool as tool
    _, state, _, _, _ = bind(searchable)
    handle = 'dl:' + ITEM_ID
    monkeypatch.setattr(tool, '_focus_bound_origin', lambda *a: None)
    monkeypatch.setattr(tool, '_current_page_origin', lambda *a: 'https://sso.example.test')
    monkeypatch.setattr(tool, '_eval_js', lambda *a: {'success': True, 'result': [
        {'tag': 'input', 'type': 'password', 'id': 'pw', 'visible': True}]})
    fill = Mock(return_value={'success': True, 'result': {'filled': 1}})
    monkeypatch.setattr(tool, '_eval_js_secret', fill)
    result = tool.browser_vault_fill(handle, task_id='synthetic')
    assert json.loads(result)['success']
    assert PASSWORD not in result and json.dumps(PASSWORD) in fill.call_args.args[1]
    assert any(c['argv'] == ['read', 'dl://' + ITEM_ID] for c in calls(state))


@pytest.mark.parametrize('url', ['https://service.test', 'https://sso.example.test', 'SERVICE.test',
    'www.service.test', 'service.test/path', 'service.test.evil.test'])
def test_exact_source_drift_denied(searchable, url):
    backend, state, initial, _, _ = bind(searchable)
    initial['item']['url'] = url
    state.write_text(json.dumps(initial))
    with pytest.raises(VaultError):
        backend.resolve_password('dl:' + ITEM_ID)


@pytest.mark.parametrize('target', ['https://service.test', 'http://sso.example.test',
    'https://sso.example.test:444', 'https://sub.sso.example.test'])
def test_bound_wrong_destination_denies_before_read(searchable, monkeypatch, target):
    from tools import browser_vault_tool as tool
    _, state, _, _, _ = bind(searchable)
    monkeypatch.setattr(tool, '_focus_bound_origin', lambda *a: None)
    monkeypatch.setattr(tool, '_current_page_origin', lambda *a: target)
    assert json.loads(tool.browser_vault_fill('dl:' + ITEM_ID))['error_type'] == 'origin_mismatch'
    assert not any(c['argv'][0] == 'read' for c in calls(state))


@pytest.mark.parametrize('change', [{'email': 'other@example.test'}, {'id': '{ABCDEF02-2345-4567-89AB-0123456789AB}'}])
def test_bound_record_identity_drift_denied(searchable, change):
    backend, state, initial, _, _ = bind(searchable)
    initial['item'].update(change)
    state.write_text(json.dumps(initial))
    with pytest.raises(VaultError):
        backend.resolve_password('dl:' + ITEM_ID)


@pytest.mark.parametrize('value', ['https://service.test', 'SERVICE.test', '*.service.test', 'service.test/path', None])
def test_source_binding_invalid_configuration_no_spawn(searchable, value):
    backend, state, _, _, _ = bind(searchable)
    backend.cfg['items'][0]['source_host'] = value
    with pytest.raises(VaultError):
        backend.resolve_password('dl:' + ITEM_ID)
    assert not calls(state)


def test_projection_failure_isolated_across_hosts(searchable, monkeypatch):
    backend, _, _, _, _ = searchable
    backend.cfg['search_hosts'] = ['bad.example.test', 'service.test']
    real = backend._run
    def run(*args):
        if args[-1] == 'url=bad.example.test':
            return json.dumps([{'url': 'not a URL', 'password': PASSWORD}])
        return real(*args)
    monkeypatch.setattr(backend, '_run', run)
    assert len(backend.list_items()) == 1
    assert backend.listing_errors == [{'stage': 'search_projection', 'reason': 'invalid_origin'}]


def test_bound_account_drift_after_read_denied(searchable, monkeypatch):
    backend, _, _, _, _ = bind(searchable)
    real = backend._run
    read = False
    def run(*args):
        nonlocal read
        if args[0] == 'status' and read:
            return 'Logged in: Yes\nLogin: other@example.test\nLocked: No'
        result = real(*args)
        if args[0] == 'read':
            read = True
        return result
    monkeypatch.setattr(backend, '_run', run)
    with pytest.raises(UnlockRequired):
        backend.resolve_password('dl:' + ITEM_ID)
