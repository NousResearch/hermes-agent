"""Real synthetic subprocess search -> registry list -> existing fill, without a vault."""
import json

from unittest.mock import Mock

import pytest

from agent.vault_backends import dashlane as module
from agent.vault_backends.dashlane import DashlaneLoginBackend
from agent.vault_backends.base import UnlockRequired
from agent.vault_store import VaultError
from tests.agent.test_dashlane_vault_backend import enrolled, calls, ITEM_ID, PASSWORD

pytestmark = pytest.mark.platforms("posix")
CANARIES = [PASSWORD, 'CANARY-OTP-SECRET', 'CANARY-OTP-URL', 'CANARY-NOTE', 'CANARY-CUSTOM']


@pytest.fixture
def searchable(enrolled):
    backend, state, initial, home, config = enrolled
    module._DISCOVERED.clear()
    backend.cfg.update(items=[], search_hosts=['service.test'])
    record = {'id': '{' + ITEM_ID + '}', 'url': 'https://service.test/login',
              'title': 'Synthetic service', 'email': 'person@example.test', 'password': PASSWORD,
              'otpSecret': CANARIES[1], 'otpUrl': CANARIES[2],
              'note': CANARIES[3], 'custom': {'value': CANARIES[4]}}
    initial.update(records=[record], item=record)
    state.write_text(json.dumps(initial))
    (home / 'config.yaml').write_text(json.dumps(config))
    yield backend, state, initial, home, config
    module._DISCOVERED.clear()


def test_no_uuid_onboarding_through_real_tool_and_fill(searchable, monkeypatch, caplog, capsys):
    from tools import browser_vault_tool as tool
    backend, state, _, _, _ = searchable
    result = tool.browser_vault_list()
    item, = json.loads(result)['items']
    assert item['origin'] == 'https://service.test'
    assert item['handle'].startswith('dl:search-') and ITEM_ID not in result
    assert 'two_factor' not in item
    assert backend.resolve_otp(item['handle']) is None
    cached = repr(module._DISCOVERED)
    for secret in CANARIES:
        assert secret not in result + cached + caplog.text + str(capsys.readouterr())
    assert [c['argv'] for c in calls(state) if c['argv'][0] != 'status'] == [
        ['password', '--output', 'json', 'url=service.test']]
    monkeypatch.setattr(tool, '_focus_bound_origin', lambda *a: None)
    monkeypatch.setattr(tool, '_current_page_origin', lambda *a: 'https://service.test')
    monkeypatch.setattr(tool, '_eval_js', lambda *a: {'success': True, 'result': [
        {'tag': 'input', 'type': 'password', 'id': 'pw', 'visible': True, 'autocomplete': 'current-password'}]})
    fill = Mock(return_value={'success': True, 'result': {'filled': 1}})
    monkeypatch.setattr(tool, '_eval_js_secret', fill)
    filled = tool.browser_vault_fill(item['handle'], task_id='synthetic')
    assert json.loads(filled)['success']
    assert json.dumps(PASSWORD) in fill.call_args.args[1]
    for secret in CANARIES:
        assert secret not in filled + caplog.text
    assert calls(state)[-2]['argv'] == ['read', 'dl://' + ITEM_ID]
    assert all(c['stdin'] == '' and not c['forbidden_env'] for c in calls(state))


@pytest.mark.parametrize('target', ['https://login.other-service.test', 'https://www.service.test',
                                    'http://service.test', 'https://service.test:444'])
def test_discovered_handle_wrong_origin_never_resolves(searchable, monkeypatch, target):
    from tools import browser_vault_tool as tool
    _, state, _, _, _ = searchable
    handle = json.loads(tool.browser_vault_list())['items'][0]['handle']
    monkeypatch.setattr(tool, '_focus_bound_origin', lambda *a: None)
    monkeypatch.setattr(tool, '_current_page_origin', lambda *a: target)
    result = json.loads(tool.browser_vault_fill(handle, task_id='synthetic'))
    assert result['error_type'] == 'origin_mismatch'
    assert not any(c['argv'][0] == 'read' for c in calls(state))


@pytest.mark.parametrize('url', ['https://evil-service.test', 'https://service.test.evil.test',
                                 'https://evil.test/service.test', 'https://www.service.test',
                                 'https://login.other-service.test/service.test'])
def test_vendor_substring_matches_are_not_exact_hosts(searchable, url):
    backend, state, initial, _, _ = searchable
    initial['records'][0]['url'] = url
    state.write_text(json.dumps(initial))
    assert backend.search_items('service.test') == []
    assert not module._DISCOVERED


@pytest.mark.parametrize('hosts', [[''], ['SERVICE.test'], ['*.service.test'], ['https://service.test'],
                                  ['service.test/path'], ['service.test,evil.test'], ['--debug'],
                                  ['service.test', 'service.test'], ['a..test'], ['service.test.'],
                                  ['a.test'] * 6, 'service.test', [None]])
def test_invalid_queries_fail_before_spawn(searchable, hosts):
    backend, state, _, _, _ = searchable
    backend.cfg['search_hosts'] = hosts
    with pytest.raises(VaultError):
        backend.list_items()
    assert not calls(state)


def test_unapproved_host_fails_before_spawn(searchable):
    backend, state, _, _, _ = searchable
    with pytest.raises(VaultError):
        backend.search_items('example.test')
    assert not calls(state)


@pytest.mark.parametrize('raw', ['{', 'null', '{}', '[null]', '[{"id":"CANARY-NOTE"}]',
                                '[{"id":"x","id":"y","password":"CANARY-NOTE"}]'])
def test_malformed_search_sanitized_and_no_handles(searchable, raw, caplog, capsys):
    backend, state, initial, _, _ = searchable
    state.write_text(json.dumps(dict(initial, search_raw=raw)))
    with pytest.raises(VaultError) as error:
        backend.search_items('service.test')
    assert not module._DISCOVERED
    assert 'CANARY' not in str(error.value) + caplog.text + str(capsys.readouterr())


@pytest.mark.parametrize('mode', ['error', 'overflow', 'timeout'])
def test_search_failure_boundaries(searchable, monkeypatch, caplog, capsys, mode):
    from tools import browser_vault_tool as tool
    _, state, initial, _, _ = searchable
    monkeypatch.setattr(module, '_TIMEOUT', .25)
    state.write_text(json.dumps(dict(initial, search_mode=mode)))
    result = tool.browser_vault_list()
    assert json.loads(result)['errors']
    assert json.loads(result)['items'] == []
    assert not module._DISCOVERED
    for secret in CANARIES:
        assert secret not in result + caplog.text + str(capsys.readouterr())


@pytest.mark.parametrize('status', ['Logged in: No',
    'Logged in: Yes\nLogin: owner@example.test\nLocked: Yes',
    'Logged in: Yes\nLogin: other@example.test\nLocked: No'])
@pytest.mark.parametrize('when', ['before', 'after'])
def test_search_account_and_lock_drift(searchable, status, when):
    backend, state, initial, _, _ = searchable
    initial['status' if when == 'before' else 'after_status'] = status
    state.write_text(json.dumps(initial))
    with pytest.raises(UnlockRequired):
        backend.search_items('service.test')
    assert not module._DISCOVERED
    if when == 'before':
        assert all(c['argv'] == ['status'] for c in calls(state))


@pytest.mark.parametrize('count', [2, 21])
def test_duplicate_or_overlimit_records_fail_closed(searchable, count):
    backend, state, initial, _, _ = searchable
    initial['records'] *= count
    state.write_text(json.dumps(initial))
    with pytest.raises(VaultError):
        backend.search_items('service.test')
    assert not module._DISCOVERED


def test_ambiguity_returns_choices_never_auto_resolves(searchable):
    from tools import browser_vault_tool as tool
    backend, state, initial, _, _ = searchable
    initial['records'].append(dict(initial['records'][0],
        id='{ABCDEF02-2345-4567-89AB-0123456789AB}', email='second@example.test'))
    state.write_text(json.dumps(initial))
    result = json.loads(tool.browser_vault_list())
    assert len(result['items']) == 2
    assert 'ask the user' in result['selection_required']
    assert len({i['handle'] for i in result['items']}) == 2
    with pytest.raises(VaultError):
        backend.resolve_password('dl:service.test')
    assert not any(c['argv'][0] == 'read' for c in calls(state))


def test_expired_config_and_profile_changed_handles_refuse(searchable, monkeypatch, tmp_path):
    from hermes_constants import set_hermes_home_override, reset_hermes_home_override
    backend, state, _, _, _ = searchable
    handle = backend.search_items('service.test')[0].id
    original_calls = len(calls(state))
    assert DashlaneLoginBackend(dict(backend.cfg)).get_meta(handle)
    backend.cfg['account'] = 'other@example.test'
    assert backend.get_meta(handle) is None
    backend.cfg['account'] = 'owner@example.test'
    token = set_hermes_home_override(tmp_path / 'other')
    try:
        assert backend.get_meta(handle) is None
    finally:
        reset_hermes_home_override(token)
    monkeypatch.setattr(module.time, 'monotonic', lambda: float('inf'))
    assert backend.get_meta(handle) is None
    with pytest.raises(VaultError):
        backend.resolve_password(handle)
    assert len(calls(state)) == original_calls


@pytest.mark.parametrize('change', [{'url': 'https://evil.test'}, {'email': 'other@example.test'},
                                   {'id': '{ABCDEF02-2345-4567-89AB-0123456789AB}'}])
def test_selected_record_drift_refuses(searchable, change):
    backend, state, initial, _, _ = searchable
    handle = backend.search_items('service.test')[0].id
    initial['item'].update(change)
    state.write_text(json.dumps(initial))
    with pytest.raises(VaultError):
        backend.resolve_password(handle)


@pytest.mark.parametrize('change', [
    {'email': ''}, {'email': ['invalid']}, {'title': {'invalid': True}},
    {'url': 'https://service.test\\@evil.test'}, {'id': ITEM_ID.lower()},
])
def test_malformed_projected_metadata_refuses(searchable, change):
    backend, state, initial, _, _ = searchable
    initial['records'][0].update(change)
    state.write_text(json.dumps(initial))
    with pytest.raises(VaultError):
        backend.search_items('service.test')
    assert not module._DISCOVERED


def test_username_projection_and_selected_fill(searchable):
    backend, state, initial, _, _ = searchable
    initial['records'][0].update(email='', login='synthetic-user')
    state.write_text(json.dumps(initial))
    meta, = backend.search_items('service.test')
    assert meta.identifier == 'synthetic-user' and meta.identifier_type == 'username'
    assert backend.resolve_password(meta.id) == PASSWORD


@pytest.mark.parametrize('count', [0, 20, 21])
def test_record_count_boundaries(searchable, count):
    backend, state, initial, _, _ = searchable
    initial['records'] = [dict(initial['records'][0],
        id='{' + f'{n:08X}-2345-4567-89AB-0123456789AB' + '}') for n in range(count)]
    state.write_text(json.dumps(initial))
    if count > 20:
        with pytest.raises(VaultError):
            backend.search_items('service.test')
        assert not module._DISCOVERED
    else:
        assert len(backend.search_items('service.test')) == count


def test_unexpected_listing_exception_does_not_escape(searchable, monkeypatch, caplog):
    from tools import browser_vault_tool as tool
    monkeypatch.setattr(DashlaneLoginBackend, 'list_items', Mock(side_effect=RuntimeError(CANARIES[1])))
    result = tool.browser_vault_list()
    assert json.loads(result)['errors']
    assert CANARIES[1] not in result + caplog.text


def test_selection_expiring_during_read_refuses_password(searchable, monkeypatch):
    backend, _, _, _, _ = searchable
    handle = backend.search_items('service.test')[0].id
    real_run = backend._run
    def delayed_read(*args):
        result = real_run(*args)
        if args[0] == 'read':
            with module._DISCOVERED_LOCK:
                scope, _, item_id, meta = module._DISCOVERED[handle]
                module._DISCOVERED[handle] = (scope, 0.0, item_id, meta)
        return result
    monkeypatch.setattr(backend, '_run', delayed_read)
    with pytest.raises(VaultError, match='expired'):
        backend.resolve_password(handle)


def test_discovered_handle_never_generates_otp(searchable, monkeypatch):
    from tools import browser_vault_tool as tool
    backend, state, _, _, _ = searchable
    handle = backend.search_items('service.test')[0].id
    before = len(calls(state))
    monkeypatch.setattr(tool, '_focus_bound_origin', lambda *a: None)
    monkeypatch.setattr(tool, '_current_page_origin', lambda *a: 'https://service.test')
    monkeypatch.setattr(tool, '_eval_js', lambda *a: {'success': True, 'result': [
        {'tag': 'input', 'type': 'text', 'id': 'otp', 'visible': True, 'autocomplete': 'one-time-code'}]})
    monkeypatch.setattr('agent.vault_backends.unlock.can_prompt_here', lambda: False)
    fill = Mock(side_effect=AssertionError('Automatic OTP is forbidden'))
    monkeypatch.setattr(tool, '_eval_js_secret', fill)
    result = json.loads(tool.browser_vault_enter_code(handle, task_id='synthetic'))
    assert result['error_type'] == 'prompt_unavailable'
    fill.assert_not_called()
    assert len(calls(state)) == before
