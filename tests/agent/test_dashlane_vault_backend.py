"""Synthetic dcli process contract and real registry/tool routing; no installed vault access."""
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from unittest.mock import Mock

import pytest

from agent.vault_backends.dashlane import DashlaneLoginBackend
from agent.vault_backends.base import UnlockRequired
from agent.vault_store import VaultError

pytestmark = pytest.mark.platforms("posix")
ITEM_ID = "ABCDEF01-2345-4567-89AB-0123456789AB"
HANDLE = "dl:" + ITEM_ID
PASSWORD = "  synthetic sentence \nwith whitespace  "
FAKE = r'''
import json, os, sys, time
from pathlib import Path
root = Path(__file__).parent
state = json.loads((root / 'state.json').read_text())
with (root / 'calls.jsonl').open('a') as log:
    log.write(json.dumps({'argv': sys.argv[1:], 'stdin': sys.stdin.read(),
        'forbidden_env': [k for k in os.environ if k.startswith(('DASHLANE_', 'DCLI_', 'NODE_'))]}) + '\n')
if state.get('mode') == 'descendant':
    # Fork avoids a second interpreter startup racing the one-second deadline.
    # The child still retains stdout and must be terminated by group cleanup.
    pid = os.fork()
    if pid == 0:
        time.sleep(30)
        os._exit(0)
    (root / 'descendant.pid').write_text(str(pid))
    sys.exit(0)
if state.get('mode') == 'timeout': time.sleep(30)
if state.get('mode') == 'overflow':
    sys.stdout.write('x' * 300000); sys.exit(0)
if state.get('mode') == 'error':
    print(state['password']); print(state['password'], file=sys.stderr); sys.exit(1)
if sys.argv[1:] == ['status']:
    print(state.get('status', 'Logged in: Yes\nLogin: owner@example.test\nLocked: No'))
elif sys.argv[1:] == ['password', '--output', 'json', 'url=service.test']:
    mode = state.get('search_mode')
    if mode == 'timeout': time.sleep(30)
    if mode == 'overflow':
        sys.stdout.write('x' * 300000); sys.exit(0)
    if mode == 'error':
        print(state['password']); print(state['password'], file=sys.stderr); sys.exit(1)
    if state.get('after_status'):
        (root / 'state.json').write_text(json.dumps(dict(state, status=state['after_status'])))
    print(state['search_raw'] if 'search_raw' in state else json.dumps(state.get('records', [])))
elif sys.argv[1:] == ['read', 'dl://' + state['id']]:
    if state.get('after_status'):
        after = dict(state, status=state['after_status'])
        (root / 'state.json').write_text(json.dumps(after))
    if 'raw' in state: print(state['raw'])
    else: print(json.dumps(state.get('item', {'id': '{' + state['id'] + '}', 'url': 'https://example.test/login',
        'email': 'person@example.test', 'password': state['password'], 'otpSecret': 'SYNTHETIC-OTP-DO-NOT-EXPOSE'})))
else: sys.exit(9)
'''


@pytest.fixture
def enrolled(tmp_path, monkeypatch):
    exe = tmp_path / "dcli"
    exe.write_text('#!' + sys.executable + '\n' + FAKE)
    exe.chmod(0o700)
    state = tmp_path / 'state.json'
    initial = {'id': ITEM_ID, 'password': PASSWORD}
    state.write_text(json.dumps(initial))
    home = tmp_path / 'hermes'
    home.mkdir()
    monkeypatch.setenv('HERMES_HOME', str(home))
    # Deliberately poison the parent environment: vendor overrides must not reach dcli.
    monkeypatch.setenv('DASHLANE_MASTER_PASSWORD', 'MUST-NOT-INHERIT')
    monkeypatch.setenv('DASHLANE_SERVICE_DEVICE_KEYS', 'MUST-NOT-INHERIT')
    monkeypatch.setenv('DCLI_STAGING_HOST', 'https://evil.test')
    monkeypatch.setenv('NODE_OPTIONS', '--inspect')
    cfg = {'enabled': True, 'binary_path': str(exe), 'account': 'owner@example.test', 'items': [
        {'id': ITEM_ID, 'label': 'Example', 'origin': 'https://example.test',
         'identifier': 'person@example.test', 'identifier_type': 'email'}]}
    config = {'vault': {'dashlane': cfg, 'onepassword': {'enabled': False}, 'bitwarden': {'enabled': False}}}
    (home / 'config.yaml').write_text(json.dumps(config))
    return DashlaneLoginBackend(cfg), state, initial, home, config


def calls(state):
    log = state.parent / 'calls.jsonl'
    return [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []


def test_discovery_is_metadata_only_and_resolution_is_exact(enrolled):
    backend, state, _, _, _ = enrolled
    metas = backend.list_items()
    assert metas[0].id == HANDLE
    assert all(call['argv'] == ['status'] for call in calls(state))
    assert backend.resolve_password(HANDLE) == PASSWORD
    assert backend.resolve_otp(HANDLE) is None
    assert [c['argv'] for c in calls(state) if c['argv'][0] != 'status'] == [['read', 'dl://' + ITEM_ID]]
    assert all(c['stdin'] == '' and c['forbidden_env'] == [] for c in calls(state))
    assert PASSWORD not in json.dumps([m.to_dict() for m in metas])


@pytest.mark.parametrize('status', ['Logged in: No', 'Logged in: Yes\nLogin: owner@example.test\nLocked: Yes',
    'Logged in: Yes\nLogin: other@example.test\nLocked: No',
    'Logged in: Yes\nLogin: owner@example.test\nLogin: other@example.test\nLocked: No', '',
    'Logged in: Yes\nLogin: owner@example.test\nLocked: No\nextra'])
def test_locked_missing_or_ambiguous_account_never_reads(enrolled, status):
    backend, state, initial, _, _ = enrolled
    state.write_text(json.dumps(dict(initial, status=status)))
    assert not backend.is_unlocked()
    with pytest.raises(UnlockRequired):
        backend.resolve_password(HANDLE)
    assert all(c['argv'] == ['status'] for c in calls(state))


@pytest.mark.parametrize('change', [
    {'id': 'OTHER'}, {'url': 'https://evil.test'}, {'email': 'other@example.test'},
    {'url': 'https://user:password@example.test'}, {'url': 'file:///private'}, {'password': ''},
])
def test_record_binding_drift_refuses(enrolled, change):
    backend, state, initial, _, _ = enrolled
    item = {'id': '{' + ITEM_ID + '}', 'url': 'https://example.test', 'email': 'person@example.test', 'password': PASSWORD}
    item.update(change)
    state.write_text(json.dumps(dict(initial, item=item)))
    with pytest.raises(VaultError) as error:
        backend.resolve_password(HANDLE)
    assert PASSWORD not in str(error.value)


@pytest.mark.parametrize('raw', ['[]', '{', 'null', '{"password":"a","password":"b"}'])
def test_malformed_output_is_sanitized(enrolled, raw):
    backend, state, initial, _, _ = enrolled
    state.write_text(json.dumps(dict(initial, raw=raw)))
    with pytest.raises(VaultError):
        backend.resolve_password(HANDLE)


def test_account_or_lock_change_during_read_refuses(enrolled):
    backend, state, initial, _, _ = enrolled
    state.write_text(json.dumps(dict(initial, after_status='Logged in: Yes\nLogin: other@example.test\nLocked: No')))
    with pytest.raises(UnlockRequired):
        backend.resolve_password(HANDLE)


@pytest.mark.parametrize('mode', ['timeout', 'overflow', 'error'])
def test_subprocess_limits_and_secret_free_errors(enrolled, monkeypatch, caplog, capsys, mode):
    backend, state, initial, _, _ = enrolled
    monkeypatch.setattr('agent.vault_backends.dashlane._TIMEOUT', 0.25)
    state.write_text(json.dumps(dict(initial, mode=mode)))
    import time
    start = time.monotonic()
    with pytest.raises(VaultError) as error:
        backend._run('status')
    assert time.monotonic() - start < 3
    assert PASSWORD not in str(error.value) + caplog.text + str(capsys.readouterr())


@pytest.mark.skipif(os.name != 'posix' or not shutil.which('ps'), reason='Requires POSIX process groups and ps')
def test_descendant_retaining_stdout_is_terminated(enrolled, monkeypatch):
    backend, state, initial, _, _ = enrolled
    monkeypatch.setattr('agent.vault_backends.dashlane._TIMEOUT', 1)
    state.write_text(json.dumps(dict(initial, mode='descendant')))
    pid_file = state.parent / 'descendant.pid'

    def terminated(pid):
        result = subprocess.run(['ps', '-p', str(pid), '-o', 'stat='],
                                capture_output=True, text=True, timeout=1)
        assert result.returncode in (0, 1)
        status = result.stdout.strip()
        # An orphan zombie has terminated but may await its platform's reaper.
        return not status or status.startswith('Z')

    start = time.monotonic()
    try:
        with pytest.raises(VaultError):
            backend._run('status')
        assert time.monotonic() - start < 3
        assert pid_file.exists(), 'Synthetic descendant never started'
        pid = int(pid_file.read_text())
        deadline = time.monotonic() + 2
        while not terminated(pid):
            assert time.monotonic() < deadline, 'Descendant survived process-group cleanup'
            time.sleep(0.02)
    finally:
        # Also clean up if the regression assertion fails.
        if pid_file.exists():
            pid = int(pid_file.read_text())
            if not terminated(pid):
                try:
                    os.kill(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass


@pytest.mark.parametrize('prompt_available', [False, True])
def test_headless_otp_requires_manual_verification(enrolled, monkeypatch, prompt_available):
    from tools import browser_vault_tool as tool
    _, state, _, _, _ = enrolled
    prompt = Mock(side_effect=AssertionError('Headless session must not prompt'))
    fill = Mock(side_effect=AssertionError('No automatic Dashlane OTP fill'))
    monkeypatch.setattr(tool, '_focus_bound_origin', lambda *a: None)
    monkeypatch.setattr(tool, '_current_page_origin', lambda *a: 'https://example.test')
    monkeypatch.setattr(tool, '_eval_js', lambda *a: {'success': True, 'result': [
        {'tag': 'input', 'type': 'text', 'id': 'otp', 'visible': True, 'autocomplete': 'one-time-code'}]})
    monkeypatch.setattr(tool, '_eval_js_secret', fill)
    monkeypatch.setattr('agent.vault_backends.unlock.get_code_prompt_callback',
                        lambda: prompt if prompt_available else None)
    monkeypatch.setattr('agent.vault_backends.unlock.can_prompt_here', lambda: False)
    result = json.loads(tool.browser_vault_enter_code(HANDLE, task_id='synthetic'))
    assert result['success'] is False
    assert result['error_type'] == 'prompt_unavailable'
    assert 'Complete verification manually' in result['error']
    assert 'secure prompt' in result['error']
    assert 'does not support automatic codes' in result['error']
    assert 'Save an authenticator key' not in result['error']
    assert 'generated automatically' not in result['error']
    prompt.assert_not_called()
    fill.assert_not_called()
    assert not calls(state)


@pytest.mark.parametrize('bad_id', [ITEM_ID.lower(), '{' + ITEM_ID + '}', 'id=' + ITEM_ID, '../title', '--debug'])
def test_invalid_ids_never_become_title_lookups(enrolled, bad_id):
    backend, state, _, _, _ = enrolled
    assert backend.get_meta('dl:' + bad_id) is None
    with pytest.raises(VaultError):
        backend.resolve_password('dl:' + bad_id)
    assert not calls(state)


def test_invalid_enrollment_never_spawns(enrolled):
    backend, state, _, _, _ = enrolled
    backend.cfg['items'].append(dict(backend.cfg['items'][0]))
    with pytest.raises(VaultError):
        backend.list_items()
    assert not calls(state)


@pytest.mark.parametrize('change', [
    {'enabled': 'true'}, {'enabled': 1}, {'enabled': False}, {'extra': 'rejected'},
    {'binary_path': './dcli'}, {'binary_path': 4}, {'account': ''}, {'account': 'owner\nother'},
    {'items': {}}, {'items': [None]},
])
def test_strict_config_never_spawns(enrolled, change):
    backend, state, _, _, _ = enrolled
    backend.cfg.update(change)
    with pytest.raises(VaultError):
        backend.list_items()
    assert not calls(state)


@pytest.mark.parametrize('cfg', [True, 'invalid', [], 42])
def test_scalar_config_fails_closed(cfg):
    with pytest.raises(VaultError):
        DashlaneLoginBackend(cfg).list_items()


@pytest.mark.parametrize('origin', ['https://example.test/', 'https://example.test:443',
    'https://example.test/path', 'https://user@example.test', 'https://@example.test',
    'https://exa mple.test', 'https://example.test\\evil', 'https://%65xample.test',
    'https://example.test:99999', 'file://example.test', 'https://example.test#fragment'])
def test_noncanonical_origin_rejected_before_cli(enrolled, origin):
    backend, state, _, _, _ = enrolled
    backend.cfg['items'][0]['origin'] = origin
    with pytest.raises(VaultError):
        backend.list_items()
    assert not calls(state)


def test_username_binding(enrolled):
    backend, state, initial, _, _ = enrolled
    backend.cfg['items'][0].update(identifier='synthetic-user', identifier_type='username')
    state.write_text(json.dumps(dict(initial, item={'id': '{' + ITEM_ID + '}',
        'url': 'https://example.test', 'login': 'synthetic-user', 'password': PASSWORD})))
    assert backend.resolve_password(HANDLE) == PASSWORD


def test_real_registry_tool_fill_and_origin_refusal(enrolled, monkeypatch, caplog):
    from agent.vault_backends import backend_for_handle
    from tools import browser_vault_tool as tool
    backend, state, _, _, _ = enrolled
    assert isinstance(backend_for_handle(HANDLE), DashlaneLoginBackend)
    listed = tool.browser_vault_list()
    assert PASSWORD not in listed and 'SYNTHETIC-OTP' not in listed
    assert 'two_factor' not in json.loads(listed)['items'][0]
    monkeypatch.setattr(tool, '_focus_bound_origin', lambda *a: None)
    monkeypatch.setattr(tool, '_current_page_origin', lambda *a: 'https://evil.test')
    result = json.loads(tool._handle_vault_fill({'handle': HANDLE}, task_id='synthetic'))
    assert result['error_type'] == 'origin_mismatch'
    assert all(c['argv'] == ['status'] for c in calls(state))
    monkeypatch.setattr(tool, '_current_page_origin', lambda *a: 'https://example.test')
    monkeypatch.setattr(tool, '_eval_js', lambda *a: {'success': True, 'result': [
        {'tag': 'input', 'type': 'password', 'id': 'pw', 'visible': True, 'autocomplete': 'current-password'}]})
    fill = Mock(return_value={'success': True, 'result': {'filled': 1}})
    monkeypatch.setattr(tool, '_eval_js_secret', fill)
    result = tool._handle_vault_fill({'handle': HANDLE}, task_id='synthetic')
    assert json.loads(result)['success']
    assert json.dumps(PASSWORD) in fill.call_args.args[1]
    assert PASSWORD not in result + caplog.text
    assert 'https://example.test' in fill.call_args.args[1]
    fill.return_value = {'success': True, 'result': {'refused': 'origin_changed'}}
    assert json.loads(tool.browser_vault_fill(HANDLE, task_id='synthetic'))['error_type'] == 'origin_changed'


def test_missing_enrollment_never_falls_back_to_cli_search(enrolled):
    backend, state, _, _, _ = enrolled
    backend.cfg['items'] = []
    assert backend.list_items() == []
    # Empty search_hosts must never implicitly opt in to provider projection.
    assert backend.get_meta('dl:service.test') is None
    with pytest.raises(VaultError):
        backend.resolve_password('dl:service.test')
    assert [c['argv'] for c in calls(state)] == [['status']]


@pytest.mark.parametrize('target', [
    'https://login.other-service.test', 'https://www.service.test',
    'http://service.test', 'https://service.test:444',
])
def test_saved_origin_does_not_authorize_other_targets(enrolled, monkeypatch, target):
    from tools import browser_vault_tool as tool
    _, state, _, home, config = enrolled
    config['vault']['dashlane']['items'][0]['origin'] = 'https://service.test'
    (home / 'config.yaml').write_text(json.dumps(config))
    monkeypatch.setattr(tool, '_focus_bound_origin', lambda *a: None)
    monkeypatch.setattr(tool, '_current_page_origin', lambda *a: target)
    fill = Mock(side_effect=AssertionError('Origin mismatch must not fill'))
    monkeypatch.setattr(tool, '_eval_js_secret', fill)
    result = json.loads(tool._handle_vault_fill({'handle': HANDLE}, task_id='synthetic'))
    assert result['error_type'] == 'origin_mismatch'
    assert all(c['argv'] == ['status'] for c in calls(state))
    fill.assert_not_called()


def test_manual_unlock_never_prompts(enrolled, monkeypatch):
    from tools import browser_vault_tool as tool
    backend, state, initial, _, _ = enrolled
    state.write_text(json.dumps(dict(initial, status='Logged in: No')))
    prompt = Mock(side_effect=AssertionError('No prompt supported'))
    monkeypatch.setattr('agent.vault_backends.unlock.get_unlock_prompt_callback', lambda: prompt)
    assert json.loads(tool.browser_vault_list())['locked'][0]['unlock'] == 'manual_cli'
    assert json.loads(tool.browser_vault_unlock('dashlane'))['error_type'] == 'manual_unlock_required'
    assert json.loads(tool.browser_vault_fill(HANDLE))['error_type'] == 'manual_unlock_required'
    prompt.assert_not_called()


def test_explicit_opt_in_and_profile_scoped_enrollment(enrolled, monkeypatch, tmp_path):
    from agent.vault_backends import backend_for_handle
    from agent.vault_backends.base import is_enabled
    from hermes_constants import set_hermes_home_override, reset_hermes_home_override
    _, _, _, home, config = enrolled
    other = tmp_path / 'other'
    other.mkdir()
    other_config = json.loads(json.dumps(config))
    other_config['vault']['dashlane']['enabled'] = False
    (other / 'config.yaml').write_text(json.dumps(other_config))
    for target, expected in [(home, True), (other, False), (home, True)]:
        token = set_hermes_home_override(target)
        try:
            assert is_enabled('dashlane') is expected
            assert (backend_for_handle(HANDLE) is not None) is expected
        finally:
            reset_hermes_home_override(token)
