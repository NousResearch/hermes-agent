"""Public fixed-code canaries through real browser_vault_list routing."""
import json
from unittest.mock import Mock

import pytest

from agent.vault_backends import dashlane as module
from tests.agent.test_dashlane_listing_diagnosis import synthetic, HOST, GENERIC
from tests.agent.test_dashlane_vault_backend import enrolled

CANARY = 'SYNTHETIC-SECRET-NEVER-EMIT'


def public_error(stage, reason):
    return {'backend': 'dashlane', 'error': GENERIC, 'stage': stage, 'reason': reason}


@pytest.mark.parametrize('stage,reason', [
    ('configuration', 'invalid_configuration'),
    ('status_before', 'status_unavailable'),
    ('status_after', 'status_unavailable'),
    ('search_process', 'process_failed'),
    ('search_json', 'invalid_json'),
    ('search_projection', 'invalid_metadata'),
])
def test_stage_canaries(synthetic, monkeypatch, caplog, capsys, stage, reason):
    from tools import browser_vault_tool as tool
    backend, payload = synthetic
    run = module.DashlaneLoginBackend._run
    searched = False

    def poisoned(self, *args):
        nonlocal searched
        if args == ('status',):
            if stage == ('status_after' if searched else 'status_before'):
                raise RuntimeError(CANARY)
        else:
            searched = True
            if stage == 'search_process':
                raise RuntimeError(CANARY)
        return run(self, *args)

    monkeypatch.setattr(module.DashlaneLoginBackend, '_run', poisoned)
    if stage == 'configuration':
        monkeypatch.setattr(module.DashlaneLoginBackend, '_enrolled', Mock(side_effect=ValueError(CANARY)))
    if stage == 'search_json':
        payload['raw'] = CANARY
    if stage == 'search_projection':
        payload['records'][0].update(email={'secret': CANARY}, password=CANARY, title=CANARY)
    result = tool.browser_vault_list()
    parsed = json.loads(result)
    assert parsed['errors'] == [public_error(stage, reason)]
    assert not parsed['items']
    assert not module._DISCOVERED
    assert CANARY not in result + caplog.text + str(capsys.readouterr())


@pytest.mark.parametrize('after', [False, True])
def test_status_mismatch_keeps_manual_unlock_and_diagnostic(synthetic, monkeypatch, after):
    from tools import browser_vault_tool as tool
    run = module.DashlaneLoginBackend._run
    searched = False

    def changed(self, *args):
        nonlocal searched
        if args != ('status',):
            searched = True
        elif searched == after:
            return 'Logged in: Yes\nLogin: ' + CANARY + '\nLocked: No'
        return run(self, *args)

    monkeypatch.setattr(module.DashlaneLoginBackend, '_run', changed)
    result = json.loads(tool.browser_vault_list())
    assert result['errors'] == [public_error('status_after' if after else 'status_before', 'account_or_lock_state')]
    assert result['locked'][0]['unlock'] == 'manual_cli'
    assert not result['items']
    assert not module._DISCOVERED
    assert CANARY not in json.dumps(result)


def test_valid_unrelated_origin_skips_bad_id_and_metadata(synthetic):
    backend, payload = synthetic
    payload['records'].append({'url': 'https://not-' + HOST, 'id': CANARY,
                               'email': {'secret': CANARY}, 'title': CANARY})
    item, = backend.search_items(HOST)
    assert item.origin == 'https://' + HOST
    assert CANARY not in repr(item)


@pytest.mark.parametrize('records,reason', [
    ({'secret': CANARY}, 'invalid_response'),
    ([CANARY], 'invalid_record'),
    ([{}] * 21, 'record_limit'),
])
def test_invalid_projection_shapes(synthetic, records, reason):
    from tools import browser_vault_tool as tool
    _, payload = synthetic
    payload['records'] = records
    result = json.loads(tool.browser_vault_list())
    assert result['errors'] == [public_error('search_projection', reason)]
    assert not module._DISCOVERED
    assert CANARY not in json.dumps(result)


def test_duplicate_identity_and_capacity(synthetic):
    backend, payload = synthetic
    payload['records'] *= 2
    with pytest.raises(module.DashlaneListingError) as exc:
        backend.search_items(HOST)
    assert module.listing_diagnostic(exc.value) == {'stage': 'search_projection', 'reason': 'duplicate_identity'}
    assert not module._DISCOVERED
    payload['records'] = payload['records'][:1]
    for i in range(1000):
        module._DISCOVERED[str(i)] = (None, float('inf'), None, None)
    with pytest.raises(module.DashlaneListingError) as exc:
        backend.search_items(HOST)
    assert module.listing_diagnostic(exc.value) == {'stage': 'search_projection', 'reason': 'selection_capacity'}
    assert len(module._DISCOVERED) == 1000


def test_mutated_or_untyped_codes_never_escape(synthetic, monkeypatch, caplog, capsys):
    from tools import browser_vault_tool as tool
    error = module.DashlaneListingError('search_json', 'invalid_json')
    error.reason = CANARY
    error.args = (CANARY,)
    monkeypatch.setattr(module.DashlaneLoginBackend, 'list_items', Mock(side_effect=error))
    result = tool.browser_vault_list()
    assert json.loads(result)['errors'] == [{'backend': 'dashlane', 'error': GENERIC}]
    assert CANARY not in result + caplog.text + str(capsys.readouterr())
