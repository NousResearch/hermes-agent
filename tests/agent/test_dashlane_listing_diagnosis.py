"""Listing diagnostic regression contracts. No real CLI, vault, network or browser calls."""
import json
from unittest.mock import Mock

import pytest

from agent.vault_backends import dashlane as module
from agent.vault_backends.dashlane import DashlaneLoginBackend
from agent.vault_store import VaultError
from tests.agent.test_dashlane_vault_backend import enrolled, ITEM_ID

HOST = 'sso.service.test'
GENERIC = 'Dashlane listing failed; check configuration, account/lock state and search limits'


@pytest.fixture
def synthetic(enrolled, monkeypatch):
    backend, _, _, home, config = enrolled
    backend.cfg.update(items=[], search_hosts=[HOST])
    (home / 'config.yaml').write_text(json.dumps(config))
    record = {'id': '{' + ITEM_ID + '}', 'url': 'https://' + HOST,
              'title': 'Synthetic service', 'email': 'synthetic@example.test'}
    payload = {'records': [record]}
    module._DISCOVERED.clear()

    def run(self, *args):
        if args == ('status',):
            return 'Logged in: Yes\nLogin: owner@example.test\nLocked: No\n'
        assert args == ('password', '--output', 'json', 'url=' + HOST)
        return payload.get('raw', json.dumps(payload['records']))

    monkeypatch.setattr(DashlaneLoginBackend, '_run', run)
    monkeypatch.setattr(module.subprocess, 'Popen', Mock(side_effect=AssertionError('Real CLI forbidden')))
    yield backend, payload
    module._DISCOVERED.clear()


def test_explicit_origin_vendor_shape_is_admitted(synthetic):
    backend, _ = synthetic
    item, = backend.search_items(HOST)
    assert item.origin == 'https://' + HOST
    assert item.identifier == 'synthetic@example.test'


@pytest.mark.parametrize('case,error', [

    ('lowercase_uuid', 'invalid_identity'),
    ('missing_identifier', 'invalid_metadata'),

    ('non_json_prefix', 'invalid_json'),
])
def test_known_rejection_routes_return_fixed_public_codes(synthetic, case, error):
    from tools import browser_vault_tool as tool
    backend, payload = synthetic
    record = payload['records'][0]
    if case == 'schemeless':
        record['url'] = HOST
    elif case == 'lowercase_uuid':
        record['id'] = record['id'].lower()
    elif case == 'missing_identifier':
        record.pop('email')
        record['secondaryLogin'] = 'synthetic@example.test'
    elif case == 'nonmatching_schemeless_poison':
        payload['records'].append(dict(record, id='{ABCDEF02-2345-4567-89AB-0123456789AB}',
                                       url='not-' + HOST))
    elif case == 'nonmatching_bad_id_poison':
        payload['records'].append(dict(record, id='invalid-synthetic-id', url='https://not-' + HOST))
    else:
        payload['raw'] = 'info: synthetic message\n' + json.dumps(payload['records'])
    with pytest.raises(VaultError, match=error):
        backend.search_items(HOST)
    assert not module._DISCOVERED
    result = json.loads(tool.browser_vault_list())
    assert result['items'] == []
    assert result['errors'] == [{'backend': 'dashlane', 'error': GENERIC,
                                 'stage': 'search_json' if case == 'non_json_prefix' else 'search_projection',
                                 'reason': error}]
    assert not module._DISCOVERED
