"""The aux client's anonymous custom arm (``provider: custom`` + a base URL) sends the ``extra_headers``
of the custom provider entry whose base URL is that endpoint, as the main agent's client does, and
never another entry's: the borrowed one-shot route and every aux task with ``provider: custom`` go here."""
from unittest.mock import MagicMock, patch

import pytest

import agent.auxiliary_client as ac

_CONFIG = {'custom_providers': [{'name': 'routed', 'base_url': 'https://gw.example.com/v1',
                                 'extra_headers': {'X-Route-Token': 'tok'}}]}


@pytest.mark.parametrize(('base_url', 'sent'), [
    ('https://gw.example.com/v1', 'tok'), ('https://GW.example.com/v1/', 'tok'),
    ('https://other.example.com/v1', None), ('https://gw.example.com.attacker.test/v1', None),
    ('https://gw.example.com/v1/extra', None)])
def test_a_custom_endpoint_carries_only_its_own_entrys_headers(monkeypatch, base_url, sent):
    monkeypatch.delenv('OPENAI_API_KEY', raising=False)
    created = {}

    def create(**kwargs):
        created.update(kwargs)
        return MagicMock()
    with patch('hermes_cli.config.load_config', return_value=_CONFIG), \
         patch('hermes_cli.config.load_config_readonly', return_value=_CONFIG), \
         patch.object(ac, '_create_openai_client', side_effect=create):
        ac.resolve_provider_client('custom', model='m', explicit_base_url=base_url, explicit_api_key='sk-x')
    assert created, 'no client was built'
    assert (created.get('default_headers') or {}).get('X-Route-Token') == sent
