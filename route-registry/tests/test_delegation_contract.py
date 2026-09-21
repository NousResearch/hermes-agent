"""Delegation base_url contract coverage for the governed local transition."""
import copy
import json
import unittest
from pathlib import Path

import yaml

HERE = Path(__file__).parents[1]
W = HERE.parent


def load_spec():
    return yaml.safe_load((HERE / 'registry/route-slots.yaml').read_text())['local_transition']


class DelegationContractTests(unittest.TestCase):
    def test_local_delegation_base_url_is_not_an_active_route_shape(self):
        """The registry spec declares provider/model/base_url route shapes only.

        The known live dezzy config carries a delegation.base_url pointing at the
        retired local URL with provider/model ollama-cloud/kimi-k2.6. Because
        its provider is not the old local provider, transform_config must leave
        it untouched (it is an availability declaration, not an active local
        route). This test pins that contract so a future widening of route
        shapes cannot silently rewrite delegation credentials.
        """
        from local_transition import transform_config
        spec = load_spec()
        doc = {
            'delegation': {
                'base_url': spec['old']['base_url'],
                'provider': 'ollama-cloud',
                'model': 'kimi-k2.6',
            },
            'providers': {'ollama-cloud': {'request_timeout_seconds': 41}},
        }
        original = copy.deepcopy(doc)
        out, changed = transform_config(doc, spec)
        self.assertEqual(out['delegation'], original['delegation'])
        self.assertNotIn('delegation', changed)
        # The unrelated cloud provider timeout declaration is untouched.
        self.assertEqual(out['providers']['ollama-cloud']['request_timeout_seconds'], 41)
        self.assertNotIn('custom_providers', out)


if __name__ == '__main__':
    unittest.main(verbosity=2)