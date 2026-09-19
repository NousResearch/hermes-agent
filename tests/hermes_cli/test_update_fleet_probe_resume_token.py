"""Only row-capable runtime evidence can make an empty fleet probe fail."""
from types import SimpleNamespace

import pytest

from hermes_cli.main import _fleet_probe_expected_runtimes
from hermes_cli.update_inventory import RuntimeRecord


@pytest.mark.parametrize('plan,pids,token,services,killed,expected', [
    (None, [], {'profiles': {'default': 4321}}, [], set(), False),
    (None, [], {'unmapped': [{'pid': 99}]}, [], set(), False),
    ([], [], {'profiles': {'work': 777}}, [], set(), False),
    (None, [], {'services': ['HermesGateway']}, [], set(), False),
    (None, [], {}, [], set(), False), (None, [], None, [], set(), False),
    ([], [], None, [], set(), False), (None, None, None, [], set(), True),
    (None, [4321], {'profiles': {'default': 4321}}, [], set(), True),
    (['gateway'], [], {'unmapped': [{'pid': 99}]}, [], set(), True),
    (['serve', 'dashboard'], [], None, [], set(), False),
    (['serve', 'gateway'], [], None, [], set(), True),
    (None, [], None, ['hermes-gateway'], set(), True),
    (None, [], None, [], {4321}, True),
])
def test_expected_rows(plan, pids, token, services, killed, expected):
    inventory = None if plan is None else SimpleNamespace(
        runtimes=[RuntimeRecord(kind=kind, profile='default') for kind in plan])
    assert _fleet_probe_expected_runtimes(inventory, pids, token, services, killed) is expected
