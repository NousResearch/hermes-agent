"""A frozen managed worker's config write-back never resurrects a branch the operator deleted."""
import pytest

from hermes_cli.config_worker_save import _three_way

_SERVER = {'command': 'stale-cmd', 'args': ['--x'], 'enabled': False}
_ENABLED = {**_SERVER, 'enabled': True}


@pytest.mark.parametrize('live, base, desired, expected', [
    # Operator removed the server entry; the worker's leaf enable must not copy it back.
    ({'mcp_servers': {}}, {'mcp_servers': {'revoked': _SERVER}},
     {'mcp_servers': {'revoked': _ENABLED}}, {'mcp_servers': {}}),
    # Operator removed the whole mcp_servers section.
    ({'model': 'm'}, {'model': 'm', 'mcp_servers': {'revoked': _SERVER}},
     {'model': 'm', 'mcp_servers': {'revoked': _ENABLED}}, {'model': 'm'}),
    # Operator replaced the branch with a different type: live wins.
    ({'mcp_servers': {'revoked': 'off'}}, {'mcp_servers': {'revoked': _SERVER}},
     {'mcp_servers': {'revoked': _ENABLED}}, {'mcp_servers': {'revoked': 'off'}}),
    # A deleted list is not rebuilt from the snapshot's stale items.
    ({}, {'command_allowlist': ['a *']}, {'command_allowlist': ['a *', 'b *']}, {}),
    # Controls: a live parent still present takes the leaf edit; a brand-new key is added.
    ({'mcp_servers': {'revoked': {**_SERVER, 'command': 'live-cmd'}}}, {'mcp_servers': {'revoked': _SERVER}},
     {'mcp_servers': {'revoked': _ENABLED}},
     {'mcp_servers': {'revoked': {**_SERVER, 'command': 'live-cmd', 'enabled': True}}}),
    ({'mcp_servers': {}}, {'mcp_servers': {}}, {'mcp_servers': {'new': _ENABLED}},
     {'mcp_servers': {'new': _ENABLED}}),
])
def test_worker_rebase_keeps_live_deletions(live, base, desired, expected):
    assert _three_way(live, desired, base) == expected
