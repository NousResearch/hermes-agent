"""A TUI session with no explicit ``display.busy_input_mode`` reports main's TUI default ``queue``
(``ui-tui/src/app/useConfigSync.ts`` TUI_BUSY_DEFAULT): typing while the agent works queues the next
prompt instead of redirecting the running turn (FINDINGS row 11). An explicit setting still wins, and
every other surface keeps the framework default ``interrupt``."""
import json
from types import SimpleNamespace

import pytest

from gateway.session_contract import SessionRef


def _connection(source, display):
    live = SimpleNamespace(source=SimpleNamespace(platform='local'), busy_input_mode=None)
    authority = SimpleNamespace(sessions={'s': live}, authorize=lambda actor, ref, capability: None,
                                runner=SimpleNamespace(), profile_id='/nonexistent')
    policy = SimpleNamespace(source=source, config=lambda *a, **k: {'display': display} if display is not None else {})
    return SimpleNamespace(authority=authority, actor=None), policy


@pytest.mark.asyncio
@pytest.mark.parametrize(('source', 'display', 'expected'), [
    ('tui', None, 'queue'),
    ('tui', {}, 'queue'),
    ('tui', {'busy_input_mode': 'interrupt'}, 'interrupt'),
    ('tui', {'busy_input_mode': 'steer'}, 'steer'),
    ('cli', None, 'interrupt'),
    ('gui', {}, 'interrupt'),
    ('cli', {'busy_input_mode': 'queue'}, 'queue'),
])
async def test_busy_default_follows_the_surface(monkeypatch, source, display, expected):
    from gateway import session_policy
    from gateway.session_busy_controls import busy_config
    connection, policy = _connection(source, display)
    monkeypatch.setattr(session_policy, 'policy_for_source', lambda runner, src: policy)
    result = await busy_config(connection, SessionRef('p', 's'), {'session_id': 's', 'key': 'busy'})
    assert result == {'key': 'busy', 'value': expected, 'scope': 'session'}, json.dumps(result)
