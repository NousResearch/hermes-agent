"""Managed workers use the session's toolset exclusions when building their agent."""
import json

from tests.gateway.test_managed_worker_route_key import _execute
from tests.gateway.test_managed_turn_contract import assignment


def test_worker_uses_frozen_toolset_exclusions(tmp_path, monkeypatch):
    frame, _ = assignment(tmp_path, monkeypatch)
    config = json.loads(frame['policy']['config_json'])
    config.setdefault('agent', {})['disabled_toolsets'] = ['kanban', 'terminal']
    frame['policy']['config_json'] = json.dumps(config)

    agent = _execute(frame, monkeypatch)

    assert agent.get('disabled_toolsets') == ['kanban', 'terminal']
