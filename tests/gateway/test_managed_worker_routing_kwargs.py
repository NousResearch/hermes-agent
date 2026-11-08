"""A managed worker (every Kanban run, opt-in CLI sessions) builds its agent with the session's
provider routing and service tier, as ``hermes chat`` did on main (``HermesCLI._init_agent``):
``provider_routing.ignore``/``data_collection`` and a static ``agent.service_tier`` reach AIAgent."""
import json
import threading
from types import SimpleNamespace

import pytest

from tests.gateway.test_managed_turn_contract import assignment

ROUTING = {'only': ['Anthropic'], 'ignore': ['DeepInfra'], 'order': ['Anthropic'], 'sort': 'price',
           'require_parameters': True, 'data_collection': 'deny'}


@pytest.mark.parametrize('tier, overrides', [('priority', {'service_tier': 'priority'}), ('', None)])
def test_managed_worker_agent_gets_the_session_provider_routing(tmp_path, monkeypatch, tier, overrides):
    from agent import managed_worker as worker
    frame, _ = assignment(tmp_path, monkeypatch)
    config = json.loads(frame['policy']['config_json'])
    config.update(provider_routing=ROUTING, openrouter={'min_coding_score': 0.8})
    config.setdefault('agent', {})['service_tier'] = tier
    config['model'] = {**config.get('model', {}), 'default': 'gpt-5.5', 'provider': 'openai',
                       'base_url': 'https://api.openai.com/v1'}
    frame = {**frame, 'policy': {**frame['policy'], 'model': 'gpt-5.5', 'config_json': json.dumps(config)}}
    built = {}

    class Agent:
        def __init__(self, **kwargs):
            built.update(kwargs)
            self.session_prompt_tokens = self.session_completion_tokens = 0
        def run_conversation(self, text, **kwargs):
            return {'final_response': 'ok', 'completed': True}

    class Store:
        def __init__(self, *args): pass
        def get_messages_as_conversation(self, sid): return []
        def flush_token_counts(self): pass
        def finish(self): pass
        def close(self): pass

    class Controls:
        def __init__(self, *args):
            self.stopped, self.finish = threading.Event(), threading.Event()
            self.finish.set()
        def approval(self, data): pass
        def clarify(self, questions): pass

    monkeypatch.setattr(worker, 'bind_worker_policy', lambda frame: None)
    monkeypatch.setattr(worker, 'discover_profile_mcp', lambda policy: None)
    monkeypatch.setattr(worker, 'retire_agent', lambda agent: None)
    monkeypatch.setattr(worker, 'WorkerControls', Controls)
    monkeypatch.setattr('agent.runtime_session_store.WorkerRPC', lambda home: lambda *a, **k: {'owner_epoch': 1})
    monkeypatch.setattr('agent.runtime_session_store.RuntimeSessionStore', Store)
    monkeypatch.setattr('tools.process_registry.process_registry.recover_from_checkpoint', lambda: None)
    monkeypatch.setattr('run_agent.AIAgent', Agent)
    monkeypatch.setattr('agent.title_generator.wait_for_title_upgrades', lambda: None)
    worker.execute(frame, SimpleNamespace(send=lambda kind, **payload: None))
    assert {k: built.get(k) for k in ('providers_allowed', 'providers_ignored', 'providers_order', 'provider_sort',
                                      'provider_require_parameters', 'provider_data_collection')} == {
        'providers_allowed': ['Anthropic'], 'providers_ignored': ['DeepInfra'], 'providers_order': ['Anthropic'],
        'provider_sort': 'price', 'provider_require_parameters': True, 'provider_data_collection': 'deny'}
    assert built.get('openrouter_min_coding_score') == 0.8
    assert built.get('service_tier') == (tier or None)
    assert built.get('request_overrides') == overrides
