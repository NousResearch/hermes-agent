"""#132648 on the path a resumed CLI/TUI chat really takes: the gateway adopts the saved row.

A row with a model but no provider was launched on the profile's main slot, which serves another model, so
every turn 404ed even though the CLI printed the right provider. Loopback providers record what each one
receives on the adopted session's next turn.
"""
import os
from pathlib import Path

import pytest

from tests.fakes.fake_llm_provider import FakeLLMServer, Text
from tests.gateway.test_legacy_session_route_turn import _adopt_and_resolve


@pytest.mark.parametrize('shared_relay', [False, True], ids=['own-endpoints', 'one-relay-two-keys'])
def test_an_adopted_row_with_no_provider_runs_on_the_entry_that_serves_its_model(monkeypatch, shared_relay):
    from run_agent import AIAgent
    with FakeLLMServer([Text('main')]) as main, FakeLLMServer([Text('entry-b ok')]) as other:
        entry_b = main if shared_relay else other
        home = Path(os.environ['HERMES_HOME'])
        monkeypatch.setattr('gateway.run._hermes_home', home)
        home.mkdir(parents=True, exist_ok=True)
        (home / 'config.yaml').write_text(
            'model:\n  default: profile-model\n  provider: custom:entry-a\n'
            'agent:\n  api_max_retries: 1\nplatform_toolsets:\n  tui: []\ncustom_providers:\n'
            f'  - name: entry-a\n    base_url: {main.base_url}\n    api_key: sk-a\n'
            f'  - name: entry-b\n    base_url: {entry_b.base_url}\n    api_key: sk-b\n    models: {{legacy-model: {{}}}}\n')
        # The row carries the main slot's URL and wire, stamped by the turns that already went there.
        _, (model, runtime) = _adopt_and_resolve(home, {'base_url': main.base_url}, source='tui')
        agent = AIAgent(model=model, **{k: runtime[k] for k in ('api_key', 'base_url', 'provider', 'api_mode')},
                        quiet_mode=True, skip_context_files=True, skip_memory=True, enabled_toolsets=['memory'])
        agent.run_conversation('hello again')

        assert [r['model'] for r in entry_b.main_requests()] == ['legacy-model']
        assert {r['auth'] for r in entry_b.requests} == {'Bearer sk-b'}  # entry-b's own key, never entry-a's
        if not shared_relay:
            assert main.requests == []
