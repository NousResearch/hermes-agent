"""Resolve real SDK clients from the local-server profile without making model requests."""
import os

import pytest


@pytest.fixture
def configured_chain(tmp_path):
    previous = os.environ.get('HERMES_HOME')
    os.environ['HERMES_HOME'] = str(tmp_path)
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    token = set_hermes_home_override(tmp_path)
    from hermes_cli.config import atomic_config_write

    entries = [{'provider': 'ollama', 'model': model, 'base_url': 'http://localhost:11434/v1'}
               for model in ('gpt-oss:20b', 'qwen3:8b')]
    atomic_config_write(tmp_path / 'config.yaml', {'auxiliary': {'title_generation': {'fallback_chain': entries}}})
    from agent import auxiliary_client as auxiliary

    clients = []
    try:
        yield auxiliary, clients, entries
    finally:
        for client in clients:
            client.close()
        reset_hermes_home_override(token)
        if previous is None:
            os.environ.pop('HERMES_HOME')
        else:
            os.environ['HERMES_HOME'] = previous


def test_actual_sdk_selection_walks_each_configured_entry_once(configured_chain):
    auxiliary, clients, entries = configured_chain
    excluded = set()
    for entry in entries:
        client, model, label = auxiliary._try_configured_fallback_chain(
            'title_generation', 'openrouter', excluded_labels=excluded)
        assert client is not None
        clients.append(client)
        assert model == entry['model'] and str(client.base_url).rstrip('/') == entry['base_url']
        assert label and label not in excluded
        excluded.add(label)
    assert auxiliary._try_configured_fallback_chain(
        'title_generation', 'openrouter', excluded_labels=excluded) == (None, None, '')


def test_attempt_budget_stops_an_otherwise_eligible_sdk_candidate(configured_chain):
    from agent.backend_identity import FailureScope
    from agent.auxiliary_fallback import _next_fallback_after_quarantine

    auxiliary, clients, entries = configured_chain
    client, model, _label = auxiliary._try_configured_fallback_chain('title_generation', 'openrouter')
    assert client is not None and model == entries[0]['model']
    clients.append(client)
    base = str(client.base_url)
    route = auxiliary._LadderRoute(
        client=client, task='title_generation', tag='', async_mode=False, base_info=base,
        resolved_provider='ollama', resolved_model=model, resolved_base_url=base,
        resolved_api_key=client.api_key, resolved_api_mode='chat_completions', final_model=model,
        main_runtime=None, route_info={'provider': 'ollama', 'model': model}, timeout=None)
    before = dict(route.route_info)
    blocked = _next_fallback_after_quarantine(
        'title_generation', 'ollama', False, route, model, FailureScope.MODEL,
        task_chain_only=True, configured_attempt_limit=0)
    assert blocked == (None, None, '') and route.route_info == before
    available, next_model, _label = _next_fallback_after_quarantine(
        'title_generation', 'ollama', False, route, model, FailureScope.MODEL,
        task_chain_only=True, configured_attempt_limit=1)
    assert available is not None and next_model == entries[1]['model']
    clients.append(available)
