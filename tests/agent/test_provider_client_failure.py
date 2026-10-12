"""Provider transport initialization failures must retain their original cause."""
from types import SimpleNamespace

import pytest

from agent.agent_runtime_helpers import _provider_supplied_client


def test_registered_provider_client_failure_is_not_generic_fallback(monkeypatch):
    import providers

    original = FileNotFoundError('worker cwd no longer exists')

    def create_client(**kwargs):
        assert kwargs['command'] == 'synthetic-acp'
        raise original

    monkeypatch.setattr(providers, 'get_provider_profile', lambda name: SimpleNamespace(
        name=name, create_client=create_client))
    with pytest.raises(FileNotFoundError) as caught:
        _provider_supplied_client(SimpleNamespace(provider='claude-acp'), {'command': 'synthetic-acp'})
    assert caught.value is original
