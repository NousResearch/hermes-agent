"""A provider the profile disabled must not block a model switch away from it.

The session froze its route on a provider the operator later turned off
(``providers.<name>.enabled: false``). That route cannot run, and it must not be the reason a
model switch fails: the session has to be able to leave the route it is not allowed to use --
the same trap ``test_a_route_that_cannot_authenticate_does_not_block_switching_away`` covers for
dead credentials.

``ProviderDisabledError`` is a ``ValueError``, so every recovery that already tolerates an
unresolvable route (the stopped local model server it also covers) keeps working unchanged.
"""
from contextlib import nullcontext

import pytest


class _Old:
    """The stored policy of a session frozen on a provider that is disabled now."""

    def __init__(self):
        self.model = 'm-model'
        self._config = {'model': {'provider': 'provx', 'default': 'm-model',
                                  'base_url': 'https://example.invalid/v1'}}

    def config(self, authority, *, secrets=True):
        return self._config

    def current_model(self, authority):
        return self.model


class _Runner:
    @staticmethod
    def _resolve_session_agent_runtime(**_kwargs):
        from hermes_cli.runtime_provider import ProviderDisabledError
        raise ProviderDisabledError("provider 'provx' is disabled in config "
                                    "(providers.provx.enabled: false)")


class _Authority:
    runner = _Runner()
    profile_id = 'profile'


def test_a_disabled_provider_does_not_block_switching_away(monkeypatch):
    """The route being replaced cannot run; the switch must still resolve it and move on."""
    from types import SimpleNamespace

    from gateway import session_policy
    from gateway.session_mutation_model import _current_route

    monkeypatch.setattr(session_policy, 'profile_read_scope', lambda *a, **k: nullcontext())
    live = SimpleNamespace(source=SimpleNamespace(chat_id='chat'), route='route')

    config, route = _current_route(_Authority(), live, _Old())

    assert config['model']['provider'] == 'provx'
    assert (route.provider, route.model) == ('provx', 'm-model')
    assert route.api_key == ''


def test_a_disabled_provider_still_refuses_to_run(monkeypatch):
    """The turn path keeps refusing a disabled provider loudly: no silent reroute."""
    from hermes_cli import runtime_provider

    monkeypatch.setattr(runtime_provider, 'load_config',
                        lambda: {'providers': {'provx': {'enabled': False}}})

    with pytest.raises(runtime_provider.ProviderDisabledError) as excinfo:
        runtime_provider.resolve_runtime_provider(requested='provx')

    assert isinstance(excinfo.value, ValueError)
    assert 'provx' in str(excinfo.value)
