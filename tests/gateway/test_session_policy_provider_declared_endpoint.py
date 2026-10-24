"""A launch ``base_url`` the picked provider itself declares passes the http(s) guard.

A provider's own transport can be a local sentinel rather than a URL to dial
(``moa://local`` for the MoA aggregator, ``acp://copilot`` for the ACP bridge,
``process://...`` for an external-process plugin). A session that ran on one persists
that endpoint as its route, and resuming that session hands the stored value back as
a launch override -- so the http(s)-only guard would refuse a route the runtime
itself accepts, and the store could not round-trip its own state (#142779).

The guard still refuses every endpoint that carries credentials, a query or a
fragment, and every endpoint a provider does not itself declare.
"""
import json

import pytest

from gateway.session_local_migration import legacy_launch_params
from gateway.session_policy import _apply_launch_overrides, build_policy
from hermes_cli.providers import ProviderDef
from hermes_state_runtime import RuntimeStoreError

CONFIG = {'model': {'provider': 'moa', 'base_url': 'https://example.invalid/v1', 'default': 'm'}}


def _apply(provider, base_url):
    params = {'base_url': base_url}
    if provider is not None:
        params['provider'] = provider
    _apply_launch_overrides(params, json.loads(json.dumps(CONFIG)))
    return True


def _refused(provider, base_url):
    params = {'base_url': base_url}
    if provider is not None:
        params['provider'] = provider
    with pytest.raises(RuntimeStoreError) as excinfo:
        _apply_launch_overrides(params, json.loads(json.dumps(CONFIG)))
    assert getattr(excinfo.value, 'reason', None) == 'invalid_params'
    assert getattr(excinfo.value, 'field', None) == 'base_url'
    return True


# -- the endpoint the provider itself declares ---------------------------------

def test_overlay_facade_endpoint_is_accepted():
    assert _apply('moa', 'moa://local')


def test_overlay_facade_endpoint_is_accepted_case_insensitively():
    assert _apply('moa', 'MOA://LOCAL')


def test_overlay_facade_endpoint_is_accepted_with_a_trailing_slash():
    assert _apply('moa', 'moa://local/')


def test_second_overlay_facade_endpoint_is_accepted():
    assert _apply('copilot-acp', 'acp://copilot')


def test_a_plain_http_endpoint_is_still_accepted():
    assert _apply('moa', 'https://example.invalid/v1')


# -- a facade is a facade only for the provider that declares it ---------------

def test_mismatched_provider_is_refused():
    assert _refused('openai', 'moa://local')


def test_a_custom_provider_does_not_borrow_the_overlay_facade():
    assert _refused('custom:metapi', 'moa://local')


def test_a_custom_provider_named_like_the_overlay_does_not_borrow_it():
    assert _refused('custom:moa', 'moa://local')


def test_a_facade_without_a_provider_is_refused():
    assert _refused(None, 'moa://local')


def test_another_host_on_the_same_scheme_is_refused():
    assert _refused('moa', 'moa://other')


# -- the guard keeps refusing what it exists for -------------------------------

def test_a_facade_with_a_path_is_refused():
    assert _refused('moa', 'moa://local/anything')


def test_a_facade_with_a_query_is_refused():
    assert _refused('moa', 'moa://local?x=1')


def test_a_facade_with_a_fragment_is_refused():
    assert _refused('moa', 'moa://local#f')


def test_a_facade_with_credentials_is_refused():
    assert _refused('moa', 'moa://u:p@local')


def test_an_http_endpoint_with_credentials_is_refused():
    assert _refused('moa', 'https://u:p@example.invalid/v1')


def test_an_http_endpoint_with_a_query_is_refused():
    assert _refused('moa', 'https://example.invalid/v1?x=1')


def test_an_http_endpoint_with_a_fragment_is_refused():
    assert _refused('moa', 'https://example.invalid/v1#f')


def test_an_empty_endpoint_is_refused():
    assert _refused('moa', '')


# -- a plugin profile's own endpoint (the #142779 shape) -----------------------

def _fake_plugin_provider(name):
    def _get_provider(canonical, *, allow_network=True):
        if canonical != 'fake-process-provider':
            return None
        return ProviderDef(id='fake-process-provider', name='fake-process-provider',
                           transport='external_process', api_key_env_vars=(),
                           base_url='process://fake-process-provider',
                           auth_type='external_process', source='plugin-profile')
    return _get_provider


def test_a_plugin_profiles_declared_endpoint_is_accepted(monkeypatch):
    monkeypatch.setattr('hermes_cli.providers.get_provider', _fake_plugin_provider('fake-process-provider'))
    assert _apply('fake-process-provider', 'process://fake-process-provider')


def test_a_plugin_profiles_endpoint_is_refused_for_another_provider(monkeypatch):
    monkeypatch.setattr('hermes_cli.providers.get_provider', _fake_plugin_provider('fake-process-provider'))
    assert _refused('moa', 'process://fake-process-provider')


def test_an_unresolvable_provider_keeps_the_refusal():
    # The plugin is gone: it can no longer declare its endpoint, so the resume is
    # refused rather than rerouted somewhere else.
    assert _refused('fake-process-provider', 'process://fake-process-provider')


# -- the end-to-end resume shape -----------------------------------------------

def test_resuming_a_stored_moa_row_builds_a_policy(tmp_path):
    row = {
        'id': '20261020_112735_2274a8', 'source': 'desktop', 'model': 'daily',
        'model_config': json.dumps({'model': 'daily', 'provider': 'moa',
                                    'base_url': 'moa://local',
                                    'api_mode': 'chat_completions'}),
    }
    params, _api_mode = legacy_launch_params(row)
    assert params['base_url'] == 'moa://local'
    policy = build_policy(params, json.loads(json.dumps(CONFIG)), private_secrets={})
    assert json.loads(policy.config_json)['model']['base_url'] == 'moa://local'
