"""The route can be stored in two columns; this covers the shape where it is stored
in ``billing_base_url`` alone and the reader defaults the provider to bare ``custom``.

``legacy_launch_params`` resurrects the per-turn usage stamp as a launch
``base_url`` when the row's route carries none, and defaults the provider to
``custom`` when the stamp names none. The launch params then pair a client-writable
provider with a non-http endpoint, which the base_url guard refuses -- and the
provider name no longer says which runtime declared the endpoint, so the
"the picked provider itself declares it" rule cannot match.

This is the residual shape of #142779 that a provider-declared-endpoint rule does
NOT cover: the row's own provider name is the only thing that could restore the
route, and it is gone by the time the endpoint is read back.
"""
import json

import pytest

from gateway.session_local_migration import legacy_launch_params
from gateway.session_policy import build_policy
from hermes_state_runtime import RuntimeStoreError

CONFIG = {'model': {'provider': 'moa', 'base_url': 'https://example.invalid/v1', 'default': 'm'}}


def _row(**overrides):
    row = {
        'id': 'gap-row', 'source': 'desktop', 'model': 'daily',
        'model_config': json.dumps({'model': 'daily', 'api_mode': 'chat_completions'}),
        'billing_base_url': 'moa://local', 'billing_provider': None,
    }
    row.update(overrides)
    return row


def test_the_stamp_alone_resurrects_the_endpoint_as_a_launch_pin():
    params, _api_mode = legacy_launch_params(_row())
    assert params['base_url'] == 'moa://local'
    # The reader cannot tell which runtime declared it, so it defaults the provider
    # to the bare client-writable class.
    assert params.get('provider') == 'custom'


def test_the_bare_custom_provider_still_refuses_the_stamp():
    """The residual shape: refused, because bare ``custom`` declares no endpoint."""
    params, _api_mode = legacy_launch_params(_row())
    with pytest.raises(RuntimeStoreError) as excinfo:
        build_policy(params, json.loads(json.dumps(CONFIG)), private_secrets={})
    assert getattr(excinfo.value, 'reason', None) == 'invalid_params'
    assert getattr(excinfo.value, 'field', None) == 'base_url'


def test_a_stamp_that_names_its_provider_is_accepted():
    """The same row, when it also records the provider that declared the endpoint."""
    params, _api_mode = legacy_launch_params(_row(billing_provider='moa'))
    assert params.get('provider') == 'moa'
    policy = build_policy(params, json.loads(json.dumps(CONFIG)), private_secrets={})
    assert json.loads(policy.config_json)['model']['base_url'] == 'moa://local'
