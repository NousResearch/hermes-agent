"""`_apply_launch_overrides`'s `base_url` validator stays strict http(s)-only.

Context (2026-11-02): a same-day production hotfix widened this validator's accepted
schemes to include ``process`` to unblock `bot_relay.deliver` refusals on external-process
providers (claude-subscription-directsdk-experimental and siblings), whose routes are
represented internally as a ``process://<provider>`` pseudo-URL.

That approach was considered and explicitly rejected by the actual upstream fix (PR #146658,
`fix(gateway): an adopted chat on an external-process provider drops the provider's process://
address`, cherry-picked onto this branch): "Let `_apply_launch_overrides` accept `process://`.
Rejected. That validator guards client-supplied launch overrides, and an external-process
address is not an endpoint a client can select. Relaxing it would let any `session.create` pin
a `process://` URL on any provider, http or not, just to carry a value the runtime ignores."

The real fix instead drops the saved pseudo-URL in `legacy_launch_params`
(`gateway/session_local_migration.py`) before it ever reaches this validator — see
`tests/gateway/test_legacy_external_process_route.py`. This test pins the validator's half of
that contract: it must keep refusing every non-http(s) scheme, including `process://`, so a
future change does not silently reintroduce the rejected widening.
"""
import pytest

from gateway.session_policy import _apply_launch_overrides
from hermes_state_runtime import RuntimeStoreError


@pytest.mark.parametrize('base_url', [
    'process://claude-subscription-directsdk-experimental',
    'acp://copilot',
    'ftp://example.com/v1',
    'http://user:pass@example.com/v1',   # credentials
    'https://example.com/v1?key=abc',    # query
    'https://example.com/v1#frag',       # fragment
])
def test_rejects_non_conforming_base_url(base_url):
    with pytest.raises(RuntimeStoreError) as exc:
        _apply_launch_overrides({'base_url': base_url}, {})
    assert exc.value.field == 'base_url'
    assert exc.value.reason == 'invalid_params'


@pytest.mark.parametrize('base_url', [
    'https://api.example.com/v1',
    'http://127.0.0.1:8080/v1',
])
def test_accepts_plain_http_s_base_url(base_url):
    config = {}
    _apply_launch_overrides({'base_url': base_url}, config)
    assert config['model']['base_url'] == base_url
