"""A profile config whose ``gateway:`` is a status scalar must not crash the managed-worker gate.

``SessionPolicy.config()`` hands back the frozen profile config verbatim (``present_sections`` only
drops ``None``), and a profile config carries its own status line ``gateway: running`` under the same
top-level key the managed-worker opt-in reads. ``managed_policy`` assumed a mapping there, so every
dispatch answered ``AttributeError: 'str' object has no attribute 'get'`` before any turn could run.
"""

from types import SimpleNamespace

import pytest


def _policy(tmp_path, gateway):
    from gateway.session_policy import build_policy
    return build_policy({'cwd': str(tmp_path), 'model': 'm', 'provider': 'custom',
                         'base_url': 'http://127.0.0.1:9/v1'},
                        {'gateway': gateway})


def _managed_policy(policy, monkeypatch):
    from gateway import session_managed_worker as smw
    monkeypatch.setattr('gateway.session_policy.policy_for_source', lambda runner, source: policy)
    live = SimpleNamespace(source=SimpleNamespace(user_id='u', chat_id='c'), route='route')
    authority = SimpleNamespace(sessions={'sid': live}, runner=None, profile_id='p')
    return smw.managed_policy(authority, SimpleNamespace(session_id='sid'))


@pytest.mark.parametrize('gateway', ['running', 'stopped', 0, [], ['running']])
def test_a_non_mapping_gateway_section_is_not_an_opt_in(tmp_path, monkeypatch, gateway):
    assert _managed_policy(_policy(tmp_path, gateway), monkeypatch) is None


def test_the_nested_opt_in_still_binds_the_policy(tmp_path, monkeypatch):
    policy = _policy(tmp_path, {'managed_workers': True})
    assert _managed_policy(policy, monkeypatch) is policy
