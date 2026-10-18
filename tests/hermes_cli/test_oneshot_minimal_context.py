from types import SimpleNamespace

import pytest

from hermes_cli.gateway_chat import GatewayClientError, _requested_policy


def test_toolsets_none_requests_an_explicit_toolless_oneshot():
    args = SimpleNamespace(oneshot="hello", toolsets="none", ignore_rules=True)

    assert _requested_policy(args) == {"toolsets": [], "ignore_rules": True}


def test_toolsets_none_is_exclusive_to_oneshot():
    chat_args = SimpleNamespace(oneshot=None, toolsets="none")
    mixed_args = SimpleNamespace(oneshot="hello", toolsets="none,terminal")

    with pytest.raises(GatewayClientError, match="only with -z/--oneshot"):
        _requested_policy(chat_args)
    with pytest.raises(GatewayClientError, match="cannot be combined"):
        _requested_policy(mixed_args)
