"""1Password browser-vault backend command contracts."""

from __future__ import annotations

import json
from unittest.mock import Mock, patch

from agent.vault_backends.onepassword import OnePasswordLoginBackend


def test_service_account_item_reads_keep_vault_in_opaque_handle():
    with patch("agent.secret_scope.get_secret", return_value="service-token"):
        backend = OnePasswordLoginBackend()

    backend._run = Mock(side_effect=[
        json.dumps([{
            "id": "item-id",
            "title": "Example",
            "vault": {"id": "vault-id", "name": "Private"},
            "urls": [{"href": "https://example.com/login"}],
        }]),
        "password\n",
        "123456\n",
    ])

    [item] = backend.list_items()
    assert item.id == "op:vault-id:item-id"
    assert "Private" not in item.id
    assert backend.resolve_password(item.id) == "password"
    assert backend.resolve_otp(item.id) == "123456"
    assert backend._run.call_args_list[1].args == (
        "item", "get", "item-id", "--vault", "vault-id",
        "--fields", "label=password", "--reveal",
    )
    assert backend._run.call_args_list[2].args == (
        "item", "get", "item-id", "--vault", "vault-id", "--otp",
    )