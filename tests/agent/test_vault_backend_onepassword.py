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


def test_legacy_handle_recovers_vault_and_keeps_all_saved_origins():
    with patch("agent.secret_scope.get_secret", return_value="service-token"):
        backend = OnePasswordLoginBackend()

    listed = json.dumps([{
        "id": "item-id", "title": "Example", "vault": {"id": "vault-id"},
        "urls": [{"href": "https://example.com/login"},
                 {"href": "https://other.example/login"}],
    }])
    backend._run = Mock(side_effect=[listed, listed, "password\n", listed, "123456\n"])

    meta = backend.get_meta("op:item-id")
    assert meta is not None
    assert meta.id == "op:vault-id:item-id"
    assert meta.allowed_origins == ("https://example.com", "https://other.example")
    assert backend.resolve_password("op:item-id") == "password"
    assert backend.resolve_otp("op:item-id") == "123456"
    reads = [call.args for call in backend._run.call_args_list if call.args[:2] == ("item", "get")]
    assert reads == [
        ("item", "get", "item-id", "--vault", "vault-id", "--fields", "label=password", "--reveal"),
        ("item", "get", "item-id", "--vault", "vault-id", "--otp"),
    ]