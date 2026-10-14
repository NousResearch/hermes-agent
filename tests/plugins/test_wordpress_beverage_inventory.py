"""Test the authenticated, fixed-endpoint DanceOffice inventory reader."""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import Mock

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from plugins.wordpress_beverage_inventory import (
    BeverageInventoryError,
    ENDPOINT,
    fetch_latest_inventory,
)


def test_success_uses_fixed_https_endpoint_and_basic_auth():
    client = Mock()
    client.get.return_value = httpx.Response(200, json={"inventory": {"id": "fixture"}})
    data = fetch_latest_inventory("agent-user", "abcd efgh", client=client)
    assert data == {"inventory": {"id": "fixture"}}
    args, kwargs = client.get.call_args
    assert args == (ENDPOINT,)
    assert ENDPOINT.startswith("https://")
    assert kwargs["auth"] == ("agent-user", "abcd efgh")
    assert kwargs["timeout"] == 15
    assert kwargs["follow_redirects"] is False
    assert kwargs["headers"]["Accept"] == "application/json"


@pytest.mark.parametrize(
    ("status", "message"),
    [
        (401, r"authentication failed \(401\)"),
        (403, r"denied the inventory capability \(403\)"),
        (404, r"endpoint was not found \(404\)"),
        (500, r"server error \(500\)"),
        (503, r"server error \(503\)"),
        (302, r"request failed \(302\)"),
    ],
)
def test_http_errors_do_not_echo_response(status, message):
    secret = "DO-NOT-LEAK-THIS-PASSWORD"
    client = Mock()
    client.get.return_value = httpx.Response(status, json={"message": secret})
    with pytest.raises(BeverageInventoryError, match=message) as err:
        fetch_latest_inventory("u", secret, client=client)
    assert secret not in str(err.value)


def test_timeout_does_not_echo_transport_details():
    client = Mock()
    client.get.side_effect = httpx.ReadTimeout("SECRET-PASSWORD")
    with pytest.raises(BeverageInventoryError, match="timed out after 15s") as err:
        fetch_latest_inventory("u", "SECRET-PASSWORD", client=client)
    assert "SECRET-PASSWORD" not in str(err.value)


def test_network_error_does_not_echo_transport_details():
    client = Mock()
    client.get.side_effect = httpx.ConnectError("SECRET-PASSWORD")
    with pytest.raises(BeverageInventoryError, match="network error") as err:
        fetch_latest_inventory("u", "SECRET-PASSWORD", client=client)
    assert "SECRET-PASSWORD" not in str(err.value)


@pytest.mark.parametrize("content", [b"not-json", b"\xff"])
def test_invalid_json_is_reported(content):
    client = Mock()
    client.get.return_value = httpx.Response(200, content=content)
    with pytest.raises(BeverageInventoryError, match="invalid JSON"):
        fetch_latest_inventory("u", "p", client=client)


@pytest.mark.parametrize("content", [b"[]", b"null", b"\"text\""])
def test_non_object_json_is_rejected(content):
    client = Mock()
    client.get.return_value = httpx.Response(200, content=content)
    with pytest.raises(BeverageInventoryError, match="expected an object"):
        fetch_latest_inventory("u", "p", client=client)


@pytest.mark.parametrize(("username", "password"), [("", "p"), ("u", ""), (" ", "p")])
def test_missing_credentials_fail_before_network(username, password):
    client = Mock()
    with pytest.raises(BeverageInventoryError, match="HERMES_WP_API_USER"):
        fetch_latest_inventory(username, password, client=client)
    client.get.assert_not_called()
