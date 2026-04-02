"""Integration tests: mitmproxy addon substitutes credentials in flows."""

from unittest.mock import MagicMock

import pytest

from cred_proxy.server import CredentialProxyAddon
from cred_proxy.store import CredStore


@pytest.fixture
def store() -> CredStore:
    return CredStore()


def _make_flow(headers: dict, body: bytes = b"") -> MagicMock:
    flow = MagicMock()
    flow.request.headers = dict(headers)
    flow.request.content = body
    return flow


def test_addon_request_substitutes_header(store: CredStore) -> None:
    """Addon replaces a hermes-proxy:// placeholder in a request header."""
    store.set("mytoken", "real-secret-value")
    addon = CredentialProxyAddon(store)

    flow = _make_flow({"Authorization": "Bearer hermes-proxy://mytoken"})
    addon.request(flow)

    assert flow.request.headers["Authorization"] == "Bearer real-secret-value"


def test_addon_request_substitutes_body(store: CredStore) -> None:
    """Addon replaces a hermes-proxy:// placeholder in the request body."""
    store.set("tok", "real-value")
    addon = CredentialProxyAddon(store)

    flow = _make_flow({}, body=b"secret=hermes-proxy://tok")
    addon.request(flow)

    assert flow.request.content == b"secret=real-value"


def test_addon_request_leaves_unknown_placeholder(store: CredStore) -> None:
    """Unknown credential names are left unchanged in headers."""
    addon = CredentialProxyAddon(store)

    flow = _make_flow({"Authorization": "Bearer hermes-proxy://unknown"})
    addon.request(flow)

    assert flow.request.headers["Authorization"] == "Bearer hermes-proxy://unknown"


def test_addon_request_no_substitution_needed(store: CredStore) -> None:
    """Addon leaves headers and body untouched when no placeholders are present."""
    addon = CredentialProxyAddon(store)

    flow = _make_flow({"Authorization": "Bearer plain-token"}, body=b"plain body")
    addon.request(flow)

    assert flow.request.headers["Authorization"] == "Bearer plain-token"
    assert flow.request.content == b"plain body"


def test_no_public_get_api(store: CredStore) -> None:
    """CredStore has no public method to retrieve stored credential values."""
    store.set("secret", "sensitive-value")

    public_methods = {
        m
        for m in dir(store)
        if not m.startswith("_") and callable(getattr(store, m))
    }
    assert public_methods == {"set", "list", "delete"}, (
        f"Unexpected public methods on CredStore: {public_methods - {'set', 'list', 'delete'}}"
    )
