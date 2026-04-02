"""Tests for cred_proxy.substitutor placeholder substitution."""

import pytest

from cred_proxy.store import CredStore
from cred_proxy.substitutor import substitute


@pytest.fixture
def store() -> CredStore:
    s = CredStore()
    s.set("mytoken", "real-token-value")
    s.set("apikey", "sk-12345")
    return s


def test_substitute_in_header_value(store: CredStore) -> None:
    result = substitute("Bearer hermes-proxy://mytoken", store)
    assert result == "Bearer real-token-value"


def test_substitute_in_json_body(store: CredStore) -> None:
    result = substitute('{"api_key": "hermes-proxy://apikey"}', store)
    assert result == '{"api_key": "sk-12345"}'


def test_unknown_name_left_unchanged(store: CredStore) -> None:
    result = substitute("hermes-proxy://does-not-exist", store)
    assert result == "hermes-proxy://does-not-exist"


def test_non_placeholder_string_unchanged(store: CredStore) -> None:
    original = "Authorization: Basic dXNlcjpwYXNz"
    assert substitute(original, store) == original


def test_multiple_placeholders_in_one_string(store: CredStore) -> None:
    result = substitute(
        "Bearer hermes-proxy://mytoken key=hermes-proxy://apikey",
        store,
    )
    assert result == "Bearer real-token-value key=sk-12345"


def test_substitute_in_query_string(store: CredStore) -> None:
    result = substitute("?token=hermes-proxy://mytoken&other=value", store)
    assert result == "?token=real-token-value&other=value"


def test_mixed_known_and_unknown(store: CredStore) -> None:
    result = substitute(
        "hermes-proxy://mytoken and hermes-proxy://missing",
        store,
    )
    assert result == "real-token-value and hermes-proxy://missing"


def test_empty_string(store: CredStore) -> None:
    assert substitute("", store) == ""
