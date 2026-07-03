"""Tests for the static gateway config ownership manifest.

These tests intentionally avoid load_gateway_config() and real profile state.
They compare static source/default contracts only.
"""

from dataclasses import fields

from gateway.config import GatewayConfig
from gateway.config_ownership_manifest import (
    DEFAULT_CONFIG_ROOT_OWNERS,
    GATEWAY_CONFIG_FIELD_OWNERS,
    GATEWAY_BRIDGED_CONFIG_PATH_OWNERS,
    KNOWN_OWNERS,
    NEEDS_OWNER_ALLOWLIST,
    owner_for_default_root,
    owner_for_gateway_config_field,
    unresolved_owner_entries,
)
from hermes_cli.config import DEFAULT_CONFIG


def test_default_config_roots_have_manifest_classification() -> None:
    missing = sorted(set(DEFAULT_CONFIG) - set(DEFAULT_CONFIG_ROOT_OWNERS))
    extra = sorted(set(DEFAULT_CONFIG_ROOT_OWNERS) - set(DEFAULT_CONFIG))

    assert missing == []
    assert extra == []
    assert all(owner in KNOWN_OWNERS for owner in DEFAULT_CONFIG_ROOT_OWNERS.values())


def test_gateway_config_fields_have_manifest_classification() -> None:
    gateway_fields = {field.name for field in fields(GatewayConfig)}

    missing = sorted(gateway_fields - set(GATEWAY_CONFIG_FIELD_OWNERS))
    extra = sorted(set(GATEWAY_CONFIG_FIELD_OWNERS) - gateway_fields)

    assert missing == []
    assert extra == []
    assert all(owner in KNOWN_OWNERS for owner in GATEWAY_CONFIG_FIELD_OWNERS.values())


def test_gateway_config_to_dict_keys_stay_manifest_aligned() -> None:
    # Constructing the default dataclass may resolve HERMES_HOME for the default
    # sessions_dir path, but it does not read config.yaml/state/credentials.
    round_trip_keys = set(GatewayConfig().to_dict())

    missing = sorted(round_trip_keys - set(GATEWAY_CONFIG_FIELD_OWNERS))
    assert missing == []

    # The manifest may classify fields that are represented by transformed
    # values in to_dict(), but every current GatewayConfig field is represented.
    assert round_trip_keys == {field.name for field in fields(GatewayConfig)}


def test_gateway_bridged_paths_are_gateway_owned_and_known() -> None:
    assert GATEWAY_BRIDGED_CONFIG_PATH_OWNERS
    assert all(
        owner == "gateway_config"
        for owner in GATEWAY_BRIDGED_CONFIG_PATH_OWNERS.values()
    )
    assert all(owner in KNOWN_OWNERS for owner in GATEWAY_BRIDGED_CONFIG_PATH_OWNERS.values())


def test_manifest_has_no_unallowlisted_needs_owner_entries() -> None:
    unresolved = unresolved_owner_entries()
    unallowlisted = sorted(set(unresolved) - set(NEEDS_OWNER_ALLOWLIST))

    assert unallowlisted == []
    assert NEEDS_OWNER_ALLOWLIST == {}


def test_lookup_helpers_return_needs_owner_for_unknown_surfaces() -> None:
    assert owner_for_default_root("not_a_default_root") == "needs_owner"
    assert owner_for_gateway_config_field("not_a_gateway_field") == "needs_owner"
