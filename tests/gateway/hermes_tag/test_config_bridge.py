from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import pytest

from gateway.hermes_tag import (
    ConfigurationError,
    ContinuityMode,
    HermesTagConfig,
    IncompleteScope,
    Sensitivity,
    database_path,
    identity_from_session_source,
    profile_state_directory,
    surface_from_session_source,
)


def test_config_defaults_are_additive_and_shadow_only():
    config = HermesTagConfig.from_mapping(None)
    assert config.enabled is False
    assert config.shadow is True
    assert config.context.enabled is False
    assert config.continuity.mode is ContinuityMode.ISOLATED
    assert config.leases.signing_secret_ref is None


@pytest.mark.parametrize("key", ["signing_secret", "hmac_key", "secret"])
def test_config_rejects_inline_top_level_signing_material(key):
    with pytest.raises(ConfigurationError, match="reference"):
        HermesTagConfig.from_mapping({key: "plaintext"})


@pytest.mark.parametrize("key", ["signing_secret", "hmac_key", "secret"])
def test_config_rejects_inline_lease_signing_material(key):
    with pytest.raises(ConfigurationError, match="signing_secret_ref"):
        HermesTagConfig.from_mapping({"leases": {key: "plaintext"}})


@pytest.mark.parametrize(
    "raw",
    [
        {"enabeld": True},
        {"context": {"max_factz": 10}},
        {"continuity": {"mdoe": "project"}},
        {"leases": {"ttl_secondz": 30}},
        {"budgets": {"daily_tokenz": 100}},
    ],
)
def test_config_rejects_unknown_fields(raw):
    with pytest.raises(ConfigurationError, match="unknown fields"):
        HermesTagConfig.from_mapping(raw)


def test_config_parses_nested_controls():
    config = HermesTagConfig.from_mapping(
        {
            "enabled": True,
            "shadow": False,
            "allow_guests": False,
            "database_filename": "tag-state.db",
            "context": {
                "enabled": True,
                "max_chars": 8000,
                "max_facts": 20,
                "sensitivity_ceiling": "confidential",
            },
            "continuity": {"mode": "project", "max_hops": 7},
            "leases": {
                "ttl_seconds": 90,
                "clock_skew_seconds": 2,
                "signing_secret_ref": "vault://hermes/tag",
            },
            "budgets": {
                "hourly_tokens": 1000,
                "daily_tokens": 5000,
                "hourly_cost_usd": 1.25,
                "daily_cost_usd": 4,
            },
        }
    )
    assert config.enabled is True
    assert config.shadow is False
    assert config.allow_guests is False
    assert config.context.sensitivity_ceiling is Sensitivity.CONFIDENTIAL
    assert config.continuity.mode is ContinuityMode.PROJECT
    assert config.leases.signing_secret_ref == "vault://hermes/tag"
    assert config.budgets.daily_tokens == 5000


@pytest.mark.parametrize(
    "raw",
    [
        {"context": {"max_facts": 0}},
        {"context": {"max_chars": 10}},
        {"continuity": {"max_hops": 100}},
        {"leases": {"ttl_seconds": 1}},
        {"budgets": {"daily_tokens": -1}},
        {"enabled": "yes"},
    ],
)
def test_config_rejects_bad_bounds_or_types(raw):
    with pytest.raises(ConfigurationError):
        HermesTagConfig.from_mapping(raw)


@pytest.mark.parametrize("filename", ["../x.db", "dir/x.db", "dir\\x.db", ".", ".."])
def test_config_database_filename_must_be_leaf(filename):
    with pytest.raises(ConfigurationError, match="filename"):
        HermesTagConfig.from_mapping({"database_filename": filename})


def test_profile_state_directory_matches_hermes_layout(tmp_path: Path):
    assert profile_state_directory(tmp_path, "default") == tmp_path.resolve()
    assert profile_state_directory(tmp_path, "worker") == (
        tmp_path / "profiles" / "worker"
    ).resolve()


@pytest.mark.parametrize("profile", ["", "../x", "a/b", "a\\b", ".."])
def test_profile_state_directory_rejects_ambiguous_profile(tmp_path, profile):
    with pytest.raises(ConfigurationError):
        profile_state_directory(tmp_path, profile)


def test_profile_state_directory_rejects_symlink_escape(tmp_path):
    profiles = tmp_path / "profiles"
    profiles.mkdir()
    outside = tmp_path.parent / f"{tmp_path.name}-outside"
    outside.mkdir()
    link = profiles / "worker"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("directory symlinks are unavailable")
    with pytest.raises(ConfigurationError, match="escapes"):
        profile_state_directory(tmp_path, "worker")


def test_database_path_is_profile_local(tmp_path):
    config = HermesTagConfig.from_mapping({"database_filename": "tag.db"})
    assert database_path(tmp_path, "worker", config) == (
        tmp_path / "profiles" / "worker" / "tag.db"
    ).resolve()


class Platform(Enum):
    SLACK = "slack"


@dataclass
class SourceObject:
    platform: Platform = Platform.SLACK
    profile: str = "default"
    team_id: str = "T1"
    channel_id: str = "C1"
    thread_ts: str = "123.4"
    user_id: str = "U1"
    display_name: str = "Axl"


def test_bridge_normalizes_attribute_source():
    source = SourceObject()
    surface = surface_from_session_source(source)
    identity = identity_from_session_source(source, surface=surface)
    assert surface.platform == "slack"
    assert surface.scope_id == "T1"
    assert surface.chat_id == "C1"
    assert surface.thread_id == "123.4"
    assert identity.external_id == "U1"
    assert identity.display_name == "Axl"


def test_bridge_normalizes_mapping_aliases():
    source = {
        "source": "discord",
        "profile": "guild-bot",
        "guild_id": "G1",
        "room_id": "C1",
        "topic_id": "TH1",
        "actor_id": "USER1",
    }
    surface = surface_from_session_source(source)
    identity = identity_from_session_source(source, surface=surface)
    assert (surface.platform, surface.profile, surface.scope_id) == (
        "discord",
        "guild-bot",
        "G1",
    )
    assert surface.thread_id == "TH1"
    assert identity.external_id == "USER1"


def test_bridge_profile_override_is_authoritative():
    source = {
        "platform": "slack",
        "profile": "wire-value",
        "workspace_id": "T1",
        "chat_id": "C1",
    }
    assert surface_from_session_source(source, profile="trusted").profile == "trusted"


@pytest.mark.parametrize(
    "missing",
    [
        {"platform": "slack", "profile": "default", "chat_id": "C1"},
        {"platform": "slack", "profile": "default", "team_id": "T1"},
        {"profile": "default", "team_id": "T1", "chat_id": "C1"},
    ],
)
def test_bridge_fails_closed_on_incomplete_surface(missing):
    with pytest.raises(IncompleteScope):
        surface_from_session_source(missing)


def test_identity_bridge_requires_authenticated_actor():
    source = {
        "platform": "slack",
        "profile": "default",
        "team_id": "T1",
        "chat_id": "C1",
    }
    surface = surface_from_session_source(source)
    with pytest.raises(IncompleteScope, match="actor"):
        identity_from_session_source(source, surface=surface)
