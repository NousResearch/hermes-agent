"""Contract tests for the kanban agent-profile registry (slice 5)."""

from __future__ import annotations

import dataclasses
import json
from types import MappingProxyType

import pytest

from hermes_cli import kanban_profiles as kp
from hermes_cli.kanban_profiles import (
    LANE_PROFILE_NAMES,
    PROFILES,
    AgentProfile,
    get_profile,
    registry_hash,
    validate_registry,
)


def test_registry_is_valid_and_complete():
    validate_registry()  # must not raise
    assert len(PROFILES) == 7
    assert set(LANE_PROFILE_NAMES) == {
        "planning",
        "research",
        "coding",
        "review",
        "security",
        "ops",
    }


def test_lane_profiles_are_distinct_and_not_orchestrator():
    lane_names = list(LANE_PROFILE_NAMES.values())
    assert len(set(lane_names)) == 6
    orchestrator = [n for n in PROFILES if PROFILES[n].role == "orchestrator"]
    assert len(orchestrator) == 1
    for name in lane_names:
        assert PROFILES[name].role != "orchestrator"


def test_registry_hash_is_stable_and_sensitive():
    h1 = registry_hash()
    h2 = registry_hash()
    assert h1 == h2
    assert len(h1) == 64
    # Any mutation to the registry content must change the hash or break
    # validation.
    original = dict(PROFILES["kanban-coding"].toolsets)
    try:
        # (a) Content drift: remove the terminal toolset while
        # may_execute_terminal stays True — a real invariant violation.
        object.__setattr__(
            PROFILES["kanban-coding"],
            "toolsets",
            MappingProxyType({t: None for t in original if t != "terminal"}),
        )
        with pytest.raises(ValueError):
            validate_registry()
        assert registry_hash() != h1
        # (b) Tampering with a mutable replacement mapping must fail
        # validation even when content still matches.
        object.__setattr__(
            PROFILES["kanban-coding"],
            "toolsets",
            dict(original),
        )
        with pytest.raises(ValueError):
            validate_registry()
        # (c) registry_version drift must change the hash.
        object.__setattr__(
            PROFILES["kanban-coding"],
            "registry_version",
            2,
        )
        object.__setattr__(
            PROFILES["kanban-coding"],
            "toolsets",
            MappingProxyType(original),
        )
        with pytest.raises(ValueError):
            validate_registry()
        assert registry_hash() != h1
    finally:
        object.__setattr__(PROFILES["kanban-coding"], "registry_version", 1)
        object.__setattr__(
            PROFILES["kanban-coding"],
            "toolsets",
            MappingProxyType(original),
        )
    validate_registry()
    assert registry_hash() == h1


def test_capability_boundaries():
    orch = get_profile("kanban-orchestrator")
    assert not orch.capabilities["may_execute_terminal"]
    assert not orch.capabilities["may_edit_code"]
    assert orch.capabilities["may_delegate"]

    coding = get_profile("kanban-coding")
    assert coding.capabilities["may_edit_code"]
    assert coding.capabilities["may_execute_terminal"]
    assert not coding.capabilities["may_approve_external_effects"]
    assert "terminal" in coding.allowed_toolsets

    ops = get_profile("kanban-ops")
    assert ops.capabilities["may_approve_external_effects"]
    assert "ops/deploy" in ops.allowed_credential_scopes

    review = get_profile("kanban-review")
    security = get_profile("kanban-security")
    for p in (review, security):
        assert "terminal" not in p.allowed_toolsets
        assert not p.capabilities["may_edit_code"]


def test_registry_immutable():
    with pytest.raises(TypeError):
        PROFILES["kanban-coding"].toolsets["browser"] = None  # type: ignore[index]
    with pytest.raises(TypeError):
        PROFILES["new-profile"] = PROFILES["kanban-coding"]  # type: ignore[index]
    profile = get_profile("kanban-coding")
    with pytest.raises(dataclasses.FrozenInstanceError):
        profile.role = "ops"  # type: ignore[misc]


def test_description_and_version_drift_rejected():
    original_desc = PROFILES["kanban-review"].description
    object.__setattr__(
        PROFILES["kanban-review"], "description", "A different valid-length description."
    )
    try:
        with pytest.raises(ValueError):
            validate_registry()
    finally:
        object.__setattr__(PROFILES["kanban-review"], "description", original_desc)
    # True / 1.0 compare equal to 1 but are not exact ints.
    for bad_version in (True, 1.0):
        object.__setattr__(PROFILES["kanban-review"], "registry_version", bad_version)
        try:
            with pytest.raises(ValueError):
                validate_registry()
        finally:
            object.__setattr__(PROFILES["kanban-review"], "registry_version", 1)
    validate_registry()


def test_expected_spec_is_deep_frozen():
    with pytest.raises(TypeError):
        kp._EXPECTED_CAPABILITIES["ops"]["may_delegate"] = True  # type: ignore[index]


def test_get_profile_fails_closed_on_unknown():
    with pytest.raises(KeyError):
        get_profile("does-not-exist")


def test_profile_shape():
    for name, p in PROFILES.items():
        assert isinstance(p, AgentProfile)
        assert p.name == name
        assert p.description and len(p.description) <= 400
        assert p.allowed_toolsets  # never empty
        assert "kanban/board" in p.allowed_credential_scopes
        assert p.registry_version == 1


def test_registry_json_serializable_for_evidence():
    payload = {
        "hash": registry_hash(),
        "profiles": sorted(PROFILES),
    }
    assert json.loads(json.dumps(payload)) == payload
