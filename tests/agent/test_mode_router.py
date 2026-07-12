"""Contract tests for the code-defined automatic mode router modes."""

from dataclasses import FrozenInstanceError

import pytest

from agent.mode_router import MODE_DEFINITIONS, ModeDefinition, UnknownModeError, get_mode


EXPECTED_MODES = {
    "thinking-expansion",
    "research-analysis",
    "execution-development",
}


def test_only_the_three_supported_modes_are_defined():
    assert set(MODE_DEFINITIONS) == EXPECTED_MODES
    assert "verification" not in MODE_DEFINITIONS


def test_every_mode_has_an_explicit_immutable_contract_with_verification():
    for name, mode in MODE_DEFINITIONS.items():
        assert isinstance(mode, ModeDefinition)
        assert mode.name == name
        assert mode.objective
        assert mode.stages
        assert mode.verification is True
        assert isinstance(mode.stages, tuple)

        with pytest.raises(FrozenInstanceError):
            mode.objective = "changed"


def test_mode_registry_is_immutable():
    with pytest.raises(TypeError):
        MODE_DEFINITIONS["custom"] = MODE_DEFINITIONS["thinking-expansion"]


def test_get_mode_returns_the_code_defined_contract():
    assert get_mode("research-analysis") is MODE_DEFINITIONS["research-analysis"]


def test_unknown_mode_fails_closed():
    with pytest.raises(UnknownModeError, match="unknown mode"):
        get_mode("custom-user-mode")
