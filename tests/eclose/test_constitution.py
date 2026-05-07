import pytest
from eclose.constitution.constraints import ConstitutionalConstraints

def test_constitution_initialization():
    constraints = ConstitutionalConstraints()
    assert constraints is not None

def test_benevolence_constraint():
    constraints = ConstitutionalConstraints()
    # Test that harmful requests are blocked
    harmful_action = {"type": "weapon_design"}
    assert not constraints.is_allowed(harmful_action)

def test_allow_good_action():
    constraints = ConstitutionalConstraints()
    good_action = {"type": "code_generation"}
    assert constraints.is_allowed(good_action)
