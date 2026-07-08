#!/usr/bin/env python3
"""A3 skill-hygiene guard E2E — proves G1/G2/G3 fire.

Proper pytest module (no env var, no import-time assertions): each test imports
the first-party module normally (pytest runs from the repo root, so these resolve
to the tree under test) and asserts the guard behavior.
"""
import importlib.util
import pathlib

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_g2_queue_note_name_and_path_exclusion():
    """G2: pending-* work-queue notes never register as loadable skills."""
    import agent.skill_utils as su
    assert su.is_queue_note_name("pending-shared-skill-patches") is True
    assert su.is_queue_note_name("pending-patch-foo") is True
    assert su.is_queue_note_name("coding-guardrails") is False
    # path-level: a pending-* skill dir's SKILL.md is excluded; a real skill is not
    assert su.is_excluded_skill_path(
        "/x/skills-shared/pending/pending-shared-skill-patches/SKILL.md") is True
    assert su.is_excluded_skill_path(
        "/x/skills-shared/coding/coding-guardrails/SKILL.md") is False


def test_g1_rejects_queue_note_named_create():
    """G1 (source guard): a create whose NAME is a reserved queue-note prefix is refused."""
    import tools.skill_manager_tool as smt
    valid = ("---\nname: pending-patch-evil\n"
             "description: a valid-looking description long enough to pass\n---\nbody")
    r = smt._create_skill("pending-patch-evil", valid, category="coding")
    assert r.get("success") is False
    assert "reserved" in r.get("error", "").lower(), r.get("error")


def test_g1_enforce_shared_placement_returns_bool():
    """G1: the placement-enforcement flag resolves to a bool (config or auto-detect)."""
    import tools.skill_manager_tool as smt
    assert isinstance(smt._enforce_shared_placement(), bool)


def test_g3_computed_officialness_api_present():
    """G3: the relocator exposes computed-officialness so it can't rmtree an official skill."""
    path = _REPO_ROOT / "scripts" / "local-skill-leak-check.py"
    spec = importlib.util.spec_from_file_location("leakcheck", path)
    lc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(lc)
    assert hasattr(lc, "is_official_skill")
    assert hasattr(lc, "load_allowlist")
