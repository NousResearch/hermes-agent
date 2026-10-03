import importlib
import sys

import pytest


def _reload_common():
    sys.modules.pop("hermes_state_common", None)
    return importlib.import_module("hermes_state_common")

def test_missing_joint_in_stale_module_uses_byte_identical_fallback(monkeypatch):
    import agent.skill_commands as skills
    monkeypatch.delattr(skills, "SKILL_EXCERPT_JOINT")
    common = _reload_common()
    assert common.SKILL_EXCERPT_JOINT == "\x1e"
    assert common._shape_preview("head\x1etail") == "head"

def test_missing_required_skill_dependency_still_fails(monkeypatch):
    import agent.skill_commands as skills
    monkeypatch.delattr(skills, "SKILL_SCAFFOLD_SQL_LIKE")
    with pytest.raises(ImportError, match="SKILL_SCAFFOLD_SQL_LIKE"):
        _reload_common()
