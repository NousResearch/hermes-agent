"""Kanban/session-preview import of ``SKILL_EXCERPT_JOINT`` must fail-open.

Kanban loads ``hermes_state`` → ``hermes_state_common``, whose module-level

    from agent.skill_commands import SKILL_EXCERPT_JOINT, ...

raises ``ImportError: cannot import name 'SKILL_EXCERPT_JOINT'`` when the
live ``agent.skill_commands`` module object does not export that name.

Two real mechanisms produce that exact error:

1. Stale ``sys.modules`` cache (same class as ``tests/test_stale_utils_module_import.py``):
   the process imported ``agent.skill_commands`` before ``SKILL_EXCERPT_JOINT``
   existed; disk may now define it, the in-memory module does not.
2. Version skew: the installed ``skill_commands.py`` is older than
   ``hermes_state_common.py`` and never exported the name.

On a healthy process the binding must be the exact live object (``"\\x1e"``).
When only the joint is missing, fall back to the byte-identical joint so
preview SQL / ``_shape_preview`` still work. Missing
``SKILL_SCAFFOLD_SQL_LIKE`` / ``describe_skill_invocation`` must still fail.
"""

import importlib
import sys
import types

import pytest


@pytest.fixture
def import_hermes_state_common():
    """Re-run ``hermes_state_common``'s module body against the current
    ``sys.modules['agent.skill_commands']`` — the kanban/session-preview path."""
    original = sys.modules.get("hermes_state_common")

    def _load():
        sys.modules.pop("hermes_state_common", None)
        return importlib.import_module("hermes_state_common")

    yield _load
    if original is not None:
        sys.modules["hermes_state_common"] = original
    else:
        sys.modules.pop("hermes_state_common", None)


class TestSkillExcerptJointImport:
    def test_live_module_binds_exact_joint_object(self, import_hermes_state_common):
        """CONTROL: when the live module has the name, use that exact object."""
        import agent.skill_commands as skill_commands

        assert hasattr(skill_commands, "SKILL_EXCERPT_JOINT")
        common = import_hermes_state_common()
        assert common.SKILL_EXCERPT_JOINT is skill_commands.SKILL_EXCERPT_JOINT
        assert common.SKILL_EXCERPT_JOINT == "\x1e"
        assert common.SKILL_SCAFFOLD_SQL_LIKE is skill_commands.SKILL_SCAFFOLD_SQL_LIKE
        assert common.describe_skill_invocation is skill_commands.describe_skill_invocation

    def test_stale_cached_module_missing_joint_still_binds(self, monkeypatch, import_hermes_state_common):
        """Stale in-memory ``skill_commands`` + fresh ``hermes_state_common`` import."""
        import agent.skill_commands as skill_commands

        assert hasattr(skill_commands, "SKILL_EXCERPT_JOINT")
        monkeypatch.delattr(skill_commands, "SKILL_EXCERPT_JOINT")

        common = import_hermes_state_common()
        assert common.SKILL_EXCERPT_JOINT == "\x1e"
        assert common.SKILL_SCAFFOLD_SQL_LIKE is skill_commands.SKILL_SCAFFOLD_SQL_LIKE
        assert common.describe_skill_invocation is skill_commands.describe_skill_invocation
        assert "\x1e" in common._PREVIEW_RAW_SELECT
        assert common._shape_preview(f"hello{common.SKILL_EXCERPT_JOINT}tail") == "hello"

    def test_version_skew_module_missing_joint_still_binds(self, monkeypatch, import_hermes_state_common):
        """Installed ``skill_commands`` never exported the joint; the other two names exist."""
        import agent.skill_commands as live

        stub = types.ModuleType("agent.skill_commands")
        stub.__file__ = getattr(live, "__file__", "skill_commands.py")
        stub.SKILL_SCAFFOLD_SQL_LIKE = live.SKILL_SCAFFOLD_SQL_LIKE
        stub.describe_skill_invocation = live.describe_skill_invocation
        monkeypatch.setitem(sys.modules, "agent.skill_commands", stub)

        common = import_hermes_state_common()
        assert common.SKILL_EXCERPT_JOINT == "\x1e"
        assert common.SKILL_SCAFFOLD_SQL_LIKE is live.SKILL_SCAFFOLD_SQL_LIKE
        assert common.describe_skill_invocation is live.describe_skill_invocation

    def test_missing_scaffold_like_still_fails(self, monkeypatch, import_hermes_state_common):
        import agent.skill_commands as skill_commands

        monkeypatch.delattr(skill_commands, "SKILL_SCAFFOLD_SQL_LIKE")
        with pytest.raises(ImportError, match=r"cannot import name 'SKILL_SCAFFOLD_SQL_LIKE'"):
            import_hermes_state_common()

    def test_missing_describe_skill_invocation_still_fails(self, monkeypatch, import_hermes_state_common):
        import agent.skill_commands as skill_commands

        monkeypatch.delattr(skill_commands, "describe_skill_invocation")
        with pytest.raises(ImportError, match=r"cannot import name 'describe_skill_invocation'"):
            import_hermes_state_common()
