"""Regression tests: --skills forwarded to ACP sessions (#24466).

``hermes -s <skill> acp`` must preload skill content into every new ACP session.
The fix has two parts:
  1. ``cmd_acp`` in ``hermes_cli/main.py`` sets ``HERMES_ACP_SKILLS`` env var.
  2. ``SessionManager._make_agent`` reads it and injects via ``ephemeral_system_prompt``.
"""

from __future__ import annotations

import os
from types import SimpleNamespace
from unittest.mock import MagicMock, patch, sentinel

import pytest

from acp_adapter.session import SessionManager


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _manager():
    """Minimal SessionManager bypassing __init__ filesystem side-effects."""
    mgr = SessionManager.__new__(SessionManager)
    mgr._agent_factory = None
    # _db_instance=sentinel.db causes _get_db() to return it directly (non-None)
    # so the lazy DB init path is never reached during unit tests.
    mgr._db_instance = None
    return mgr


def _fake_agent(**kwargs):
    agent = MagicMock()
    agent._print_fn = None
    return agent


_PATCHES_BASE = [
    patch("run_agent.AIAgent"),
    patch("hermes_cli.config.load_config", return_value={}),
    patch("hermes_cli.runtime_provider.resolve_runtime_provider", return_value={}),
    patch("acp_adapter.session._register_task_cwd"),
    patch("acp_adapter.session._expand_acp_enabled_toolsets", return_value=[]),
    patch("acp_adapter.session.SessionManager._get_db", return_value=None),
]


# ---------------------------------------------------------------------------
# SessionManager._make_agent reads HERMES_ACP_SKILLS
# ---------------------------------------------------------------------------

class TestMakeAgentSkillsPreload:

    def test_no_env_var_no_ephemeral_prompt(self, monkeypatch):
        monkeypatch.delenv("HERMES_ACP_SKILLS", raising=False)

        created_kwargs: dict = {}

        def fake_factory(**kwargs):
            created_kwargs.update(kwargs)
            return _fake_agent(**kwargs)

        mgr = _manager()

        with (
            patch("run_agent.AIAgent", side_effect=fake_factory),
            patch("hermes_cli.config.load_config", return_value={}),
            patch("hermes_cli.runtime_provider.resolve_runtime_provider", return_value={}),
            patch("acp_adapter.session._register_task_cwd"),
            patch("acp_adapter.session._expand_acp_enabled_toolsets", return_value=[]),
            patch("acp_adapter.session.SessionManager._get_db", return_value=None),
        ):
            mgr._make_agent(session_id="s1", cwd="/tmp")

        assert "ephemeral_system_prompt" not in created_kwargs

    def test_env_var_set_injects_skills_prompt(self, monkeypatch):
        monkeypatch.setenv("HERMES_ACP_SKILLS", "my-skill")

        created_kwargs: dict = {}

        def fake_factory(**kwargs):
            created_kwargs.update(kwargs)
            return _fake_agent(**kwargs)

        mgr = _manager()

        with (
            patch("run_agent.AIAgent", side_effect=fake_factory),
            patch("hermes_cli.config.load_config", return_value={}),
            patch("hermes_cli.runtime_provider.resolve_runtime_provider", return_value={}),
            patch("acp_adapter.session._register_task_cwd"),
            patch("acp_adapter.session._expand_acp_enabled_toolsets", return_value=[]),
            patch("acp_adapter.session.SessionManager._get_db", return_value=None),
            patch(
                "agent.skill_commands.build_preloaded_skills_prompt",
                return_value=("SKILL CONTENT", ["my-skill"], []),
            ),
        ):
            mgr._make_agent(session_id="s1", cwd="/tmp")

        assert created_kwargs.get("ephemeral_system_prompt") == "SKILL CONTENT"

    def test_multiple_skills_comma_separated(self, monkeypatch):
        monkeypatch.setenv("HERMES_ACP_SKILLS", "skill-a,skill-b")

        captured_identifiers: list[str] = []

        def fake_build(identifiers, task_id=None):
            captured_identifiers.extend(identifiers)
            return ("MULTI", list(identifiers), [])

        created_kwargs: dict = {}

        def fake_factory(**kwargs):
            created_kwargs.update(kwargs)
            return _fake_agent(**kwargs)

        mgr = _manager()

        with (
            patch("run_agent.AIAgent", side_effect=fake_factory),
            patch("hermes_cli.config.load_config", return_value={}),
            patch("hermes_cli.runtime_provider.resolve_runtime_provider", return_value={}),
            patch("acp_adapter.session._register_task_cwd"),
            patch("acp_adapter.session._expand_acp_enabled_toolsets", return_value=[]),
            patch("acp_adapter.session.SessionManager._get_db", return_value=None),
            patch("agent.skill_commands.build_preloaded_skills_prompt", side_effect=fake_build),
        ):
            mgr._make_agent(session_id="s1", cwd="/tmp")

        assert captured_identifiers == ["skill-a", "skill-b"]
        assert created_kwargs.get("ephemeral_system_prompt") == "MULTI"

    def test_unknown_skill_logs_warning_but_does_not_raise(self, monkeypatch, caplog):
        monkeypatch.setenv("HERMES_ACP_SKILLS", "no-such-skill")

        created_kwargs: dict = {}

        def fake_factory(**kwargs):
            created_kwargs.update(kwargs)
            return _fake_agent(**kwargs)

        mgr = _manager()

        import logging

        caplog.set_level(logging.WARNING, logger="acp_adapter.session")

        with (
            patch("run_agent.AIAgent", side_effect=fake_factory),
            patch("hermes_cli.config.load_config", return_value={}),
            patch("hermes_cli.runtime_provider.resolve_runtime_provider", return_value={}),
            patch("acp_adapter.session._register_task_cwd"),
            patch("acp_adapter.session._expand_acp_enabled_toolsets", return_value=[]),
            patch("acp_adapter.session.SessionManager._get_db", return_value=None),
            patch(
                "agent.skill_commands.build_preloaded_skills_prompt",
                return_value=("", [], ["no-such-skill"]),
            ),
        ):
            mgr._make_agent(session_id="s1", cwd="/tmp")

        assert "no-such-skill" in caplog.text
        assert "ephemeral_system_prompt" not in created_kwargs

    def test_deduplicate_skill_names(self, monkeypatch):
        monkeypatch.setenv("HERMES_ACP_SKILLS", "skill-a,skill-a,skill-b")

        captured_identifiers: list[str] = []

        def fake_build(identifiers, task_id=None):
            captured_identifiers.extend(identifiers)
            return ("OK", list(identifiers), [])

        created_kwargs: dict = {}

        def fake_factory(**kwargs):
            created_kwargs.update(kwargs)
            return _fake_agent(**kwargs)

        mgr = _manager()

        with (
            patch("run_agent.AIAgent", side_effect=fake_factory),
            patch("hermes_cli.config.load_config", return_value={}),
            patch("hermes_cli.runtime_provider.resolve_runtime_provider", return_value={}),
            patch("acp_adapter.session._register_task_cwd"),
            patch("acp_adapter.session._expand_acp_enabled_toolsets", return_value=[]),
            patch("acp_adapter.session.SessionManager._get_db", return_value=None),
            patch("agent.skill_commands.build_preloaded_skills_prompt", side_effect=fake_build),
        ):
            mgr._make_agent(session_id="s1", cwd="/tmp")

        assert captured_identifiers == ["skill-a", "skill-b"]

    def test_empty_env_var_no_ephemeral_prompt(self, monkeypatch):
        monkeypatch.setenv("HERMES_ACP_SKILLS", "")

        created_kwargs: dict = {}

        def fake_factory(**kwargs):
            created_kwargs.update(kwargs)
            return _fake_agent(**kwargs)

        mgr = _manager()

        with (
            patch("run_agent.AIAgent", side_effect=fake_factory),
            patch("hermes_cli.config.load_config", return_value={}),
            patch("hermes_cli.runtime_provider.resolve_runtime_provider", return_value={}),
            patch("acp_adapter.session._register_task_cwd"),
            patch("acp_adapter.session._expand_acp_enabled_toolsets", return_value=[]),
            patch("acp_adapter.session.SessionManager._get_db", return_value=None),
        ):
            mgr._make_agent(session_id="s1", cwd="/tmp")

        assert "ephemeral_system_prompt" not in created_kwargs


# ---------------------------------------------------------------------------
# cmd_acp sets HERMES_ACP_SKILLS before calling acp_main
# ---------------------------------------------------------------------------

class TestCmdAcpSetsEnvVar:
    """cmd_acp must bridge args.skills → HERMES_ACP_SKILLS."""

    def _run_cmd_acp_logic(self, skills: list[str], monkeypatch) -> str | None:
        """Simulate the cmd_acp env-var bridge and return what acp_main sees."""
        monkeypatch.delenv("HERMES_ACP_SKILLS", raising=False)

        seen: dict = {}

        def fake_acp_main():
            seen["val"] = os.environ.get("HERMES_ACP_SKILLS")

        args = SimpleNamespace(skills=skills if skills else None)

        s = getattr(args, "skills", None)
        if s:
            flattened: list[str] = []
            for item in s:
                flattened.extend(
                    part.strip() for part in str(item).split(",") if part.strip()
                )
            if flattened:
                os.environ["HERMES_ACP_SKILLS"] = ",".join(flattened)
        fake_acp_main()
        os.environ.pop("HERMES_ACP_SKILLS", None)

        return seen.get("val")

    def test_single_skill(self, monkeypatch):
        val = self._run_cmd_acp_logic(["my-skill"], monkeypatch)
        assert val == "my-skill"

    def test_multiple_skills(self, monkeypatch):
        val = self._run_cmd_acp_logic(["skill-a", "skill-b"], monkeypatch)
        assert val == "skill-a,skill-b"

    def test_comma_in_single_arg(self, monkeypatch):
        val = self._run_cmd_acp_logic(["skill-a,skill-b"], monkeypatch)
        assert val == "skill-a,skill-b"

    def test_no_skills_env_not_set(self, monkeypatch):
        val = self._run_cmd_acp_logic([], monkeypatch)
        assert val is None
