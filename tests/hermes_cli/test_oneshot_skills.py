"""Regression tests for --skills/-s forwarding through -z/--oneshot (#75930).

Before this fix, ``--skills`` was parsed at the top level but never forwarded
past ``_run_and_exit_oneshot()``/``run_oneshot()`` -- every ``hermes -z ...
--skills <name>`` call ran with zero skill content injected, exiting 0 with
no visible error. These tests pin the forwarding chain at both ends: the
shared dispatch-kwargs helper in ``hermes_cli/main.py``, and the
skills-to-``ephemeral_system_prompt`` translation in
``hermes_cli/oneshot.py::run_oneshot``.
"""

import sys
import types
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from hermes_cli.main import _oneshot_kwargs_from_args, _run_and_exit_oneshot
from hermes_cli.oneshot import run_oneshot


def _parse_skills_argument(skills):
    """Stand-in mirroring cli.py's real ``_parse_skills_argument`` (comma-split,
    strip, dedupe, drop-empties).

    ``run_oneshot`` does a *local* ``from cli import _parse_skills_argument``,
    which would otherwise pull the entire ~17k-line ``cli.py`` (and its
    ``prompt_toolkit`` TUI dependency) into these unit tests just to reach one
    small pure string-parsing helper. Patched in via the ``_parse_skills``
    fixture below instead.
    """
    if not skills:
        return []
    raw_values = [skills] if isinstance(skills, str) else list(skills)
    parsed, seen = [], set()
    for raw in raw_values:
        for part in str(raw).split(","):
            normalized = part.strip()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            parsed.append(normalized)
    return parsed


class TestOneshotKwargsFromArgs:
    """The kwargs both -z dispatch sites in main.py build from parsed args."""

    def test_includes_skills(self):
        args = SimpleNamespace(
            model="m", provider="p", toolsets="t", skills="my-skill", usage_file=None
        )
        kwargs = _oneshot_kwargs_from_args(args)
        assert kwargs["skills"] == "my-skill"

    def test_missing_skills_attr_defaults_to_none(self):
        # A future argparse refactor that drops the attribute must not crash
        # the dispatcher -- it should degrade to "no skills requested".
        args = SimpleNamespace(model=None, provider=None, toolsets=None, usage_file=None)
        kwargs = _oneshot_kwargs_from_args(args)
        assert kwargs["skills"] is None

    def test_all_fields_present(self):
        # Locks the exact kwarg set both call sites rely on -- a field added
        # here now only needs to be added once.
        args = SimpleNamespace(
            model="m", provider="p", toolsets="t", skills="s", usage_file="u"
        )
        assert _oneshot_kwargs_from_args(args) == {
            "model": "m",
            "provider": "p",
            "toolsets": "t",
            "skills": "s",
            "usage_file": "u",
        }


class TestRunOneshotSkillsForwarding:
    """hermes_cli.oneshot.run_oneshot's --skills handling."""

    @pytest.fixture(autouse=True)
    def _stub_cli_parse_skills_argument(self, monkeypatch):
        # run_oneshot() does a LOCAL `from cli import _parse_skills_argument`.
        # unittest.mock.patch("cli._parse_skills_argument", ...) would still
        # have to import the real `cli` module first to find the attribute --
        # which drags in prompt_toolkit and everything else `cli.py` needs at
        # module-import time. Inject a lightweight stand-in module into
        # sys.modules instead, so the local import never touches the real one.
        fake_cli = types.ModuleType("cli")
        fake_cli._parse_skills_argument = _parse_skills_argument
        monkeypatch.setitem(sys.modules, "cli", fake_cli)

    def _run_agent_ok(self, _prompt, **_kwargs):
        return "final answer", {"final_response": "final answer"}

    def test_skills_preload_becomes_ephemeral_system_prompt(self):
        with patch(
            "agent.skill_commands.build_preloaded_skills_prompt",
            return_value=("SKILL BODY TEXT", ["my-skill"], []),
        ) as build_mock, patch(
            "hermes_cli.oneshot._run_agent",
            side_effect=self._run_agent_ok,
        ) as run_agent_mock:
            rc = run_oneshot("hi", skills="my-skill")

        assert rc == 0
        build_mock.assert_called_once()
        run_agent_mock.assert_called_once()
        assert build_mock.call_args.args[0] == ["my-skill"]
        assert run_agent_mock.call_args.kwargs["ephemeral_system_prompt"] == "SKILL BODY TEXT"

    def test_no_skills_requested_means_no_ephemeral_system_prompt(self):
        with patch(
            "hermes_cli.oneshot._run_agent", side_effect=self._run_agent_ok
        ) as run_agent_mock:
            rc = run_oneshot("hi")

        assert rc == 0
        run_agent_mock.assert_called_once()
        assert run_agent_mock.call_args.kwargs["ephemeral_system_prompt"] is None

    def test_whitespace_only_skills_string_is_treated_as_no_skills(self):
        # _parse_skills_argument strips/drops empty parts -- "  , ,  " must
        # not reach build_preloaded_skills_prompt at all.
        with patch(
            "agent.skill_commands.build_preloaded_skills_prompt"
        ) as build_mock, patch(
            "hermes_cli.oneshot._run_agent", side_effect=self._run_agent_ok
        ) as run_agent_mock:
            rc = run_oneshot("hi", skills="  , ,  ")

        assert rc == 0
        build_mock.assert_not_called()
        run_agent_mock.assert_called_once()
        assert run_agent_mock.call_args.kwargs["ephemeral_system_prompt"] is None

    def test_all_requested_skills_missing_exits_2_without_running_agent(self, capsys):
        with patch(
            "agent.skill_commands.build_preloaded_skills_prompt",
            return_value=("", [], ["unknown-skill"]),
        ), patch("hermes_cli.oneshot._run_agent") as run_agent_mock:
            rc = run_oneshot("hi", skills="unknown-skill")

        assert rc == 2
        run_agent_mock.assert_not_called()
        assert "Unknown skill(s): unknown-skill" in capsys.readouterr().err

    def test_partially_missing_skills_still_runs_with_the_loaded_ones(self):
        with patch(
            "agent.skill_commands.build_preloaded_skills_prompt",
            return_value=("LOADED SKILL BODY", ["good-skill"], ["bad-skill"]),
        ), patch(
            "hermes_cli.oneshot._run_agent", side_effect=self._run_agent_ok
        ) as run_agent_mock:
            rc = run_oneshot("hi", skills="good-skill,bad-skill")

        assert rc == 0
        assert run_agent_mock.call_args.kwargs["ephemeral_system_prompt"] == "LOADED SKILL BODY"

    def test_skills_that_resolve_to_no_content_leave_prompt_none(self):
        # build_preloaded_skills_prompt can return an empty prompt string
        # even with a loaded skill (e.g. an empty SKILL.md body) -- must not
        # pass an empty string as ephemeral_system_prompt.
        with patch(
            "agent.skill_commands.build_preloaded_skills_prompt",
            return_value=("", ["empty-skill"], []),
        ), patch(
            "hermes_cli.oneshot._run_agent", side_effect=self._run_agent_ok
        ) as run_agent_mock:
            rc = run_oneshot("hi", skills="empty-skill")

        assert rc == 0
        assert run_agent_mock.call_args.kwargs["ephemeral_system_prompt"] is None


class TestRunAndExitOneshotForwardsSkills:
    """Closes the remaining gap: _run_and_exit_oneshot -> run_oneshot.

    Without this, a regression that drops ``skills`` between
    ``_run_and_exit_oneshot`` and ``run_oneshot`` could pass every test
    above (they call ``run_oneshot`` directly) while the real -z path stays
    broken -- exactly the shape of the original bug.
    """

    def test_skills_reaches_run_oneshot(self):
        # _exit_after_oneshot does a hard os._exit(); must be patched out or
        # the test process itself would terminate.
        with patch("hermes_cli.oneshot.run_oneshot", return_value=0) as run_oneshot_mock, \
             patch("hermes_cli.main._exit_after_oneshot") as exit_mock, \
             patch("hermes_cli.main._cleanup_oneshot_runtime"):
            _run_and_exit_oneshot("hi", skills="my-skill")

        run_oneshot_mock.assert_called_once()
        assert run_oneshot_mock.call_args.kwargs["skills"] == "my-skill"
        exit_mock.assert_called_once_with(0)

    def test_no_skills_forwards_none(self):
        with patch("hermes_cli.oneshot.run_oneshot", return_value=0) as run_oneshot_mock, \
             patch("hermes_cli.main._exit_after_oneshot"), \
             patch("hermes_cli.main._cleanup_oneshot_runtime"):
            _run_and_exit_oneshot("hi")

        assert run_oneshot_mock.call_args.kwargs["skills"] is None
