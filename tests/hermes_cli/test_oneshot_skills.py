"""Tests for hermes -z --skills preloading (issue #31548).

The oneshot path must honor -s/--skills with the same contract as
interactive chat (cli.py main()):
  - all requested skills load        -> prompt built, run proceeds
  - SOME skills missing              -> warn on stderr, continue with loaded
  - ALL requested skills missing     -> fatal error on stderr, exit 2
  - no skills requested              -> no-op
"""

from unittest import mock

import pytest

from hermes_cli.oneshot import _build_oneshot_skills_prompt, run_oneshot


def _patch_builder(monkeypatch, prompt, loaded, missing):
    """Route the agent.skill_commands import to a stub module."""
    import sys
    import types

    stub = types.ModuleType("agent.skill_commands")
    calls = {}

    def build_preloaded_skills_prompt(identifiers, task_id=None):
        calls["identifiers"] = list(identifiers)
        calls["task_id"] = task_id
        return prompt, loaded, missing

    stub.build_preloaded_skills_prompt = build_preloaded_skills_prompt
    agent_pkg = sys.modules.get("agent") or types.ModuleType("agent")
    monkeypatch.setitem(sys.modules, "agent", agent_pkg)
    monkeypatch.setitem(sys.modules, "agent.skill_commands", stub)
    return calls


class TestBuildOneshotSkillsPrompt:
    def test_no_skills_is_noop(self):
        prompt, err = _build_oneshot_skills_prompt(None)
        assert prompt == ""
        assert err is None

    def test_empty_entries_are_noop(self):
        prompt, err = _build_oneshot_skills_prompt(" , ,")
        assert prompt == ""
        assert err is None

    def test_all_loaded_returns_prompt(self, monkeypatch):
        calls = _patch_builder(
            monkeypatch, "SKILL PROMPT", ["github-operations"], []
        )
        prompt, err = _build_oneshot_skills_prompt("github-operations")
        assert prompt == "SKILL PROMPT"
        assert err is None
        assert calls["identifiers"] == ["github-operations"]

    def test_comma_and_repeat_flags_deduplicate(self, monkeypatch):
        calls = _patch_builder(monkeypatch, "P", ["a", "b"], [])
        prompt, err = _build_oneshot_skills_prompt(["a,b", "a"])
        assert err is None
        assert calls["identifiers"] == ["a", "b"]

    def test_partial_missing_warns_and_continues(self, monkeypatch, capsys):
        _patch_builder(monkeypatch, "PARTIAL PROMPT", ["real-skill"], ["typo-skill"])
        prompt, err = _build_oneshot_skills_prompt("real-skill,typo-skill")
        assert prompt == "PARTIAL PROMPT"
        assert err is None  # NOT fatal — mirrors interactive chat
        stderr = capsys.readouterr().err
        assert "typo-skill" in stderr
        assert "real-skill" in stderr

    def test_all_missing_is_fatal(self, monkeypatch):
        _patch_builder(monkeypatch, "", [], ["nope-1", "nope-2"])
        prompt, err = _build_oneshot_skills_prompt("nope-1,nope-2")
        assert prompt == ""
        assert err is not None
        assert "nope-1" in err
        assert "nope-2" in err

    def test_import_failure_is_fatal_not_silent(self, monkeypatch):
        import sys

        monkeypatch.setitem(sys.modules, "agent", None)
        monkeypatch.setitem(sys.modules, "agent.skill_commands", None)
        prompt, err = _build_oneshot_skills_prompt("anything")
        assert prompt == ""
        assert err is not None


class TestRunOneshotSkillsWiring:
    def test_all_missing_exits_2_before_agent(self, monkeypatch, capsys):
        _patch_builder(monkeypatch, "", [], ["ghost"])
        with mock.patch("hermes_cli.oneshot._run_agent") as agent:
            rc = run_oneshot("hi", skills="ghost")
        assert rc == 2
        agent.assert_not_called()
        assert "ghost" in capsys.readouterr().err

    def test_skills_prompt_forwarded_to_run_agent(self, monkeypatch):
        _patch_builder(monkeypatch, "THE PROMPT", ["real"], [])
        with mock.patch(
            "hermes_cli.oneshot._run_agent", return_value=("ok", {})
        ) as agent:
            rc = run_oneshot("hi", skills="real")
        assert rc == 0
        assert agent.call_args.kwargs["skills_prompt"] == "THE PROMPT"

    def test_no_skills_forwards_empty_prompt(self):
        with mock.patch(
            "hermes_cli.oneshot._run_agent", return_value=("ok", {})
        ) as agent:
            rc = run_oneshot("hi")
        assert rc == 0
        assert agent.call_args.kwargs["skills_prompt"] == ""

    def test_partial_missing_still_runs(self, monkeypatch, capsys):
        _patch_builder(monkeypatch, "PARTIAL", ["real"], ["typo"])
        with mock.patch(
            "hermes_cli.oneshot._run_agent", return_value=("ok", {})
        ) as agent:
            rc = run_oneshot("hi", skills="real,typo")
        assert rc == 0
        assert agent.call_args.kwargs["skills_prompt"] == "PARTIAL"
        assert "typo" in capsys.readouterr().err

    def test_current_run_agent_contract_tuple(self):
        # Guard the (response, result) return contract this fix relies on —
        # a fake agent asserting the old str-only contract regressed PR #31591.
        with mock.patch(
            "hermes_cli.oneshot._run_agent",
            return_value=("final text", {"api_calls": 1}),
        ):
            rc = run_oneshot("hi")
        assert rc == 0
