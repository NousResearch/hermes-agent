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


SENTINEL = "ONESHOT-SKILLS-E2E-SENTINEL-31548 use the frobnicate protocol"


def _make_real_skill(hermes_home, name, body_sentinel=SENTINEL):
    """Write a real SKILL.md under <hermes_home>/skills/<name>/."""
    skill_dir = hermes_home / "skills" / name
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        f"""\
---
name: {name}
description: Integration regression fixture for issue #31548.
---

# {name}

{body_sentinel}
""",
        encoding="utf-8",
    )
    return skill_dir


def _fake_runtime(**overrides):
    """Minimal resolve_runtime_provider() shape for a mocked-agent run."""
    runtime = {
        "api_key": "test-key",
        "base_url": "http://127.0.0.1:1",
        "provider": "openai",
        "api_mode": "chat_completions",
        "credential_pool": None,
    }
    runtime.update(overrides)
    return runtime


class TestOneshotSkillsEndToEnd:
    """Integration regression (review feedback on #63814): no skill-loader
    stubs. A REAL skill on disk under a temp HERMES_HOME must be discovered
    by the actual agent.skill_commands loader and its body must reach the
    ``ephemeral_system_prompt`` handed to AIAgent. Only the external
    boundaries (provider resolution, the agent itself) are mocked."""

    @pytest.fixture(autouse=True)
    def _restore_logging(self):
        # run_oneshot() calls logging.disable(CRITICAL) globally; undo it so
        # this test can't degrade later files in shared-process runs.
        import logging

        yield
        logging.disable(logging.NOTSET)

    def _run(self, tmp_path, monkeypatch, skills):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.delenv("HERMES_INFERENCE_MODEL", raising=False)

        agent_cls = mock.MagicMock()
        agent_cls.return_value.run_conversation.return_value = {
            "final_response": "done",
            "api_calls": 1,
        }
        with mock.patch(
            "hermes_cli.runtime_provider.resolve_runtime_provider",
            return_value=_fake_runtime(),
        ), mock.patch("run_agent.AIAgent", agent_cls):
            rc = run_oneshot("hi", skills=skills)
        return rc, agent_cls

    def test_real_skill_body_reaches_agent_ephemeral_prompt(
        self, tmp_path, monkeypatch
    ):
        _make_real_skill(tmp_path, "oneshot-e2e-skill")

        rc, agent_cls = self._run(tmp_path, monkeypatch, "oneshot-e2e-skill")

        assert rc == 0
        agent_cls.assert_called_once()
        prompt = agent_cls.call_args.kwargs["ephemeral_system_prompt"]
        # The actual on-disk skill body — not a stub — must arrive verbatim.
        assert SENTINEL in prompt
        assert "oneshot-e2e-skill" in prompt

    def test_missing_skill_with_real_loader_exits_2_before_agent(
        self, tmp_path, monkeypatch, capsys
    ):
        (tmp_path / "skills").mkdir(parents=True)  # empty skills dir

        rc, agent_cls = self._run(tmp_path, monkeypatch, "no-such-skill")

        assert rc == 2
        agent_cls.assert_not_called()
        assert "no-such-skill" in capsys.readouterr().err

    def test_no_skills_requested_passes_none_ephemeral_prompt(
        self, tmp_path, monkeypatch
    ):
        rc, agent_cls = self._run(tmp_path, monkeypatch, None)

        assert rc == 0
        assert agent_cls.call_args.kwargs["ephemeral_system_prompt"] is None
