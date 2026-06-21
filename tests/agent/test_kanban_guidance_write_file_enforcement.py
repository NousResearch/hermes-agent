"""Regression coverage for #29654 — ``KANBAN_GUIDANCE`` previously told
workers to ``cd $HERMES_KANBAN_WORKSPACE before any file operations``
but never said "use the ``write_file`` tool to actually create the
deliverable". Weak models (deepseek-v4-flash/pro, grok-4.3, GLM,
gemma) read the prompt as "describe what you would write" and shipped
empty workspaces with beautiful prose summaries.

These tests pin every clause of the rewrite:

* Step 2 explicitly names ``write_file`` (and ``patch``).
* Step 2 contains the "do not only describe" anti-hallucination clause.
* The ``Do NOT`` section grows a bullet that forbids claiming you
  wrote a file without a matching tool call.
* The hallucination phrasing matches the failure mode the reporter
  named — describing file contents in prose / comment / summary.
* The system prompt actually surfaces the new wording end-to-end
  when ``HERMES_KANBAN_TASK`` is set (i.e. when the worker is
  dispatched by the kanban dispatcher).
* The companion ``kanban-worker`` skill repeats the rule with a
  worked example, so workers that load the skill see both layers.
* The size-cap test still passes (the cap was bumped to 5 KB).
"""

from __future__ import annotations

from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Prompt-string contract (pure — no agent boot needed)
# ---------------------------------------------------------------------------


class TestKanbanGuidanceMentionsWriteFile:
    """The static ``KANBAN_GUIDANCE`` constant must name the file-write
    tool explicitly and forbid hallucinating file output."""

    def test_step_2_mentions_write_file_tool(self):
        from agent.prompt_builder import KANBAN_GUIDANCE
        assert "write_file" in KANBAN_GUIDANCE, (
            "KANBAN_GUIDANCE must name the write_file tool explicitly so "
            "weak models don't pattern-match 'describe the file' as the "
            "deliverable (#29654)"
        )

    def test_step_2_mentions_patch_tool_too(self):
        """``patch`` is the targeted-edit counterpart; mentioning both
        prevents a model that interprets 'write_file = full overwrite'
        from skipping the tool entirely on edit-shaped tasks."""
        from agent.prompt_builder import KANBAN_GUIDANCE
        assert "`patch`" in KANBAN_GUIDANCE

    def test_step_2_uses_directive_capitalisation(self):
        """The step header must visually signal urgency — weak models
        skim headings, so ``ACTUALLY write deliverables`` is part of
        the contract."""
        from agent.prompt_builder import KANBAN_GUIDANCE
        assert "ACTUALLY write deliverables" in KANBAN_GUIDANCE

    def test_step_2_calls_out_hallucination_failure_mode(self):
        """The literal anti-hallucination clause — describing file
        contents in prose / comment / summary is explicitly named as
        the failure mode (#29654's reporter cited exactly this)."""
        from agent.prompt_builder import KANBAN_GUIDANCE
        # The substring is split across the source string concatenation;
        # check for the substantive phrase rather than an exact line.
        assert "hallucination" in KANBAN_GUIDANCE.lower()
        assert "fails review" in KANBAN_GUIDANCE.lower()

    def test_step_2_names_workspace_path_pattern(self):
        """Workers need a concrete path shape, not just 'put it
        somewhere' — name the canonical ``$HERMES_KANBAN_WORKSPACE/<filename>``
        pattern."""
        from agent.prompt_builder import KANBAN_GUIDANCE
        assert "$HERMES_KANBAN_WORKSPACE/<filename>" in KANBAN_GUIDANCE

    def test_do_not_section_has_hallucination_bullet(self):
        from agent.prompt_builder import KANBAN_GUIDANCE
        assert "Do NOT" in KANBAN_GUIDANCE
        # The forbid-claim-without-call rule must appear after the
        # ``Do NOT`` header.
        do_not_idx = KANBAN_GUIDANCE.find("Do NOT")
        assert do_not_idx > 0
        do_not_section = KANBAN_GUIDANCE[do_not_idx:]
        assert "Do not claim you created or modified a file" in do_not_section

    def test_describing_is_not_writing_phrase(self):
        """The ``Do NOT`` bullet ends with the maxim ``Describing file
        contents is not the same as writing them.`` — this is the line
        weak models seem to internalise best."""
        from agent.prompt_builder import KANBAN_GUIDANCE
        assert (
            "Describing file contents is not the same as writing them"
            in KANBAN_GUIDANCE
        )

    def test_size_cap_still_honoured(self):
        """Pin the same bound the dedicated sanity test in
        ``tests/tools/test_kanban_tools.py`` asserts, so this file
        catches a runaway prompt-bloat without depending on that test
        being run."""
        from agent.prompt_builder import KANBAN_GUIDANCE
        assert 1_500 < len(KANBAN_GUIDANCE) < 5_120

    def test_step_ordering_preserved(self):
        """The lifecycle is "1. Orient → 2. Work → 3. Heartbeat → 4.
        Block → 5. Complete → 6. Spawn follow-up" — the rewrite must
        not have reordered them by accident."""
        from agent.prompt_builder import KANBAN_GUIDANCE
        idx1 = KANBAN_GUIDANCE.find("1. **Orient")
        idx2 = KANBAN_GUIDANCE.find("2. **Work inside the workspace")
        idx3 = KANBAN_GUIDANCE.find("3. **Heartbeat")
        idx4 = KANBAN_GUIDANCE.find("4. **Block")
        idx5 = KANBAN_GUIDANCE.find("5. **Complete")
        idx6 = KANBAN_GUIDANCE.find("6. **If follow-up")
        assert 0 < idx1 < idx2 < idx3 < idx4 < idx5 < idx6, (
            f"Lifecycle steps reordered or missing — got positions "
            f"{[idx1, idx2, idx3, idx4, idx5, idx6]}"
        )


# ---------------------------------------------------------------------------
# End-to-end — the new wording actually lands in the rendered system prompt
# when the worker is dispatched (``HERMES_KANBAN_TASK`` set).
# ---------------------------------------------------------------------------


class TestKanbanGuidanceInWorkerSystemPrompt:
    """The reporter's actual experience is "the worker model never made
    the ``write_file`` call". Verify the new wording reaches the system
    prompt the model actually sees."""

    @pytest.fixture
    def worker_agent(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HERMES_KANBAN_TASK", "t_repro_29654")
        home = tmp_path / ".hermes"
        home.mkdir()
        monkeypatch.setenv("HERMES_HOME", str(home))
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        from run_agent import AIAgent
        return AIAgent(
            api_key="test",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )

    def test_worker_prompt_includes_write_file(self, worker_agent):
        prompt = worker_agent._build_system_prompt()
        assert "write_file" in prompt

    def test_worker_prompt_includes_hallucination_warning(self, worker_agent):
        prompt = worker_agent._build_system_prompt()
        assert "hallucination" in prompt.lower()

    def test_worker_prompt_includes_describe_not_write_maxim(
        self, worker_agent
    ):
        prompt = worker_agent._build_system_prompt()
        assert (
            "Describing file contents is not the same as writing them"
            in prompt
        )

    def test_worker_prompt_still_includes_pre_existing_signals(
        self, worker_agent
    ):
        """Guardrail: the rewrite must not have nuked the existing
        lifecycle signals that ``test_kanban_guidance_in_worker_prompt``
        already pins. Re-checking here so this regression file is
        self-contained."""
        prompt = worker_agent._build_system_prompt()
        for signal in (
            "Kanban task execution protocol",
            "kanban_show()",
            "kanban_complete",
            "kanban_block",
            "kanban_create",
        ):
            assert signal in prompt, f"missing pre-existing signal: {signal!r}"


# ---------------------------------------------------------------------------
# Non-worker sessions are untouched — the rewrite is opt-in via
# ``HERMES_KANBAN_TASK``.
# ---------------------------------------------------------------------------


class TestNormalSessionUnaffected:
    def test_normal_chat_has_no_kanban_write_file_hint(
        self, monkeypatch, tmp_path
    ):
        """If the worker dispatcher isn't running, the system prompt
        must not be cluttered with kanban-specific write_file
        guidance — this is dispatcher-only context."""
        monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
        home = tmp_path / ".hermes"
        home.mkdir()
        monkeypatch.setenv("HERMES_HOME", str(home))
        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        from run_agent import AIAgent
        a = AIAgent(
            api_key="test",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
        prompt = a._build_system_prompt()
        assert "Kanban task execution protocol" not in prompt
        assert "ACTUALLY write deliverables" not in prompt


# ---------------------------------------------------------------------------
# Skill file mirrors the rule (defence-in-depth — workers that load
# kanban-worker via ``--skills`` see the deeper example).
# ---------------------------------------------------------------------------


class TestKanbanWorkerSkillMirrorsRule:
    @pytest.fixture
    def skill_text(self) -> str:
        path = Path(__file__).resolve().parents[2] / "skills" / "devops" / "kanban-worker" / "SKILL.md"
        assert path.exists(), f"missing {path}"
        return path.read_text(encoding="utf-8")

    def test_skill_has_producing_deliverables_section(self, skill_text):
        assert "## Producing deliverables" in skill_text

    def test_skill_mentions_write_file_tool(self, skill_text):
        assert "write_file" in skill_text

    def test_skill_shows_right_vs_wrong_example(self, skill_text):
        """The skill must contrast a correct ``write_file(...)`` call
        with the failure mode (claiming completion without a call),
        so weak-model workers have a concrete pattern to anchor on."""
        assert "# Right" in skill_text
        assert "# Wrong" in skill_text

    def test_skill_self_check_before_complete(self, skill_text):
        """The skill prescribes the operational check workers should
        run before ``kanban_complete``."""
        assert "Self-check before `kanban_complete`" in skill_text

    def test_skill_names_known_weak_models(self, skill_text):
        """Naming the model families that the reporter caught
        hallucinating (deepseek/grok/GLM/gemma) gives the model a
        precise reason to take the rule seriously when it sees its
        own family listed."""
        for model_family in ("deepseek", "grok"):
            assert model_family in skill_text.lower()
