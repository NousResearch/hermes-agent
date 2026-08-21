"""Tests for /upskill — automatic session→skill sweep.

/upskill is the automatic inverse of /learn: instead of the user naming a
source to distill, the agent reviews what it actually did in this session and
PROPOSES candidate reusable skills (including small/multi-step ones). Like
/learn it has no engine and no model tool — it builds a standards-guided prompt
the live agent runs as a normal turn. These are the load-bearing contracts:
propose-before-save, dedupe, and a noise bar that rejects one-off tasks.
"""

from agent.learn_prompt import _AUTHORING_STANDARDS, _SOURCE_HYGIENE
from agent.upskill_prompt import build_upskill_prompt


class TestBuildUpskillPrompt:
    def test_is_a_sweep_not_a_open_ended_learn(self):
        prompt = build_upskill_prompt()
        low = prompt.lower()
        # It must NOT read like the open-ended /learn ("gather sources you named").
        assert "propose" in low
        assert "sweep" in low or "sweeping" in low
        # Explicitly says it proposes candidates for approval before saving.
        assert "propose" in low and "approval" in low

    def test_proposes_before_saving_anything(self):
        # The core contract: nothing is saved without user confirmation.
        prompt = build_upskill_prompt()
        low = prompt.lower()
        assert "do not save anything yet" in low
        assert "only after approval" in low

    def test_clusters_repeated_procedures_including_trivial_ones(self):
        prompt = build_upskill_prompt()
        # The whole point: even small/multi-step workflows get swept.
        assert "trivial" in prompt.lower() or "small" in prompt.lower()
        assert "reused" in prompt.lower() or "reusable" in prompt.lower()

    def test_has_a_noise_bar_rejecting_one_off_tasks(self):
        # Must not propose low-value one-off tasks as skills.
        prompt = build_upskill_prompt()
        assert "one-off" in prompt
        assert "noise" in prompt.lower()

    def test_dedupes_against_existing_skills(self):
        prompt = build_upskill_prompt()
        # Never propose a near-duplicate; extend an existing skill instead.
        low = prompt.lower()
        assert "skills_list" in low
        assert "skill_view" in low
        assert "dedup" in low or "exists" in low

    def test_embeds_source_hygiene_and_authoring_standards(self):
        prompt = build_upskill_prompt()
        assert _SOURCE_HYGIENE in prompt
        assert _AUTHORING_STANDARDS in prompt

    def test_optional_scope_emphasis_is_load_bearing(self):
        prompt = build_upskill_prompt("focus only on the WiNG console workflow")
        assert "WiNG console workflow" in prompt
        assert "focus the sweep on it" in prompt.lower()

    def test_empty_scope_runs_full_session_sweep(self):
        full = build_upskill_prompt()
        scoped = build_upskill_prompt("only the git stuff")
        # A scope hint must narrow/zero in on it; no scope = general sweep.
        assert "git stuff" in scoped
        # Both still carry the same safe defaults.
        assert _AUTHORING_STANDARDS in full and _AUTHORING_STANDARDS in scoped

    def test_can_conclude_nothing_worth_saving(self):
        # Honest negative result is allowed — avoid inventing low-value skills.
        prompt = build_upskill_prompt()
        assert "genuinely nothing worth saving" in prompt


class TestUpskillRegistryWiring:
    def test_upskill_is_registered_and_resolves(self):
        from hermes_cli.commands import resolve_command

        cmd = resolve_command("upskill")
        assert cmd is not None
        assert cmd.name == "upskill"

    def test_upskill_is_not_cli_only(self):
        # /upskill should be available on the gateway too, like /learn.
        from hermes_cli.commands import resolve_command

        assert not resolve_command("upskill").cli_only

    def test_upskill_handler_exists_on_cli_mixin(self):
        from hermes_cli.cli_commands_mixin import CLICommandsMixin

        assert hasattr(CLICommandsMixin, "_handle_upskill_command")
