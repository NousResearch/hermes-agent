"""Tests for the built-in /plan command (formerly the bundled `plan` skill).

Covers the shared prompt builder (agent.plan_prompt.build_plan_prompt) and the
registry wiring that makes /plan a first-class command on every surface —
CLI, gateway messengers, and TUI. The skill-to-builtin move exists precisely
so /plan survives the Telegram/Discord command-menu caps that trimmed it as
an alphabetical skill entry.
"""

from agent.plan_prompt import build_plan_prompt

class TestBuildPlanPrompt:
    def test_task_is_included_verbatim(self):
        task = "migrate the auth provider to OIDC with zero downtime"
        prompt = build_plan_prompt(task)
        assert task in prompt

class TestPlanRegistryWiring:
    def test_plan_is_registered_and_resolves(self):
        from hermes_cli.commands import resolve_command

        cmd = resolve_command("plan")
        assert cmd is not None
        assert cmd.name == "plan"


class TestImplementPrompt:
    """/implement freezes the approved plan into the turn: what the user reviewed is what runs."""

    def test_newest_plan_is_inlined_and_named_fragments_pick_older_ones(self, tmp_path):
        import os
        from agent.implement_prompt import build_implement_prompt

        plans = tmp_path / ".hermes" / "plans"
        plans.mkdir(parents=True)
        old = plans / "2026-11-01_090000-oauth.md"
        new = plans / "2026-11-16_090000-billing.md"
        old.write_text("# OAuth plan\n1. add the PKCE flow\n", encoding="utf-8")
        new.write_text("# Billing plan\n1. add invoices\n", encoding="utf-8")
        os.utime(old, (1, 1))

        prompt, label = build_implement_prompt("", str(tmp_path))
        assert label == os.path.join(".hermes", "plans", new.name)
        assert "1. add invoices" in prompt and "PKCE" not in prompt

        prompt, label = build_implement_prompt("oauth", str(tmp_path))
        assert label.endswith(old.name) and "1. add the PKCE flow" in prompt

    def test_no_saved_plan_tells_the_agent_to_look_instead_of_improvising(self, tmp_path):
        from agent.implement_prompt import build_implement_prompt

        prompt, label = build_implement_prompt("", str(tmp_path))
        assert label == ""
        assert ".hermes" in prompt and "/plan <task>" in prompt
