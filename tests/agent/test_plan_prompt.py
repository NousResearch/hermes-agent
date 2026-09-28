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




