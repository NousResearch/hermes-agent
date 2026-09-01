"""`hermes doctor` must be able to answer "is my prompt cache actually on?".

Before this section existed the only signal was an ENABLED banner in
`agent_init`, which has no DISABLED counterpart and is suppressed under
`quiet_mode` (the ACP adapter forces it on). An operator whose gateway route
silently resolves to `(False, False)` had nothing to run.

The section is offline by construction — it resolves the same destination the
agent will use and asks `anthropic_prompt_cache_policy` — so these tests stub
only the two resolvers and never touch the network.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

GATEWAY = "https://gateway.example.test"


def _run_section(capsys, *, api_mode, model, provider="custom", cache_ttl="5m"):
    """Render the Prompt Caching section alone and return (stdout, issues)."""
    from hermes_cli import doctor as doctor_mod

    issues: list[str] = []

    with (
        patch(
            "hermes_cli.runtime_provider.resolve_runtime_provider",
            return_value={
                "provider": provider,
                "base_url": GATEWAY,
                "api_mode": api_mode,
                "api_key": "sk-test",
            },
        ),
        patch(
            "hermes_cli.config.load_config",
            return_value={"model": {"default": model}},
        ),
        patch(
            "hermes_cli.config.load_config_readonly",
            return_value={"prompt_caching": {"cache_ttl": cache_ttl}},
        ),
    ):
        doctor_mod._prompt_caching_section(issues)

    return capsys.readouterr().out, issues


@pytest.fixture(autouse=True)
def _require_extracted_section():
    """The section must be callable on its own, not welded into run_doctor."""
    from hermes_cli import doctor as doctor_mod

    assert hasattr(doctor_mod, "_prompt_caching_section"), (
        "expected doctor to expose the prompt-caching check as its own helper"
    )


class TestDoctorPromptCachingSection:
    def test_reports_ok_on_the_native_wire(self, capsys):
        out, issues = _run_section(
            capsys, api_mode="anthropic_messages", model="claude-sonnet-4-5"
        )
        assert "Prompt caching" in out
        assert "ENABLED" in out
        assert issues == []

    def test_flags_an_unset_api_mode_and_names_the_config_key(self, capsys):
        """The regression this section exists for."""
        out, issues = _run_section(capsys, api_mode="", model="claude-sonnet-4-5")

        assert "DISABLED" in out
        assert "model.api_mode: anthropic_messages" in out
        # The consequence must be spelled out, not left as an inference.
        assert "re-billed as uncached input" in out
        assert len(issues) == 1
        assert "Prompt caching is off" in issues[0]

    def test_flags_the_openai_wire(self, capsys):
        out, issues = _run_section(
            capsys, api_mode="chat_completions", model="claude-sonnet-4-5"
        )
        assert "DISABLED" in out
        assert len(issues) == 1

    def test_an_explicit_cache_ttl_optout_is_reported_but_not_an_issue(self, capsys):
        """A deliberate operator choice must not read as something to fix."""
        out, issues = _run_section(
            capsys,
            api_mode="anthropic_messages",
            model="claude-sonnet-4-5",
            cache_ttl="off",
        )
        assert "DISABLED" in out
        assert "prompt_caching.cache_ttl" in out
        assert "re-billed" not in out
        assert issues == []

    def test_a_resolver_failure_degrades_to_a_warning(self, capsys):
        """doctor must never crash on a route it cannot resolve."""
        from hermes_cli import doctor as doctor_mod

        issues: list[str] = []
        with patch(
            "hermes_cli.runtime_provider.resolve_runtime_provider",
            side_effect=RuntimeError("no provider configured"),
        ):
            doctor_mod._prompt_caching_section(issues)

        out = capsys.readouterr().out
        assert "Could not resolve prompt caching status" in out
        assert issues == []
