"""Regression tests for user-facing terminal failure remedies."""

from types import SimpleNamespace

from agent.turn_failure_copy import nonretryable_copy


def test_copilot_acp_auth_failure_names_copilot_cli_login():
    """An ACP subprocess owns its login; Hermes cannot re-add its credential."""
    copy = nonretryable_copy(
        SimpleNamespace(is_auth=True, reason=SimpleNamespace(value="auth")),
        provider="copilot-acp",
        model="copilot-acp",
        summary="Copilot ACP session/prompt failed: Authentication required",
    )

    assert "`copilot login`" in copy
    assert "hermes auth add copilot-acp" not in copy
