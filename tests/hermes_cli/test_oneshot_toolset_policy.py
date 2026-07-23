"""Security regressions for one-shot AIAgent toolset policy propagation."""

from unittest.mock import patch

import pytest


@pytest.mark.parametrize(
    ("configured", "expected"),
    [
        pytest.param("memory", ["memory"], id="bare-string"),
        pytest.param(
            ["*", "deny-provider-store"],
            ["*", "deny-provider-store"],
            id="list",
        ),
        pytest.param(None, None, id="none"),
    ],
)
def test_oneshot_forwards_global_disabled_toolsets(configured, expected):
    from hermes_cli import oneshot

    captured = {}

    class FakeAgent:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def run_conversation(self, *_args, **_kwargs):
            return {"final_response": "ok"}

    config = {
        "model": {"default": "test/model"},
        "agent": {"disabled_toolsets": configured},
    }
    runtime = {
        "api_key": "test-key",
        "base_url": "https://example.test/v1",
        "provider": "test",
        "api_mode": "chat_completions",
        "credential_pool": None,
    }
    with (
        patch("hermes_cli.config.load_config", return_value=config),
        patch("hermes_cli.runtime_provider.resolve_runtime_provider", return_value=runtime),
        patch("run_agent.AIAgent", FakeAgent),
        patch.object(oneshot, "_create_session_db_for_oneshot", return_value=None),
    ):
        text, result = oneshot._run_agent("check policy")

    assert text == "ok"
    assert result["final_response"] == "ok"
    assert captured["disabled_toolsets"] == expected
