from types import SimpleNamespace

from agent.codex_runtime import _codex_app_server_config_overrides


def test_codex_app_server_uses_hermes_model_and_reasoning_controls():
    agent = SimpleNamespace(
        model="gpt-5.6-terra",
        reasoning_config={"enabled": True, "effort": "xhigh"},
    )

    assert _codex_app_server_config_overrides(agent) == [
        "-c", 'model="gpt-5.6-terra"',
        "-c", 'model_reasoning_effort="xhigh"',
    ]


def test_codex_app_server_does_not_force_an_unknown_reasoning_default():
    agent = SimpleNamespace(model="gpt-5.6-terra", reasoning_config=None)

    assert _codex_app_server_config_overrides(agent) == [
        "-c", 'model="gpt-5.6-terra"',
    ]
