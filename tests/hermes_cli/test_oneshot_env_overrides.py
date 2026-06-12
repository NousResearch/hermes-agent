from hermes_cli.oneshot import (
    _fast_mode_request_overrides,
    _parse_env_reasoning_config,
    _parse_env_service_tier,
)


def test_oneshot_reads_reasoning_effort_from_env(monkeypatch):
    monkeypatch.setenv("HERMES_INFERENCE_REASONING_EFFORT", "minimal")

    assert _parse_env_reasoning_config() == {"enabled": True, "effort": "minimal"}


def test_oneshot_reads_service_tier_from_env(monkeypatch):
    monkeypatch.setenv("HERMES_INFERENCE_SERVICE_TIER", "fast")
    assert _parse_env_service_tier() == "priority"

    monkeypatch.setenv("HERMES_INFERENCE_SERVICE_TIER", "off")
    assert _parse_env_service_tier() is None


def test_oneshot_builds_fast_mode_overrides_for_openai_model():
    assert _fast_mode_request_overrides("gpt-5.4-mini", "priority") == {
        "service_tier": "priority"
    }
