from __future__ import annotations


def test_oneshot_passes_effective_max_tokens_to_agent(monkeypatch):
    import hermes_cli.oneshot as oneshot

    captured = {}

    class FakeAgent:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.suppress_status_output = False
            self.stream_delta_callback = None
            self.tool_gen_callback = None

        def run_conversation(self, prompt):
            return {"final_response": "ok"}

    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {
            "model": {
                "default": "gpt-test",
                "provider": "custom",
                "max_tokens": 16384,
            }
        },
    )
    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        lambda **_kwargs: {
            "provider": "custom",
            "api_mode": "chat_completions",
            "base_url": "https://local.test/v1",
            "api_key": "test-key",
            "max_output_tokens": 12000,
        },
    )
    monkeypatch.setattr("hermes_cli.oneshot._create_session_db_for_oneshot", lambda: None)
    monkeypatch.setattr("hermes_cli.oneshot.get_fallback_chain", lambda _cfg: [])
    monkeypatch.setattr("hermes_cli.tools_config._get_platform_tools", lambda _cfg, _platform: set())
    monkeypatch.setattr("run_agent.AIAgent", FakeAgent)

    text, result = oneshot._run_agent("hello")

    assert text == "ok"
    assert result["final_response"] == "ok"
    assert captured["max_tokens"] == 16384
