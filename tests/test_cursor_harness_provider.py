from __future__ import annotations


def test_cursor_model_prefix_resolves_cursor_runtime(monkeypatch):
    from hermes_cli import runtime_provider

    monkeypatch.setattr(runtime_provider, "resolve_requested_provider", lambda requested=None: "auto")
    monkeypatch.setattr(runtime_provider, "_get_model_config", lambda: {"default": "cursor/default"})

    runtime = runtime_provider.resolve_runtime_provider()

    assert runtime["provider"] == "cursor-harness"
    assert runtime["api_mode"] == "cursor_harness"
    assert runtime["base_url"] == "cursor://harness"


def test_cursor_provider_request_resolves_cursor_runtime(monkeypatch):
    from hermes_cli import runtime_provider

    monkeypatch.setattr(runtime_provider, "resolve_requested_provider", lambda requested=None: "cursor-harness")
    monkeypatch.setattr(runtime_provider, "_get_model_config", lambda: {"default": "cursor/composer-2"})

    runtime = runtime_provider.resolve_runtime_provider()

    assert runtime["provider"] == "cursor-harness"
    assert runtime["api_mode"] == "cursor_harness"
    assert runtime["requested_provider"] == "cursor-harness"


def test_cursor_model_mapping_and_prompt_rendering():
    from agent.cursor_harness_adapter import cursor_model_from_hermes, render_cursor_prompt

    assert cursor_model_from_hermes("cursor/default") is None
    assert cursor_model_from_hermes("cursor/composer-2") == "composer-2"

    prompt = render_cursor_prompt(
        messages=[{"role": "user", "content": "Fix the test"}],
        system_prompt="Hermes system context",
        current_turn_user_idx=0,
        user_injections=["Memory context"],
        hermes_session_id="hermes-1",
        model="cursor/default",
        platform="cli",
    )

    assert "<hermes_system_context>" in prompt
    assert "Fix the test" in prompt
    assert "Memory context" in prompt
    assert "hermes-1" in prompt
