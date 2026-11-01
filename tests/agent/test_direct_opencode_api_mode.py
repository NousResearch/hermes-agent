"""Direct AIAgent construction resolves the OpenCode per-model wire.

The primary (/model switch) and fallback paths already derive ``api_mode`` from
``opencode_model_api_mode``; direct construction without an explicit api_mode
fell through to ``chat_completions``, so Responses-only models (muse-spark)
500ed on /chat/completions (same bug class as #102148, init path).
"""

from __future__ import annotations


def _agent(provider, model, base_url, api_mode=None):
    from run_agent import AIAgent

    return AIAgent(
        provider=provider,
        base_url=base_url,
        api_key="sk-test-opencode",
        model=model,
        api_mode=api_mode,
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
    )


def test_direct_go_muse_resolves_codex_responses():
    agent = _agent("opencode-go", "muse-spark-1.3-contributor", "https://opencode.ai/zen/go/v1")
    assert agent.api_mode == "codex_responses"
    assert agent.provider == "opencode-go"


def test_direct_zen_muse_resolves_codex_responses():
    agent = _agent("opencode-zen", "muse-spark-1.2", "https://opencode.ai/zen/v1")
    assert agent.api_mode == "codex_responses"
    assert agent.provider == "opencode-zen"


def test_direct_go_chat_model_stays_chat_completions():
    agent = _agent("opencode-go", "deepseek-v4-flash", "https://opencode.ai/zen/go/v1")
    assert agent.api_mode == "chat_completions"


def test_direct_explicit_pin_wins_over_table():
    agent = _agent(
        "opencode-go",
        "muse-spark-1.3-contributor",
        "https://opencode.ai/zen/go/v1",
        api_mode="chat_completions",
    )
    assert agent.api_mode == "chat_completions"


def test_direct_non_opencode_untouched(monkeypatch):
    # Credential gate only (no network at init); keeps the focus on api_mode.
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test-openrouter")
    agent = _agent("openrouter", "openrouter/pareto-code", None)
    assert agent.api_mode == "chat_completions"
    assert agent.provider == "openrouter"
