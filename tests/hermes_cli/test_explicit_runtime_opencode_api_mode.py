"""Regression tests for issue #21419.

`_resolve_explicit_runtime()` must re-derive `api_mode` from the effective
model (target_model > model_cfg.default) for opencode-zen / opencode-go,
the same way `_resolve_runtime_from_pool_entry()` already does. Without
this, a stale `api_mode: anthropic_messages` from a previous Claude
session leaks onto a non-Claude opencode model, sending Anthropic-format
requests to a chat_completions endpoint and getting 404.
"""

from hermes_cli.runtime_provider import _resolve_explicit_runtime


def _resolve(provider: str, model_default: str, configured_api_mode: str = "anthropic_messages",
             *, target_model: str = None):
    return _resolve_explicit_runtime(
        provider=provider,
        requested_provider=provider,
        model_cfg={"default": model_default, "api_mode": configured_api_mode},
        explicit_api_key="sk-test",
        explicit_base_url="https://opencode.ai/zen/v1" if provider == "opencode-zen" else "https://opencode.ai/zen/go/v1",
        target_model=target_model,
    )


def test_opencode_zen_non_claude_overrides_stale_anthropic_api_mode():
    """The bug: config has api_mode=anthropic_messages from previous Claude session,
    but the active model is non-Claude — must resolve to chat_completions."""
    runtime = _resolve("opencode-zen", "big-pickle", configured_api_mode="anthropic_messages")
    assert runtime is not None
    assert runtime["api_mode"] == "chat_completions"


def test_opencode_zen_claude_model_resolves_anthropic_messages():
    runtime = _resolve("opencode-zen", "claude-sonnet-4-5", configured_api_mode="chat_completions")
    assert runtime is not None
    assert runtime["api_mode"] == "anthropic_messages"
    # The /v1 strip must apply: Anthropic SDK appends /v1/messages, so the
    # base_url should end at /zen (no trailing /v1) to avoid /v1/v1/messages.
    assert not runtime["base_url"].endswith("/v1")
    assert runtime["base_url"].endswith("/zen")


def test_opencode_zen_gpt_model_resolves_codex_responses():
    runtime = _resolve("opencode-zen", "gpt-5", configured_api_mode="anthropic_messages")
    assert runtime is not None
    assert runtime["api_mode"] == "codex_responses"


def test_opencode_go_minimax_resolves_anthropic_messages():
    runtime = _resolve("opencode-go", "minimax-m2", configured_api_mode="chat_completions")
    assert runtime is not None
    assert runtime["api_mode"] == "anthropic_messages"


def test_opencode_go_non_minimax_resolves_chat_completions():
    runtime = _resolve("opencode-go", "glm-4.6", configured_api_mode="anthropic_messages")
    assert runtime is not None
    assert runtime["api_mode"] == "chat_completions"


def test_target_model_overrides_model_cfg_default():
    """A mid-session /model switch passes target_model — that must win
    over the persisted config default."""
    runtime = _resolve(
        "opencode-zen",
        "claude-sonnet-4-5",  # stale persisted default
        configured_api_mode="anthropic_messages",
        target_model="big-pickle",  # the model we are switching TO
    )
    assert runtime is not None
    assert runtime["api_mode"] == "chat_completions"


def test_non_opencode_provider_unchanged():
    """Sanity: providers that are not opencode-zen / opencode-go must still
    honour the configured api_mode (no behavioural change)."""
    runtime = _resolve_explicit_runtime(
        provider="kilocode",
        requested_provider="kilocode",
        model_cfg={"default": "anything", "api_mode": "anthropic_messages"},
        explicit_api_key="sk-test",
        explicit_base_url="https://api.kilo.ai/api/gateway",
    )
    assert runtime is not None
    assert runtime["api_mode"] == "anthropic_messages"