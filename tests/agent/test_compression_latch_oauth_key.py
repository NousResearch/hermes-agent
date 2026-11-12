"""#138469: once the summary fallback latch is set, the pinned main-model route must not pin
the compressor's stored OAuth-shaped api_key. That token is revoked in place when rotated
elsewhere (another process refreshes it, or the Claude Code CLI re-auths), and an explicit
api_key overrides the credential pool inside ``call_llm`` — so every later compaction 401s
for the rest of the session. Static keys (Console ``sk-ant-api*``, custom-endpoint secrets)
have no rotate-and-revoke lifecycle and stay pinned.
"""

from unittest.mock import patch

from agent.context_compressor import ContextCompressor

# Shape prefixes only, assembled at runtime: the classification under test is the token's
# prefix shape, never a real credential value.
_ANTHROPIC_OAUTH_PREFIX = "sk-ant-" + "oaut-2-"
_JWT_PREFIX = "eyJ" + "hbGciOiJSUzI1NiJ9"
_ANTHROPIC_CONSOLE_PREFIX = "sk-ant-" + "api03-"


def _latched(provider: str = "anthropic", api_key: str = "", base_url: str = "") -> ContextCompressor:
    with patch("agent.context_compressor.get_model_context_length", return_value=100000):
        c = ContextCompressor(
            model="claude-opus-5-5", quiet_mode=True, base_url=base_url,
            api_key=api_key, provider=provider,
        )
    c._summary_model_fallen_back = True
    return c


def test_latched_route_does_not_pin_anthropic_oauth_token():
    c = _latched(api_key=_ANTHROPIC_OAUTH_PREFIX + "stale-token")

    call_kwargs: dict = {}
    c._apply_summary_route(call_kwargs)

    assert "api_key" not in call_kwargs
    assert call_kwargs["provider"] == "anthropic"
    assert call_kwargs["model"] == "claude-opus-5-5"


def test_latched_route_does_not_pin_jwt_shaped_token():
    c = _latched(provider="openai-codex", api_key=_JWT_PREFIX + ".stale.payload")

    call_kwargs: dict = {}
    c._apply_summary_route(call_kwargs)

    assert "api_key" not in call_kwargs
    assert call_kwargs["provider"] == "openai-codex"


def test_latched_route_still_pins_console_api_key():
    c = _latched(api_key=_ANTHROPIC_CONSOLE_PREFIX + "static-console-key")

    call_kwargs: dict = {}
    c._apply_summary_route(call_kwargs)

    assert call_kwargs["api_key"] == _ANTHROPIC_CONSOLE_PREFIX + "static-console-key"


def test_latched_route_still_pins_custom_endpoint_key():
    static_key = "sk-or-" + "v1-static"
    c = _latched(provider="custom", api_key=static_key, base_url="https://proxy.example/v1")

    call_kwargs: dict = {}
    c._apply_summary_route(call_kwargs)

    assert call_kwargs["api_key"] == static_key
    assert call_kwargs["base_url"] == "https://proxy.example/v1"


def test_unlatched_route_is_untouched():
    c = _latched(api_key=_ANTHROPIC_OAUTH_PREFIX + "stale-token")
    c._summary_model_fallen_back = False

    call_kwargs: dict = {}
    c._apply_summary_route(call_kwargs)

    assert call_kwargs == {}
