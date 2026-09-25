"""A custom provider can opt in to keeping signed thinking blocks (#120723).

Third-party Anthropic-compatible endpoints strip every thinking block because the
signatures are proprietary to Anthropic -- but a trusted proxy that re-signs on
the way out (signed replay) needs the blocks kept, or every intermediate turn
loses its reasoning and the prompt-cache prefix diverges from what the proxy
saw. The provider entry's ``preserve_thinking: true`` opts in; absent or false
keeps today's strip behavior (fail-closed default).
"""

from types import SimpleNamespace

from agent.anthropic_adapter import build_anthropic_kwargs
from agent.anthropic_message_convert import convert_messages_to_anthropic
from agent.transports import get_transport
from hermes_cli.config_providers import (
    _custom_provider_entry_to_provider_config,
    _normalize_custom_provider_entry,
    get_custom_provider_extra_headers,
    get_custom_provider_preserve_thinking,
)

SIG = "sig-proxy"
RELAY = "https://relay.example.internal/anthropic"  # non-anthropic.com -> third-party


def _thinking_turns(base_url, model="claude-sonnet-4", preserve_thinking=None):
    """Normalize a signed thinking turn, replay it mid-conversation, return its blocks."""
    response = SimpleNamespace(
        content=[
            SimpleNamespace(type="thinking", thinking="five: 5 11 27 63 88", signature=SIG),
            SimpleNamespace(type="text", text="5 27 88"),
        ],
        stop_reason="end_turn",
        usage=None,
    )
    normalized = get_transport("anthropic_messages").normalize_response(response)
    stored = {
        "role": "assistant",
        "content": normalized.content or "",
        "reasoning_details": (normalized.provider_data or {}).get("reasoning_details"),
    }
    messages = [
        {"role": "user", "content": "q1"},
        stored,
        {"role": "user", "content": "q2"},
    ]
    _sys, out = convert_messages_to_anthropic(
        messages, base_url=base_url, model=model, preserve_thinking=preserve_thinking)
    assistant = [m for m in out if m.get("role") == "assistant"][0]
    return [b for b in assistant["content"] if isinstance(b, dict) and b.get("type") == "thinking"]


def test_relay_strips_thinking_by_default():
    """Without the opt-in a third-party endpoint keeps today's strip behavior."""
    assert _thinking_turns(RELAY, preserve_thinking=False) == []


def test_relay_opt_in_keeps_signed_blocks():
    """preserve_thinking keeps the signed block on the latest turn, signature intact."""
    blocks = _thinking_turns(RELAY, preserve_thinking=True)
    assert blocks and blocks[0].get("signature") == SIG


def test_direct_anthropic_ignores_the_flag():
    """The flag only relaxes the third-party strip; direct Anthropic is untouched."""
    assert _thinking_turns(None, preserve_thinking=False)
    assert _thinking_turns(None, preserve_thinking=True)


def _relay_entry(**extra):
    entry = {"name": "relay", "base_url": RELAY, "api_mode": "anthropic_messages"}
    entry.update(extra)
    return entry


def test_provider_entry_parses_and_preserves_the_flag():
    """The flag round-trips the provider config chain; only a real bool opts in."""
    assert get_custom_provider_preserve_thinking(
        RELAY, custom_providers=[_relay_entry(preserve_thinking=True)]) is True
    # Fail-closed: absent / false / non-bool all keep the strip behavior.
    for value in (None, False, "false", "yes", 1):
        entry = {} if value is None else {"preserve_thinking": value}
        assert get_custom_provider_preserve_thinking(
            RELAY, custom_providers=[_relay_entry(**entry)]) is False
    # Non-matching routes are unaffected.
    assert get_custom_provider_preserve_thinking(
        "https://other.example.internal/anthropic",
        custom_providers=[_relay_entry(preserve_thinking=True)]) is False


def test_flag_survives_normalization_and_translation():
    """Normalize (legacy list + v12 providers) and the v12 translation keep the flag."""
    normalized = _normalize_custom_provider_entry(_relay_entry(preserve_thinking=True))
    assert normalized["preserve_thinking"] is True
    translated = _custom_provider_entry_to_provider_config(normalized)
    assert translated["preserve_thinking"] is True


def test_flag_coexists_with_extra_headers():
    """The trusted-proxy auth shape: extra_headers still merge, the flag still reads."""
    entry = _relay_entry(preserve_thinking=True, extra_headers={"x-proxy-auth": "tok"})
    providers = [entry]
    assert get_custom_provider_extra_headers(RELAY, custom_providers=providers) == {
        "x-proxy-auth": "tok"}
    assert get_custom_provider_preserve_thinking(RELAY, custom_providers=providers) is True


def test_build_kwargs_resolves_flag_from_config(monkeypatch):
    """build_anthropic_kwargs consults the provider config; explicit False wins."""
    monkeypatch.setattr(
        "hermes_cli.config.get_custom_provider_preserve_thinking",
        lambda base_url: True)
    response = SimpleNamespace(
        content=[
            SimpleNamespace(type="thinking", thinking="five: 5 11 27 63 88", signature=SIG),
            SimpleNamespace(type="text", text="5 27 88"),
        ],
        stop_reason="end_turn",
        usage=None,
    )
    normalized = get_transport("anthropic_messages").normalize_response(response)
    messages = [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": normalized.content or "",
         "reasoning_details": (normalized.provider_data or {}).get("reasoning_details")},
        {"role": "user", "content": "q2"},
    ]

    def _thinking_blocks(**overrides):
        kwargs = build_anthropic_kwargs(
            model="claude-sonnet-4", messages=messages, tools=None, max_tokens=1024,
            reasoning_config=None, base_url=RELAY, **overrides)
        assistant = [m for m in kwargs["messages"] if m.get("role") == "assistant"][0]
        return [b for b in assistant["content"] if isinstance(b, dict) and b.get("type") == "thinking"]

    # Config says preserve -> signed blocks survive the third-party strip.
    blocks = _thinking_blocks()
    assert blocks and blocks[0].get("signature") == SIG
    # An explicit False (session-level off-switch) beats the config opt-in.
    assert _thinking_blocks(preserve_thinking=False) == []
