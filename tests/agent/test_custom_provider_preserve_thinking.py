"""A custom provider can opt in to keeping signed thinking blocks (#120723).

Third-party Anthropic-compatible endpoints strip every thinking block because the
signatures are proprietary to Anthropic -- but a trusted proxy that re-signs on
the way out (signed replay) needs the blocks kept, or every intermediate turn
loses its reasoning and the prompt-cache prefix diverges from what the proxy
saw. The provider entry's ``preserve_thinking: true`` opts in; absent or false
keeps today's strip behavior (fail-closed default).
"""

import copy
from types import SimpleNamespace

import pytest

from agent.anthropic_adapter import build_anthropic_kwargs
from agent.anthropic_message_convert import convert_messages_to_anthropic
from agent.anthropic_thinking_policy import (
    anthropic_thinking_route,
    native_anthropic_preserves_prior_thinking,
)
from agent.transports import get_transport
from hermes_cli.config_providers import (
    _custom_provider_entry_to_provider_config,
    _normalize_custom_provider_entry,
    get_custom_provider_extra_headers,
    get_custom_provider_preserve_thinking,
)

SIG = "sig-proxy"
RELAY = "https://relay.example.internal/anthropic"  # non-anthropic.com -> third-party
# Rework tests resolve the opt-in from the provider config (not the kwarg), so they point at
# their own host and monkeypatch the config lookup for it.
REWORK_RELAY = "https://trusted-proxy.example.internal/anthropic"


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


def test_opt_in_maps_the_route_to_native():
    """The opt-in IS the native mapping (#123021 rework): a third-party endpoint with
    ``preserve_thinking=True`` resolves through ``anthropic_thinking_route`` as ``native`` —
    the converter, the signature-rejection recovery and every native call site then share one
    contract. Kimi/DeepSeek keep their own routes (their judgment outranks the opt-in)."""
    assert anthropic_thinking_route(RELAY, "claude-sonnet-4", preserve_thinking=True) == "native"
    assert anthropic_thinking_route(RELAY, "claude-sonnet-4", preserve_thinking=False) == "third_party"
    assert anthropic_thinking_route(RELAY, "claude-sonnet-4", preserve_thinking=None) == "third_party"
    assert anthropic_thinking_route(None, "claude-sonnet-4", preserve_thinking=True) == "native"
    assert anthropic_thinking_route(RELAY, "kimi-k2.5", preserve_thinking=True) == "kimi"
    assert anthropic_thinking_route(RELAY, "deepseek-r2", preserve_thinking=True) == "deepseek"


def test_opt_in_benefits_match_the_native_capability_split():
    """The mapped route must behave EXACTLY like direct Anthropic for the same model
    (#123021): preserve-prior models (Opus 4.5+/Sonnet 4.6+) keep signed thinking on every
    assistant turn — the prompt-cache prefix then never diverges from what native computes —
    while last-turn-only models (older Claude) take the native latest-turn path. No separate
    trusted-proxy arm exists; the shared classifier decides."""
    for model, expected in (("claude-opus-4.5", 2), ("claude-sonnet-4.6", 2),
                            ("claude-sonnet-4", 1), ("claude-haiku-4.5", 1)):
        _sys, out = convert_messages_to_anthropic(
            _multi_turn_conversation(2), base_url=REWORK_RELAY, model=model, preserve_thinking=True)
        assert _signed_block_count(out) == expected, model


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


def _stored_signed_turn():
    """One normalized assistant turn carrying a signed thinking block."""
    response = SimpleNamespace(
        content=[
            SimpleNamespace(type="thinking", thinking="five: 5 11 27 63 88", signature=SIG),
            SimpleNamespace(type="text", text="5 27 88"),
        ],
        stop_reason="end_turn",
        usage=None,
    )
    normalized = get_transport("anthropic_messages").normalize_response(response)
    return {
        "role": "assistant",
        "content": normalized.content or "",
        "reasoning_details": (normalized.provider_data or {}).get("reasoning_details"),
    }


def _signed_block_count(out):
    return sum(1 for m in out if m.get("role") == "assistant"
               for b in (m.get("content") or [])
               if isinstance(b, dict) and b.get("type") == "thinking" and b.get("signature"))


def _multi_turn_conversation(turns):
    turn = _stored_signed_turn()
    messages = [{"role": "user", "content": "warmup"}]
    for i in range(turns):
        messages.append(copy.deepcopy(turn))
        messages.append({"role": "user", "content": f"q{i}"})
    return messages


@pytest.mark.parametrize("turns", [2, 4, 8])
def test_opt_in_preserve_prior_model_keeps_every_turn(turns):
    """Rework invariant (#123021): behind a trusted proxy a preserve-prior model (Opus 4.5+ /
    Sonnet 4.6+) keeps its signed thinking on EVERY assistant turn, exactly like direct
    Anthropic — the prompt-cache prefix then never diverges from what native computes
    (latest-turn-only cache-broke ~84% of requests; keep-all replayed ~18k requests over
    5 days with zero signature rejections). The old trusted-proxy latest-turn arm keeps 1."""
    _sys, out = convert_messages_to_anthropic(
        _multi_turn_conversation(turns), base_url=RELAY, model="claude-opus-4.5",
        preserve_thinking=True)
    assert _signed_block_count(out) == turns, [
        [b.get("type") for b in (m.get("content") or []) if isinstance(b, dict)]
        for m in out if m.get("role") == "assistant"]


@pytest.mark.parametrize("turns", [2, 4, 8])
def test_opt_in_last_turn_only_model_keeps_only_the_latest_turn(turns):
    """Rework invariant (#123021): behind a trusted proxy, last-turn-only models (Haiku /
    Claude 3 / pre-4.5 Opus / pre-4.6 Sonnet) follow the NATIVE latest-turn behavior — their
    stale historical signatures would answer 400 on a verbatim pass-through. No separate
    trusted-proxy arm: the shared route classifier decides."""
    _sys, out = convert_messages_to_anthropic(
        _multi_turn_conversation(turns), base_url=RELAY, model="claude-sonnet-4",
        preserve_thinking=True)
    assert _signed_block_count(out) == 1, [
        [b.get("type") for b in (m.get("content") or []) if isinstance(b, dict)]
        for m in out if m.get("role") == "assistant"]


def test_kimi_precedence_untouched_by_the_opt_in():
    """Kimi handling outranks the opt-in: a Kimi-family model keeps its replay-as-is semantics,
    so BOTH turns keep their blocks (the latest-turn relaxation must not re-shape Kimi turns)."""
    _sys, out = convert_messages_to_anthropic(
        _multi_turn_conversation(2), base_url=RELAY, model="kimi-k2.5", preserve_thinking=True)
    assert _signed_block_count(out) == 2


def test_deepseek_precedence_untouched_by_the_opt_in():
    """DeepSeek handling outranks the opt-in: signed blocks stay stripped (DeepSeek rejects
    signed ones), the opt-in must not resurrect them."""
    _sys, out = convert_messages_to_anthropic(
        _multi_turn_conversation(2), base_url=RELAY, model="deepseek-r2", preserve_thinking=True)
    assert _signed_block_count(out) == 0


def test_invalidated_signature_still_downgrades_under_opt_in():
    """Orphan-stripping a tool_use kills that turn's signatures (they were signed against the
    original content, and the #123583-style strip flags it); under the opt-in the keep path
    demotes the thinking to text instead of replaying a stale signature into a 400 loop."""
    response = SimpleNamespace(
        content=[
            SimpleNamespace(type="thinking", thinking="five: 5 11 27 63 88", signature=SIG),
            SimpleNamespace(type="tool_use", id="tu_1", name="shell", input={"cmd": "ls"}),
        ],
        stop_reason="tool_use",
        usage=None,
    )
    normalized = get_transport("anthropic_messages").normalize_response(response)
    turn = {
        "role": "assistant",
        "content": normalized.content or "",
        "reasoning_details": (normalized.provider_data or {}).get("reasoning_details"),
        "tool_calls": [{"id": "tu_1", "type": "function",
                        "function": {"name": "shell", "arguments": "{\"cmd\": \"ls\"}"}}],
    }
    messages = [{"role": "user", "content": "q1"}, turn, {"role": "user", "content": "q2"}]
    _sys, out = convert_messages_to_anthropic(
        messages, base_url=RELAY, model="claude-sonnet-4", preserve_thinking=True)
    assert _signed_block_count(out) == 0
    assistant = [m for m in out if m.get("role") == "assistant"][0]
    assert any(isinstance(b, dict) and b.get("type") == "text" and "five" in (b.get("text") or "")
               for b in assistant["content"])


def test_transport_seam_resolves_the_opt_in(monkeypatch):
    """convert_messages at the transport seam resolves the same opt-in build_anthropic_kwargs
    does (explicit kwarg wins, else the provider entry) — a plugin calling the seam directly
    must not silently get the strip path."""
    transport = get_transport("anthropic_messages")
    turn = _stored_signed_turn()
    messages = [{"role": "user", "content": "q1"}, turn, {"role": "user", "content": "q2"}]

    monkeypatch.setattr(
        "hermes_cli.config.get_custom_provider_preserve_thinking", lambda base_url: True)
    _sys, out = transport.convert_messages(copy.deepcopy(messages), base_url=RELAY)
    assert _signed_block_count(out) == 1

    monkeypatch.setattr(
        "hermes_cli.config.get_custom_provider_preserve_thinking", lambda base_url: False)
    _sys, out = transport.convert_messages(copy.deepcopy(messages), base_url=RELAY)
    assert _signed_block_count(out) == 0


def test_opt_in_resolved_from_config_matches_the_kwarg(monkeypatch):
    """The native mapping resolves the provider opt-in at the ROUTE, so the converter, the
    signature-rejection recovery and the native call sites (context accounting, sanitization,
    turn assembly) all see the SAME decision as build_anthropic_kwargs — the #123021 benefit
    that made the rework preferable to a converter-only arm."""
    monkeypatch.setattr(
        "hermes_cli.config_providers.get_custom_provider_preserve_thinking",
        lambda base_url, custom_providers=None, config=None: base_url == REWORK_RELAY)
    messages = _multi_turn_conversation(2)
    # Config-resolved opt-in (preserve_thinking=None) == explicit kwarg: keep-all for a
    # preserve-prior model, and the shared classifier agrees.
    _sys, out = convert_messages_to_anthropic(
        copy.deepcopy(messages), base_url=REWORK_RELAY, model="claude-opus-4.5",
        preserve_thinking=None)
    assert _signed_block_count(out) == 2
    _sys, kwarg_out = convert_messages_to_anthropic(
        copy.deepcopy(messages), base_url=REWORK_RELAY, model="claude-opus-4.5",
        preserve_thinking=True)
    assert _signed_block_count(kwarg_out) == 2
    assert anthropic_thinking_route(REWORK_RELAY, "claude-opus-4.5") == "native"
    assert native_anthropic_preserves_prior_thinking(REWORK_RELAY, "claude-opus-4.5")


def test_kimi_and_deepseek_outrank_the_config_resolved_opt_in(monkeypatch):
    """The opt-in never rewrites the Kimi / DeepSeek contracts, whichever way it arrives."""
    assert anthropic_thinking_route(None, "kimi-k2.5", preserve_thinking=True) == "kimi"
    # Third-party + DeepSeek-thinking model: the deepseek judgment wins even with the opt-in.
    assert anthropic_thinking_route(RELAY, "deepseek-r2", preserve_thinking=True) == "deepseek"
    assert anthropic_thinking_route(RELAY, "deepseek-r2") == "deepseek"  # judgment, opt-in irrelevant
    _sys, out = convert_messages_to_anthropic(
        _multi_turn_conversation(2), base_url=RELAY, model="kimi-k2.5", preserve_thinking=True)
    assert _signed_block_count(out) == 2
    _sys, out = convert_messages_to_anthropic(
        _multi_turn_conversation(2), base_url=RELAY, model="deepseek-r2", preserve_thinking=True)
    assert _signed_block_count(out) == 0
