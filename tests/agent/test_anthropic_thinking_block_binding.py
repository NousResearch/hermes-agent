"""Thinking-prefix block binding (Claude 5.1+) — port of anomalyco/opencode#47884.

Claude 5.1+ binds each thinking-block signature to the full conversation prefix; a replay
behind a changed prefix answers HTTP 400 unless the request opts into
``thinking.block_binding.prefix_mismatch_behavior="drop_block"`` under the
``thinking-binding-controls-2026-08-01`` beta. Hermes mutates the prefix in sanctioned ways
(compression, orphan tool-call stripping), so the opt-in defaults on for enforcing models.
"""

import pytest

from agent.anthropic_adapter import build_anthropic_kwargs


_MSGS = [{"role": "user", "content": "hello"}]


@pytest.mark.parametrize(
    "model, expected",
    [
        ("claude-fable-5-1", True),
        ("claude-fable-5.1", True),
        ("anthropic/claude-fable-5.1", True),
        ("claude-fable-5-1@20260901", True),
        ("claude-sonnet-6", True),
        # A snapshot date is not a minor version.
        ("claude-opus-5-20260901", False),
        ("claude-opus-4-8", False),
        ("claude-fable-5", False),
        ("claude-haiku-5-1", False),  # no extended thinking
        ("kimi-k2.5", False),
    ],
)
def test_block_binding_defaults_by_model_version(model, expected):
    """Enforcing models (5.1+) get drop_block + the binding beta; older/foreign models are
    untouched (some deployments reject the unknown parameter — opencode#46848)."""
    kwargs = build_anthropic_kwargs(
        model=model, messages=list(_MSGS), tools=None, max_tokens=1024,
        reasoning_config={"enabled": True, "effort": "medium"},
    )
    thinking = kwargs.get("thinking") or {}
    binding = thinking.get("block_binding")
    header = kwargs.get("extra_headers", {}).get("anthropic-beta", "")
    if expected:
        assert binding == {"prefix_mismatch_behavior": "drop_block"}
        assert "thinking-binding-controls-2026-08-01" in header
        # The rebuilt header must keep the always-on betas (it OVERRIDES the client header).
        assert "interleaved-thinking-2025-05-14" in header
    else:
        assert binding is None
        assert "thinking-binding-controls-2026-08-01" not in header


def test_block_binding_respects_disable_and_third_party_endpoints():
    """An explicit thinking disable keeps the omission (nothing replays), and third-party
    Anthropic-compatible endpoints — where signatures are stripped anyway — never receive the
    unknown parameter/beta. reasoning_config=None still binds: adaptive models think by default.
    (claude-sonnet-6 accepts a true disable; mandatory-thinking families like claude-fable omit the
    disable and keep thinking on, so they correctly keep the binding.)"""
    disabled = build_anthropic_kwargs(
        model="claude-sonnet-6", messages=list(_MSGS), tools=None, max_tokens=1024,
        reasoning_config={"enabled": False},
    )
    assert "block_binding" not in (disabled.get("thinking") or {})

    third_party = build_anthropic_kwargs(
        model="claude-fable-5-1", messages=list(_MSGS), tools=None, max_tokens=1024,
        reasoning_config={"enabled": True, "effort": "medium"},
        base_url="https://api.minimax.io/anthropic",
    )
    assert "block_binding" not in (third_party.get("thinking") or {})
    assert "thinking-binding-controls" not in third_party.get("extra_headers", {}).get("anthropic-beta", "")

    no_reasoning = build_anthropic_kwargs(
        model="claude-fable-5-1", messages=list(_MSGS), tools=None, max_tokens=1024,
        reasoning_config=None,
    )
    assert (no_reasoning.get("thinking") or {}).get("block_binding") == {
        "prefix_mismatch_behavior": "drop_block"
    }
