"""Regression: AIAgent.__init__ must route Kimi /coding to anthropic_messages.

Kimi Code's ``api.kimi.com/coding`` endpoint speaks the **Anthropic Messages**
protocol (it accepts Claude Code's native request shape), even on URLs like
``.../coding/v1`` that do NOT carry the ``/anthropic`` suffix.

Three other resolution paths already detect this correctly:

- ``hermes_cli.providers.determine_api_mode`` (the canonical resolver)
- ``hermes_cli.runtime_provider._detect_api_mode_for_url``
- ``tools.delegate_tool`` child-runtime resolution

But the hand-rolled api_mode inference inside ``agent.agent_init`` (the block
that runs in ``AIAgent.__init__``) omitted the rule, so a freshly constructed
agent pointed at ``api.kimi.com/coding`` fell through to ``chat_completions``.
That sends OpenAI-shaped requests to an Anthropic-Messages endpoint — the wrong
wire protocol — which is exactly the routing-drift class this guards against.

The contract asserted here is an *invariant between two resolvers*: whatever
``determine_api_mode(provider, base_url)`` decides for the Kimi /coding
endpoint, a real ``AIAgent`` constructed with that same provider/base_url
(and no explicit ``api_mode``) must agree. This is not a snapshot of a literal
value — if the canonical mapping ever changes, both sides move together.
"""

from __future__ import annotations

from pathlib import Path

import pytest


def _prep_home(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / ".env").write_text("", encoding="utf-8")
    (tmp_path / "config.yaml").write_text("", encoding="utf-8")


def _make_agent(tmp_path: Path, **overrides):
    from run_agent import AIAgent

    kwargs = dict(
        model="kimi-k2.5",
        api_key="sk-kimi-dummy",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
        platform="cli",
    )
    kwargs.update(overrides)
    return AIAgent(**kwargs)


# URLs the Kimi /coding endpoint is reached through. None carry an explicit
# /anthropic suffix except the last; the bug was that the non-/anthropic forms
# (the common ones) fell through to chat_completions.
_KIMI_CODING_URLS = [
    "https://api.kimi.com/coding",
    "https://api.kimi.com/coding/v1",
    "https://api.kimi.com/coding/",
    "https://api.kimi.com/coding/anthropic",
]


@pytest.mark.parametrize("base_url", _KIMI_CODING_URLS)
@pytest.mark.parametrize("provider", ["kimi-coding", "custom"])
def test_kimi_coding_infers_anthropic_messages(monkeypatch, tmp_path, provider, base_url):
    """A fresh AIAgent on Kimi /coding must use the anthropic_messages mode."""
    _prep_home(tmp_path, monkeypatch)
    agent = _make_agent(tmp_path, provider=provider, base_url=base_url)
    assert agent.api_mode == "anthropic_messages", (
        f"provider={provider!r} base_url={base_url!r} resolved to "
        f"{agent.api_mode!r}; Kimi /coding speaks Anthropic Messages."
    )


@pytest.mark.parametrize("base_url", _KIMI_CODING_URLS)
@pytest.mark.parametrize("provider", ["kimi-coding", "custom"])
def test_agent_api_mode_matches_canonical_resolver(monkeypatch, tmp_path, provider, base_url):
    """Behavior contract: AIAgent's inferred api_mode == determine_api_mode().

    The two resolvers must not drift. determine_api_mode is the canonical
    source of truth used by the model-switch / runtime pipelines; the
    in-__init__ inference must produce the same wire protocol for the same
    provider + base_url.
    """
    from hermes_cli.providers import determine_api_mode

    _prep_home(tmp_path, monkeypatch)
    expected = determine_api_mode(provider, base_url)
    agent = _make_agent(tmp_path, provider=provider, base_url=base_url)
    assert agent.api_mode == expected, (
        f"AIAgent inferred {agent.api_mode!r} but determine_api_mode said "
        f"{expected!r} for provider={provider!r} base_url={base_url!r}."
    )


def test_explicit_api_mode_override_is_respected(monkeypatch, tmp_path):
    """An explicit api_mode the user passes always wins over inference."""
    _prep_home(tmp_path, monkeypatch)
    agent = _make_agent(
        tmp_path,
        provider="kimi-coding",
        base_url="https://api.kimi.com/coding/v1",
        api_mode="chat_completions",
    )
    assert agent.api_mode == "chat_completions"


def test_kimi_chat_endpoint_stays_chat_completions(monkeypatch, tmp_path):
    """Don't over-fire: Kimi's OpenAI-compatible /v1 endpoint is NOT /coding.

    The kimi-coding provider has dual endpoints — sk-kimi-* keys hit
    api.kimi.com/coding (Anthropic), legacy keys hit api.moonshot.ai/v1
    (OpenAI chat completions). The fix must key on the /coding path, not the
    provider name, so the chat endpoint is left untouched.
    """
    _prep_home(tmp_path, monkeypatch)
    agent = _make_agent(
        tmp_path,
        provider="kimi-coding",
        base_url="https://api.moonshot.ai/v1",
    )
    assert agent.api_mode == "chat_completions"
