"""A proxy Responses route must not replay a whole session's sealed reasoning (#119336 follow-up).

``reasoning.encrypted_content`` is sealed to the backend identity that minted it. First-party
issuers keep that identity, so they replay their full history. A proxy or aggregator route (its
issuer kind starts with ``other:``) can rotate it between turns or across a gateway restart, and the
provider then answers the replayed blob with HTTP 400 ``invalid_encrypted_content``.

These tests pin the window that bounds that damage: on a proxy route only the
``codex_proxy_replay_turns`` most recent assistant turns carrying reasoning send their encrypted
sidecar. Assistant text is never affected, and first-party issuers are untouched.

The window limits blast radius only. It does not decide when to stop replaying: that verdict belongs
to ``AIAgent._disable_codex_reasoning_replay`` (``agent/turn_recovery.py``), which #130787 made a
one-strike, session-wide rule.
"""
from __future__ import annotations

import pytest

from agent.codex_responses_adapter import _chat_messages_to_responses_input
from agent.codex_proxy_replay_window import (
    DEFAULT_PROXY_REPLAY_TURNS, proxy_replay_max_turns, proxy_replay_turns_from_config,
)

_ZEN_BASE_URL = "https://opencode.ai/zen/go/v1"
_ZEN_KIND = f"other:{_ZEN_BASE_URL}"
_MODEL = "gpt-5-codex"


def _history(turns: int, *, issuer_kind: str = _ZEN_KIND) -> list[dict]:
    """`turns` assistant turns, each with one sealed reasoning item stamped for `issuer_kind`."""
    messages: list[dict] = []
    for i in range(turns):
        messages.append({"role": "user", "content": f"ask {i}"})
        messages.append({
            "role": "assistant", "content": f"answer {i}",
            "codex_reasoning_items": [
                {
                    "type": "reasoning", "id": f"rs_{i}", "encrypted_content": f"ENC-{i}",
                    "_issuer_kind": issuer_kind, "_issuer_model": _MODEL,
                }
            ],
        })
    return messages


def _wire(messages, *, issuer_kind: str = _ZEN_KIND, cap=None) -> list[dict]:
    return _chat_messages_to_responses_input(
        messages, current_issuer_kind=issuer_kind, current_issuer_model=_MODEL,
        proxy_replay_max_turns=cap,
    )


def _encs(items: list[dict]) -> list[str]:
    return [i["encrypted_content"] for i in items if i.get("type") == "reasoning"]


def _assistant_text(items: list[dict]) -> list[str]:
    return [i.get("content") for i in items if i.get("role") == "assistant"]


class _Agent:
    """Minimal agent: only the replay window is read off it."""

    def __init__(self, *, proxy_replay_turns=None):
        if proxy_replay_turns is not None:
            self.codex_proxy_replay_turns = proxy_replay_turns

    def __getattr__(self, name):
        return None


def test_proxy_issuer_replays_only_the_last_n_turns():
    items = _wire(_history(10), cap=2)

    assert _encs(items) == ["ENC-8", "ENC-9"]


def test_older_turns_keep_their_text_and_lose_only_the_sealed_sidecar():
    items = _wire(_history(4), cap=2)

    # Every answer still reaches the wire; only the two oldest reasoning blobs are withheld.
    assert _assistant_text(items) == ["answer 0", "answer 1", "answer 2", "answer 3"]
    assert _encs(items) == ["ENC-2", "ENC-3"]


def test_first_party_issuer_replays_the_whole_history():
    """Codex/xAI/GitHub issuers keep a stable sealing identity, so the window must not touch them."""
    history = _history(10, issuer_kind="codex_backend")

    assert _encs(_wire(history, issuer_kind="codex_backend", cap=2)) == [f"ENC-{i}" for i in range(10)]


def test_no_cap_leaves_every_issuer_uncapped():
    history = _history(6)

    assert _encs(_wire(history, cap=None)) == [f"ENC-{i}" for i in range(6)]


def test_turns_without_reasoning_do_not_consume_the_window():
    """Only turns that actually carry a sealed blob compete for the window. A long stretch of
    plain-text turns must not push the reasoning out of range."""
    messages = [
        {"role": "user", "content": "ask 0"},
        {"role": "assistant", "content": "answer 0"},
        {"role": "user", "content": "ask 1"},
        {
            "role": "assistant", "content": "answer 1",
            "codex_reasoning_items": [
                {"type": "reasoning", "id": "rs_a", "encrypted_content": "ENC-A",
                 "_issuer_kind": _ZEN_KIND, "_issuer_model": _MODEL},
            ],
        },
    ]

    assert _encs(_wire(messages, cap=1)) == ["ENC-A"]


@pytest.mark.parametrize("cap", [0, 1, 2, 5, 50])
def test_any_cap_keeps_at_most_that_many_reasons(cap):
    assert len(_encs(_wire(_history(10), cap=cap))) == min(cap, 10)


def test_zero_cap_sends_no_sealed_reasoning_at_all():
    items = _wire(_history(10), cap=0)

    assert _encs(items) == []
    assert _assistant_text(items) == [f"answer {i}" for i in range(10)]


def test_window_is_honored_from_config_and_defaults_to_two():
    assert DEFAULT_PROXY_REPLAY_TURNS == 2
    assert proxy_replay_max_turns(_Agent()) is None  # no key = uncapped
    assert proxy_replay_turns_from_config(None) == 2
    assert proxy_replay_turns_from_config(5) == 5
    assert proxy_replay_turns_from_config("3") == 3
    assert proxy_replay_turns_from_config(0) == 0
    assert proxy_replay_max_turns(_Agent(proxy_replay_turns=5)) == 5


@pytest.mark.parametrize("raw", ["-1", "abc", "", True, 1.5])
def test_invalid_window_falls_back_to_the_default(raw):
    assert proxy_replay_turns_from_config(raw) == DEFAULT_PROXY_REPLAY_TURNS


def test_wider_window_sends_more_history():
    assert _encs(_wire(_history(10), cap=5)) == [f"ENC-{i}" for i in range(5, 10)]


def test_config_default_matches_the_module_default():
    from hermes_cli.config_defaults import DEFAULT_CONFIG

    assert DEFAULT_CONFIG["agent"]["codex_proxy_replay_turns"] == DEFAULT_PROXY_REPLAY_TURNS == 2


def test_config_value_reaches_the_wire(tmp_path, monkeypatch):
    """Production path: the key is read through the real config loader, so the window that reaches
    the wire is the one the operator set."""
    import textwrap

    from hermes_cli.config import load_config

    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text(
        textwrap.dedent(
            """
            agent:
              codex_proxy_replay_turns: 4
            """
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    agent = _Agent(proxy_replay_turns=load_config().get("agent", {}).get("codex_proxy_replay_turns"))

    assert proxy_replay_max_turns(agent) == 4
    assert _encs(_wire(_history(10), cap=proxy_replay_max_turns(agent))) == [f"ENC-{i}" for i in range(6, 10)]
