#!/usr/bin/env python
"""Deterministic A/B harness for provider-authoritative compression pressure.

Run from the repository root:
    python evals/token_accounting/ab_provider_usage_gates.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agent.model_metadata import (
    anchored_context_tokens,
    capture_usage_anchor,
    estimate_messages_tokens_rough,
    provider_prompt_with_delta,
)


THRESHOLD = 50_000
_AGENT_GATES = ("preflight", "idle", "pre_api", "post_tool")


def _msg(role: str, content: str, **extra) -> dict:
    return {"role": role, "content": content, **extra}


def _agent_case(prompt_tokens: int) -> dict:
    messages = [
        _msg("user", "x" * 400_000),
        _msg(
            "assistant",
            "checkpoint",
            codex_reasoning_items=[
                {"type": "reasoning", "encrypted_content": "z" * 1_000_000}
            ],
        ),
    ]
    anchor = capture_usage_anchor(prompt_tokens, 900_000, messages)
    messages.extend([_msg("assistant", "short reply"), _msg("tool", "small result")])
    pressure = anchored_context_tokens(messages, anchor)
    assert pressure is not None
    return {
        "rough_whole_context": estimate_messages_tokens_rough(messages),
        "provider_plus_delta": pressure,
        "gates": {gate: pressure >= THRESHOLD for gate in _AGENT_GATES},
    }


def _gateway_case(prompt_tokens: int) -> dict:
    history = [_msg("user", "x" * 400_000), _msg("assistant", "short reply")]
    pressure = provider_prompt_with_delta(prompt_tokens, history[-1:])
    assert pressure is not None
    return {
        "rough_whole_context": estimate_messages_tokens_rough(history),
        "provider_plus_delta": pressure,
        "gates": {"gateway_hygiene": pressure >= THRESHOLD},
    }


def main() -> None:
    report = {
        "under": {"agent": _agent_case(12_000), "gateway": _gateway_case(12_000)},
        "over": {"agent": _agent_case(60_000), "gateway": _gateway_case(60_000)},
    }
    assert not any(
        fired
        for surface in report["under"].values()
        for fired in surface["gates"].values()
    )
    assert all(
        fired
        for surface in report["over"].values()
        for fired in surface["gates"].values()
    )
    assert report["under"]["agent"]["rough_whole_context"] >= THRESHOLD
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
