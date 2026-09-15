"""Transcript fixtures for the compaction eval harness.

Real transcripts are supplied by path (never committed). This module loads
them, estimates tokens the same way the harness scores them, and can generate
a small synthetic transcript so CI smoke tests run without real data.
"""
from __future__ import annotations

import json
import random
from typing import Any, Dict, List


def estimate_tokens(msg: Dict[str, Any]) -> int:
    """Chars/4 estimate, matching the harness's scoring convention."""
    total = len(msg.get("content") or "") if isinstance(msg.get("content"), str) else 0
    tc = msg.get("tool_calls")
    if tc:
        total += len(json.dumps(tc, default=str))
    return total // 4


def total_tokens(messages: List[Dict[str, Any]]) -> int:
    return sum(estimate_tokens(m) for m in messages)


def load_transcript(path: str, cap_tokens: int | None = None) -> List[Dict[str, Any]]:
    """Load a transcript JSON ({"messages": [...]}) and optionally cap it.

    The cap takes the chronological prefix, then drops trailing assistant
    tool_calls whose results were cut off so the input is well-formed.
    """
    data = json.load(open(path, encoding="utf-8"))
    msgs = data["messages"] if isinstance(data, dict) else data
    if cap_tokens is None:
        return msgs
    prefix: List[Dict[str, Any]] = []
    running = 0
    for m in msgs:
        t = estimate_tokens(m)
        if running + t > cap_tokens and len(prefix) > 10:
            break
        prefix.append(m)
        running += t
    while prefix and prefix[-1].get("tool_calls"):
        prefix.pop()
    return prefix


def synthetic_transcript(n_turns: int = 60, seed: int = 7) -> List[Dict[str, Any]]:
    """Deterministic fake transcript with plantable facts for smoke tests.

    Every 10th turn plants a distinctive fact ("The deploy code for region
    N is XYZ") so smoke tests can assert recall mechanics without an LLM.
    """
    rng = random.Random(seed)
    msgs: List[Dict[str, Any]] = [
        {"role": "system", "content": "You are a test agent."},
        {"role": "user", "content": "Work through the checklist and remember the codes."},
    ]
    for i in range(n_turns):
        fact = ""
        if i % 10 == 0:
            fact = f" The deploy code for region {i // 10} is Z{rng.randint(1000, 9999)}."
        msgs.append({
            "role": "assistant",
            "content": f"Working on step {i}.{fact}",
            "tool_calls": [{
                "id": f"c{i}",
                "function": {"name": "terminal", "arguments": json.dumps({"command": f"echo step {i}"})},
            }],
        })
        msgs.append({
            "role": "tool",
            "tool_call_id": f"c{i}",
            "content": ("step output " * 200) + f"result-{i}",
        })
    msgs.append({"role": "assistant", "content": "Checklist complete."})
    return msgs
