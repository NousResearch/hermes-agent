"""Synthetic message-list generators with predictable shape & token weight."""
from __future__ import annotations

from typing import Any, Dict, List


def make_loop_session(
    n_iterations: int,
    paths: tuple[str, ...] = ("/a.py", "/b.py", "/c.py"),
    chars_per_read: int = 4_000,
) -> List[Dict[str, Any]]:
    """N iterations of: user → assistant(read X) → tool → assistant(patch X) → tool.

    Cycles through ``paths`` so each path gets read+patched many times.
    This is the dedup-friendly shape: lots of redundant reads.

    Returns OpenAI-shaped messages including system + user-prelude.
    """
    msgs: List[Dict[str, Any]] = [
        {"role": "system", "content": "You are an assistant."},
        {"role": "user", "content": "Refactor the codebase."},
    ]
    for i in range(n_iterations):
        path = paths[i % len(paths)]
        cid_r, cid_p = f"r{i}", f"p{i}"
        msgs += [
            {"role": "assistant", "content": f"Reading {path}.",
             "tool_calls": [{"id": cid_r, "function": {
                 "name": "read_file",
                 "arguments": f'{{"path":"{path}"}}'}}]},
            {"role": "tool", "tool_call_id": cid_r,
             "content": ("x" * chars_per_read) + f"\n# rev {i}"},
            {"role": "assistant", "content": f"Patching {path}.",
             "tool_calls": [{"id": cid_p, "function": {
                 "name": "patch",
                 "arguments": f'{{"mode":"replace","path":"{path}",'
                              f'"old_string":"x","new_string":"y"}}'}}]},
            {"role": "tool", "tool_call_id": cid_p, "content": "OK"},
        ]
    msgs.append({"role": "user", "content": "Now summarize."})
    return msgs


def make_neutral_session(n_turns: int, chars_per_turn: int = 1_500) -> List[Dict[str, Any]]:
    """N turns of unique user/assistant exchanges — no resource reuse.

    Stresses the path where dedup should NOT fire and we shouldn't
    accidentally lose information.
    """
    msgs: List[Dict[str, Any]] = [
        {"role": "system", "content": "You are an assistant."},
    ]
    for i in range(n_turns):
        msgs += [
            {"role": "user", "content": f"Question {i}: " + ("q" * chars_per_turn)},
            {"role": "assistant", "content": f"Answer {i}: " + ("a" * chars_per_turn)},
        ]
    return msgs


def make_parallel_tool_session(n_groups: int, fanout: int = 3) -> List[Dict[str, Any]]:
    """N groups of (assistant emits ``fanout`` parallel tool_calls → ``fanout`` tool_results)."""
    msgs: List[Dict[str, Any]] = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "Read three files."},
    ]
    for g in range(n_groups):
        cids = [f"g{g}c{j}" for j in range(fanout)]
        msgs.append({
            "role": "assistant", "content": "",
            "tool_calls": [
                {"id": cid, "function": {
                    "name": "read_file",
                    "arguments": f'{{"path":"/grp{g}_{j}.py"}}'}}
                for j, cid in enumerate(cids)
            ],
        })
        for j, cid in enumerate(cids):
            msgs.append({"role": "tool", "tool_call_id": cid,
                         "content": f"file g{g}/{j} content"})
    msgs.append({"role": "user", "content": "Now summarize."})
    return msgs


def make_multimodal_session(n_image_turns: int) -> List[Dict[str, Any]]:
    """User sends images, assistant analyzes. Verifies vision messages
    flow through compaction without corruption."""
    msgs: List[Dict[str, Any]] = [
        {"role": "system", "content": "Vision-capable assistant."},
    ]
    for i in range(n_image_turns):
        msgs += [
            {"role": "user", "content": [
                {"type": "text", "text": f"What is in image {i}?"},
                {"type": "image_url",
                 "image_url": {"url": f"data:image/png;base64,IMG{i}=="}},
            ]},
            {"role": "assistant", "content": f"It's a cat in image {i}."},
        ]
    msgs.append({"role": "user", "content": "Done."})
    return msgs


def estimate_serialized_bytes(messages: List[Dict[str, Any]]) -> int:
    """Sum the serialized JSON byte length of every message.

    Approximates the wire cost of shipping the prompt to the local
    server. Local Qwen has no partial-prefix KV reuse, so wire bytes
    correlate strongly with prefill wall-clock.
    """
    import json
    return sum(len(json.dumps(m, ensure_ascii=False)) for m in messages)
