"""Benchmark: wire tokens per provider request, with and without tool-result projection.

Reproduces the "long agentic session" profile: a system prompt, ~20 turns of coding work
with 40 tool calls, and ~770 KB of accumulated tool output. Everything is measured with the
repository's own estimator (``agent.model_metadata.estimate_messages_tokens_rough``) against
the real projection path — no provider calls, no invented latency numbers.

    python scripts/bench_tool_result_projection.py

Per request it reports the wire size, and — because prefix caching makes *re-processing* the
cost, not size — the byte-level diff against the previous request: how many leading messages
the provider can reuse from cache, how many tokens it must re-process, and which requests
rewrote history behind the new content (a cache break).
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.model_metadata import estimate_messages_tokens_rough as estimate  # noqa: E402
from agent.tool_result_projection import ProjectionState, project_stale_tool_results  # noqa: E402

SYSTEM_PROMPT = (
    "You are a coding agent working in a repository. You have tools: terminal, read_file, "
    "write_file, search_files, web_search, browser, execute_code, delegate_task."
) * 40  # ~1.5 KB, standing in for the real system prompt + memory block

# (tool, args, result size in KB) — a plausible 20-turn coding session.
WORKLOAD = (
    [("read_file", '{"path": "src/module_%d.py"}' % i, 25) for i in range(12)]
    + [("terminal", '{"command": "pytest -q tests/part_%d"}' % i, 40) for i in range(8)]
    + [("search_files", '{"pattern": "handl.*_%d"}' % i, 12) for i in range(10)]
    + [("web_extract", '{"urls": ["https://example.com/doc_%d"]}' % i, 18) for i in range(6)]
    + [("terminal", '{"command": "git log --oneline -%d"}' % (i + 5), 5) for i in range(4)]
)


def build_history():
    """System + user, then one assistant tool_call / tool result pair per workload entry."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "Fix the failing tests in this repo and report what changed."},
    ]
    for idx, (tool, args, kb) in enumerate(WORKLOAD):
        call_id = f"call_{idx:02d}"
        messages.append({
            "role": "assistant",
            "content": "" if idx % 4 else f"Checking part {idx}.",
            "tool_calls": [{"id": call_id, "type": "function",
                            "function": {"name": tool, "arguments": args}}],
        })
        body = f"$ {tool} {args}\n" + (f"line {idx} of output\n" * ((kb * 1024) // 20))
        messages.append({"role": "tool", "tool_call_id": call_id, "content": body[: kb * 1024]})
        if idx % 2 == 1:
            messages.append({"role": "assistant", "content": f"Part {idx} done, moving on."})
    return messages


def _agent(context_length: int, *, caching: bool):
    from types import SimpleNamespace

    compressor = SimpleNamespace(
        context_length=context_length,
        tool_result_projection="auto",
        tool_result_projection_min_tokens=0,
        tool_result_projection_min_result_chars=4000,
        tool_result_projection_tail_ratio=0.025,
        protect_last_n=20,          # NOT what bounds the tail: the message floor/cap are internal
    )
    # ``_use_prompt_caching`` is the explicit marker policy, and it is what the two arms below turn
    # on and off. A route with provider-side automatic caching is classified the same way the
    # ``caching=True`` arm is — via the cached input tokens the provider reports — so read the
    # "prompt caching on" numbers as what such a route gets, not only as the Anthropic-marker route.
    agent = SimpleNamespace(context_compressor=compressor, _use_prompt_caching=caching)
    agent._tool_result_projection_state = ProjectionState()
    return agent


def _common_prefix_len(previous, current) -> int:
    """How many leading messages are byte-identical, i.e. reusable from the provider cache."""
    limit = min(len(previous), len(current))
    for idx in range(limit):
        if json.dumps(previous[idx], sort_keys=True) != json.dumps(current[idx], sort_keys=True):
            return idx
    return limit


def measure(window: int, *, caching: bool, project: bool):
    """Walk the whole session forward, one request per message added (a request after every
    tool call is what a real agentic loop does). Returns per-request measurements."""
    base = build_history()
    agent = _agent(window, caching=caching) if project else None
    rows, previous = [], None
    for end in range(3, len(base) + 1):
        wire = [dict(m) for m in base[:end]]
        if project:
            project_stale_tool_results(agent, wire)
        prefix = _common_prefix_len(previous, wire) if previous is not None else 0
        reused = estimate(previous[:prefix]) if previous is not None and prefix else 0
        total = estimate(wire)
        rows.append({
            "index": end,
            "wire_tokens": total,
            "reused_tokens": reused,
            "reprocessed_tokens": max(0, total - reused),
            # History rewritten behind the new content = a prompt-cache break.
            "break": previous is not None and prefix < len(previous),
        })
        previous = wire
    return rows


def summarise(label: str, rows):
    sizes = [r["wire_tokens"] for r in rows]
    reprocessed = [r["reprocessed_tokens"] for r in rows]
    breaks = sum(1 for r in rows if r["break"])
    billed = sum(r["reused_tokens"] * 0.10 + r["reprocessed_tokens"] * 1.25 for r in rows)
    print(f"{label:<12} mean wire {statistics.mean(sizes):>9,.0f}   "
          f"p50 {statistics.median(sizes):>9,.0f}   "
          f"last {sizes[-1]:>9,.0f}   |   re-processed {sum(reprocessed):>10,.0f}   "
          f"cache breaks {breaks:>3}   modelled billed {billed:>11,.0f}")
    return {"mean": statistics.mean(sizes), "last": sizes[-1],
            "reprocessed": sum(reprocessed), "breaks": breaks, "billed": billed}


def report(window: int, *, caching: bool):
    baseline = summarise("baseline", measure(window, caching=caching, project=False))
    projected = summarise("projected", measure(window, caching=caching, project=True))
    extra = projected["reprocessed"] / baseline["reprocessed"] - 1
    print(f"             → wire {100 * (1 - projected['mean'] / baseline['mean']):.1f}% less/request, "
          f"{100 * (1 - projected['last'] / baseline['last']):.1f}% less at the end, "
          f"re-processed {'+' if extra >= 0 else ''}{extra * 100:.1f}% "
          f"(the cache breaks above), modelled billed input "
          f"{100 * (1 - projected['billed'] / baseline['billed']):.1f}% less\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--windows", default="200000,1000000")
    args = parser.parse_args()
    print(f"session: {len(WORKLOAD)} tool calls, "
          f"{sum(kb for _, _, kb in WORKLOAD)} KB of tool output, "
          f"{len(build_history())} messages "
          f"({len(build_history()) - 2} requests measured)")
    print("modelled billed input assumes cache write 1.25x / cache read 0.10x of the input price\n")
    for window in (int(w) for w in args.windows.split(",")):
        for caching in (False, True):
            print(f"=== window {window:,} tokens · prompt caching {'on' if caching else 'off'} ===")
            report(window, caching=caching)


if __name__ == "__main__":
    main()
