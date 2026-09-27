"""fast-jev-compaction as a compaction eval arm.

Python port of https://github.com/tamaratran/fast-jev-compaction (MIT), which
replaces the compaction summary with per-tool-call keep/drop decisions from
TypeSafe's Jev "System One" model: the whole history (tool results replaced by
a size note) is sent as `state`, two `noul` questions per candidate call ask
whether the call and whether its verbatim result must stay, and the transcript
is rebuilt with nothing rewritten. Text messages are never touched.

Differences from the TypeScript original are format-only: Hermes transcripts
carry `tool_calls` on assistant rows and one `role: tool` row per result, so a
"message" here is one chat row and `preserve_recent_messages` counts rows.

Transport is OpenRouter's Decisions API (`POST /api/alpha/decisions`, model
`~typesafe/jev-latest`), so the eval needs only OPENROUTER_API_KEY.
"""
from __future__ import annotations

import copy
import json
import math
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

STATE_CONTEXT = (
    "A coding assistant conversation is being compacted to free context. `history` is the "
    "whole conversation so far, oldest first; tool outputs are replaced by a short `result` "
    "note and long texts may be abridged. Each question asks whether one tool call, or the "
    "full output of that call, still needs to stay in the history verbatim. Whatever is not "
    "kept is deleted permanently, but the assistant can always re-run a tool or re-read a file."
)
INPUT_CHARS = (1000, 200, 60)
TEXT_HEAD = 400
TEXT_TAIL = 150
REQUEST_OVERHEAD_TOKENS = 20
OPENROUTER_DECISIONS_URL = "https://openrouter.ai/api/alpha/decisions"
OPENROUTER_JEV_MODEL = "~typesafe/jev-latest"

_TOKEN_PIECES = re.compile(r"[A-Za-z]+|\d+|[^\sA-Za-z\d]")
_WS = re.compile(r"\s+")


def estimate_tokens(text: str) -> int:
    """Port of the plugin's tokenizer-free estimate (letters/6, digits/2, symbols 0.9)."""
    tokens = 0.0
    for piece in _TOKEN_PIECES.findall(text):
        c = piece[0]
        if c.isdigit():
            tokens += len(piece) / 2
        elif c.isascii() and c.isalpha():
            tokens += 1 + (len(piece) - 1) // 6
        else:
            tokens += 0.9
    return math.ceil(tokens)


def truncate(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: max(0, limit - 1)] + "…"


def abridge(text: str, head: int, tail: int) -> str:
    if len(text) <= head + tail + 40:
        return text
    return f"{text[:head]}\n[… {len(text) - head - tail} chars omitted …]\n{text[-tail:]}"


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)


def message_text(m: Dict[str, Any]) -> str:
    c = m.get("content")
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        return "\n".join(p.get("text", "") for p in c if isinstance(p, dict) and p.get("type") == "text")
    return ""


@dataclass
class ToolCall:
    id: str
    tool_call_id: str
    tool: str
    input: Any
    call_index: int
    result_index: int
    result_chars: int
    is_error: bool
    pinned: bool


@dataclass
class Decision:
    id: str
    tool: str
    keep_call: float
    keep_result: float
    action: str  # keep | drop_result | drop_call
    reason: str  # pinned | kept | result_dropped | call_dropped


@dataclass
class JevOptions:
    goal: str = ""
    keep_threshold: float = 0.5
    preserve_recent_messages: int = 6
    max_state_tokens: int = 25_000
    max_request_tokens: int = 30_000
    truncate_head_chars: int = 300
    # Eval-only extension (not in the plugin): instead of thresholding, keep
    # whole call+result pairs in rank order until `result_budget_tokens` of
    # tool content is retained; everything else is dropped. `select="jev"`
    # ranks by Jev's keep_result, `select="recency"` by position and never
    # calls Jev — the control that tells whether Jev's ranking carries signal.
    select: Optional[str] = None
    result_budget_tokens: int = 0


@dataclass
class JevUsage:
    requests: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    models: List[str] = field(default_factory=list)


def is_pinned(index: int, total: int, preserve: int) -> bool:
    return index == 0 or index >= total - preserve


def _parse_arguments(fn: Dict[str, Any]) -> Any:
    args = fn.get("arguments")
    if isinstance(args, str):
        try:
            return json.loads(args)
        except Exception:
            return {"arguments": args}
    return args if args is not None else {}


def collect_tool_calls(messages: List[Dict[str, Any]], preserve: int) -> List[ToolCall]:
    results: Dict[str, int] = {}
    for idx, m in enumerate(messages):
        if m.get("role") == "tool" and m.get("tool_call_id"):
            results[m["tool_call_id"]] = idx
    calls: List[ToolCall] = []
    total = len(messages)
    for idx, m in enumerate(messages):
        for tc in m.get("tool_calls") or []:
            tcid = tc.get("id")
            if tcid not in results:
                continue
            ridx = results[tcid]
            fn = tc.get("function") or {}
            rtext = message_text(messages[ridx])
            calls.append(ToolCall(
                id=f"t{len(calls) + 1}",
                tool_call_id=tcid,
                tool=fn.get("name") or tc.get("name") or "tool",
                input=_parse_arguments(fn),
                call_index=idx,
                result_index=ridx,
                result_chars=len(rtext),
                is_error=bool(re.match(r"\s*(\{\"error\"|Error\b|error:)", rtext[:40], re.I)),
                pinned=is_pinned(idx, total, preserve) or is_pinned(ridx, total, preserve),
            ))
    return calls


def _input_text(inp: Any, limit: int) -> str:
    return truncate(_json(inp), limit)


def _result_note(call: ToolCall) -> str:
    return f"{'error' if call.is_error else 'ok'}, {call.result_chars} chars (omitted)"


def _compact_call(call: ToolCall) -> str:
    if isinstance(call.input, dict):
        parts = []
        for k, v in call.input.items():
            text = v if isinstance(v, str) else _json({k: v})[:200]
            flat = _WS.sub(" ", text)
            parts.append(f"{k}={flat}")
        inp = " ".join(parts)
    else:
        inp = _json(call.input)
    return f"{call.id} {call.tool} {truncate(inp, INPUT_CHARS[2])} → {'error' if call.is_error else 'ok'} {call.result_chars}ch"


def _history_entries(messages, calls: List[ToolCall], input_chars: int) -> List[Dict[str, Any]]:
    by_msg: Dict[int, List[ToolCall]] = {}
    for c in calls:
        by_msg.setdefault(c.call_index, []).append(c)
    entries = []
    for i, m in enumerate(messages):
        tcs = [{"id": c.id, "tool": c.tool, "input": _input_text(c.input, input_chars), "result": _result_note(c)}
               for c in by_msg.get(i, [])]
        text = message_text(m)
        if m.get("role") == "tool":
            text = ""  # results are represented by the call's note, never verbatim
        if not text.strip() and not tcs:
            continue
        entry: Dict[str, Any] = {"i": i, "role": m.get("role"), "text": text}
        if tcs:
            entry["tool_calls"] = tcs
        entries.append(entry)
    return entries


def goal_from_messages(messages) -> str:
    prompts = [message_text(m) for m in messages if m.get("role") == "user" and message_text(m).strip()]
    return "\n".join(truncate(p, 500) for p in prompts[-3:])


def fit_state(messages, calls: List[ToolCall], opt: JevOptions) -> Dict[str, Any]:
    """Port of fitState: shrink the whole-history state in stages until it fits."""
    goal = opt.goal or goal_from_messages(messages)
    total = len(messages)

    def state_of(history):
        return {"context": STATE_CONTEXT, "goal": goal, "history": history}

    def entry_tokens(e):
        return estimate_tokens(_json(e)) + 1

    base = estimate_tokens(_json(state_of([])))
    history: List[Dict[str, Any]] = []
    per: List[int] = []
    tokens = 0

    def rebuild(limit):
        nonlocal history, per, tokens
        history = _history_entries(messages, calls, limit)
        per = [entry_tokens(e) for e in history]
        tokens = base + sum(per)

    def fits():
        return tokens <= opt.max_state_tokens

    def fitted(h, stage):
        return {"state": state_of(h), "tokens": tokens, "stage": stage}

    def shrink(index, change):
        nonlocal tokens
        change(history[index])
        now = entry_tokens(history[index])
        tokens += now - per[index]
        per[index] = now

    rebuild(INPUT_CHARS[0])
    if fits():
        return fitted(history, "full")
    for limit in INPUT_CHARS[1:]:
        rebuild(limit)
        if fits():
            return fitted(history, f"inputs<={limit}")

    def pinned(e):
        return is_pinned(e["i"], total, opt.preserve_recent_messages)

    idx = list(range(len(history)))
    order = [i for i in idx if not pinned(history[i])] + [i for i in idx if pinned(history[i])]

    for i in order:
        if len(history[i]["text"]) <= TEXT_HEAD + TEXT_TAIL + 40:
            continue
        shrink(i, lambda e: e.__setitem__("text", abridge(e["text"], TEXT_HEAD, TEXT_TAIL)))
        if fits():
            return fitted(history, "texts abridged")

    for i in order:
        e = history[i]
        if pinned(e) or not e["text"]:
            continue
        original = len(message_text(messages[e["i"]]))
        shrink(i, lambda e, n=original: e.__setitem__("text", f"[… {n} chars omitted …]"))
        if fits():
            return fitted(history, "old messages collapsed")

    by_msg: Dict[int, List[ToolCall]] = {}
    for c in calls:
        by_msg.setdefault(c.call_index, []).append(c)
    for i in order:
        e = history[i]
        own = by_msg.get(e["i"])
        if pinned(e) or not own:
            continue
        shrink(i, lambda e, own=own: e.__setitem__("tool_calls", [_compact_call(c) for c in own]))
        if fits():
            return fitted(history, "old calls compacted")

    left = set()
    for i in order:
        e = history[i]
        if pinned(e) or e.get("tool_calls"):
            continue
        left.add(i)
        tokens -= per[i]
        if fits():
            return fitted([h for j, h in enumerate(history) if j not in left], "old messages left out")

    remaining = [h for j, h in enumerate(history) if j not in left]
    merged: List[Dict[str, Any]] = []

    def foldable(e):
        return not pinned(e) and not e["text"] and isinstance((e.get("tool_calls") or [None])[0], str)

    for e in remaining:
        prev = merged[-1] if merged else None
        if prev and foldable(prev) and foldable(e) and prev["role"] == e["role"]:
            prev["tool_calls"] = list(prev["tool_calls"]) + list(e["tool_calls"])
            continue
        merged.append(dict(e))
    history = merged
    per = [entry_tokens(e) for e in history]
    tokens = base + sum(per)
    if fits():
        return fitted(history, "old calls merged")
    raise ValueError(f"history too large for Jev (~{tokens} tokens after truncation, limit {opt.max_state_tokens})")


def questions_for(call: ToolCall) -> Dict[str, Any]:
    return {
        f"call_{call.id}": {
            "type": "noul",
            "instructions": (
                f"Tool call {call.id} ({call.tool}) should stay in the history: knowing this call "
                "was made, with its input, still matters for what the assistant does next"),
        },
        f"result_{call.id}": {
            "type": "noul",
            "instructions": (
                f"The full output of tool call {call.id} ({call.tool}, {call.result_chars} chars) should "
                "stay in the history verbatim: the assistant still needs its contents and re-running "
                "the tool would not do"),
        },
    }


def batch_calls(calls: List[ToolCall], state_tokens: int, opt: JevOptions) -> List[List[ToolCall]]:
    budget = opt.max_request_tokens - state_tokens - REQUEST_OVERHEAD_TOKENS
    batches: List[List[ToolCall]] = []
    cur: List[ToolCall] = []
    cur_tokens = 0
    for c in calls:
        t = estimate_tokens(_json(questions_for(c)))
        if cur and cur_tokens + t > budget:
            batches.append(cur)
            cur, cur_tokens = [], 0
        if not cur and t > budget:
            raise ValueError(f"state leaves no room for questions (~{state_tokens} of {opt.max_request_tokens} tokens)")
        cur.append(c)
        cur_tokens += t
    if cur:
        batches.append(cur)
    return batches


def decide_call(call: ToolCall, keep_call: float, keep_result: float, opt: JevOptions) -> Decision:
    base = dict(id=call.id, tool=call.tool, keep_call=keep_call, keep_result=keep_result)
    if call.pinned:
        return Decision(**base, action="keep", reason="pinned")
    if keep_result >= opt.keep_threshold:
        return Decision(**base, action="keep", reason="kept")
    if keep_call >= opt.keep_threshold:
        return Decision(**base, action="drop_result", reason="result_dropped")
    return Decision(**base, action="drop_call", reason="call_dropped")


def _truncated_result(text: str, is_error: bool, head: int) -> str:
    if len(text) <= head + 120:
        return text
    lead = f"{text[:head]}\n" if head > 0 else ""
    return (f"{lead}[fast-jev-compaction truncated {len(text) - head} chars of this tool result"
            f"{' (error)' if is_error else ''}; re-run the tool if needed]")


def apply_decisions(messages, decisions: List[Decision], calls: List[ToolCall], head: int):
    """Rebuild the Hermes transcript: dropped calls vanish with their result row,
    dropped results keep a bounded head + note, untouched rows are the same objects."""
    by_id = {c.id: c for c in calls}
    actions: Dict[str, str] = {}
    for d in decisions:
        c = by_id.get(d.id)
        if c and d.action != "keep":
            actions[c.tool_call_id] = d.action
    errors = {c.tool_call_id: c.is_error for c in calls}
    kept = []
    for m in messages:
        if m.get("role") == "tool":
            act = actions.get(m.get("tool_call_id"))
            if act == "drop_call":
                continue
            if act == "drop_result":
                nm = dict(m)
                nm["content"] = _truncated_result(message_text(m), errors.get(m.get("tool_call_id"), False), head)
                kept.append(nm)
                continue
            kept.append(m)
            continue
        tcs = m.get("tool_calls") or []
        if not any(actions.get(tc.get("id")) == "drop_call" for tc in tcs):
            kept.append(m)
            continue
        remaining = [tc for tc in tcs if actions.get(tc.get("id")) != "drop_call"]
        if not remaining and not message_text(m).strip():
            continue
        nm = dict(m)
        if remaining:
            nm["tool_calls"] = remaining
        else:
            nm.pop("tool_calls", None)
        kept.append(nm)
    return kept


def openrouter_asker(api_key: Optional[str] = None, model: str = OPENROUTER_JEV_MODEL,
                     url: str = OPENROUTER_DECISIONS_URL, timeout: float = 120.0) -> Callable:
    """`ask(state, questions) -> response dict` over OpenRouter's Decisions API."""
    import urllib.request

    key = api_key or os.environ.get("OPENROUTER_API_KEY") or ""
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY is not configured")

    def ask(state, questions):
        body = json.dumps({"model": model, "state": state, "questions": questions}).encode()
        req = urllib.request.Request(url, data=body, headers={
            "Authorization": f"Bearer {key}", "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/NousResearch/hermes-agent", "X-Title": "hermes compaction eval",
        })
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read().decode())
        except urllib.error.HTTPError as e:  # pragma: no cover - network
            raise RuntimeError(f"Jev request failed ({e.code}): {e.read().decode()[:200]}") from e

    return ask


class JevCompactor:
    """Eval-arm compressor with the ContextCompressor.compress() call shape."""

    def __init__(self, asker: Optional[Callable] = None, options: Optional[JevOptions] = None,
                 concurrency: int = 4):
        self.asker = asker or openrouter_asker()
        self.opt = options or JevOptions()
        self.concurrency = concurrency
        self.usage = JevUsage()
        self.stats: Dict[str, Any] = {}
        self.decisions: List[Decision] = []
        self._last_summary_error: Optional[str] = None

    def _ask_batch(self, state, batch: List[ToolCall]) -> Dict[str, Dict[str, float]]:
        questions: Dict[str, Any] = {}
        for c in batch:
            questions.update(questions_for(c))
        resp = self.asker(state, questions)
        answers = resp.get("answers") if isinstance(resp, dict) else None
        if not isinstance(answers, dict):
            raise ValueError("Jev response is missing answers")
        usage = resp.get("usage") or {}
        self.usage.requests += 1
        self.usage.input_tokens += int(usage.get("input_tokens") or 0)
        self.usage.output_tokens += int(usage.get("output_tokens") or 0)
        self.usage.cost_usd += float(usage.get("cost") or 0.0)
        if resp.get("model") and resp["model"] not in self.usage.models:
            self.usage.models.append(resp["model"])

        def noul(name):
            a = answers.get(name)
            if not isinstance(a, dict) or not isinstance(a.get("noul"), (int, float)):
                raise ValueError(f"Invalid Jev answer for {name}")
            return float(a["noul"])

        return {c.id: {"keep_call": noul(f"call_{c.id}"), "keep_result": noul(f"result_{c.id}")} for c in batch}

    def _budget_decisions(self, messages, calls, candidates, answers) -> List[Decision]:
        """Keep ranked call+result pairs until the tool-content budget is spent; drop the rest."""
        opt = self.opt

        def pair_tokens(c: ToolCall) -> int:
            return (c.result_chars + len(_json(c.input))) // 4

        if opt.select == "jev":
            ranked = sorted(candidates, key=lambda c: -answers.get(c.id, {}).get("keep_result", 0.0))
        else:
            ranked = sorted(candidates, key=lambda c: -c.result_index)
        kept, spent = set(), 0
        for c in ranked:
            t = pair_tokens(c)
            if spent + t > opt.result_budget_tokens:
                continue
            kept.add(c.id)
            spent += t
        out = []
        for c in calls:
            a = answers.get(c.id, {})
            base = dict(id=c.id, tool=c.tool, keep_call=a.get("keep_call", 1.0), keep_result=a.get("keep_result", 1.0))
            if c.pinned:
                out.append(Decision(**base, action="keep", reason="pinned"))
            elif c.id in kept:
                out.append(Decision(**base, action="keep", reason="kept"))
            else:
                out.append(Decision(**base, action="drop_call", reason="call_dropped"))
        return out

    def compress(self, messages: List[Dict[str, Any]], current_tokens: int = 0, force: bool = True):
        t0 = time.time()
        opt = self.opt
        calls = collect_tool_calls(messages, opt.preserve_recent_messages)
        candidates = [c for c in calls if not c.pinned]
        answers: Dict[str, Dict[str, float]] = {}
        fitted = {"tokens": 0, "stage": ""}
        batches: List[List[ToolCall]] = []
        if candidates and opt.select != "recency":
            fitted = fit_state(messages, calls, opt)
            batches = batch_calls(candidates, fitted["tokens"], opt)
            with ThreadPoolExecutor(max_workers=self.concurrency) as pool:
                for result in pool.map(lambda b: self._ask_batch(fitted["state"], b), batches):
                    answers.update(result)
        if opt.select:
            self.decisions = self._budget_decisions(messages, calls, candidates, answers)
        else:
            self.decisions = [
                decide_call(c, *(answers.get(c.id, {}).get(k, 1.0) for k in ("keep_call", "keep_result")), opt)
                for c in calls
            ]
        kept = apply_decisions(messages, self.decisions, calls, opt.truncate_head_chars)
        reasons = [d.reason for d in self.decisions]
        self.stats = {
            "messages_before": len(messages), "messages_after": len(kept),
            "calls": len(calls), "kept": reasons.count("kept"),
            "results_dropped": reasons.count("result_dropped"),
            "calls_dropped": reasons.count("call_dropped"), "pinned": reasons.count("pinned"),
            "state_tokens": fitted["tokens"], "state_stage": fitted["stage"],
            "requests": len(batches), "seconds": round(time.time() - t0, 1),
        }
        return kept


def fake_asker(keep_call: float = 0.2, keep_result: float = 0.1) -> Callable:
    """Deterministic asker for offline tests: every candidate gets the same answer."""
    def ask(state, questions):
        return {"model": "fake-jev", "answers": {n: {"type": "noul", "noul": keep_result if n.startswith("result_") else keep_call}
                                                 for n in questions},
                "usage": {"input_tokens": estimate_tokens(_json(state)), "output_tokens": len(questions), "cost": 0.0}}
    return ask


__all__ = [
    "JevCompactor", "JevOptions", "JevUsage", "apply_decisions", "batch_calls", "collect_tool_calls",
    "decide_call", "estimate_tokens", "fake_asker", "fit_state", "openrouter_asker", "questions_for",
]
