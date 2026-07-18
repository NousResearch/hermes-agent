#!/usr/bin/env python3
"""Offline replay of real session shapes through background compression.

Task 9 of the async-compression plan: exercise the full candidate lifecycle
(freeze → prepare → validate → apply/fallback) against realistic transcript
shapes WITHOUT sending any message to a provider and WITHOUT executing any
tool present in the history. The summariser is a deterministic stand-in, so
every run is reproducible and free.

Two input sources:

* built-in scenarios covering the plan's matrix (simple, tools, subagent,
  concurrent arrivals, already-compressed, in-place, legacy rotation,
  summariser failure, timeout, reset during preparation);
* sanitized session exports dropped into ``tests/fixtures/async_compression/``
  as ``*.json`` files (see the README there for the export procedure) — each
  is replayed through the in-place apply scenario.

Structural gate (the canary blocker): every check below must pass for every
scenario. Any failure exits non-zero.

  suffix_intact            live suffix survives byte-for-byte (same objects)
  tool_pairs_valid         no orphaned tool_call / tool result in the result
  no_duplicates            active DB rows match the final transcript 1:1
  history_searchable       pre-compaction turns stay archived + FTS-findable
  summary_within_budget    prepared prefix is smaller than what it replaced
  sentinels_preserved      IDs, paths, numbers, user corrections survive

Usage:
  python scripts/replay_async_compression.py            # builtin + fixtures
  python scripts/replay_async_compression.py --json     # machine-readable
  python scripts/replay_async_compression.py --scenario tools_closed_groups
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import tempfile
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

FIXTURES_DIR = REPO_ROOT / "tests" / "fixtures" / "async_compression"

# Sentinel tokens that must survive compression verbatim (the plan's
# "IDs, caminhos, números e correções do usuário preservados").
SENTINELS = (
    "/srv/app/config/settings.py",
    "ticket #48213",
    "R$ 1.234,56",
    "correction: the endpoint is /v2/checkout not /v1/checkout",
)

SUMMARY_MARKER = "[CONTEXT COMPACTION]"


# ── transcript builders ────────────────────────────────────────────────────


def _turns(n: int, prefix: str = "message") -> List[Dict[str, Any]]:
    out = []
    for i in range(n):
        role = "user" if i % 2 == 0 else "assistant"
        out.append({"role": role, "content": f"{prefix} {i}"})
    return out


def _tool_group(idx: int, name: str = "read_file", content: str = "") -> List[Dict[str, Any]]:
    call_id = f"call_{name}_{idx}"
    return [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": call_id,
                "type": "function",
                "function": {"name": name, "arguments": "{}"},
            }],
        },
        {
            "role": "tool",
            "tool_call_id": call_id,
            "name": name,
            "content": content or f"tool result {idx}",
        },
    ]


def _sentinel_tail() -> List[Dict[str, Any]]:
    """Recent-turn messages carrying the must-survive sentinels."""
    return [
        {"role": "user",
         "content": f"please edit {SENTINELS[0]} for {SENTINELS[1]}"},
        {"role": "assistant",
         "content": f"done — the refund of {SENTINELS[2]} is configured"},
        {"role": "user", "content": SENTINELS[3]},
        {"role": "assistant", "content": "acknowledged, using /v2/checkout"},
    ]


def _simple_conversation() -> List[Dict[str, Any]]:
    return [{"role": "system", "content": "sys"}] + _turns(14) + _sentinel_tail()


def _tools_conversation() -> List[Dict[str, Any]]:
    msgs = [{"role": "system", "content": "sys"}] + _turns(4)
    for i in range(4):
        msgs.extend(_tool_group(i))
        msgs.append({"role": "assistant", "content": f"analysed result {i}"})
        msgs.append({"role": "user", "content": f"continue {i}"})
    return msgs + _sentinel_tail()


def _subagent_conversation() -> List[Dict[str, Any]]:
    msgs = [{"role": "system", "content": "sys"}] + _turns(4)
    msgs.extend(_tool_group(0, name="delegate_task",
                            content="subagent finished: refactored 3 files"))
    msgs.append({"role": "assistant", "content": "subagent work merged"})
    msgs.extend(_turns(6, prefix="follow-up"))
    return msgs + _sentinel_tail()


def _already_compressed_conversation() -> List[Dict[str, Any]]:
    msgs = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": f"{SUMMARY_MARKER} earlier summary of turns 1-40"},
    ]
    msgs += _turns(12, prefix="post-summary")
    return msgs + _sentinel_tail()


# ── scenario definition ────────────────────────────────────────────────────


@dataclass
class Scenario:
    name: str
    messages_builder: Callable[[], List[Dict[str, Any]]]
    mode: str = "apply"          # apply | failure | timeout | reset
    in_place: bool = True
    late_messages: int = 0       # messages arriving between prepare and apply
    description: str = ""


BUILTIN_SCENARIOS: List[Scenario] = [
    Scenario("simple_operational", _simple_conversation,
             description="plain user/assistant conversation"),
    Scenario("tools_closed_groups", _tools_conversation,
             description="conversation with closed tool-call groups"),
    Scenario("subagent_delegation", _subagent_conversation,
             description="conversation containing a delegate_task round"),
    Scenario("messages_during_execution", _simple_conversation,
             late_messages=3,
             description="messages arrive after the prefix was frozen"),
    Scenario("already_compressed", _already_compressed_conversation,
             description="transcript already carries a compaction summary"),
    Scenario("in_place_session", _tools_conversation, in_place=True,
             description="in-place compaction (same session id)"),
    Scenario("legacy_rotation_session", _tools_conversation, in_place=False,
             description="legacy rotation (parent → child session)"),
    Scenario("summariser_failure", _simple_conversation, mode="failure",
             description="worker raises; foreground must be untouched"),
    Scenario("summariser_timeout", _simple_conversation, mode="timeout",
             description="worker never finishes in time; sync fallback"),
    Scenario("reset_during_preparation", _simple_conversation, mode="reset",
             description="/reset lands while the worker is in flight"),
]


@dataclass
class ReplayResult:
    name: str
    ok: bool
    checks: Dict[str, bool] = field(default_factory=dict)
    detail: str = ""


# ── validations ────────────────────────────────────────────────────────────


def _canonical(msgs: List[Dict[str, Any]]) -> str:
    from agent.async_context_compression import _semantic_view
    return json.dumps([_semantic_view(m) for m in msgs], sort_keys=True,
                      ensure_ascii=False, default=repr)


def _tool_pairs_valid(msgs: List[Dict[str, Any]]) -> bool:
    open_ids: set = set()
    for m in msgs:
        if not isinstance(m, dict):
            return False
        if m.get("role") == "assistant" and m.get("tool_calls"):
            for tc in m["tool_calls"]:
                tc_id = tc.get("id") if isinstance(tc, dict) else None
                if tc_id:
                    open_ids.add(tc_id)
        elif m.get("role") == "tool":
            tc_id = m.get("tool_call_id")
            if tc_id not in open_ids:
                return False  # orphan result
            open_ids.discard(tc_id)
    return not open_ids  # no orphan calls


def _rough_tokens(msgs: List[Dict[str, Any]]) -> int:
    return max(1, len(json.dumps(msgs, ensure_ascii=False, default=repr)) // 4)


def _transcript_text(msgs: List[Dict[str, Any]]) -> str:
    return "\n".join(str(m.get("content", "")) for m in msgs if isinstance(m, dict))


# ── replay core ────────────────────────────────────────────────────────────


def _make_agent(session_db, session_id):
    os.environ.setdefault("OPENROUTER_API_KEY", "replay-offline-key")
    from run_agent import AIAgent
    agent = AIAgent(
        api_key="replay-offline-key",
        base_url="https://openrouter.ai/api/v1",
        model="test/model",
        quiet_mode=True,
        session_db=session_db,
        session_id=session_id,
        skip_context_files=True,
        skip_memory=True,
    )
    agent._compression_feasibility_checked = True
    return agent


def replay_scenario(scenario: Scenario) -> ReplayResult:
    """Run one scenario in a throwaway SessionDB. Never touches real state."""
    from agent.async_context_compression import (
        BackgroundCompressionConfig,
        BackgroundCompressionController,
        CandidateState,
    )
    from agent.conversation_compression import apply_prepared_candidate
    from hermes_state import SessionDB

    checks: Dict[str, bool] = {}
    with tempfile.TemporaryDirectory(prefix="hermes-replay-") as tmpdir:
        db = SessionDB(db_path=Path(tmpdir) / "state.db")
        sid = f"replay-{scenario.name}"
        db.create_session(sid, "cli", model="test/model")
        agent = _make_agent(db, sid)
        agent.compression_in_place = scenario.in_place
        try:
            messages = [copy.deepcopy(m) for m in scenario.messages_builder()]
            agent._flush_messages_to_session_db(messages, [])
            baseline = copy.deepcopy(messages)

            cfg = BackgroundCompressionConfig.from_dict({
                "enabled": True, "shadow_only": False,
                "min_delta_tokens": 0, "min_frozen_messages": 2,
            })
            ctl = BackgroundCompressionController(cfg)
            release = threading.Event()

            def prepare_fn(prefix):
                if scenario.mode == "failure":
                    raise RuntimeError("summariser failure (replay)")
                if scenario.mode in ("timeout", "reset"):
                    # Deterministic stand-in for a hung/slow provider call.
                    assert release.wait(timeout=10.0)
                return [
                    {"role": "user",
                     "content": f"{SUMMARY_MARKER} deterministic replay summary"},
                    copy.deepcopy(prefix[-1]),
                ]

            prefix_count = len(messages) - 5
            checks["preparation_started"] = ctl.try_start_preparation(
                session_id=sid, messages=messages, prefix_count=prefix_count,
                current_turn=1, source_prompt_tokens=200_000,
                prepare_fn=prepare_fn,
            )

            if scenario.mode == "reset":
                ctl.reset()
                release.set()
                checks["settled"] = ctl.wait_until_settled(timeout=10.0)
                checks["candidate_discarded"] = ctl.peek_candidate() is None
                checks["transcript_unchanged"] = messages == baseline
                checks["db_rows_unchanged"] = (
                    len(db.get_messages(sid)) == len(baseline)
                )
            elif scenario.mode == "timeout":
                # Apply attempt while the worker is still "hung": the loop
                # must defer and the synchronous fallback stays available.
                checks["apply_deferred"] = ctl.take_valid_candidate(
                    session_id=sid, messages=messages, current_turn=2
                ) is None
                ctl.cancel()
                release.set()
                checks["settled"] = ctl.wait_until_settled(timeout=10.0)
                checks["late_result_discarded"] = ctl.peek_candidate() is None
                checks["transcript_unchanged"] = messages == baseline
                checks["db_rows_unchanged"] = (
                    len(db.get_messages(sid)) == len(baseline)
                )
            elif scenario.mode == "failure":
                checks["settled"] = ctl.wait_until_settled(timeout=10.0)
                checks["worker_failed_quietly"] = ctl.state is CandidateState.FAILED
                checks["no_candidate"] = ctl.take_valid_candidate(
                    session_id=sid, messages=messages, current_turn=2
                ) is None
                checks["transcript_unchanged"] = messages == baseline
                checks["db_rows_unchanged"] = (
                    len(db.get_messages(sid)) == len(baseline)
                )
            else:  # apply
                checks["settled"] = ctl.wait_until_settled(timeout=10.0)
                for i in range(scenario.late_messages):
                    messages.append({
                        "role": "user" if i % 2 == 0 else "assistant",
                        "content": f"arrived during preparation {i} "
                                   f"(keep {SENTINELS[1]})",
                    })
                    agent._flush_messages_to_session_db(messages, [])

                cand = ctl.take_valid_candidate(
                    session_id=sid, messages=messages, current_turn=2
                )
                checks["candidate_valid"] = cand is not None
                if cand is not None:
                    live_suffix = messages[cand.prefix_message_count:]
                    suffix_canon = _canonical(live_suffix)
                    frozen_tokens = _rough_tokens(
                        messages[:cand.prefix_message_count]
                    )
                    prepared_tokens = _rough_tokens(
                        list(cand.prepared_messages)
                    )

                    result = apply_prepared_candidate(
                        agent, cand, messages, "sys", controller=ctl
                    )
                    checks["apply_committed"] = result is not None
                    if result is not None:
                        new_messages, _prompt = result
                        prepared_n = len(cand.prepared_messages)
                        got_suffix = new_messages[
                            prepared_n:prepared_n + len(live_suffix)
                        ]
                        checks["suffix_intact"] = (
                            _canonical(got_suffix) == suffix_canon
                            and all(a is b for a, b in
                                    zip(got_suffix, live_suffix))
                        )
                        checks["tool_pairs_valid"] = _tool_pairs_valid(new_messages)
                        checks["summary_within_budget"] = (
                            prepared_tokens < frozen_tokens
                        )
                        text = _transcript_text(new_messages)
                        checks["sentinels_preserved"] = all(
                            s in text for s in SENTINELS
                        )

                        if scenario.in_place:
                            checks["same_session_id"] = agent.session_id == sid
                            active = db.get_messages(sid)
                            checks["no_duplicates"] = (
                                len(active) == len(new_messages)
                            )
                            checks["history_archived"] = (
                                db.has_archived_messages(sid)
                            )
                            # Pre-compaction turns must remain searchable —
                            # probe with the real content of an archived
                            # (frozen-prefix) message from this transcript.
                            probe = ""
                            for m in baseline[1:cand.prefix_message_count]:
                                if isinstance(m.get("content"), str) and m["content"].strip():
                                    probe = m["content"].strip()[:40]
                                    break
                            hits = db.search_messages(f'"{probe}"', limit=5)
                            checks["history_searchable"] = any(
                                h.get("session_id") == sid for h in hits
                            )
                        else:
                            checks["rotated_to_child"] = agent.session_id != sid
                            child = db.get_session(agent.session_id)
                            checks["child_linked_to_parent"] = bool(
                                child and child.get("parent_session_id") == sid
                            )
                            parent_rows = db.get_messages(sid)
                            checks["parent_history_preserved"] = (
                                len(parent_rows) >= len(baseline)
                            )
                checks["lock_released"] = (
                    db.get_compression_lock_holder(sid) is None
                )

            ctl.shutdown(wait=True)
            ok = all(checks.values())
            return ReplayResult(scenario.name, ok, checks,
                                scenario.description)
        finally:
            try:
                agent.close()
            except Exception:
                pass
            db.close()


def _fixture_scenarios() -> List[Scenario]:
    out: List[Scenario] = []
    if not FIXTURES_DIR.is_dir():
        return out
    for path in sorted(FIXTURES_DIR.glob("*.json")):
        def _load(p=path):
            data = json.loads(p.read_text(encoding="utf-8"))
            if not isinstance(data, list) or len(data) < 8:
                raise ValueError(f"fixture {p.name}: expected a list of >=8 messages")
            # Never execute anything from the history: strip nothing, run
            # nothing — messages are treated as opaque transcript data.
            return data + _sentinel_tail()
        out.append(Scenario(
            name=f"fixture_{path.stem}", messages_builder=_load,
            description=f"sanitized export {path.name}",
        ))
    return out


def run_all(only: Optional[str] = None) -> List[ReplayResult]:
    scenarios = BUILTIN_SCENARIOS + _fixture_scenarios()
    if only:
        scenarios = [s for s in scenarios if s.name == only]
        if not scenarios:
            raise SystemExit(f"unknown scenario: {only}")
    return [replay_scenario(s) for s in scenarios]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--scenario", help="run a single scenario by name")
    parser.add_argument("--json", action="store_true",
                        help="machine-readable output")
    args = parser.parse_args()

    results = run_all(only=args.scenario)
    if args.json:
        print(json.dumps(
            [{"name": r.name, "ok": r.ok, "checks": r.checks,
              "description": r.detail} for r in results],
            indent=2, ensure_ascii=False,
        ))
    else:
        for r in results:
            status = "PASS" if r.ok else "FAIL"
            print(f"[{status}] {r.name} — {r.detail}")
            if not r.ok:
                for check, value in r.checks.items():
                    if not value:
                        print(f"        ✗ {check}")
    failed = [r for r in results if not r.ok]
    print(f"\n{len(results) - len(failed)}/{len(results)} scenarios passed")
    if failed:
        print("STRUCTURAL GATE FAILED — canary is blocked.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
