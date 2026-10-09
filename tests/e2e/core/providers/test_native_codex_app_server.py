"""codex_app_server wire conformance: real ``hermes chat -q`` against a fake ``codex app-server``.

The fake (``tests/fakes/providers/codex_app_server.py``) speaks newline-delimited JSON-RPC over stdio,
validates every request/response Hermes sends against the codex-cli 0.147 app-server schema (serde-style
``-32600 Invalid request`` on missing/mistyped fields; unknown fields recorded because the real server
silently drops them) and records the transcript per app-server PID. Selected via
``model.openai_runtime: codex_app_server`` + ``model.codex_bin``.

Happy-path lifecycle: tool item + approval round trip, reasoning projection, ``--resume`` in a new process
(``thread/resume`` vs. the history-seed fallback) and codex-native compaction.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor

import pytest

from tests.e2e.core._pending_fixes import known_gate
from tests.e2e.core.providers._native_helpers import KnownSymptom, messages, tool_calls_of
from tests.fakes.providers.codex_app_server import CodexRun, run_codex_scenario

pytestmark = [
    pytest.mark.skipif(sys.platform == "win32", reason="POSIX sh wrapper for the fake codex binary"),
    # CodexRun.cleanup() SIGKILLs any app-server that outlived its CLI (reparented to init by then).
    pytest.mark.live_system_guard_bypass,
]

# Red on current main for a tracked, open bug: key -> (the bug's own failure-message pattern, reason).
KNOWN: dict[str, tuple[str, str]] = {
    "compaction_row": (r"^compaction boundary persisted as assistant content: \[.*contextCompaction",
                       "#121301 native contextCompaction item persisted as a raw-JSON assistant message"),
}

YOLO = ["--yolo"]
SEED_MARKER = "Prior conversation from this Hermes session"

SCENARIOS = {
    # a + b: reasoning, a command needing approval, usage, final answer; then --resume in a new process.
    "tools": dict(
        turns=[
            {"steps": [{"kind": "reasoning", "summary": ["R-SUMMARY-alpha weighing the canary"]},
                       {"kind": "command", "command": "echo CANARY-1", "output": "CANARY-1-OUT\n"},
                       {"kind": "usage", "input": 1234, "cached": 200, "output": 30},
                       {"kind": "message", "text": "FINAL-ANSWER-ONE"}]},
            {"steps": [{"kind": "message", "text": "FINAL-ANSWER-TWO"}]},
        ],
        runs=[{"prompt": "run the canary USER-ONE", "args": YOLO}, {"prompt": "and now USER-TWO", "args": YOLO}],
    ),
    # Resume fallback: the rollout is gone for run 2 (history seed on thread/start), back for run 3.
    "seed": dict(
        turns=[
            {"steps": [{"kind": "reasoning", "summary": ["R-SEED-PRIVATE never replayed"]},
                       {"kind": "command", "command": "echo SEED-CMD", "output": "SEED-OUT\n"},
                       {"kind": "message", "text": "SEED-FINAL-ONE"}]},
            {"steps": [{"kind": "message", "text": "SEED-FINAL-TWO"}]},
            {"steps": [{"kind": "message", "text": "SEED-FINAL-THREE"}]},
        ],
        runs=[{"prompt": "first prompt USER-ONE", "args": YOLO, "then": {"forget_threads": True}},
              {"prompt": "second prompt USER-TWO", "args": YOLO, "then": {"forget_threads": False}},
              {"prompt": "third prompt USER-THREE", "args": YOLO}],
    ),
    # c: codex compacts natively mid-turn (contextCompaction item); the next process resumes the same thread.
    "compact": dict(
        turns=[
            {"steps": [{"kind": "message", "text": "C-ONE"}, {"kind": "compaction"},
                       {"kind": "usage", "input": 500}, {"kind": "message", "text": "C-TWO"}]},
            {"steps": [{"kind": "message", "text": "C-THREE"}]},
        ],
        runs=[{"prompt": "compact me", "args": YOLO}, {"prompt": "after compact", "args": YOLO}],
    ),
}


@pytest.fixture(scope="module")
def runs(tmp_path_factory) -> Iterator[dict[str, CodexRun]]:
    with ThreadPoolExecutor(max_workers=len(SCENARIOS)) as pool:
        futures = {name: pool.submit(run_codex_scenario, tmp_path_factory.mktemp(f"codex_{name}"), **spec)
                   for name, spec in SCENARIOS.items()}
        done = {name: future.result() for name, future in futures.items()}
    yield done
    for run in done.values():
        run.cleanup()


def _started_thread_id(run: CodexRun, index: int) -> str:
    """Thread id the fake returned to a thread/start in the ``index``-th app-server process."""
    starts = {m["id"] for m in run.process_requests(index, "thread/start")}
    ids = [e["msg"]["result"]["thread"]["id"] for e in run.process_entries(index)
           if e.get("dir") == "out" and e["msg"].get("id") in starts and "result" in e["msg"]]
    assert len(ids) == 1, f"expected one successful thread/start in process {index}, got {ids}"
    return ids[0]


def _rows(run: CodexRun) -> list[dict]:
    return messages(run.home, run.session_id)


def test_every_request_is_schema_valid_and_handshake_ordered(runs):
    for name, run in runs.items():
        run.fake.assert_wire_clean()
        for index, _ in enumerate(run.results):
            inbound = [e["msg"] for e in run.process_entries(index) if e.get("dir") == "in"]
            methods = [m.get("method") for m in inbound]
            assert methods[:2] == ["initialize", "initialized"], f"{name}#{index} handshake order: {methods}"
            assert "id" not in inbound[1], f"{name}#{index}: `initialized` must be a notification"
            assert inbound[0]["params"]["clientInfo"]["name"], f"{name}#{index}: empty clientInfo.name"


def test_command_approval_round_trip_and_tool_rows(runs):
    run = runs["tools"]
    assert run.results[0].returncode == 0, run.results[0].describe()
    assert run.results[0].stdout.count("FINAL-ANSWER-ONE") == 1, run.results[0].describe()

    first = run.process_entries(0)
    request = next(e["msg"] for e in first if e.get("dir") == "out"
                   and e["msg"].get("method") == "item/commandExecution/requestApproval")
    replies = [e for e in first if e.get("reply_to") == "item/commandExecution/requestApproval"]
    assert [r["msg"]["id"] for r in replies] == [request["id"]], f"approval reply ids: {replies}"
    assert replies[0]["msg"]["result"] == {"decision": "accept"}, replies[0]

    rows = _rows(run)
    calls = [(row, call) for row in rows for call in tool_calls_of(row)]
    assert len(calls) == 1, f"expected one projected tool call, rows: {rows}"
    call_row, call = calls[0]
    assert json.loads(call["function"]["arguments"])["command"] == "echo CANARY-1", call
    results = [r for r in rows if r["role"] == "tool"]
    assert [r["tool_call_id"] for r in results] == [call["id"]], f"tool result not paired: {results}"
    assert "CANARY-1-OUT" in results[0]["content"], results[0]
    finals = [r for r in rows if r["role"] == "assistant" and r["content"] == "FINAL-ANSWER-ONE"]
    assert len(finals) == 1 and finals[0]["id"] > results[0]["id"], f"final answer row missing/out of order: {rows}"


def test_reasoning_projected_once_and_never_as_content(runs):
    run = runs["tools"]
    rows = _rows(run)
    carriers = [r for r in rows if "R-SUMMARY-alpha" in (r.get("reasoning") or "")]
    assert len(carriers) == 1, f"reasoning must attach to exactly one row (after resume too): {rows}"
    assert carriers[0]["role"] == "assistant" and tool_calls_of(carriers[0]), carriers[0]
    assert not [r for r in rows if "R-SUMMARY-alpha" in (r.get("content") or "")], "reasoning leaked into content"
    assert "R-SUMMARY-alpha" not in run.output, "reasoning printed in quiet mode"


def test_resume_in_new_process_resumes_the_same_thread(runs):
    run = runs["tools"]
    assert run.results[1].returncode == 0 and "FINAL-ANSWER-TWO" in run.results[1].stdout, run.results[1].describe()
    thread_id = _started_thread_id(run, 0)
    assert run.fake.spawned_pids()[0] != run.fake.spawned_pids()[1]
    assert run.process_requests(1, "thread/start") == [], "resume must not start a fresh thread"
    resumes = run.process_requests(1, "thread/resume")
    assert [r["params"]["threadId"] for r in resumes] == [thread_id], resumes
    assert SEED_MARKER not in (resumes[0]["params"].get("developerInstructions") or ""), \
        "a resumed thread already holds the history; seeding it again duplicates the conversation"
    turn_starts = run.process_requests(1, "turn/start")
    assert [(t["params"]["threadId"], t["params"]["input"]) for t in turn_starts] == \
        [(thread_id, [{"type": "text", "text": "and now USER-TWO"}])], turn_starts
    contents = [(r["role"], r["content"]) for r in _rows(run) if r["content"]]
    assert contents.count(("assistant", "FINAL-ANSWER-ONE")) == 1 and contents[-2:] == [
        ("user", "and now USER-TWO"), ("assistant", "FINAL-ANSWER-TWO")], contents


def test_resume_fallback_seeds_history_then_rebinds_new_thread(runs):
    run = runs["seed"]
    assert [r.returncode for r in run.results] == [0, 0, 0], "\n".join(r.describe() for r in run.results)
    old_thread = _started_thread_id(run, 0)
    assert [r["params"]["threadId"] for r in run.process_requests(1, "thread/resume")] == [old_thread]
    new_thread = _started_thread_id(run, 1)
    seed = run.process_requests(1, "thread/start")[0]["params"].get("developerInstructions") or ""
    assert SEED_MARKER in seed, "fallback thread/start must carry the prior conversation"
    history = seed[seed.index(SEED_MARKER):]
    for needle in ("first prompt USER-ONE", "SEED-OUT", "SEED-FINAL-ONE"):
        assert history.count(needle) == 1, f"{needle!r} seeded {history.count(needle)}x:\n{history}"
    assert "R-SEED-PRIVATE" not in history, "reasoning must not be replayed as conversation text"
    assert "second prompt USER-TWO" not in history, "the new user turn goes in turn/start, not the seed"
    assert [t["params"]["threadId"] for t in run.process_requests(1, "turn/start")] == [new_thread]
    # Run 3: the session is now bound to the replacement thread.
    assert [r["params"]["threadId"] for r in run.process_requests(2, "thread/resume")] == [new_thread]
    assert run.process_requests(2, "thread/start") == []
    contents = [r["content"] for r in _rows(run) if r["role"] == "assistant" and r["content"]]
    assert contents == ["SEED-FINAL-ONE", "SEED-FINAL-TWO", "SEED-FINAL-THREE"], contents


def test_native_compaction_keeps_thread_and_transcript(runs):
    run = runs["compact"]
    assert [r.returncode for r in run.results] == [0, 0], "\n".join(r.describe() for r in run.results)
    assert "C-TWO" in run.results[0].stdout and "C-THREE" in run.results[1].stdout
    thread_id = _started_thread_id(run, 0)
    assert [r["params"]["threadId"] for r in run.process_requests(1, "thread/resume")] == [thread_id], \
        "codex-native compaction must not retire the thread"
    assert run.process_requests(1, "thread/start") == []
    assert run.fake.requests("thread/compact/start") == [], "native mode: Hermes must not compact on top of codex"
    rows = _rows(run)
    assert {r["session_id"] for r in messages(run.home)} == {run.session_id}, "session was split"
    texts = [r["content"] for r in rows if r["role"] == "assistant" and r["content"] in ("C-ONE", "C-TWO", "C-THREE")]
    assert texts == ["C-ONE", "C-TWO", "C-THREE"], f"transcript rewritten or duplicated: {rows}"


def test_native_compaction_is_not_persisted_as_assistant_text(runs):
    run = runs["compact"]
    assert [r.returncode for r in run.results] == [0, 0], "\n".join(r.describe() for r in run.results)
    rows = _rows(run)
    assert any(r["content"] == "C-TWO" for r in rows), f"post-compaction answer not persisted: {rows}"
    leaked = [r["content"] for r in rows if "contextCompaction" in (r.get("content") or "")]
    with known_gate(KNOWN, "compaction_row", raises=KnownSymptom):
        if leaked:
            raise KnownSymptom(f"compaction boundary persisted as assistant content: {leaked}")
