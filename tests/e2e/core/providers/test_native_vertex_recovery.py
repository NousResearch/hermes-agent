"""Vertex AI recovery paths: compaction in a signed reasoning session, and a mid-stream drop.

Real ``hermes chat -q`` turns against the fake Vertex (``tests/fakes/providers/vertex.py``), which
rejects exactly what Gemini 3 rejects after history surgery: orphaned tool results, unanswered
calls, and a current-turn function call without its thought signature (or with a signature it
never issued). Scenarios run concurrently in a module fixture:

* ``compaction`` — ``--reasoning high``, every step a signed ``read_file`` call, a tiny
  ``compression.threshold_tokens`` and large reported prompt tokens, across a ``--resume``.
* ``drop``       — the stream after a signed tool step is cut mid-body (incomplete chunked
  TLS response); the retry must resend a valid request and persist the answer once.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable

import pytest

pytest.importorskip("google.auth", reason="Vertex minting needs google-auth (CI installs it)")

from tests.e2e.core.providers._native_helpers import (  # noqa: E402
    ChatResult,
    assert_no_duplicate_assistant_text,
    latest_session,
    make_home,
    messages,
    run_chat,
    tool_calls_of,
)
from tests.fakes.providers.vertex import (  # noqa: E402
    PROJECT,
    REGION,
    SA_EMBEDDED_PROJECT,
    Call,
    Drop,
    FakeVertex,
    Say,
    hermes_setup,
)

KNOWN: dict[str, str] = {}

SUMMARY_MARK = "VERTEX-SUMMARY-OK"
TURN1_FINAL = "Compaction turn one done."
TURN2_FINAL = "Compaction turn two done."
DROP_PARTIAL = "DROPPED-PARTIAL-ANSWER that never finishes"
DROP_FINAL = "Recovered answer after the stream dropped."
# Reported prompt tokens stay far above ``threshold_tokens`` so the compressor must run.
CONTEXT_LENGTH = 64_000
THRESHOLD_TOKENS = 12_000
ARGS = ("-t", "file", "--reasoning", "high")


def _prompt_tokens(body: dict[str, Any]) -> int:
    return 20_000 + len(json.dumps(body.get("messages", []))) // 4


def _summary(_rec: dict[str, Any]) -> Say:
    return Say(f"## Goal\nKeep reading the seeded files ({SUMMARY_MARK}).\n## Progress\n- Read several files.\n", chunk_chars=64)


def _fake(tmp: Path, name: str, script: list[Any], **kw: Any) -> FakeVertex:
    fake = FakeVertex(tmp / name / "fake", project=PROJECT, region=REGION, sa_project=SA_EMBEDDED_PROJECT,
                      script=script, **kw)
    fake.start()
    return fake


def run_compaction(tmp: Path) -> dict[str, Any]:
    script: list[Any] = [Call([("read_file", {"path": f"f{i}.txt"})]) for i in range(4)] + [Say(TURN1_FINAL)]
    script += [Call([("read_file", {"path": f"f{i}.txt"})]) for i in range(4, 7)] + [Say(TURN2_FINAL)]
    fake = _fake(tmp, "compaction", script, aux=_summary, prompt_tokens_fn=_prompt_tokens)
    setup = hermes_setup(fake, context_length=CONTEXT_LENGTH, extra_config={
        "compression": {"threshold_tokens": THRESHOLD_TOKENS, "protect_last_n": 4}})
    nh = make_home(tmp / "compaction" / "h", **setup)
    for i in range(7):
        (nh.project / f"f{i}.txt").write_text(f"file {i} " + "lorem ipsum dolor " * 300 + "\n", encoding="utf-8")
    turn1 = run_chat(nh, "Read f0.txt through f3.txt.", env=fake.child_env(), args=ARGS)
    sid = latest_session(nh) if turn1.returncode == 0 else None
    turn2 = run_chat(nh, "Now read f4.txt through f6.txt.", env=fake.child_env(), args=ARGS, resume=sid) if sid else None
    return {"fake": fake, "nh": nh, "turn1": turn1, "turn2": turn2}


def run_drop(tmp: Path) -> dict[str, Any]:
    fake = _fake(tmp, "drop", [Call([("read_file", {"path": "note.txt"})]), Drop(DROP_PARTIAL, after_chars=22), Say(DROP_FINAL)])
    nh = make_home(tmp / "drop" / "h", **hermes_setup(fake))
    (nh.project / "note.txt").write_text("note: KIWI-13\n", encoding="utf-8")
    return {"fake": fake, "nh": nh, "turn": run_chat(nh, "Read note.txt.", env=fake.child_env(), args=ARGS)}


SCENARIOS: dict[str, Callable[[Path], dict[str, Any]]] = {"compaction": run_compaction, "drop": run_drop}


@pytest.fixture(scope="module")
def results(tmp_path_factory: pytest.TempPathFactory) -> Any:
    tmp = tmp_path_factory.mktemp("vertex_recovery")
    with ThreadPoolExecutor(max_workers=len(SCENARIOS)) as pool:
        futures = {name: pool.submit(fn, tmp) for name, fn in SCENARIOS.items()}
        out = {name: f.result() for name, f in futures.items()}
    yield out
    for res in out.values():
        res["fake"].stop()


def _ok(turn: ChatResult | None, what: str, answer: str) -> None:
    assert turn is not None and turn.returncode == 0, f"{what} failed:\n{turn.describe() if turn else 'not run'}"
    assert answer in turn.stdout, turn.describe()


def _assert_pairs(rows: list[dict[str, Any]]) -> None:
    calls = [tc["id"] for r in rows if r["role"] == "assistant" for tc in tool_calls_of(r)]
    results = [r["tool_call_id"] for r in rows if r["role"] == "tool"]
    assert sorted(calls) == sorted(results), f"persisted tool pairs broken: calls={calls} results={results}"


def test_compacted_signed_session_stays_valid_for_vertex(results: dict[str, Any]) -> None:
    """The summary call goes to Vertex with the minted bearer, and every request after compaction is
    accepted: tool pairs intact, current-turn calls still signed, each signature on its own call."""
    res = results["compaction"]
    fake = res["fake"]
    _ok(res["turn1"], "turn 1", TURN1_FINAL)
    _ok(res["turn2"], "turn 2 (--resume)", TURN2_FINAL)
    assert not fake.rejected(), json.dumps([(r["status"], r["rejected"]) for r in fake.rejected()], indent=1)
    aux = fake.aux_requests()
    assert aux, "compaction never called the summarizer (threshold not crossed?)"
    minted = {f"Bearer {t}" for t in fake.minted_tokens()}
    assert {r["auth"] for r in aux} <= minted
    first_aux = fake.requests.index(aux[0])
    after = [r for r in fake.requests[first_aux:] if r["kind"] == "main"]
    assert after and any(SUMMARY_MARK in json.dumps(r["body"]["messages"]) for r in after), (
        "no post-compaction request carries the summary")
    for rec in after:
        for msg in rec["body"]["messages"]:
            for tc in msg.get("tool_calls") or []:
                sig = (tc.get("extra_content") or {}).get("google", {}).get("thought_signature")
                if sig is not None:
                    assert fake.signature_by_call.get(tc["id"]) == sig, f"signature moved to another call: {tc['id']}"


def test_compacted_session_rows_keep_tool_pairs(results: dict[str, Any]) -> None:
    res = results["compaction"]
    nh = res["nh"]
    _ok(res["turn2"], "turn 2 (--resume)", TURN2_FINAL)
    rows = messages(nh, latest_session(nh))
    _assert_pairs(rows)
    assert_no_duplicate_assistant_text(rows, TURN2_FINAL)
    assert any(r["role"] == "assistant" and r.get("content") == TURN2_FINAL for r in rows)


def test_stream_drop_retried_without_duplicate_content(results: dict[str, Any]) -> None:
    """The response after a signed tool step dies mid-body: Hermes resends the same valid request
    (signature intact), prints the recovered answer, and persists it exactly once."""
    res = results["drop"]
    fake, nh = res["fake"], res["nh"]
    _ok(res["turn"], "drop turn", DROP_FINAL)
    assert not fake.rejected(), [r["rejected"] for r in fake.rejected()]
    mains = fake.main_requests()
    assert [r["response"] for r in mains] == ["Call", "Drop", "Say"], [r.get("response") for r in mains]
    assert mains[2]["body"]["messages"] == mains[1]["body"]["messages"], "the retry changed the conversation"
    rows = messages(nh, latest_session(nh))
    assert [r["role"] for r in rows] == ["user", "assistant", "tool", "assistant"], rows
    partial = DROP_PARTIAL[:22]
    assert not [r["id"] for r in rows if partial in (r.get("content") or "")], "the dropped partial was persisted"
    assert_no_duplicate_assistant_text(rows, DROP_FINAL)
    assert partial not in res["turn"].stdout
