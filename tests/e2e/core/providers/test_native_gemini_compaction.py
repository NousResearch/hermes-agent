"""Gemini native wire conformance: auto-compaction inside a thought-signature session.

Turn 1 (process A) reads two bulky files (two signed functionCall steps) and answers. Turn 2
(process B, ``--resume``) reads a third file; that response reports a huge ``promptTokenCount``, so
Hermes compacts mid-turn (tiny ``compression.threshold_tokens``) before the next step, then the
model makes one more signed call and answers.

The fake validates every request like Google: roles alternate, every functionResponse follows
its functionCall (count / name / id), and each functionCall step of the current turn carries a
signature Google issued. The tests additionally require that compaction really happened on the
wire and that every functionCall that survives it still carries its own signature verbatim.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass

import pytest

from tests.e2e.core.providers import _native_helpers as nh
from tests.fakes.providers.gemini_native import (
    HERMES_ENV,
    Call,
    Calls,
    GeminiFake,
    Recorded,
    Text,
    hermes_model,
)

SUMMARY_MARK = "GEMINI-SUMMARY-CHECKPOINT"
ANSWER_1 = "Turn one answer GEMINI-CMP-ONE"
ANSWER_2 = "Turn two answer GEMINI-CMP-TWO"
ECHO = "GEMINI-CMP-ECHO-3391"
BIG_PROMPT = 50_000  # far above threshold_tokens below: compaction must fire

SUMMARY = (f"## Goal\nKeep helping with the files ({SUMMARY_MARK}).\n## Progress\n### Done\n"
           "- Read big1.txt and big2.txt.\n## Next Steps\n- Continue with big3.txt.\n")


def _summary_route(rec: Recorded) -> Text | None:
    """The compaction summariser call is the one generate call that declares no tools."""
    return None if (rec.body or {}).get("tools") else Text(SUMMARY, signed=False)


@dataclass
class Run:
    home: nh.NativeHome
    fake: GeminiFake
    turns: list[nh.ChatResult]

    def calls(self) -> list[Recorded]:
        return self.fake.generate_calls()

    def split(self) -> tuple[Recorded, Recorded, list[Recorded]]:
        """(last main request before the summary, first after it, every main request after it)."""
        calls = self.calls()
        idx = next((i for i, c in enumerate(calls) if c.reply.startswith("route:")), None)
        assert idx is not None, f"no compaction summary request reached Google: {[c.reply for c in calls]}"
        before = [c for c in calls[:idx] if c.reply.startswith("script:")]
        after = [c for c in calls[idx + 1:] if c.reply.startswith("script:")]
        assert before and after, [c.reply for c in calls]
        return before[-1], after[0], after


@pytest.fixture(scope="module")
def run(tmp_path_factory: pytest.TempPathFactory) -> Run:
    root = tmp_path_factory.mktemp("gemini_compaction")
    home = nh.make_home(root, hermes_model(context_length=64_000), env_file=HERMES_ENV,
                        extra_config={"compression": {"threshold_tokens": 12_000, "protect_last_n": 4}})
    rng = random.Random(7)
    words = "alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima mike oscar".split()
    paths = []
    for i in range(1, 4):
        path = home.project / f"big{i}.txt"
        lines = [" ".join(rng.choice(words) for _ in range(14)) for _ in range(260)]
        path.write_text("\n".join(lines) + f"\nBIG-END-{i}\n", encoding="utf-8")
        paths.append(str(path))
    script = [
        Calls([Call("read_file", {"path": paths[0]})]),
        Calls([Call("read_file", {"path": paths[1]})]),
        Text(ANSWER_1),
        Calls([Call("read_file", {"path": paths[2]})], prompt_tokens=BIG_PROMPT),
        Calls([Call("terminal", {"command": f"echo {ECHO}"})], prompt_tokens=BIG_PROMPT),
        Text(ANSWER_2, prompt_tokens=3_000),
    ]
    with GeminiFake(root / "fake", script, route=_summary_route) as fake:
        # Outcomes are asserted by the tests (a rejected request must name the broken contract).
        first = nh.run_chat(home, "Read big1.txt and big2.txt.", env=fake.child_env())
        second = nh.run_chat(home, "Now read big3.txt, then echo the marker.", env=fake.child_env(),
                             resume=nh.latest_session(home))
    return Run(home, fake, [first, second])


def test_compaction_happens_on_the_wire(run: Run) -> None:
    before, after, _ = run.split()
    # Without compaction the next request = previous request + (functionCall, functionResponse).
    assert len(after.contents) < len(before.contents) + 2, (len(before.contents), len(after.contents))
    assert SUMMARY_MARK in after.all_text(), "compaction summary never reached the next request"
    assert all(t.returncode == 0 for t in run.turns), [t.describe() for t in run.turns]
    assert ANSWER_2 in run.turns[1].stdout, run.turns[1].describe()


def test_post_compaction_requests_are_valid_and_signed(run: Run) -> None:
    """No request (before or after compaction) was rejected, and every functionCall still on the
    wire after compaction carries the exact signature Google minted for that call id."""
    assert run.fake.rejections() == [], run.fake.rejections()
    _, _, after = run.split()
    issued = run.fake.call_signatures
    for rec in after:
        call_ids = {p["functionCall"].get("id") for p in rec.parts("functionCall")}
        for part in rec.parts("functionCall"):
            cid = part["functionCall"].get("id")
            assert cid in issued, f"functionCall id {cid!r} was never issued by Google"
            assert part.get("thoughtSignature") == issued[cid], f"signature for {cid} lost/altered: {part}"
        for part in rec.parts("functionResponse"):
            assert part["functionResponse"].get("id") in call_ids, f"orphan functionResponse: {part}"
    last = after[-1]
    echoed = [p["functionResponse"] for p in last.parts("functionResponse")]
    assert any(ECHO in json.dumps(r["response"]) for r in echoed), echoed


def test_persisted_transcript_keeps_pairs_and_signatures(run: Run) -> None:
    rows = nh.messages(run.home)  # active rows across the (possibly split) session lineage
    open_ids: set[str] = set()
    for row in rows:
        for tc in nh.tool_calls_of(row) if row["role"] == "assistant" else []:
            open_ids.add(str(tc.get("id")))
        if row["role"] == "tool":
            assert row.get("tool_call_id") in open_ids, f"orphan tool row {row.get('tool_call_id')}"
    nh.assert_no_duplicate_assistant_text(rows, ANSWER_2)
    assert any(ANSWER_2 in (r.get("content") or "") for r in rows if r["role"] == "assistant")
