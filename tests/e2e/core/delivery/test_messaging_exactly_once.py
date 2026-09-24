"""C12 — messaging-platform delivery exactly-once, through the REAL gateway and the REAL agent.

One real ``GatewayRunner`` (child process, real ``start()``) serves three fake platforms that honour
the ``BasePlatformAdapter`` contract with different wire semantics — a Telegram-like editable chat
(4096 cap, streaming on), a Discord-like one (2000 cap, threads, streaming on) and a non-editable
one (streaming off). The model is ``tests/fakes/fake_llm_provider.py`` scripted per inbound message
(tool calls, long answers, slow streams). Faults are injected at the platform wire: send timeouts,
sent-but-ack-lost, 429 with retry_after, connection resets, a slow turn with a busy follow-up,
interrupts, and SIGKILL of the gateway process between provider completion and the send.

Invariants asserted after every scenario (they hold for every correct implementation):

* every inbound message gets exactly one COMPLETE final visible reply — no duplicate, no truncated
  or stale partial rendition left visible, never silence;
* a redelivery that might duplicate an already-visible reply is never silent: any second copy
  carries a recovery marker (the ledger's documented honest at-least-once), and a delivery that the
  platform provably never saw is not marked at all;
* the final visible text equals the persisted assistant text (whitespace-normalised, split chunks
  re-joined), and the answer is persisted exactly once;
* no inbound message is processed twice (one user row, one model turn per inbound id);
* after a crash, catch-up delivers every pending obligation exactly once and does not re-run the
  already-answered turn.
"""

from __future__ import annotations

import contextlib
import os
import re
import sqlite3
import sys
import threading
from pathlib import Path
from typing import Dict, List, Optional

import pytest

from tests.e2e.core.delivery._fake_platform import (
    GatewayProcess,
    read_jsonl,
    footer,
    header,
    norm,
    persisted_answers,
    persisted_user_rows,
    visible_copies,
    wait_until,
)
from tests.e2e.core.delivery._pending_fixes import expect_gap, known_failure
from tests.fakes.fake_llm_provider import FakeLLMServer, StallMidStream, Text, ToolCall

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="SIGKILL + process-group restart harness")

PLATFORMS = {"fk_tg": "telegram", "fk_dc": "discord", "fk_ne": "noedit"}
STREAMING = {"fk_tg": True, "fk_dc": True, "fk_ne": False}
EXTRA_CONFIG = (
    "display:\n"
    "  platforms:\n"
    "    fk_ne:\n"
    "      streaming: false\n"
)

_TOKEN_RE = re.compile(r"\[in:([A-Za-z0-9_.-]+)\]")


class Director:
    """Scripts the fake model per inbound token and records which turns each token started."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.scripts: Dict[str, List] = {}
        self.turns: Dict[str, int] = {}
        self.resumes: Dict[str, int] = {}

    def script(self, token: str, *responses) -> None:
        with self._lock:
            self.scripts[token] = list(responses)

    @staticmethod
    def answer(aid: str, body: str, **kw) -> Text:
        return Text(f"{header(aid)} {body} {footer(aid)}", **kw)

    def __call__(self, record: dict):
        request = record["body"]
        users = [m for m in request.get("messages", []) if m.get("role") == "user"]

        def text_of(m) -> str:
            c = m.get("content")
            if isinstance(c, list):
                c = " ".join(p.get("text", "") for p in c if isinstance(p, dict))
            return c or ""

        last = text_of(users[-1]) if users else ""
        # A resumed turn's recovery note can arrive merged with the original tagged user message
        # (the unanswered user row and the note are consecutive user turns). It is still a resume,
        # not a second run of that token.
        resumed = "previous turn was interrupted" in last.lower()
        tokens = [] if resumed else _TOKEN_RE.findall(last)
        # A tool-result continuation re-sends the same last user message: it is the same turn.
        continuation = request["messages"][-1].get("role") == "tool"
        with self._lock:
            if not tokens:
                # No inbound token in the newest user message: a gateway-synthesized turn (restart
                # auto-resume). Attribute it to the newest inbound in the history.
                earlier = [t for m in users for t in _TOKEN_RE.findall(text_of(m))]
                owner = earlier[-1] if earlier else "none"
                if not continuation:
                    self.resumes[owner] = self.resumes.get(owner, 0) + 1
                return self.answer(f"R-{owner}-{self.resumes.get(owner, 0)}", "resumed turn")
            token = tokens[-1]
            if not continuation:
                self.turns[token] = self.turns.get(token, 0) + 1
            queue = self.scripts.get(token)
            if queue:
                return queue.pop(0)
            n = self.turns.get(token, 0)
            return self.answer(f"{token}-extra{n}", "unscripted extra turn")


@pytest.fixture(scope="module")
def director():
    return Director()


@pytest.fixture(scope="module")
def llm(director):
    with FakeLLMServer(director) as srv:
        yield srv


@pytest.fixture(scope="module")
def gw(tmp_path_factory, llm):
    keep = os.environ.get("C12_SPOOL_ROOT")  # debugging: keep the child's home/spool after the run
    root = Path(keep) / f"gw-{os.getpid()}" if keep else tmp_path_factory.mktemp("c12-gw")
    proc = GatewayProcess(root, platforms=PLATFORMS, llm_base_url=llm.base_url,
                          extra_config=EXTRA_CONFIG)
    proc.start()
    yield proc
    proc.stop()


def ledger_rows(db_path: Path, chat: str) -> List[tuple]:
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=30)
    try:
        return conn.execute("SELECT state, attempts, content FROM delivery_obligations WHERE chat_id=?",
                            (chat,)).fetchall()
    except sqlite3.OperationalError:
        return []
    finally:
        conn.close()


def ledger_quiescent(db_path: Path, chat: str) -> bool:
    return all(state in ("delivered", "abandoned") for state, _, _ in ledger_rows(db_path, chat))


def dump(gw: GatewayProcess, platform: str, chat: str) -> str:
    lines = [f"--- visible in {platform}:{chat}"]
    for m in gw.platform_view().visible(platform, chat):
        lines.append(f"  [{m.message_id} edits={m.edits}] {m.text[:160]!r}")
    lines.append("--- ops")
    for op in gw.ops():
        if op.get("platform") == platform and op.get("chat_id") == chat:
            lines.append(f"  {op['op']} {op.get('content', '')[:100]!r}")
    lines.append(f"--- ledger {ledger_rows(gw.db_path, chat)}")
    fired = read_jsonl(gw.spool / "faults_fired.jsonl")[-4:]
    lines.append(f"--- last faults fired {[(f['op'], f['kind'], f['content'][:40]) for f in fired]}")
    return "\n".join(lines)


def assert_exactly_once(gw: GatewayProcess, director: Director, platform: str, chat: str, token: str,
                        aid: str, *, marked_duplicates_allowed: bool = False,
                        first_copy_may_be_marked: bool = False) -> None:
    ctx = dump(gw, platform, chat)
    msgs = gw.platform_view().visible(platform, chat)
    copies = visible_copies(msgs, aid)
    persisted = persisted_answers(gw.db_path, aid)
    assert len(persisted) == 1, f"answer {aid} persisted {len(persisted)}x\n{ctx}"
    want = norm(persisted[0])
    complete = [c for c in copies if c.complete]
    partial = [c for c in copies if not c.complete]
    assert complete, f"no complete visible reply for {aid} (silently lost or truncated)\n{ctx}"
    assert not partial, f"a truncated/stale partial rendition of {aid} stayed visible: {partial}\n{ctx}"
    for c in complete:
        if c.text != want:
            at = next((i for i, (a, b) in enumerate(zip(c.text, want)) if a != b), min(len(c.text), len(want)))
            raise AssertionError(
                f"visible reply != persisted assistant text (whitespace-insensitive), first diff at {at}:\n"
                f"visible:   ...{c.text[max(0, at - 60):at + 60]!r}\npersisted: ...{want[max(0, at - 60):at + 60]!r}\n{ctx}")
    unmarked = [c for c in complete if not c.marked]
    marked = [c for c in complete if c.marked]
    if marked_duplicates_allowed:
        assert len(unmarked) <= 1, f"{len(unmarked)} unmarked copies of {aid} (silent duplicate)\n{ctx}"
    elif first_copy_may_be_marked:
        assert len(complete) == 1, f"{len(complete)} visible copies of {aid}, expected exactly one\n{ctx}"
    else:
        assert len(unmarked) == 1 and not marked, (
            f"expected exactly one unmarked reply for {aid}: unmarked={len(unmarked)} marked={len(marked)}\n{ctx}")
    stray = [m.text[:80] for m in msgs if "<<R-" in m.text or "-extra" in m.text.split(">>")[0]]
    assert not stray, f"an extra model turn (re-run / auto-resume) replied in this chat: {stray}\n{ctx}"
    assert not director.resumes.get(token), f"inbound {token} was re-run by auto-resume\n{ctx}"
    rows = persisted_user_rows(gw.db_path, f"[in:{token}]")
    assert len(rows) == 1, f"inbound {token} persisted {len(rows)}x as a user turn\n{ctx}"
    assert director.turns.get(token) == 1, f"inbound {token} started {director.turns.get(token)} model turns\n{ctx}"


LONG = " ".join(f"word{i:04d}" for i in range(900))  # ~8.1k chars: splits on both caps
SLOW = dict(chunk_chars=6, delay_per_chunk=0.03)  # a ~2 s stream: preview + several edits


def _say(body: str, **kw):
    return lambda aid: [Director.answer(aid, body, **kw)]


# fault id -> (fault kind | None, op, target, responses(aid), expectation)
#   op: "send" | "edit" | "any"; target: "footer" = the op carrying the FINAL text,
#   "header" = the first op of the reply (the stream preview's first send)
#   expectation: exact = one unmarked copy, nothing else; one_copy = exactly one complete copy (a
#   retry of a never-seen send may carry the honest "may be a duplicate" marker); marked_dupes = the
#   platform may hold a second copy (ack lost) but every extra copy is marked.
FAULTS = {
    "clean": (None, "any", "footer", _say("short answer."), "exact"),
    "tool_then_answer": (None, "any", "footer",
                         lambda aid: [ToolCall("terminal", {"command": "echo hi"}),
                                      Director.answer(aid, "after the tool, done.")], "exact"),
    # The whole 8.1k reply in ONE stream chunk: the first send overflows every cap and is split
    # once, with nothing streamed after the split.
    "long_split": (None, "any", "footer", _say(LONG, chunk_chars=10_000), "exact"),
    # 2.5k chunks 1 s apart (edit interval 0.05 s), so every chunk lands in its own consumer tick:
    # the Discord-like first send already overflows (split, tail kept as the live preview) and the
    # next chunks extend that tail; the Telegram-like one overflows on an edit and seals.
    "long_streamed": (None, "any", "footer", _say(LONG, chunk_chars=2500, delay_per_chunk=1.0), "exact"),
    # Two 4.5k bursts 3 s apart (edit interval 0.05 s): the first send overflows (split, tail kept as
    # the live preview), then the second burst leaves a remainder over the limit after sealing.
    "burst_overflow": (None, "any", "footer", _say(LONG, chunk_chars=4500, delay_per_chunk=3.0), "exact"),
    "slow_stream": (None, "any", "footer", _say("a slowly streamed answer " * 6, **SLOW), "exact"),
    "rate_limit_429": ("rate_limit", "any", "footer", _say("throttled once then fine."), "exact"),
    "conn_reset": ("conn_error", "any", "footer", _say("connection reset once."), "exact"),
    "send_timeout": ("timeout", "any", "footer", _say("platform never saw the first try."), "one_copy"),
    # One chunk: on streaming platforms the consumer's FIRST send then carries the whole reply (and
    # the footer the fault targets), so the ack is always lost on that send, never on a later edit.
    "ack_lost": ("ack_lost", "any", "footer", _say("sent but the ack vanished.", chunk_chars=10_000), "marked_dupes"),
    "stream_429_final_edit": ("rate_limit", "any", "footer", _say("stream throttled " * 8, **SLOW), "exact"),
    "stream_timeout_final_edit": ("timeout", "any", "footer", _say("stream final lost " * 8, **SLOW),
                                  "one_copy"),
    "stream_ack_lost_final_edit": ("ack_lost", "any", "footer", _say("stream ack lost " * 8, **SLOW),
                                   "marked_dupes"),
    # The first preview send times out; four more chunks follow, each in its own consumer tick.
    "stream_timeout_first_send": ("timeout", "send", "header",
                                  _say("preview never landed " * 8, chunk_chars=42, delay_per_chunk=0.5),
                                  "one_copy"),
}


# Streaming gaps #120315 fixes: an overflowing first send keeps the " (n/n)" indicator on the live
# tail that later chunks extend / a sealed remainder over the limit is re-sent; a failed first send
# leaves an uneditable partial preview next to the final. Every cell here fails every run while the
# gap is open (the chunk pacing above puts each chunk in its own consumer tick).
STREAM_OVERFLOW_GAP = "LIVE GAP (fixed by #120315): streamed overflow / failed first send"
STREAM_OVERFLOW_GAP_CELLS = {
    ("burst_overflow", "fk_tg"), ("burst_overflow", "fk_dc"),
    ("long_streamed", "fk_dc"),
    ("stream_timeout_first_send", "fk_tg"), ("stream_timeout_first_send", "fk_dc"),
}


STREAM_ACK_LOST_GAP = (
    "LIVE GAP (#53449/#25349 family): on a streaming platform the stream consumer's first send is the "
    "whole short reply; when the platform accepts it but the ack is lost (timeout), the consumer reports "
    "nothing delivered and the gateway's final send re-sends it UNMARKED -> the user sees the reply "
    "twice with no 'may be a duplicate' marker. The non-streaming path hands the same timeout to the "
    "delivery ledger, which redelivers with the marker.")
# No fix PR yet: a run-time xfail on this exact failure (see _pending_fixes.known_failure), so the
# cell passes on its own the day the gap closes.
STREAM_ACK_LOST_SIGNATURE = r"^\d+ unmarked copies of A-fk_(tg|dc)\.ack_lost \(silent duplicate\)"
# Tokens whose cell actually XFAILed on a known gap this run: excluded from the whole-run audit.
XFAILED_TOKENS: set = set()


def faults_fired(gw: GatewayProcess, aid: str) -> List[dict]:
    chat = "m-" + aid[2:].replace(".", "-", 1)
    return [f for f in read_jsonl(gw.spool / "faults_fired.jsonl") if f.get("chat_id") == chat]


@pytest.mark.parametrize("platform", list(PLATFORMS))
@pytest.mark.parametrize("fault", list(FAULTS))
def test_delivery_fault_matrix(gw, director, platform, fault, request):
    kind, op, target, build, expect = FAULTS[fault]
    token = f"{platform}.{fault}"
    chat = f"m-{platform}-{fault}"
    aid = f"A-{token}"
    director.script(token, *build(aid))
    if kind is not None:
        if target == "footer":
            gw.fault(platform=platform, op=op, kind=kind, contains=footer(aid), chat_id=chat)
        else:  # the reply's first platform call, whatever it carries (a stream preview is a prefix)
            gw.fault(platform=platform, op=op, kind=kind, chat_id=chat)
    gw.inject(platform, f"[in:{token}] please answer", message_id=f"in-{token}", chat_id=chat)
    gw.wait_idle([chat], f"{token} turn to finish")
    wait_until(lambda: ledger_quiescent(gw.db_path, chat), f"{token} ledger to settle",
               proc=gw.proc, log=gw.log)
    wait_until(lambda: any(c.complete for c in visible_copies(gw.platform_view().visible(platform, chat), aid)),
               f"{token} complete reply\n" + dump(gw, platform, chat), proc=gw.proc, log=gw.log)
    if kind is not None:
        assert faults_fired(gw, aid), f"injected {kind} never hit a platform call\n{dump(gw, platform, chat)}"
    if (fault, platform) in STREAM_OVERFLOW_GAP_CELLS:
        expect_gap(request, 120315, STREAM_OVERFLOW_GAP)
    guard = (known_failure(STREAM_ACK_LOST_SIGNATURE, STREAM_ACK_LOST_GAP,
                           on_xfail=lambda: XFAILED_TOKENS.add(token))
             if fault == "ack_lost" and STREAMING[platform] else contextlib.nullcontext())
    with guard:
        assert_exactly_once(gw, director, platform, chat, token, aid,
                            marked_duplicates_allowed=expect == "marked_dupes",
                            first_copy_may_be_marked=expect == "one_copy")


# Concurrency: slow turn + busy follow-up, interrupt, parallel threads -------------------------


def _first_op_seen(gw: GatewayProcess, platform: str, chat: str) -> bool:
    return any(o.get("platform") == platform and o.get("chat_id") == chat and o["op"] in ("send", "edit")
               for o in gw.ops())


def _order(gw: GatewayProcess, platform: str, chat: str, aid: str) -> int:
    for i, m in enumerate(gw.platform_view().visible(platform, chat)):
        if header(aid) in m.text:
            return i
    return -1


VERY_SLOW = dict(chunk_chars=6, delay_per_chunk=0.08)


@pytest.mark.parametrize("platform", ["fk_tg", "fk_ne"])
def test_slow_turn_with_queued_followup(gw, director, platform):
    """A follow-up sent with /queue while the first turn streams: both inbound messages get exactly
    one complete reply, in arrival order, and neither turn runs twice."""
    t1, t2 = f"{platform}.q1", f"{platform}.q2"
    chat = f"q-{platform}"
    director.script(t1, Director.answer(f"A-{t1}", "first, slowly " * 10, **VERY_SLOW))
    director.script(t2, Director.answer(f"A-{t2}", "second answer."))
    gw.inject(platform, f"[in:{t1}] first", message_id=f"in-{t1}", chat_id=chat)
    wait_until(lambda: director.turns.get(t1) == 1, f"{t1} model turn to start", proc=gw.proc, log=gw.log)
    gw.inject(platform, f"/queue [in:{t2}] second", message_id=f"in-{t2}", chat_id=chat)
    gw.wait_idle([chat], "both turns to drain")
    wait_until(lambda: ledger_quiescent(gw.db_path, chat), "ledger to settle", proc=gw.proc, log=gw.log)
    for tok in (t1, t2):
        assert_exactly_once(gw, director, platform, chat, tok, f"A-{tok}")
    assert 0 <= _order(gw, platform, chat, f"A-{t1}") < _order(gw, platform, chat, f"A-{t2}"), dump(gw, platform, chat)


@pytest.mark.parametrize("platform", ["fk_tg", "fk_ne"])
def test_slow_turn_interrupted_by_followup(gw, director, platform):
    """A plain follow-up interrupts the running turn (default busy mode). The follow-up gets exactly
    one complete reply; the interrupted turn never leaves a second or stale-partial copy behind and
    is not re-run; nothing is persisted twice."""
    t1, t2 = f"{platform}.i1", f"{platform}.i2"
    chat = f"i-{platform}"
    director.script(t1, Director.answer(f"A-{t1}", "long interrupted answer " * 30, **VERY_SLOW))
    director.script(t2, Director.answer(f"A-{t2}", "the follow-up answer."))
    gw.inject(platform, f"[in:{t1}] first", message_id=f"in-{t1}", chat_id=chat)
    wait_until(lambda: director.turns.get(t1) == 1, f"{t1} model turn to start", proc=gw.proc, log=gw.log)
    if STREAMING[platform]:
        wait_until(lambda: _first_op_seen(gw, platform, chat), "first preview", proc=gw.proc, log=gw.log)
    gw.inject(platform, f"[in:{t2}] actually, do this instead", message_id=f"in-{t2}", chat_id=chat)
    gw.wait_idle([chat], "interrupt + follow-up to drain")
    wait_until(lambda: ledger_quiescent(gw.db_path, chat), "ledger to settle", proc=gw.proc, log=gw.log)
    print(dump(gw, platform, chat))
    assert_exactly_once(gw, director, platform, chat, t2, f"A-{t2}")
    assert director.turns.get(t1) == 1, dump(gw, platform, chat)
    assert len(persisted_user_rows(gw.db_path, f"[in:{t1}]")) == 1, dump(gw, platform, chat)
    copies = visible_copies(gw.platform_view().visible(platform, chat), f"A-{t1}")
    assert len([c for c in copies if c.complete]) <= 1, dump(gw, platform, chat)


def test_parallel_threads_are_independent(gw, director):
    """Two Discord threads in one channel run slow turns at the same time: each gets exactly one
    complete reply, delivered into its own thread."""
    chat = "par-dc"
    toks = ["fk_dc.th1", "fk_dc.th2"]
    for tok in toks:
        director.script(tok, Director.answer(f"A-{tok}", f"reply for {tok} " * 8, **SLOW))
    for n, tok in enumerate(toks):
        gw.inject("fk_dc", f"[in:{tok}] go", message_id=f"in-{tok}", chat_id=chat, thread_id=f"thr{n}")
    gw.wait_idle([chat], "both threads to finish")
    for n, tok in enumerate(toks):
        wait_until(lambda: ledger_quiescent(gw.db_path, chat), "ledger to settle", proc=gw.proc, log=gw.log)
        assert_exactly_once(gw, director, "fk_dc", chat, tok, f"A-{tok}")
        threads = {m.thread_id for m in gw.platform_view().visible("fk_dc", chat) if header(f"A-{tok}") in m.text}
        assert threads == {f"thr{n}"}, f"{tok} delivered into {threads}\n{dump(gw, 'fk_dc', chat)}"


# Inbound re-delivery ---------------------------------------------------------------------------


@pytest.mark.parametrize("when", ["during_turn", "after_turn", "after_reconnect"])
def test_redelivered_inbound_id_processed_once(gw, director, when):
    """The platform re-delivers an inbound id (webhook retry, websocket resume, reconnect replay):
    the message is processed exactly once and answered exactly once."""
    platform = "fk_tg"
    tok = f"{platform}.redeliver-{when}"
    chat = f"r-{when}"
    director.script(tok, Director.answer(f"A-{tok}", "answer once " * 6, **SLOW))
    text, mid = f"[in:{tok}] hello", f"in-{tok}"
    gw.inject(platform, text, message_id=mid, chat_id=chat)
    if when == "during_turn":
        wait_until(lambda: director.turns.get(tok) == 1, "turn start", proc=gw.proc, log=gw.log)
        gw.inject(platform, text, message_id=mid, chat_id=chat)
    gw.wait_idle([chat], f"{tok} turn")
    if when == "after_turn":
        gw.inject(platform, text, message_id=mid, chat_id=chat)
    if when == "after_reconnect":
        before = gw.rpc("reconnect", platform=platform)["id"]
        wait_until(lambda: gw.rpc("adapter_id", platform=platform)["id"] not in (before, None),
                   "the reconnect watcher to install a fresh adapter", proc=gw.proc, log=gw.log)
        gw.inject(platform, text, message_id=mid, chat_id=chat)
    gw.wait_idle([chat], f"{tok} after re-delivery")
    wait_until(lambda: ledger_quiescent(gw.db_path, chat), "ledger to settle", proc=gw.proc, log=gw.log)
    assert_exactly_once(gw, director, platform, chat, tok, f"A-{tok}")


# Crash between provider completion and the send: SIGKILL + restart on the same state ------------

# crash point -> (fault op, fault kind)
CRASH_POINTS = {
    # the turn returned (persisted), the final text not yet handed to the delivery ledger
    "completed_not_ledgered": ("pre_ledger", "hold"),
    # the final send/edit is in flight and the platform never saw it
    "send_in_flight": ("any", "hold"),
    # the turn is persisted and the platform accepted the final send/edit; the ack died with the process
    "sent_ack_lost": ("any", "hold_after"),
    # streaming: the platform accepted a preview edit that already shows the complete answer, while
    # the provider stream (so the turn) is still open -> nothing persisted when the process dies
    "stream_accepted_unpersisted": ("any", "hold_after"),
}

# completed_not_ledgered: the turn is persisted and the handler returned, but the final text never
# reached the delivery ledger; recovery must hand the persisted reply to the ledger rather than
# re-run the turn. Either way exactly one complete reply must show.

STREAM_CRASH_AFTER_ACCEPT_GAP = (
    "LIVE GAP: streaming turn, the platform already shows the complete final answer, SIGKILL before "
    "the turn is persisted -> restart auto-resumes and the model answers AGAIN; the second answer is "
    "unmarked and the first (visible) answer was never persisted (visible != transcript).")
# No fix PR yet: run-time xfail only when the visible pair is exactly the gap's shape, the held
# original answer AND an auto-resumed re-answer, both unmarked.
STREAM_CRASH_AFTER_ACCEPT_SIGNATURE = (
    r"(?s)^(2 unmarked complete replies after catch-up|both the held answer and an auto-resumed "
    r"re-answer are visible)\n.*<<A-fk_tg\.crash-stream_accepted_unpersisted>>.*"
    r"<<R-fk_tg\.crash-stream_accepted_unpersisted-\d>>")


def _replies(gw: GatewayProcess, platform: str, chat: str, token: str, aid: str):
    msgs = gw.platform_view().visible(platform, chat)
    originals = visible_copies(msgs, aid)
    resumes = [c for n in range(1, 6) for c in visible_copies(msgs, f"R-{token}-{n}")]
    return originals, resumes


CRASH_MATRIX = [("completed_not_ledgered", "fk_ne"),  # streaming finals are sent before the handler returns
                ("send_in_flight", "fk_ne"), ("send_in_flight", "fk_tg"),
                ("sent_ack_lost", "fk_ne"), ("sent_ack_lost", "fk_tg"),
                ("stream_accepted_unpersisted", "fk_tg")]


@pytest.fixture
def fresh_gw(tmp_path, llm):
    """A gateway on its OWN home: one unclean restart per home keeps the restart-loop breaker and
    the 120 s recency fallback of other scenarios out of the picture."""
    keep = os.environ.get("C12_SPOOL_ROOT")
    root = Path(keep) / f"crash-{tmp_path.name}" if keep else tmp_path / "gw"
    proc = GatewayProcess(root, platforms=PLATFORMS, llm_base_url=llm.base_url, extra_config=EXTRA_CONFIG)
    proc.start()
    yield proc
    proc.stop()


@pytest.mark.parametrize("point,platform", CRASH_MATRIX)
def test_crash_between_completion_and_send(fresh_gw, director, point, platform):
    """kill -9 the gateway at a point between provider completion and the platform ack, restart it
    on the same HERMES_HOME. Recovery may be the delivery ledger (redeliver the held answer) or,
    for a turn that never finished, auto-resume (a fresh model turn): either way the inbound ends
    with exactly ONE complete visible reply that matches the transcript, and any second copy the
    platform may hold is marked as a possible duplicate."""
    gw = fresh_gw
    op, kind = CRASH_POINTS[point]
    streaming = STREAMING[platform]
    token = f"{platform}.crash-{point}"
    chat = f"k-{platform}-{point}"
    aid = f"A-{token}"
    answer = Director.answer(aid, "answer computed before the crash.")
    if point == "stream_accepted_unpersisted":  # the whole answer streams, then the stream stays open
        answer = StallMidStream(text=answer.text, after_chars=len(answer.text), seconds=60)
    director.script(token, answer)
    gw.fault(platform=platform, op=op, kind=kind, contains=footer(aid), chat_id=chat)
    gw.inject(platform, f"[in:{token}] question", message_id=f"in-{token}", chat_id=chat)
    wait_until(lambda: any(footer(aid) in h["content"] for h in gw.holds()), f"{token} to reach the crash point",
               proc=gw.proc, log=gw.log)
    if point == "sent_ack_lost":  # the stream's footer edit can land before the turn is persisted
        wait_until(lambda: persisted_answers(gw.db_path, aid), f"{token} answer persisted before the kill",
                   proc=gw.proc, log=gw.log)
    gw.kill9()
    gw.start()

    def replied() -> bool:
        originals, resumes = _replies(gw, platform, chat, token, aid)
        return any(c.complete for c in originals + resumes)

    wait_until(replied, f"{token} catch-up reply after restart\n" + dump(gw, platform, chat),
               proc=gw.proc, log=gw.log)
    gw.wait_idle([chat], f"{token} to settle after restart")
    wait_until(lambda: ledger_quiescent(gw.db_path, chat), f"{token} ledger to settle", proc=gw.proc, log=gw.log)
    # Every wait above must have succeeded before a known gap may XFAIL the cell: only the final
    # assertions run under the run-time xfail, so a lost reply or a failed restart stays red.
    guard = (known_failure(STREAM_CRASH_AFTER_ACCEPT_SIGNATURE, STREAM_CRASH_AFTER_ACCEPT_GAP)
             if point == "stream_accepted_unpersisted" else contextlib.nullcontext())
    with guard:
        _assert_single_final_reply(gw, director, platform, chat, token, aid, point, streaming)


def _assert_single_final_reply(gw, director, platform, chat, token, aid, point, streaming) -> None:
    ctx = dump(gw, platform, chat)
    originals, resumes = _replies(gw, platform, chat, token, aid)
    complete = [c for c in originals + resumes if c.complete]
    unmarked = [c for c in complete if not c.marked]
    assert len(unmarked) <= 1, f"{len(unmarked)} unmarked complete replies after catch-up\n{ctx}"
    assert len(complete) - len(unmarked) <= (1 if point == "sent_ack_lost" else 0) or not unmarked, (
        f"unexpected extra marked copies\n{ctx}")
    # A crash mid-stream may leave the interrupted preview on screen; nothing else may be partial.
    assert not [c for c in resumes if not c.complete], ctx
    if not streaming:
        assert not [c for c in originals if not c.complete], ctx
    assert not (resumes and [c for c in originals if c.complete]), (
        f"both the held answer and an auto-resumed re-answer are visible\n{ctx}")
    final = resumes[0] if resumes else next(c for c in originals if c.complete)
    final_aid = aid if not resumes else f"R-{token}-{director.resumes.get(token, 0)}"
    persisted = persisted_answers(gw.db_path, final_aid)
    assert len(persisted) == 1 and norm(persisted[0]) == final.text, (
        f"visible final reply is not the (single) persisted assistant text: persisted={persisted}\n{ctx}")
    assert director.turns.get(token) == 1, ctx
    assert len(persisted_user_rows(gw.db_path, f"[in:{token}]")) == 1, ctx


RESUME_EXPECTED: set = set()


def test_zz_whole_run_audit(gw, director):
    """Across EVERY chat of the module (every scenario, every restart): no inbound ended with two
    unmarked complete replies (held answer, extra turn or auto-resume), and no already-answered
    inbound was re-run by a LATER restart's auto-resume."""
    visible = gw.platform_view().visible()
    head = re.compile(r"<<((?:A-|R-)?[A-Za-z0-9_.-]+?)>>")
    problems = []
    gaps = set(XFAILED_TOKENS)  # cells that XFAILed on a known gap this run
    for token in sorted(director.turns):
        if token in gaps or ".crash-" in token:  # crash scenarios run on their own homes
            continue
        ids = {m.group(1) for v in visible for m in head.finditer(v.text)
               if m.group(1) in (f"A-{token}",) or m.group(1).startswith((f"R-{token}-", f"{token}-extra"))}
        unmarked = 0
        for aid in ids:
            by_chat: Dict[str, list] = {}
            for v in visible:
                by_chat.setdefault((v.platform, v.chat_id), []).append(v)
            for msgs in by_chat.values():
                unmarked += sum(1 for c in visible_copies(msgs, aid) if c.complete and not c.marked)
        if unmarked > 1:
            problems.append(f"{token}: {unmarked} unmarked complete replies ({sorted(ids)})")
        if director.resumes.get(token) and token not in RESUME_EXPECTED:
            problems.append(f"{token}: re-run by auto-resume {director.resumes[token]}x after it was answered")
    assert not problems, "\n".join(problems)


def test_zzz_unclean_restart_reruns_nothing(gw, director):
    """Runs LAST on the module gateway: after every scenario above has been answered, an unclean
    restart with NOTHING in flight must not re-run or re-deliver anything."""
    tok = "fk_tg.quiet-restart"
    director.script(tok, Director.answer(f"A-{tok}", "answered long before the crash."))
    gw.inject("fk_tg", f"[in:{tok}] hi", message_id=f"in-{tok}", chat_id="quiet")
    gw.wait_idle(None, "every chat idle before the crash", settle=5)
    wait_until(lambda: all(state in ("delivered", "abandoned") for state, _, _ in all_ledger_rows(gw.db_path)),
               "ledger quiescent before the crash", proc=gw.proc, log=gw.log)
    assert_exactly_once(gw, director, "fk_tg", "quiet", tok, f"A-{tok}")
    before = {m.message_id for m in gw.platform_view().visible()}
    # The Director is module-wide: crash cells on their own homes may legitimately resume their
    # in-flight turn, so only what THIS restart adds counts.
    turns, resumes = dict(director.turns), dict(director.resumes)
    gw.kill9()
    gw.start()
    # A buggy restart shows its first extra reply within a few seconds of boot; a correct one never
    # does. Waiting for the bad outcome (not for idleness) keeps the check honest on a slow box.
    try:
        wait_until(lambda: [m for m in gw.platform_view().visible() if m.message_id not in before],
                   "an unsolicited post-restart reply", timeout=30, proc=gw.proc, log=gw.log)
    except AssertionError as exc:
        if "timed out" not in str(exc):
            raise
    gw.wait_idle(None, "the restarted gateway to go idle", settle=5)
    new = [m for m in gw.platform_view().visible() if m.message_id not in before]
    assert not new, f"an unclean restart with nothing in flight delivered {len(new)} new messages, e.g.:\n" + \
        "\n".join(f"  {m.chat_id}: {m.text[:100]!r}" for m in new[:8])
    assert director.turns == turns and director.resumes == resumes, (
        f"model re-run after restart: {director.resumes} (before the kill: {resumes})")




def all_ledger_rows(db_path: Path) -> List[tuple]:
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=30)
    try:
        return conn.execute("SELECT state, attempts, chat_id FROM delivery_obligations").fetchall()
    except sqlite3.OperationalError:
        return []
    finally:
        conn.close()
