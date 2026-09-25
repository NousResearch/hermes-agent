"""Compaction persistence under multi-process contention (issue class C1, "state.db corrupting from
compaction").

Real ``AIAgent`` processes drive real turns (user -> terminal tool call -> answer) through the loopback fake
provider on ONE shared ``state.db`` — once per journal mode Hermes deploys (WAL, and the DELETE mode the
production ``apply_wal_with_fallback`` picks on a WAL-reset-vulnerable SQLite, see ``_helpers``) — while other
processes write to, read and open/close the same file:

* a gateway-like agent with rolling micro-compaction on (every turn folds the oldest exchange into a summary
  and commits it through ``archive_and_compact``);
* a TUI-like agent that runs ``/compress here 2`` mid-session (in-place manual compaction, kept tail);
* a plain gateway writer on another session, a dashboard reader, and a CLI-style open/close churner.

Fault matrix: clean, ``kill -9`` of the micro-compacting agent mid-turn, and ``kill -9`` timed into its
compaction (the summary stream / commit window). A fresh process then resumes the session and runs more
turns. Invariants after every episode:

* no acknowledged turn is lost: its user message is in the live transcript exactly once (user turns are
  never absorbed by compaction), its tool result and answer are still recoverable — live (``active=1``) or
  as compacted history (``compacted=1``) — and never live twice;
* the exchanges ``/compress here N`` promised to keep are live exactly once;
* canonical row counts only grow (compaction archives, it never deletes) — sampled continuously from outside
  and by the reader; ``integrity_check`` ok; FTS mirrors canonical rows; the store stays in the arm's journal
  mode; no ``(deleted)`` store or WAL held, nor a persistent deleted SHM descriptor;
* the resumed process sends the model every acknowledged user turn exactly once (persisted == sent);
* the plain writer's acked appends are stored exactly once.
"""

from __future__ import annotations

import contextlib
import json
import random
import re
import sqlite3
import sys
import threading
import time
from dataclasses import dataclass

import pytest

from tests.e2e.core.sqlite._helpers import (
    JOURNAL_MODES,
    SEED_ENV,
    Chamber,
    base_seed,
    busy_summary,
    compress_journal,
    counts,
    episode_seed,
    exactly_once_problems,
    fts_problems,
    integrity_rows,
    journal_mode_problems,
    skip_unless_deployable,
    token_flags,
)
from tests.fakes.fake_llm_provider import FakeLLMServer, Text, ToolCall, write_hermes_home

pytestmark = [
    pytest.mark.skipif(not sys.platform.startswith("linux"), reason="/proc fd scan and POSIX lock semantics"),
]

Q_TOKEN = re.compile(r"TA[a-z]+\d+N\d+Q")
MICRO_CONFIG = "compression:\n  micro_compact: true\n  target_ratio: 0.1\n  protect_last_n: 4\n"
DB_CONFIG = "database:\n  journal_mode: wal\n"
ANSWER_FILLER = "lorem " * 1200  # big enough that the compressible window is non-empty at 64K context


def _text(content) -> str:
    if isinstance(content, list):
        return " ".join(str(p.get("text", "")) for p in content if isinstance(p, dict))
    return str(content or "")


def _responder(record: dict):
    """Main turns: a user turn gets a terminal tool call, the tool result gets the final answer."""
    messages = record["body"]["messages"]
    last_user = next(_text(m.get("content")) for m in reversed(messages) if m.get("role") == "user")
    base = Q_TOKEN.findall(last_user)[-1][:-1]
    if messages[-1].get("role") == "tool":
        return Text(f"{base}A answer. {ANSWER_FILLER}", chunk_chars=2000)
    return ToolCall("terminal", {"command": f"echo {base}T"})


class _Aux:
    """Summaries for compaction; flags every micro-compaction summary request of the gateway agent."""

    def __init__(self):
        self.summaries = 0
        self._cond = threading.Condition()

    def wait_beyond(self, seen: int, still_running) -> bool:
        """Block until a summary request newer than ``seen`` arrives (False if the agent exited first)."""
        with self._cond:
            while self.summaries <= seen:
                if not still_running():
                    return False
                self._cond.wait(0.05)
            return True

    def __call__(self, record: dict):
        body = json.dumps(record["body"])
        if re.search(r"TAg\d+N\d+T", body):  # the exchange text carries the tool result -> a summary call
            with self._cond:
                self.summaries += 1
                self._cond.notify_all()
        return Text("Rolling summary: the user asked numbered questions and each was answered.",
                    chunk_chars=6, delay_per_chunk=0.01)


@dataclass
class Rig:
    ch: Chamber
    server: FakeLLMServer
    aux: _Aux
    tui_env: dict


@pytest.fixture(scope="module", params=JOURNAL_MODES)
def rig(request, tmp_path_factory):
    skip_unless_deployable(request.param)
    aux = _Aux()
    server = FakeLLMServer(_responder, aux=aux)
    server.start()
    ch = Chamber(tmp_path_factory.mktemp(f"compaction-{request.param}"), journal=request.param)
    ctx = "  context_length: 128000\n"
    write_hermes_home(ch.hermes_home, server.base_url, extra_config=MICRO_CONFIG + DB_CONFIG)
    tui_home = ch.root / "tui-home"
    write_hermes_home(tui_home / ".hermes", server.base_url, extra_config=DB_CONFIG)
    for home in (ch.hermes_home, tui_home / ".hermes"):
        cfg = home / "config.yaml"
        cfg.write_text(cfg.read_text(encoding="utf-8").replace(ctx, "  context_length: 64000\n"), encoding="utf-8")
    yield Rig(ch, server, aux, {"HOME": str(tui_home), "HERMES_HOME": str(tui_home / ".hermes")})
    ch.shutdown()
    server.stop()


class _GrowOnly(threading.Thread):
    """Samples the canonical row count from outside every ~50 ms; compaction must never shrink it."""

    def __init__(self, ch: Chamber):
        super().__init__(daemon=True)
        self.ch, self.stop, self.violations, self.samples = ch, threading.Event(), [], 0

    def run(self):
        last = 0
        while not self.stop.is_set():
            try:
                n = counts(self.ch.db)["__total__"]
            except sqlite3.OperationalError:
                n = last  # busy under a checkpoint: skip the sample, never guess
            if n < last:
                self.violations.append(f"canonical rows shrank {last} -> {n}")
            last = max(last, n)
            self.samples += 1
            self.stop.wait(0.05)


def _agent(rig: Rig, name, *, sid, tag, turns, env=None, **kw):
    rig.ch.spawn("agent", name, env=env, base_url=rig.server.base_url, session_id=sid, tag=tag, turns=turns, **kw)
    rig.ch.wait_event(name, "ready", deadline=90)


def _turn_problems(ch: Chamber, run: str, *, compacting: bool) -> list[str]:
    intents, acked = ch.journal(run)
    problems = []
    for base in sorted(acked):
        q, t, a = token_flags(ch.db, base + "Q"), token_flags(ch.db, base + "T"), token_flags(ch.db, base + "A")
        live_q = sum(1 for f in q if f[0] == 1)
        if live_q != 1:
            problems.append(f"{run}: acked user turn {base}Q is live {live_q}x (rows {q})")
        for label, flags in (("tool result", t), ("answer", a)):
            live = sum(1 for f in flags if f[0] == 1)
            kept = sum(1 for f in flags if f[0] == 1 or f[1] == 1)
            if kept < 1:
                problems.append(f"{run}: acked {label} of {base} is unrecoverable: every row is "
                                f"archived-as-superseded (active=0, compacted=0): {flags}")
            if live > 1:
                problems.append(f"{run}: acked {label} of {base} is live {live}x: {flags}")
            if not compacting and live != 1:
                problems.append(f"{run}: {label} of {base} not live without any compaction: {flags}")
    for base in sorted(set(intents) - acked):  # the turn a SIGKILL interrupted
        live_q = sum(1 for f in token_flags(ch.db, base + "Q") if f[0] == 1)
        if live_q > 1:
            problems.append(f"{run}: in-flight user turn {base}Q is live {live_q}x")
    for kept in compress_journal(ch, run):
        for base in kept:
            for suffix in "QTA":
                flags = token_flags(ch.db, base + suffix)
                if sum(1 for f in flags if f[0] == 1) != 1:
                    problems.append(f"{run}: /compress here kept {base}{suffix} but it is live "
                                    f"{sum(1 for f in flags if f[0] == 1)}x (rows {flags})")
    return problems


def _resume_problems(rig: Rig, sid: str, runs: list[str], first_request: int) -> list[str]:
    """The fresh process must send the model every acked user turn of the session exactly once."""
    ch = rig.ch
    requests = rig.server.main_requests()[first_request:]
    if not requests:
        return [f"resume of {sid} sent no request"]
    sent = " ".join(_text(m.get("content")) for m in requests[-1]["messages"])
    problems = []
    for run in runs:
        for base in sorted(ch.journal(run)[1]):
            n = sent.count(base + "Q")
            if n != 1:
                problems.append(f"resumed request carries acked user turn {base}Q {n}x (persisted != sent)")
    return problems


def _compacted_rows(ch: Chamber, sid: str) -> int:
    conn = sqlite3.connect(f"file:{ch.db}?mode=ro", uri=True, timeout=30.0)
    try:
        return conn.execute("SELECT count(*) FROM messages WHERE session_id = ? AND compacted = 1",
                            (sid,)).fetchone()[0]
    finally:
        conn.close()


def _reap_ok(ch: Chamber, name: str, deadline: float = 120.0) -> None:
    rc = ch.reap(name, deadline=deadline)
    errors = [f"{e.get('error')}\n{e.get('tb', '')}" for e in ch.events(name) if e.get("event") == "error"]
    assert rc == 0, f"{name} exited {rc}: {errors}\n{ch.stderr(name)}"


@pytest.mark.parametrize("fault", ["clean", "kill9_agent_mid_turn", "kill9_during_micro_compaction"])
def test_compaction_episode(rig, fault):
    ch = rig.ch
    seed = episode_seed(f"compaction:{fault}")
    rng = random.Random(seed)
    ctx = f"[journal={ch.mode} fault={fault} seed={seed} base={base_seed()}; replay: {SEED_ENV}={base_seed()}]"
    gw_sid, tui_sid = f"{fault}-gw", f"{fault}-tui"
    gw, tui, plain, reader = f"{fault}-agent-gw", f"{fault}-agent-tui", f"{fault}-plain", f"{fault}-reader"
    sampler = _GrowOnly(ch)
    sampler.start()
    try:
        _agent(rig, gw, sid=gw_sid, tag="g", turns=5)
        _agent(rig, tui, sid=tui_sid, tag="t", turns=5, compress_at=3, keep=2, env=rig.tui_env)
        micro = [e for e in ch.events(gw) if e.get("event") == "ready"][0]["micro"]
        assert micro, f"{ctx} gateway agent did not load micro_compact from its config"
        ch.spawn("writer", plain, tag="p", sessions=[f"{fault}-plain"], source="telegram", pace=0.01)
        ch.spawn("reader", reader, pace=0.05)
        ch.spawn("churn", f"{fault}-churn", iterations=30)

        if fault == "kill9_agent_mid_turn":
            ch.wait_acks(gw, rng.randint(2, 4), deadline=120)
            time.sleep(rng.uniform(0.0, 0.8))  # fault-injection jitter inside the next turn, not a sync
            ch.kill9(gw)
        elif fault == "kill9_during_micro_compaction":
            ch.wait_acks(gw, rng.randint(2, 3), deadline=120)
            # Turn k's summary precedes ack k, so the next one belongs to a later turn's finalize.
            if rig.aux.wait_beyond(rig.aux.summaries, lambda: ch.procs[gw].poll() is None):
                time.sleep(rng.uniform(0.0, 0.3))  # inside the summary stream or its archive_and_compact commit
            ch.kill9(gw)
        else:
            _reap_ok(ch, gw)
        _reap_ok(ch, tui)
        _reap_ok(ch, f"{fault}-churn")
        ch.request_stop(plain)
        _reap_ok(ch, plain)

        # A fresh process resumes the gateway session and keeps going (micro-compaction still on).
        first = len(rig.server.main_requests())
        resumed = f"{fault}-agent-gw-resume"
        _agent(rig, resumed, sid=gw_sid, tag="g", turns=1, resume=True)
        _reap_ok(ch, resumed)
        ch.request_stop(reader)
        _reap_ok(ch, reader)
    finally:
        sampler.stop.set()
        sampler.join(timeout=10)
        # The rig is shared by every fault: a failed episode's live roles must not leak into the next.
        for name, _proc in ch.live():
            if name.startswith(fault):
                with contextlib.suppress(AssertionError):
                    ch.stop(name, deadline=30.0)

    problems: list[str] = []
    problems += [f"{n}: {e.get('error')}\n{e.get('tb', '')}" for n, e in ch.errors() if n.startswith(fault)]
    problems += [f"{n} (pid {pid}) held {link}" for n, pid, link in ch.deleted_hits_snapshot()]
    rows = integrity_rows(ch.db)
    if rows != ["ok"]:
        problems.append(f"integrity_check: {rows[:5]}")
    problems += journal_mode_problems(ch, prefix=fault)
    problems += sampler.violations
    problems += _turn_problems(ch, gw, compacting=True)
    problems += _turn_problems(ch, resumed, compacting=True)
    problems += _turn_problems(ch, tui, compacting=True)
    problems += _resume_problems(rig, gw_sid, [gw], first)
    problems += exactly_once_problems(ch)
    problems += fts_problems(ch.db, [])
    if not compress_journal(ch, tui):
        problems.append("the TUI agent never ran /compress here")
    for sid in (gw_sid, tui_sid):  # vacuity guard: a compaction commit really landed in this episode
        if not _compacted_rows(ch, sid):
            problems.append(f"no compacted history rows in {sid}: compaction never committed")
    busy = busy_summary(ch, fault)
    assert not problems, f"{ctx} ({sampler.samples} row-count samples, busy waits {busy})\n" + "\n".join(problems)
