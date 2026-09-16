"""Acceptance + invariant tests for agentpod-stop-check.

These drive the REAL Hermes runtime integration, not just the helper:

* the plugin is installed into an isolated ``HERMES_HOME`` and loaded through
  the real ``hermes_cli.plugins.discover_plugins()`` discovery path;
* supervision context is established through the real ``pre_llm_call``
  dispatch (``hermes_cli.lifecycle.invoke_hook``), the same call the runtime
  makes in ``agent/turn_context.py``;
* the fallback path runs through the real ``agent.turn_finalizer.finalize_turn``
  (the exact function that fires ``transform_llm_output`` once per turn);
* the continuation path runs through the real
  ``hermes_cli.plugins.get_pre_verify_continue_message()`` aggregator AND,
  in ``test_20``, through the REAL ``AIAgent.run_conversation`` loop — proving
  a no-edit turn continues into a tool call and then completes;
* boards are isolated temp SQLite boards created through the installed
  ``hermes_cli.kanban_db`` interface — no real tenant/card is touched;
* owner evidence uses tiny fixture processes this test owns and a temp process
  registry file; no process outside this test is ever inspected or signalled.

Tests 1-10 are the original acceptance scenarios (updated for replace-not-
append and user-message scoping). Tests 11-21 are the independent review's
adversarial observations (``/tmp/rev_t_cf118fa9/adversarial_test.py``, defects
A1-A5, B1, C1-C4, D1-D2, E2, F1-F2) converted from "assert the defect" into
"assert the required invariant".

Run:
    scripts/run_tests.sh contrib/den-plugins/agentpod-stop-check/test_stop_check.py -q
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[3]
# RED-GREEN: check the previous plugin revision out over this directory and
# re-run this file to see these invariants fail (command in the receipt). The
# `exists()` guards below are what let an older revision (no owners.py) load.
PLUGIN_SRC = Path(__file__).resolve().parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

SESSION = "sess-agentpod-supervisor"
OTHER_SESSION = "sess-somebody-else"
QUIET = "Checked the board — no material change since the last sweep."
SUPERVISION_MSG = "sweep the board and tell me where the project stands"
UNRELATED_MSG = "how much disk space is left on the mac?"
GATE_AUTHORITY = "den"


# --------------------------------------------------------------- harness ---

@pytest.fixture
def home(tmp_path, monkeypatch):
    h = tmp_path / "hermes-home"
    h.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(h))
    monkeypatch.delenv("HERMES_KANBAN_DB", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_BOARD", raising=False)
    return h


@pytest.fixture
def procs():
    """Tiny fixture processes owned by this test."""
    started: list[subprocess.Popen] = []

    def spawn(seconds: int = 120) -> subprocess.Popen:
        p = subprocess.Popen(
            [sys.executable, "-c", f"import time; time.sleep({seconds})"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        started.append(p)
        return p

    yield spawn
    for p in started:
        try:
            p.kill()
            p.wait(timeout=5)
        except Exception:
            pass


def _board(home: Path):
    from hermes_cli import kanban_db as kb

    return kb.connect(home / "board.db"), kb


def write_registry(home: Path, entries: list[dict]) -> None:
    """Write the isolated process-registry checkpoint this test owns."""
    (home / "processes.json").write_text(json.dumps(entries), encoding="utf-8")


def registry_entry(proc, *, task_id: str, command: str | None = None, **over) -> dict:
    """A registry row for a process THIS TEST spawned, with real identity."""
    from gateway.status import get_process_start_time

    entry = {
        "session_id": f"proc_{proc.pid}",
        "command": command or f"gtimeout 2700 pi --print 'work {task_id}'",
        "pid": proc.pid,
        "pid_scope": "host",
        "host_start_time": get_process_start_time(proc.pid),
        "cwd": f"/tmp/worktrees/{task_id}",
        "started_at": time.time(),
        "task_id": "",
        "session_key": "",
        "notify_on_complete": True,
        "watcher_interval": 5,
    }
    entry.update(over)
    return entry


def install_runtime(home: Path, *, extra_cfg: dict | None = None):
    """Install + load the plugin through the real discovery path."""
    from hermes_cli import plugins as P

    pdir = home / "plugins" / "agentpod-stop-check"
    if pdir.exists():
        shutil.rmtree(pdir)
    pdir.mkdir(parents=True)
    for name in ("__init__.py", "plugin.yaml", "stopcheck.py", "owners.py"):
        if (PLUGIN_SRC / name).exists():
            shutil.copy(PLUGIN_SRC / name, pdir / name)

    cfg = {
        "enabled": True,
        "db_path": str(home / "board.db"),
        "session_ids": [SESSION],
        "gate_authorities": [GATE_AUTHORITY],
        "heartbeat_stale_seconds": 900,
        "max_continuations": 2,
        "max_report_chars": 700,
        "process_registry_path": str(home / "processes.json"),
        "ledger_path": str(home / "stopcheck-ledger.json"),
    }
    cfg.update(extra_cfg or {})
    (home / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "plugins": {"enabled": ["agentpod-stop-check"]},
                "agentpod_stop_check": cfg,
            }
        )
    )
    P.discover_plugins(force=True)
    return P


def set_turn_context(user_message: str, *, session_id: str = SESSION) -> None:
    """Fire the REAL pre_llm_call dispatch the runtime uses (turn_context.py)."""
    from hermes_cli.lifecycle import invoke_hook

    invoke_hook(
        "pre_llm_call",
        session_id=session_id,
        task_id=None,
        turn_id="turn-1",
        user_message=user_message,
        conversation_history=[],
        is_first_turn=True,
        model="test-model",
        platform="telegram",
        parent_session_id="",
        sender_id="",
    )


def run_turn(
    final_response: str,
    *,
    session_id: str = SESSION,
    interrupted: bool = False,
    user_message: str = SUPERVISION_MSG,
    set_context: bool = True,
):
    """Drive the REAL turn finalizer (the transform_llm_output fire site)."""
    from unittest.mock import MagicMock

    from agent.turn_finalizer import finalize_turn

    if set_context:
        set_turn_context(user_message, session_id=session_id)

    class _Budget:
        remaining = 50

        def __getattr__(self, _n):
            return 0

    agent = MagicMock()
    agent.max_iterations = 100
    agent.iteration_budget = _Budget()
    agent.session_id = session_id
    agent.model = "test-model"
    agent.platform = "telegram"
    agent.provider = "test"
    agent.base_url = ""
    agent.quiet_mode = True
    agent._interrupt_message = None
    agent._response_was_previewed = False
    agent._skill_nudge_interval = 0
    agent._iters_since_skill = 0
    agent._db_flush_scan_prefix = 0
    agent._tool_guardrail_halt_decision = None
    agent._turn_completion_explainer_enabled = False
    agent._file_mutation_verifier_enabled = False
    agent._turn_received_provider_response = True
    agent._stream_callback = None
    agent.valid_tool_names = set()
    agent._drain_pending_steer.return_value = None
    messages = [
        {"role": "user", "content": user_message},
        {"role": "assistant", "content": final_response},
    ]
    return finalize_turn(
        agent,
        final_response=final_response,
        api_call_count=1,
        interrupted=interrupted,
        failed=False,
        messages=messages,
        conversation_history=[],
        effective_task_id=None,
        turn_id="turn-1",
        user_message=user_message,
        original_user_message=user_message,
        _should_review_memory=False,
        _turn_exit_reason="stop",
    )


def fire_pre_verify(*, session_id: str = SESSION, final_response: str = QUIET,
                    changed_paths=None, attempt: int = 0):
    """Drive the REAL pre_verify aggregator the conversation loop calls."""
    from hermes_cli.plugins import get_pre_verify_continue_message

    return get_pre_verify_continue_message(
        session_id=session_id, platform="telegram", model="m",
        coding=True, attempt=attempt, final_response=final_response,
        changed_paths=list(changed_paths or []),
    )


def iso(delta_seconds: int) -> str:
    return (
        datetime.now(timezone.utc) + timedelta(seconds=delta_seconds)
    ).strftime("%Y-%m-%dT%H:%M:%SZ")


def live_card(conn, kb, procs, title="PR worker card", assignee="software-engineer"):
    """A live canonical executor: claimed + live owned pid (liveness only)."""
    tid = kb.create_task(conn, title=title, assignee=assignee)
    kb.claim_task(conn, tid, claimer="fixture-claimer")
    kb._set_worker_pid(conn, tid, procs(120).pid)
    return tid


def helper_verdict(home: Path, **kw):
    from contrib_stopcheck import stopcheck  # type: ignore

    kw.setdefault("cfg", {})
    kw["cfg"] = {
        "gate_authorities": [GATE_AUTHORITY],
        "process_registry_path": str(home / "processes.json"),
        **kw["cfg"],
    }
    return stopcheck.evaluate_board(db_path=str(home / "board.db"), **kw)


@pytest.fixture(autouse=True)
def _stopcheck_import_alias():
    """Import the plugin's modules directly for helper-level assertions."""
    import importlib.util
    import types

    pkg = types.ModuleType("contrib_stopcheck")
    pkg.__path__ = [str(PLUGIN_SRC)]
    sys.modules["contrib_stopcheck"] = pkg
    for name in ("owners", "stopcheck", "__init__"):
        if not (PLUGIN_SRC / f"{name}.py").exists():
            continue
        mod_name = "contrib_stopcheck." + {
            "__init__": "plugin", "owners": "owners", "stopcheck": "stopcheck",
        }[name]
        spec = importlib.util.spec_from_file_location(mod_name, PLUGIN_SRC / f"{name}.py")
        mod = importlib.util.module_from_spec(spec)
        mod.__package__ = "contrib_stopcheck"
        sys.modules[mod_name] = mod
        spec.loader.exec_module(mod)
        setattr(pkg, mod_name.split(".")[-1], mod)
    yield
    for k in [k for k in sys.modules if k.startswith("contrib_stopcheck")]:
        del sys.modules[k]


# ------------------------------------------- 1-10: acceptance scenarios ---

def test_1_live_pr_worker_cannot_hide_neglected_blocked_sibling(home, procs):
    """(1) One live worker is NOT whole-board coverage."""
    conn, kb = _board(home)
    good = live_card(conn, kb, procs)
    bad = kb.create_task(conn, title="stale sibling", assignee="software-engineer")
    kb.block_task(conn, bad, reason="EM hold")
    install_runtime(home)

    result = run_turn(QUIET)

    assert result["response_transformed"] is True
    text = result["final_response"]
    assert "STOP-CHECK" in text
    assert bad in text, text
    assert good not in text.split("attended")[0]


def test_2_stale_supervisor_hold_yields_actionable_next_step(home):
    """(2) A stale generic supervisor hold must produce a concrete action."""
    conn, kb = _board(home)
    tid = kb.create_task(conn, title="held card", assignee="software-engineer")
    kb.block_task(conn, tid, reason="supervisor hold")
    kb.add_comment(conn, tid, author="supervisor", body="still on it, will check later")
    install_runtime(home)

    v = helper_verdict(home)
    assert v.ok and not v.quiet_allowed
    f = next(f for f in v.findings if f.task_id == tid)
    assert f.kind in ("stale_hold", "unowned_blocker", "owner_unknown")
    assert any(w in f.next_action for w in ("re-dispatch", "resolve", "qualify"))
    # A repeated comment is not execution proof.
    assert "still on it" not in f.detail

    parked = kb.create_task(conn, title="parked card", assignee="reviewer")
    kb.schedule_task(conn, parked, reason="later")
    v2 = helper_verdict(home)
    p = next(f for f in v2.findings if f.task_id == parked)
    assert p.kind == "stale_hold" and "verifiable wake" in p.next_action

    text = run_turn(QUIET)["final_response"]
    assert f"- {tid}" in text
    assert any(w in text for w in ("resolve", "re-dispatch", "route", "qualify"))


def test_3_qualified_human_gates_allow_quiet(home):
    """(3) Authorised, current human/external gates permit a quiet turn."""
    conn, kb = _board(home)
    a = kb.create_task(conn, title="needs a human decision", assignee="cto")
    kb.block_task(conn, a, reason="user must authorise spend", kind="needs_input")
    b = kb.create_task(conn, title="no credentials", assignee="reviewer")
    kb.block_task(conn, b, reason="no access", kind="capability")
    c = kb.create_task(conn, title="external vendor", assignee="reviewer")
    kb.block_task(conn, c, reason="vendor")
    kb.add_comment(
        conn, c, author=GATE_AUTHORITY,
        body=f"STOP-CHECK-GATE: vendor must reply until={iso(86400)}",
    )
    install_runtime(home)

    v = helper_verdict(home)
    assert v.ok and v.quiet_allowed, [f.line() for f in v.findings]
    result = run_turn(QUIET)
    assert result["final_response"] == QUIET
    assert result["response_transformed"] is False


def test_4_stopped_owner_yields_handoff_to_same_owner(home, procs):
    """(4) Owner run ended while the card is unfinished -> handoff, no dupes."""
    conn, kb = _board(home)
    tid = live_card(conn, kb, procs, title="worker died", assignee="software-engineer")
    kb.reclaim_task(conn, tid)  # closes the run; card returns unfinished
    install_runtime(home)

    v = helper_verdict(home)
    f = next(f for f in v.findings if f.task_id == tid)
    assert f.kind == "owner_stopped"
    assert "software-engineer" in f.next_action
    assert "no duplicate worker" in f.next_action
    assert "STOP-CHECK" in run_turn(QUIET)["final_response"]


def test_5_overdue_checkpoint_forces_action(home, procs):
    """(5) An expired checkpoint is work, not a wait."""
    conn, kb = _board(home)
    tid = kb.create_task(conn, title="checkpointed card", assignee="reviewer")
    kb.block_task(conn, tid, reason="waiting")
    kb.add_comment(
        conn, tid, author="supervisor",
        body=f"STOP-CHECK-CHECKPOINT: {iso(-3600)} wake=dispatcher owner=reviewer",
    )
    install_runtime(home)

    v = helper_verdict(home)
    f = next(f for f in v.findings if f.task_id == tid)
    assert f.kind == "overdue_checkpoint"
    assert "do not simply extend the deadline" in f.next_action

    # A FUTURE checkpoint whose wake target really exists is attended.
    worker = procs(120)
    write_registry(home, [registry_entry(worker, task_id="other_card")])
    kb.add_comment(
        conn, tid, author="supervisor",
        body=f"STOP-CHECK-CHECKPOINT: {iso(3600)} wake=process:proc_{worker.pid}",
    )
    assert helper_verdict(home).quiet_allowed

    # ...but a checkpoint with no wake at all is not.
    kb.add_comment(
        conn, tid, author="supervisor",
        body=f"STOP-CHECK-CHECKPOINT: {iso(3600)} owner=reviewer",
    )
    v3 = helper_verdict(home)
    assert next(f for f in v3.findings if f.task_id == tid).kind == "unverified_wake"


def test_6_continuations_are_bounded_and_write_nothing(home):
    """(6) Concurrent wakes: bounded continuations, no board writes, no dispatch."""
    conn, kb = _board(home)
    tid = kb.create_task(conn, title="unattended", assignee="software-engineer")
    kb.block_task(conn, tid, reason="hold")
    install_runtime(home)
    set_turn_context(SUPERVISION_MSG)

    before = len(kb.list_tasks(conn, include_archived=True))
    results: list = []
    lock = threading.Lock()

    def fire():
        r = fire_pre_verify(changed_paths=["a.py"])
        with lock:
            results.append(r)

    threads = [threading.Thread(target=fire) for _ in range(8)]
    [t.start() for t in threads]
    [t.join(timeout=60) for t in threads]

    granted = [r for r in results if r]
    assert len(granted) == 2, f"expected the configured cap, got {len(granted)}"
    assert len(kb.list_tasks(conn, include_archived=True)) == before  # no new cards
    assert all("STOP-CHECK" in g for g in granted)
    # The text must not claim anything was executed: it asks, it never reports
    # a dispatch, a spawn, or a de-duplication it did not perform.
    for g in granted:
        assert "started nothing and wrote nothing" in g
        for lie in ("dispatched ", "spawned", "i have started", "no duplicates"):
            assert lie not in g.lower(), (lie, g)


def test_7_failed_or_empty_board_read_is_explicit_error(home):
    """(7) A refused/empty read is an error, never 'nothing to do'."""
    _board(home)  # creates an EMPTY board
    install_runtime(home)
    v = helper_verdict(home)
    assert v.ok is False and not v.quiet_allowed
    assert "zero cards" in (v.error or "")
    text = run_turn(QUIET)["final_response"]
    assert "STOP-CHECK ERROR" in text
    assert "not evidence of no work" in text

    from contrib_stopcheck import stopcheck  # type: ignore

    bad = stopcheck.evaluate_board(db_path="/proc/nonexistent/dir/board.db")
    assert bad.ok is False and not bad.quiet_allowed


def test_8_whole_board_coverage_allows_quiet(home, procs):
    """(8) Legitimate coverage across the WHOLE board -> quiet is allowed."""
    conn, kb = _board(home)
    live_card(conn, kb, procs)
    gated = kb.create_task(conn, title="human gate", assignee="cto")
    kb.block_task(conn, gated, reason="decision", kind="needs_input")
    later = kb.create_task(conn, title="scheduled", assignee="reviewer")
    kb.block_task(conn, later, reason="waiting for deploy window")
    worker = procs(120)
    write_registry(home, [registry_entry(worker, task_id=later)])
    done = kb.create_task(conn, title="finished", assignee="reviewer")
    kb.claim_task(conn, done, claimer="fixture")
    kb.complete_task(conn, done, result="done")
    install_runtime(home)

    v = helper_verdict(home)
    assert v.quiet_allowed, [f.line() for f in v.findings]
    result = run_turn(QUIET)
    assert result["final_response"] == QUIET
    assert result["response_transformed"] is False


def test_9_out_of_scope_session_and_user_stop_win(home):
    """(9) Opt-in scope only; an interrupted (user /stop) turn is untouched."""
    conn, kb = _board(home)
    tid = kb.create_task(conn, title="unattended", assignee="software-engineer")
    kb.block_task(conn, tid, reason="hold")
    install_runtime(home)

    other = run_turn(QUIET, session_id=OTHER_SESSION)
    assert other["final_response"] == QUIET
    assert other["response_transformed"] is False

    stopped = run_turn(QUIET, interrupted=True)
    assert "STOP-CHECK" not in (stopped["final_response"] or "")

    install_runtime(home, extra_cfg={"enabled": False})
    assert run_turn(QUIET)["final_response"] == QUIET


def test_10_mutation_removing_the_runtime_integration_fails_this_test(home, procs):
    """(10) Mutation control: drop the hook -> the fallback gate disappears."""
    conn, kb = _board(home)
    live_card(conn, kb, procs)
    bad = kb.create_task(conn, title="stale sibling", assignee="software-engineer")
    kb.block_task(conn, bad, reason="EM hold")

    install_runtime(home)
    assert "STOP-CHECK" in run_turn(QUIET)["final_response"]  # armed

    pdir = home / "plugins" / "agentpod-stop-check"
    src = (pdir / "__init__.py").read_text()
    mutated = src.replace(
        'ctx.register_hook("transform_llm_output", on_transform_llm_output)',
        "pass  # MUTATION: enforcement hook removed",
    )
    assert mutated != src
    (pdir / "__init__.py").write_text(mutated)
    from hermes_cli import plugins as P

    P.discover_plugins(force=True)

    text = run_turn(QUIET)["final_response"]
    assert text == QUIET, "gate must vanish when the runtime hook is removed"


# ------------------------- 11-21: converted adversarial invariants (B1-B7) ---

def test_11_quiet_text_is_replaced_not_appended(home):
    """B1/C3: the false 'no material change' claim must not ship at all."""
    conn, kb = _board(home)
    tid = kb.create_task(conn, title="unattended", assignee="software-engineer")
    kb.block_task(conn, tid, reason="hold")
    install_runtime(home)

    text = run_turn(QUIET)["final_response"]
    assert "no material change" not in text.lower()
    assert not text.startswith(QUIET[:20])
    assert text.startswith("STOP-CHECK")
    # D2: it must fit the platform budget it will never be shortened into.
    assert len(text) <= 700, len(text)


def test_12_live_external_pi_owner_is_not_idle_and_liveness_is_not_progress(home, procs):
    """B3/B1(review): a verified external owner is attendance; a dead one is not.

    Both directions in one test: legitimate external work must not be called
    idle, and the evidence must say *live*, never *progressing*.
    """
    conn, kb = _board(home)
    tid = kb.create_task(conn, title="external review", assignee="reviewer")
    worker = procs(120)
    write_registry(home, [registry_entry(worker, task_id=tid)])
    install_runtime(home)

    v = helper_verdict(home)
    assert v.quiet_allowed, [f.line() for f in v.findings]
    att = next(a for a in v.attended if a.task_id == tid)
    assert att.reason == "live_external_owner"
    assert "liveness, not progress" in att.detail
    assert "progress" not in att.reason

    # Dead owner: same registry row, process gone -> explicit owner_stopped.
    worker.kill()
    worker.wait(timeout=5)
    v2 = helper_verdict(home)
    f = next(f for f in v2.findings if f.task_id == tid)
    assert f.kind == "owner_stopped"

    # An alive pid whose identity does NOT confirm is neither attendance nor
    # death: it is unknown, and must be qualified rather than declared either.
    other = procs(120)
    write_registry(
        home, [registry_entry(other, task_id=tid, host_start_time=1)]
    )
    v3 = helper_verdict(home)
    f3 = next(f for f in v3.findings if f.task_id == tid)
    assert f3.kind == "owner_unknown"
    assert "recycled" in f3.detail and "NOT evidence" in f3.detail

    # Small start-time drift (observed on macOS/psutil: a consistent 1.00s
    # offset on live workers) must NOT read as a recycled pid — exact equality
    # would declare a running owner dead.
    worker2 = procs(120)
    base = registry_entry(worker2, task_id=tid)
    base["host_start_time"] = int(base["host_start_time"]) - 100
    write_registry(home, [base])
    v4 = helper_verdict(home)
    assert v4.quiet_allowed, [f.line() for f in v4.findings]


def test_12b_owner_past_its_own_deadline_is_a_finding(home, procs):
    """Deadline is checked, not assumed: a live-but-overdue owner is work."""
    conn, kb = _board(home)
    tid = kb.create_task(conn, title="long runner", assignee="reviewer")
    worker = procs(120)
    entry = registry_entry(
        worker, task_id=tid, command=f"gtimeout 60 pi --print 'work {tid}'"
    )
    entry["started_at"] = time.time() - 600  # bound expired 9 minutes ago
    write_registry(home, [entry])
    install_runtime(home)

    v = helper_verdict(home)
    f = next(f for f in v.findings if f.task_id == tid)
    assert f.kind == "owner_overdue"
    assert "past its own deadline" in f.detail
    assert "poll" in f.next_action


def test_13_dead_owner_is_not_hidden_by_a_future_marker(home, procs):
    """A4: a dead owner plus a self-written future checkpoint is NOT attended."""
    conn, kb = _board(home)
    tid = kb.create_task(conn, title="worker card", assignee="software-engineer")
    worker = procs(120)
    write_registry(home, [registry_entry(worker, task_id=tid)])
    kb.add_comment(
        conn, tid, author="software-engineer",
        body=f"STOP-CHECK-CHECKPOINT: {iso(86400)} wake=dispatcher",
    )
    worker.kill()
    worker.wait(timeout=5)
    install_runtime(home)

    v = helper_verdict(home)
    f = next(f for f in v.findings if f.task_id == tid)
    assert f.kind == "owner_stopped"
    assert "unbacked" in f.detail


def test_14_worker_written_and_expired_gates_must_requalify(home):
    """A1/A2: a self-written gate does not authorise a human wait, and a
    resolved/expired gate does not survive."""
    conn, kb = _board(home)
    install_runtime(home)

    # A1 — the card's own worker writes the gate.
    self_gate = kb.create_task(conn, title="self gated", assignee="software-engineer")
    kb.block_task(conn, self_gate, reason="hold")
    kb.add_comment(
        conn, self_gate, author="software-engineer",
        body=f"STOP-CHECK-GATE: waiting on the user until={iso(86400)}",
    )
    # A2a — an authority gate that has expired.
    expired = kb.create_task(conn, title="expired gate", assignee="reviewer")
    kb.block_task(conn, expired, reason="hold")
    kb.add_comment(
        conn, expired, author=GATE_AUTHORITY,
        body=f"STOP-CHECK-GATE: approve the spend until={iso(-3600)}",
    )
    # A2b — an authority gate explicitly resolved later.
    resolved = kb.create_task(conn, title="resolved gate", assignee="reviewer")
    kb.block_task(conn, resolved, reason="hold")
    kb.add_comment(
        conn, resolved, author=GATE_AUTHORITY,
        body=f"STOP-CHECK-GATE: approve the spend until={iso(86400)}",
    )
    kb.add_comment(
        conn, resolved, author=GATE_AUTHORITY,
        body="STOP-CHECK-GATE-RESOLVED: user approved the spend, resumed",
    )

    v = helper_verdict(home)
    kinds = {f.task_id: f.kind for f in v.findings}
    assert kinds.get(self_gate) == "unqualified_gate"
    assert kinds.get(expired) == "unqualified_gate"
    assert resolved in kinds  # the historical gate no longer covers the card

    # Requalification must never authorise the restricted action itself.
    for tid in (self_gate, expired):
        action = next(f.next_action for f in v.findings if f.task_id == tid)
        assert "do NOT perform the gated action" in action
        assert "bypass" in action


def test_15_stale_typed_hold_requalifies_but_stays_restricted(home):
    """A5: a typed capability/needs_input hold is not permanently immune."""
    conn, kb = _board(home)
    tid = kb.create_task(conn, title="old capability park", assignee="reviewer")
    kb.block_task(conn, tid, reason="no card on file", kind="capability")
    install_runtime(home)

    fresh = helper_verdict(home)
    assert fresh.quiet_allowed, [f.line() for f in fresh.findings]

    # Same board, 400 days later.
    old = helper_verdict(home, now=int(time.time()) + 400 * 86400)
    f = next(f for f in old.findings if f.task_id == tid)
    assert f.kind == "stale_hold"
    assert "requalify" in f.next_action
    assert "NOT permission to perform the held action" in f.next_action


def test_16_wake_targets_must_exist(home, procs):
    """A3: a wake is verified against the real target, not a word enum."""
    conn, kb = _board(home)
    install_runtime(home)

    def checkpointed(title, wake):
        tid = kb.create_task(conn, title=title, assignee="reviewer")
        kb.block_task(conn, tid, reason="waiting")
        kb.add_comment(
            conn, tid, author="supervisor",
            body=f"STOP-CHECK-CHECKPOINT: {iso(3600)} wake={wake}",
        )
        return tid

    bare = checkpointed("bare cron word", "cron")
    missing = checkpointed("missing job", "cron:no-such-job")
    ghost = checkpointed("ghost process", "process:proc_deadbeef")
    undispatchable = checkpointed("not dispatchable", "dispatcher")

    v = helper_verdict(home)
    kinds = {f.task_id: f.kind for f in v.findings}
    details = {f.task_id: f.detail for f in v.findings}
    assert kinds.get(bare) == "unverified_wake"
    assert "bare word is not a wake" in details[bare]
    assert kinds.get(missing) == "unverified_wake"
    assert "does not exist" in details[missing]
    assert kinds.get(ghost) == "unverified_wake"
    assert kinds.get(undispatchable) == "unverified_wake"
    assert "not dispatchable" in details[undispatchable]

    # A real, enabled, armed cron job IS a verified wake.
    from cron import jobs as cron_jobs

    cron_jobs.ensure_dirs()
    store = cron_jobs._current_cron_store()
    store.jobs_file.write_text(json.dumps({"jobs": [{
        "id": "board-sweep",
        "name": "board sweep",
        "prompt": "sweep",
        "schedule": {"type": "interval", "minutes": 60},
        "enabled": True,
        "next_run_at": (datetime.now(timezone.utc) + timedelta(minutes=30)).isoformat(),
    }]}), encoding="utf-8")
    real = checkpointed("real cron wake", "cron:board-sweep")
    v2 = helper_verdict(home)
    assert real not in {f.task_id for f in v2.findings}, [f.line() for f in v2.findings]


def test_17_unknown_owner_is_explicit_and_actionable_not_quiet(home):
    """Review point: unknown evidence is a bounded qualification, not silence."""
    conn, kb = _board(home)
    tid = kb.create_task(conn, title="assigned but unverifiable", assignee="reviewer")
    # An external owner moved the card to 'review' without a kanban run — the
    # exact shape that used to be mislabelled 'idle'.
    conn.execute("UPDATE tasks SET status = 'review' WHERE id = ?", (tid,))
    conn.commit()
    kb.add_comment(conn, tid, author="reviewer", body="picking this up now")
    kb.add_comment(conn, tid, author="reviewer", body="still working, looks good")
    install_runtime(home)

    v = helper_verdict(home)
    assert next(t for t in kb.list_tasks(conn) if t.id == tid).status == "review"
    f = next(f for f in v.findings if f.task_id == tid)
    assert f.kind == "owner_unknown"
    assert "comments are not evidence" in f.detail
    assert "qualify" in f.next_action and "bounded" in f.next_action
    assert not v.quiet_allowed  # unknown is never quiet


def test_18_scope_is_the_user_message_and_the_project(home):
    """C1/C2/E2: unrelated questions, stop/topic-change, and other projects."""
    conn, kb = _board(home)
    mine = kb.create_task(conn, title="my project card", assignee="software-engineer")
    kb.block_task(conn, mine, reason="hold")
    install_runtime(home)

    # C1 — an unrelated question whose ANSWER happens to look quiet.
    unrelated = run_turn(
        "Your disk has 212 GB free — no further action needed.",
        user_message=UNRELATED_MSG,
    )
    assert unrelated["response_transformed"] is False
    assert "STOP-CHECK" not in unrelated["final_response"]

    # C2 — a same-session user stop / topic change wins immediately.
    stopped = run_turn(QUIET, user_message="stop the board sweep, forget it for now")
    assert stopped["response_transformed"] is False
    assert stopped["final_response"] == QUIET

    # No recorded user message at all -> inert (never inferred from the answer).
    from contrib_stopcheck import plugin  # type: ignore

    plugin.reset_state()
    blind = run_turn(QUIET, set_context=False)
    assert blind["response_transformed"] is False

    # A supervision message still enforces.
    assert "STOP-CHECK" in run_turn(QUIET)["final_response"]

    # E2 — another project's cards are never read.
    other = kb.create_task(conn, title="other project", assignee="someone")
    kb.block_task(conn, other, reason="hold")
    conn.execute("UPDATE tasks SET project_id = ? WHERE id = ?", ("other-proj", other))
    conn.commit()
    v = helper_verdict(home, cfg={"project_id": None})
    assert {f.task_id for f in v.findings} >= {mine, other}
    scoped = helper_verdict(home, cfg={"project_id": "agentpod"})
    conn.execute("UPDATE tasks SET project_id = ? WHERE id = ?", ("agentpod", mine))
    conn.commit()
    scoped = helper_verdict(home, cfg={"project_id": "agentpod"})
    assert {f.task_id for f in scoped.findings} == {mine}


def test_19_quiet_paraphrases_and_hook_order_cannot_bypass_enforcement(home):
    """B7/D1/D2: no regex to evade, and no transform ordering to hide behind."""
    conn, kb = _board(home)
    tid = kb.create_task(conn, title="unattended", assignee="software-engineer")
    kb.block_task(conn, tid, reason="hold")
    install_runtime(home)

    # B7 — paraphrases the old regex missed are all enforced now, because the
    # trigger is the user's message, not the answer's phrasing.
    for paraphrase in (
        "All wrapped up on my side.",
        "Nothing pressing right now.",
        "Board looks quiet; I'll pick things up when something lands.",
        "We're in good shape — I'll wait for the workers.",
        "No blockers worth escalating this sweep.",
    ):
        out = run_turn(paraphrase)
        assert out["response_transformed"] is True, paraphrase
        assert tid in out["final_response"]

    # D1 — a transform plugin registered BEFORE us preempts the text...
    pdir = home / "plugins" / "aaa-dummy"
    pdir.mkdir(parents=True)
    (pdir / "plugin.yaml").write_text(
        "name: aaa-dummy\nversion: 0.0.1\ndescription: ordering probe\n"
    )
    (pdir / "__init__.py").write_text(
        "def t(response_text='', **kw):\n"
        "    return 'REWRITTEN BY AAA-DUMMY'\n"
        "def register(ctx):\n"
        "    ctx.register_hook('transform_llm_output', t)\n"
    )
    cfg = yaml.safe_load((home / "config.yaml").read_text())
    cfg["plugins"]["enabled"] = ["aaa-dummy", "agentpod-stop-check"]
    (home / "config.yaml").write_text(yaml.safe_dump(cfg))
    from hermes_cli import plugins as P

    P.discover_plugins(force=True)
    preempted = run_turn(QUIET)["final_response"]
    assert preempted == "REWRITTEN BY AAA-DUMMY"  # documented transform semantics

    # ...but enforcement does NOT live there: pre_verify still continues the
    # turn under the same adverse ordering, so the turn cannot end quietly.
    from contrib_stopcheck import plugin  # type: ignore

    plugin.reset_state()
    (home / "stopcheck-ledger.json").unlink(missing_ok=True)
    set_turn_context(SUPERVISION_MSG)
    msg = fire_pre_verify(changed_paths=[])
    assert msg and "STOP-CHECK" in msg and tid in msg


def test_20_no_edit_turn_really_continues_into_a_tool_call(home, monkeypatch):
    """F2 + core extension: the REAL conversation loop continues a no-edit turn.

    Not "the aggregator returned a string" — the actual ``AIAgent`` loop takes
    the continuation, runs another model turn that calls a tool, executes the
    tool through the real dispatch, and only then completes.
    """
    from unittest.mock import patch
    from types import SimpleNamespace

    conn, kb = _board(home)
    tid = kb.create_task(conn, title="unattended", assignee="software-engineer")
    kb.block_task(conn, tid, reason="hold")
    install_runtime(home, extra_cfg={"max_continuations": 1})

    cfg = yaml.safe_load((home / "config.yaml").read_text())
    cfg["agent"] = {"pre_verify_on_no_edit_turns": True, "max_verify_nudges": 3}
    (home / "config.yaml").write_text(yaml.safe_dump(cfg))

    from run_agent import AIAgent

    with (
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        agent = AIAgent(
            session_id=SESSION, api_key="k", base_url="https://example.invalid/v1",
            provider="openai-compat", model="test/model", max_iterations=6,
            quiet_mode=True, skip_context_files=True, skip_memory=True,
        )
    agent._cached_system_prompt = "stable test prompt"
    agent._session_db = None
    agent._session_json_enabled = False
    agent.save_trajectories = False
    agent.compression_enabled = False
    agent._cleanup_task_resources = lambda *_a, **_kw: None
    agent._save_trajectory = lambda *_a, **_kw: None
    agent.valid_tool_names = {"kanban_show"}

    calls: list[str] = []
    tool_calls_made: list[dict] = []

    def _msg(content=None, tool_calls=None):
        return SimpleNamespace(content=content, tool_calls=tool_calls, reasoning=None)

    def model_call(_api_kwargs):
        calls.append("api")
        if len(calls) == 1:
            # Turn 1: no file edits at all, quiet conclusion.
            return SimpleNamespace(
                choices=[SimpleNamespace(message=_msg(QUIET), finish_reason="stop")],
                model="test/model", usage=None,
            )
        if len(calls) == 2:
            # Turn 2 (post-continuation): the agent ACTS — a real tool call.
            tc = SimpleNamespace(
                id="call_1", type="function",
                function=SimpleNamespace(
                    name="kanban_show", arguments=json.dumps({"task_id": tid})
                ),
            )
            return SimpleNamespace(
                choices=[SimpleNamespace(message=_msg(None, [tc]), finish_reason="tool_calls")],
                model="test/model", usage=None,
            )
        return SimpleNamespace(
            choices=[SimpleNamespace(
                message=_msg(f"Worked {tid}: re-dispatched to its owner."),
                finish_reason="stop")],
            model="test/model", usage=None,
        )

    def fake_tool(name, args, *a, **kw):
        tool_calls_made.append({"name": name, "args": args})
        return json.dumps({"ok": True, "task": tid})

    agent._interruptible_api_call = model_call
    set_turn_context(SUPERVISION_MSG)

    with patch("run_agent.handle_function_call", side_effect=fake_tool):
        result = agent.run_conversation(SUPERVISION_MSG)

    assert len(calls) >= 3, calls
    assert tool_calls_made and tool_calls_made[0]["name"] == "kanban_show"
    assert tid in result["final_response"]
    assert "no material change" not in result["final_response"].lower()
    # Alternation is preserved and no user-visible synthetic turn is left behind.
    roles = [m["role"] for m in result["messages"]]
    for a, b in zip(roles, roles[1:]):
        assert not (a == b == "user"), roles


def test_21_continuation_budget_is_shared_across_processes(home):
    """F1: the cap is a real ledger, not a module global in one process."""
    conn, kb = _board(home)
    tid = kb.create_task(conn, title="unattended", assignee="software-engineer")
    kb.block_task(conn, tid, reason="hold")
    install_runtime(home, extra_cfg={"max_continuations": 1})
    set_turn_context(SUPERVISION_MSG)

    assert fire_pre_verify(changed_paths=[])          # 1st: granted
    assert fire_pre_verify(changed_paths=[]) is None  # cap reached in-process

    # A genuinely separate OS process, same HERMES_HOME, same session/state.
    probe = f"""
import os, sys, json
sys.path.insert(0, {str(REPO)!r})
os.environ["HERMES_HOME"] = {str(home)!r}
from hermes_cli import plugins as P
P.discover_plugins(force=True)
from hermes_cli.lifecycle import invoke_hook
invoke_hook("pre_llm_call", session_id={SESSION!r}, user_message={SUPERVISION_MSG!r})
from hermes_cli.plugins import get_pre_verify_continue_message
out = get_pre_verify_continue_message(
    session_id={SESSION!r}, platform="telegram", model="m", coding=True,
    attempt=0, final_response={QUIET!r}, changed_paths=[])
print("GRANTED" if out else "DENIED")
"""
    r = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, timeout=180,
        env={**os.environ, "HERMES_HOME": str(home)},
    )
    assert r.returncode == 0, r.stderr[-2000:]
    assert "DENIED" in r.stdout, r.stdout + r.stderr[-2000:]


def test_22_fallback_text_always_fits_the_budget_and_keeps_the_disclosure(home):
    """D2: the report is sized for the platform budget it will never be
    shortened into, and what it drops it says it dropped."""
    conn, kb = _board(home)
    ids = []
    for i in range(12):
        tid = kb.create_task(conn, title=f"card {i}", assignee="software-engineer")
        kb.block_task(conn, tid, reason="hold")
        ids.append(tid)
    install_runtime(home, extra_cfg={"max_report_chars": 400})

    text = run_turn(QUIET)["final_response"]
    assert len(text) <= 400, len(text)
    assert "more unattended not shown" in text
    assert "nothing above was executed or dispatched" in text
    # The true total is still reported honestly in the head.
    assert "12 unattended of 12 unfinished" in text


def test_23_duplicate_supervision_turns_stay_bounded_and_identical(home):
    """Repeated quiet sweeps must not grow, spawn, or write anything."""
    conn, kb = _board(home)
    tid = kb.create_task(conn, title="unattended", assignee="software-engineer")
    kb.block_task(conn, tid, reason="hold")
    install_runtime(home, extra_cfg={"max_continuations": 1})
    before = len(kb.list_tasks(conn, include_archived=True))
    comments_before = len(kb.list_comments(conn, tid))

    texts = [run_turn(QUIET)["final_response"] for _ in range(3)]
    assert len(set(texts)) == 1, texts
    assert len(kb.list_tasks(conn, include_archived=True)) == before
    assert len(kb.list_comments(conn, tid)) == comments_before

    # The continuation budget is spent once for this board state, not per turn.
    set_turn_context(SUPERVISION_MSG)
    assert fire_pre_verify(changed_paths=[])
    assert fire_pre_verify(changed_paths=[]) is None
