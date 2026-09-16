"""Acceptance tests for agentpod-stop-check.

These drive the REAL Hermes runtime integration, not just the helper:

* the plugin is installed into an isolated ``HERMES_HOME`` and loaded through
  the real ``hermes_cli.plugins.discover_plugins()`` discovery path;
* the enforcement path is exercised through the real
  ``agent.turn_finalizer.finalize_turn`` (the exact function that fires
  ``transform_llm_output`` once per turn);
* the continuation path is exercised through the real
  ``hermes_cli.plugins.get_pre_verify_continue_message()`` aggregator that
  ``agent/conversation_loop.py`` calls;
* boards are isolated temp SQLite boards created through the installed
  ``hermes_cli.kanban_db`` interface — no real tenant/card is touched;
* "live executor" evidence uses tiny fixture processes this test owns
  (``python -c 'time.sleep(...)'``), started and reaped here.

Run:
    ~/.hermes/hermes-agent/venv/bin/python -m pytest \
        contrib/den-plugins/agentpod-stop-check/test_stop_check.py -q
"""
from __future__ import annotations

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
PLUGIN_SRC = Path(__file__).resolve().parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

SESSION = "sess-agentpod-supervisor"
OTHER_SESSION = "sess-somebody-else"
QUIET = "Checked the board — no material change since the last sweep."


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


def install_runtime(home: Path, *, extra_cfg: dict | None = None, register_patch=None):
    """Install + load the plugin through the real discovery path."""
    from hermes_cli import plugins as P

    pdir = home / "plugins" / "agentpod-stop-check"
    if pdir.exists():
        shutil.rmtree(pdir)
    pdir.mkdir(parents=True)
    for name in ("__init__.py", "plugin.yaml", "stopcheck.py"):
        shutil.copy(PLUGIN_SRC / name, pdir / name)

    cfg = {
        "enabled": True,
        "db_path": str(home / "board.db"),
        "session_ids": [SESSION],
        "heartbeat_stale_seconds": 900,
        "max_continuations": 2,
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


def run_turn(final_response: str, *, session_id: str = SESSION, interrupted: bool = False):
    """Drive the REAL turn finalizer (the transform_llm_output fire site)."""
    from unittest.mock import MagicMock

    from agent.turn_finalizer import finalize_turn

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
        {"role": "user", "content": "status?"},
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
        user_message="status?",
        original_user_message="status?",
        _should_review_memory=False,
        _turn_exit_reason="stop",
    )


def iso(delta_seconds: int) -> str:
    return (
        datetime.now(timezone.utc) + timedelta(seconds=delta_seconds)
    ).strftime("%Y-%m-%dT%H:%M:%SZ")


def live_card(conn, kb, procs, title="PR worker card", assignee="software-engineer"):
    """A genuinely progressing canonical executor: claimed + live owned pid."""
    tid = kb.create_task(conn, title=title, assignee=assignee)
    kb.claim_task(conn, tid, claimer="fixture-claimer")
    kb._set_worker_pid(conn, tid, procs(120).pid)
    return tid


def helper_verdict(home: Path, **kw):
    from contrib_stopcheck import stopcheck  # type: ignore

    return stopcheck.evaluate_board(db_path=str(home / "board.db"), **kw)


@pytest.fixture(autouse=True)
def _stopcheck_import_alias():
    """Import the plugin's modules directly for helper-level assertions."""
    import importlib.util
    import types

    pkg = types.ModuleType("contrib_stopcheck")
    pkg.__path__ = [str(PLUGIN_SRC)]
    sys.modules["contrib_stopcheck"] = pkg
    for name in ("stopcheck", "__init__"):
        mod_name = "contrib_stopcheck." + ("plugin" if name == "__init__" else name)
        spec = importlib.util.spec_from_file_location(mod_name, PLUGIN_SRC / f"{name}.py")
        mod = importlib.util.module_from_spec(spec)
        mod.__package__ = "contrib_stopcheck"
        sys.modules[mod_name] = mod
        spec.loader.exec_module(mod)
        setattr(pkg, "stopcheck" if name == "stopcheck" else "plugin", mod)
    yield
    for k in [k for k in sys.modules if k.startswith("contrib_stopcheck")]:
        del sys.modules[k]


# ----------------------------------------------------------------- tests ---

def test_1_progressing_pr_worker_cannot_hide_neglected_blocked_sibling(home, procs):
    """(1) One progressing worker is NOT whole-board coverage."""
    conn, kb = _board(home)
    good = live_card(conn, kb, procs)
    bad = kb.create_task(conn, title="stale sibling", assignee="software-engineer")
    kb.block_task(conn, bad, reason="EM hold")
    install_runtime(home)

    result = run_turn(QUIET)

    assert result["response_transformed"] is True
    text = result["final_response"]
    assert "SUPERVISOR STOP-CHECK" in text
    assert bad in text, text
    assert good not in text.split("attended (no action):")[0]
    assert "progressing_executor" in text


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
    assert f.kind in ("stale_hold", "unowned_blocker")
    assert "re-dispatch" in f.next_action or "resolve" in f.next_action
    # A repeated comment is not execution proof.
    assert "still on it" not in f.detail

    # A 'scheduled' park is not dispatchable, so it is a hold too.
    parked = kb.create_task(conn, title="parked card", assignee="reviewer")
    kb.schedule_task(conn, parked, reason="later")
    v2 = helper_verdict(home)
    p = next(f for f in v2.findings if f.task_id == parked)
    assert p.kind == "stale_hold" and "supported wake" in p.next_action

    text = run_turn(QUIET)["final_response"]
    assert "-> next:" in text and tid in text


def test_3_genuine_human_gates_allow_quiet(home):
    """(3) Real human/external gates stay gated and permit a quiet turn."""
    conn, kb = _board(home)
    a = kb.create_task(conn, title="needs a human decision", assignee="cto")
    kb.block_task(conn, a, reason="user must authorise spend", kind="needs_input")
    b = kb.create_task(conn, title="no credentials", assignee="reviewer")
    kb.block_task(conn, b, reason="no access", kind="capability")
    c = kb.create_task(conn, title="external vendor", assignee="reviewer")
    kb.block_task(conn, c, reason="vendor")
    kb.add_comment(conn, c, author="supervisor", body="STOP-CHECK-GATE: vendor must reply")
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
    assert "SUPERVISOR STOP-CHECK" in run_turn(QUIET)["final_response"]


def test_5_overdue_checkpoint_forces_action(home):
    """(5) An expired checkpoint is work, not a wait."""
    conn, kb = _board(home)
    tid = kb.create_task(conn, title="checkpointed card", assignee="reviewer")
    kb.block_task(conn, tid, reason="waiting")
    kb.add_comment(
        conn, tid, author="supervisor",
        body=f"STOP-CHECK-CHECKPOINT: {iso(-3600)} wake=kanban-wake owner=reviewer",
    )
    install_runtime(home)

    v = helper_verdict(home)
    f = next(f for f in v.findings if f.task_id == tid)
    assert f.kind == "overdue_checkpoint"
    assert "do not simply extend the deadline" in f.next_action

    # A FUTURE checkpoint with a real wake is attended...
    kb.add_comment(
        conn, tid, author="supervisor",
        body=f"STOP-CHECK-CHECKPOINT: {iso(3600)} wake=kanban-wake owner=reviewer",
    )
    assert helper_verdict(home).quiet_allowed
    # ...but only if the wake really exists.
    kb.add_comment(
        conn, tid, author="supervisor",
        body=f"STOP-CHECK-CHECKPOINT: {iso(3600)} owner=reviewer",
    )
    v3 = helper_verdict(home)
    assert next(f for f in v3.findings if f.task_id == tid).kind == "checkpoint_without_wake"


def test_6_concurrent_wakes_do_not_duplicate_dispatch(home):
    """(6) Concurrent event/heartbeat turns: bounded continuations, no writes."""
    conn, kb = _board(home)
    tid = kb.create_task(conn, title="unattended", assignee="software-engineer")
    kb.block_task(conn, tid, reason="hold")
    install_runtime(home)
    from contrib_stopcheck import plugin  # type: ignore

    plugin.reset_state()
    before = len(kb.list_tasks(conn, include_archived=True))
    results: list = []
    lock = threading.Lock()

    def fire():
        from hermes_cli.plugins import get_pre_verify_continue_message

        r = get_pre_verify_continue_message(
            session_id=SESSION, platform="telegram", model="m",
            coding=True, attempt=0, final_response=QUIET, changed_paths=["a.py"],
        )
        with lock:
            results.append(r)

    threads = [threading.Thread(target=fire) for _ in range(8)]
    [t.start() for t in threads]
    [t.join(timeout=60) for t in threads]

    granted = [r for r in results if r]
    assert len(granted) == 2, f"expected the configured cap, got {len(granted)}"
    assert len(kb.list_tasks(conn, include_archived=True)) == before  # no new cards
    assert all("SUPERVISOR STOP-CHECK" in g for g in granted)


def test_7_failed_or_empty_board_read_is_explicit_error(home):
    """(7) A refused/empty read is an error, never 'nothing to do'."""
    _board(home)  # creates an EMPTY board
    install_runtime(home)
    v = helper_verdict(home)
    assert v.ok is False and not v.quiet_allowed
    assert "zero cards" in (v.error or "")
    text = run_turn(QUIET)["final_response"]
    assert "SUPERVISOR STOP-CHECK ERROR" in text
    assert "not evidence of no work" in text

    # Hard read failure (unreadable path) -> still explicit, never quiet.
    from contrib_stopcheck import stopcheck  # type: ignore

    bad = stopcheck.evaluate_board(db_path="/proc/nonexistent/dir/board.db")
    assert bad.ok is False and not bad.quiet_allowed


def test_8_whole_board_coverage_allows_quiet(home, procs):
    """(8) Legitimate progress across the WHOLE board -> quiet is allowed."""
    conn, kb = _board(home)
    live_card(conn, kb, procs)
    gated = kb.create_task(conn, title="human gate", assignee="cto")
    kb.block_task(conn, gated, reason="decision", kind="needs_input")
    later = kb.create_task(conn, title="scheduled", assignee="reviewer")
    kb.block_task(conn, later, reason="waiting for deploy window")
    kb.add_comment(
        conn, later, author="supervisor",
        body=f"STOP-CHECK-CHECKPOINT: {iso(7200)} wake=cron owner=reviewer",
    )
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

    # Another session (e.g. after /new, or any other profile/session).
    other = run_turn(QUIET, session_id=OTHER_SESSION)
    assert other["final_response"] == QUIET
    assert other["response_transformed"] is False

    # A non-quiet answer in the scoped session is never rewritten either.
    working = run_turn("Dispatched t_x to the reviewer; PR #1 is open.")
    assert "SUPERVISOR STOP-CHECK" not in working["final_response"]

    # User interrupt: the runtime never fires the transform.
    stopped = run_turn(QUIET, interrupted=True)
    assert "SUPERVISOR STOP-CHECK" not in (stopped["final_response"] or "")

    # Disabled config is fully inert.
    install_runtime(home, extra_cfg={"enabled": False})
    assert run_turn(QUIET)["final_response"] == QUIET


def test_10_mutation_removing_the_runtime_integration_fails_this_test(home, procs):
    """(10) Documentation alone cannot pass: drop the hook -> gate disappears.

    Same board as test 1, but the plugin is installed with its
    ``transform_llm_output`` registration removed. The runtime then delivers
    the quiet conclusion unchanged — proving test 1 passes because of the real
    hook wiring, not because of prose.
    """
    conn, kb = _board(home)
    live_card(conn, kb, procs)
    bad = kb.create_task(conn, title="stale sibling", assignee="software-engineer")
    kb.block_task(conn, bad, reason="EM hold")

    install_runtime(home)
    assert "SUPERVISOR STOP-CHECK" in run_turn(QUIET)["final_response"]  # armed

    # Mutate: strip the enforcement hook from the installed plugin source.
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
