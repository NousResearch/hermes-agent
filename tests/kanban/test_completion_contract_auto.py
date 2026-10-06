"""Regression tests for --completion-contract auto (fixes #126626).

SLM/small-model workers run as pure text-in-text-out functions: no lifecycle
kanban tools, no KANBAN_GUIDANCE, minimal prompt (task body), and the
dispatcher auto-completes the card to done from captured output on clean rc=0
instead of booking a protocol violation.
"""
from __future__ import annotations

import subprocess
import time
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd
from hermes_cli.kanban_pr_acceptance import (
    AUTO_COMPLETION_CONTRACT,
    validate_contract,
)
from hermes_cli.quiet_single_query import KANBAN_WORKER_EXIT_TRAILER


@pytest.fixture
def kanban_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_KANBAN_CRASH_GRACE_SECONDS", "0")
    monkeypatch.setattr(kb, "_pid_alive", lambda _pid: False)
    kbd._recent_worker_exits.clear()
    kb.init_db()
    return home


def _dead_worker_with_log(conn, tid: str, pid: int, rc: int, output: str = "the model said something") -> None:
    host = kb._claimer_id().split(":", 1)[0]
    kb.claim_task(conn, tid, claimer=f"{host}:w{pid}")
    conn.execute(
        "UPDATE tasks SET worker_pid=?, worker_started_at=NULL, started_at=? WHERE id=?",
        (pid, int(time.time()) - 120, tid),
    )
    conn.commit()
    log = kb.worker_log_path(tid)
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "a", encoding="utf-8") as f:
        f.write(f"{output}\n\n{KANBAN_WORKER_EXIT_TRAILER}{rc}\n")


# --- CLI / DB validation ---


def test_validate_contract_accepts_auto():
    assert validate_contract("auto") == "auto"
    assert validate_contract(None) == "local-only"
    assert validate_contract("local-only") == "local-only"
    assert validate_contract("acme/repo") == "acme/repo"
    assert validate_contract("https://github.com/acme/repo/pull/7") == "https://github.com/acme/repo/pull/7"


def test_validate_contract_rejects_invalid():
    with pytest.raises(ValueError):
        validate_contract("not a contract!!")
    with pytest.raises(ValueError):
        validate_contract("")


def test_create_task_stores_auto_contract(kanban_home):
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="slm job", body="do it", completion_contract="auto")
        task = kb.get_task(conn, tid)
        assert task.completion_contract == "auto"


def test_auto_contract_skips_pr_acceptance(kanban_home):
    """Auto cards complete like local-only: no gh evidence required."""
    from hermes_cli.kanban_pr_acceptance_store import prepare_acceptance

    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="auto", completion_contract="auto")
        assert prepare_acceptance(conn, tid, None, {}) is None
        assert kb.complete_task(conn, tid, result="done-output")
        assert kb.get_task(conn, tid).status == "done"


# --- Worker toolset / guidance exclusion ---


def test_auto_worker_tool_selection_omits_kanban_tools(monkeypatch):
    from model_tools import _select_tool_names

    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_auto_1")
    monkeypatch.setenv("HERMES_KANBAN_COMPLETION_CONTRACT", "auto")
    names = _select_tool_names(["terminal", "file", "kanban"], None, quiet_mode=True)
    assert "terminal" in names
    assert not any(n.startswith("kanban_") for n in names)


def test_non_auto_worker_keeps_kanban_toolset_force_add(monkeypatch):
    """Without the auto pin, dispatcher workers still get the kanban toolset."""
    from model_tools import _select_tool_names

    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_normal_1")
    monkeypatch.delenv("HERMES_KANBAN_COMPLETION_CONTRACT", raising=False)
    names = _select_tool_names(["terminal"], None, quiet_mode=True)
    assert any(n.startswith("kanban_") for n in names)


def test_auto_worker_check_fn_hides_lifecycle_tools(monkeypatch):
    from tools.kanban_tools import _check_kanban_mode

    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_auto_2")
    monkeypatch.setenv("HERMES_KANBAN_COMPLETION_CONTRACT", "auto")
    assert _check_kanban_mode() is False


def test_auto_worker_gets_no_kanban_guidance(monkeypatch):
    """agent_init records empty guidance and the system-prompt fallback stays empty."""
    from agent.prompt_builder import KANBAN_GUIDANCE
    from agent.system_prompt import _tool_guidance_block

    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_auto_3")
    monkeypatch.setenv("HERMES_KANBAN_COMPLETION_CONTRACT", "auto")

    class FakeAgent:
        valid_tool_names = {"terminal", "kanban_show"}
        _kanban_worker_guidance = ""  # as agent_init sets for auto workers
        _memory_enabled = False
        _user_profile_enabled = False
        _memory_store = None
        _memory_manager = None

    block = _tool_guidance_block(FakeAgent())
    assert block is None or KANBAN_GUIDANCE not in (block or "")

    # Fallback path (guidance resolved lazily) must also stay empty for auto.
    class FallbackAgent:
        valid_tool_names = {"terminal", "kanban_show"}
        _memory_enabled = False
        _user_profile_enabled = False
        _memory_store = None
        _memory_manager = None

    block2 = _tool_guidance_block(FallbackAgent())
    assert block2 is None or KANBAN_GUIDANCE not in (block2 or "")


def test_auto_worker_disables_stop_nudge_and_recovery(monkeypatch):
    from agent.kanban_stop import kanban_stop_nudge_enabled
    from agent.kanban_turn_recovery import kanban_turn_recovery_enabled

    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_auto_4")
    monkeypatch.setenv("HERMES_KANBAN_COMPLETION_CONTRACT", "auto")
    assert kanban_stop_nudge_enabled() is False
    assert kanban_turn_recovery_enabled() is False


# --- Worker prompt / spawn environment ---


def _make_task(assignee="worker", **kwargs):
    base = dict(
        id="t_spawn_auto",
        title="Summarize this",
        body="Summarize the file.",
        assignee=assignee,
        status="running",
        priority=0,
        created_by="test",
        created_at=1,
        started_at=None,
        completed_at=None,
        workspace_kind="dir",
        workspace_path=None,
        claim_lock="lock",
        claim_expires=None,
        tenant=None,
        current_run_id=7,
    )
    base.update(kwargs)
    return kb.Task(**base)


def test_worker_prompt_uses_body_for_auto():
    task = _make_task(completion_contract="auto")
    prompt = kbd._worker_prompt(task)
    assert "Summarize the file." in prompt
    assert "Summarize this" in prompt
    assert not prompt.startswith("work kanban task")


def test_worker_prompt_legacy_for_non_auto():
    task = _make_task(completion_contract="local-only")
    assert kbd._worker_prompt(task) == "work kanban task t_spawn_auto"


def test_worker_argv_uses_minimal_prompt_for_auto(monkeypatch, tmp_path):
    root = tmp_path / ".hermes"
    (root / "profiles" / "worker").mkdir(parents=True)
    root.joinpath("config.yaml").write_text("{}\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(root))
    monkeypatch.setattr(kbd, "_resolve_hermes_argv", lambda: ["hermes"])
    monkeypatch.setattr(kbd, "_resolve_worker_cli_toolsets", lambda _h: None)

    task = _make_task(completion_contract="auto")
    argv = kbd._worker_argv(task, "worker", None)
    assert argv[-3:-1] == ["chat", "-q"]
    # argv ends with ["chat", "-q", prompt]: prompt is task body, not the indirection.
    assert argv[-1] != f"work kanban task {task.id}"
    assert "Summarize the file." in argv[-1]


def test_default_spawn_pins_auto_contract_env(monkeypatch, tmp_path):
    root = tmp_path / ".hermes"
    (root / "profiles" / "worker").mkdir(parents=True)
    root.joinpath("config.yaml").write_text("{}\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(root))
    monkeypatch.setattr(kbd, "_resolve_hermes_argv", lambda: ["hermes"])
    # _default_spawn registers the proc in kbd._live_worker_procs; isolate it
    # so the FakeProc (no .poll()) never leaks into the zombie reaper of a
    # later test on the Windows branch (#130916 review).
    monkeypatch.setattr(kbd, "_live_worker_procs", {})
    captured = {}

    class FakeProc:
        pid = 4242

    def fake_popen(cmd, *args, **kwargs):
        captured["cmd"] = list(cmd)
        captured["env"] = dict(kwargs.get("env") or {})
        return FakeProc()

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    pid = kbd._default_spawn(_make_task(completion_contract="auto"), str(workspace))
    assert pid == 4242
    assert captured["env"]["HERMES_KANBAN_COMPLETION_CONTRACT"] == "auto"
    assert captured["env"]["HERMES_KANBAN_TASK"] == "t_spawn_auto"
    assert captured["cmd"][-1] != "work kanban task t_spawn_auto"


# --- Dispatcher auto-completion ---


def test_dispatcher_auto_completes_auto_task_on_clean_exit(kanban_home):
    with kbc.connect() as conn:
        tid = kb.create_task(
            conn, title="auto job", body="produce text", assignee="a",
            completion_contract="auto",
        )
        _dead_worker_with_log(conn, tid, 80001, 0, output="slm final answer")

        crashed = kbd.detect_crashed_workers(conn)

        task = kb.get_task(conn, tid)
        assert task.status == "done"
        assert task.result == "slm final answer"
        assert tid not in crashed
        assert tid in getattr(kbd.detect_crashed_workers, "_last_auto_completed", [])
        # No protocol-violation bookkeeping for an auto-completed card.
        kinds = [e.kind for e in kb.list_events(conn, tid)]
        assert "protocol_violation" not in kinds
        assert "completed" in kinds
        runs = kb.list_runs(conn, tid)
        assert runs and runs[-1].outcome == "completed"


def test_dispatcher_auto_complete_uses_worker_output_as_summary(kanban_home):
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="t", body="b", assignee="a", completion_contract="auto")
        _dead_worker_with_log(conn, tid, 80002, 0, output="answer text here")
        kbd.detect_crashed_workers(conn)
        runs = kb.list_runs(conn, tid)
        assert runs[-1].summary == "answer text here"


def test_empty_output_falls_back_to_protocol_violation(kanban_home):
    """An auto card with no captured output keeps the legacy violation path."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="t", body="b", assignee="a", completion_contract="auto")
        # Log carries only the exit trailer, no model text.
        host = kb._claimer_id().split(":", 1)[0]
        kb.claim_task(conn, tid, claimer=f"{host}:w81001")
        conn.execute(
            "UPDATE tasks SET worker_pid=?, worker_started_at=NULL, started_at=? WHERE id=?",
            (81001, int(time.time()) - 120, tid),
        )
        conn.commit()
        log = kb.worker_log_path(tid)
        log.parent.mkdir(parents=True, exist_ok=True)
        with open(log, "a", encoding="utf-8") as f:
            f.write(f"\n{KANBAN_WORKER_EXIT_TRAILER}0\n")
        # Force empty trimmed output even if chrome-stripping leaves whitespace.
        monkeypatch_output = None
        import hermes_cli.kanban_db_dispatch as _kbd

        orig = _kbd._worker_final_output
        _kbd._worker_final_output = lambda *a, **k: ""
        try:
            crashed = _kbd.detect_crashed_workers(conn)
        finally:
            _kbd._worker_final_output = orig
        task = kb.get_task(conn, tid)
        assert task.status == "ready"
        assert tid in crashed
        assert "protocol_violation" in [e.kind for e in kb.list_events(conn, tid)]


def test_non_auto_task_still_protocol_violation(kanban_home):
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="t", body="b", assignee="a", completion_contract="local-only")
        _dead_worker_with_log(conn, tid, 80003, 0, output="some output")
        crashed = kbd.detect_crashed_workers(conn)
        assert tid in crashed
        assert kb.get_task(conn, tid).status == "ready"
        assert "protocol_violation" in [e.kind for e in kb.list_events(conn, tid)]


def test_nonzero_exit_does_not_auto_complete(kanban_home):
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="t", body="b", assignee="a", completion_contract="auto")
        _dead_worker_with_log(conn, tid, 80004, 1, output="partial output")
        crashed = kbd.detect_crashed_workers(conn)
        task = kb.get_task(conn, tid)
        assert task.status != "done"
        assert tid in crashed
        assert tid not in getattr(kbd.detect_crashed_workers, "_last_auto_completed", [])


def test_auto_complete_visible_in_dispatch_result(kanban_home):
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="t", body="b", assignee="a", completion_contract="auto")
        _dead_worker_with_log(conn, tid, 80005, 0, output="dispatch output")
        result = kbd.dispatch_once(conn)
        assert tid in result.auto_completed
        assert kb.get_task(conn, tid).status == "done"
