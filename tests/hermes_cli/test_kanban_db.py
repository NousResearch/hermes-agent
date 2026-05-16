"""Tests for the Kanban DB layer (hermes_cli.kanban_db)."""

from __future__ import annotations

import concurrent.futures
import json
import os
import time
from pathlib import Path

import pytest
import yaml

from hermes_cli import kanban_db as kb


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME with an empty kanban DB."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


# ---------------------------------------------------------------------------
# Schema / init
# ---------------------------------------------------------------------------

def test_init_db_is_idempotent(kanban_home):
    # Second call should not error or drop data.
    with kb.connect() as conn:
        kb.create_task(conn, title="persisted")
    kb.init_db()
    with kb.connect() as conn:
        tasks = kb.list_tasks(conn)
    assert len(tasks) == 1
    assert tasks[0].title == "persisted"


def test_init_creates_expected_tables(kanban_home):
    with kb.connect() as conn:
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()
    names = {r["name"] for r in rows}
    assert {"tasks", "task_links", "task_comments", "task_events"} <= names


# ---------------------------------------------------------------------------
# Task creation + status inference
# ---------------------------------------------------------------------------

def test_create_task_no_parents_is_ready(kanban_home):
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="ship it", assignee="alice")
        t = kb.get_task(conn, tid)
    assert t is not None
    assert t.status == "ready"
    assert t.assignee == "alice"
    assert t.workspace_kind == "scratch"


def test_create_task_with_parent_is_todo_until_parent_done(kanban_home):
    with kb.connect() as conn:
        p = kb.create_task(conn, title="parent")
        c = kb.create_task(conn, title="child", parents=[p])
        assert kb.get_task(conn, c).status == "todo"
        kb.complete_task(conn, p, result="ok")
        assert kb.get_task(conn, c).status == "ready"


def test_create_task_unknown_parent_errors(kanban_home):
    with kb.connect() as conn, pytest.raises(ValueError, match="unknown parent"):
        kb.create_task(conn, title="orphan", parents=["t_ghost"])


def test_workspace_kind_validation(kanban_home):
    with kb.connect() as conn, pytest.raises(ValueError, match="workspace_kind"):
        kb.create_task(conn, title="bad ws", workspace_kind="cloud")


# ---------------------------------------------------------------------------
# Transactional profile model config primitive
# ---------------------------------------------------------------------------


def _write_profile_config(home: Path, profile: str, text: str) -> Path:
    if profile == "default":
        cfg = home / "config.yaml"
    else:
        cfg = home / "profiles" / profile / "config.yaml"
        cfg.parent.mkdir(parents=True, exist_ok=True)
    cfg.write_text(text, encoding="utf-8")
    return cfg


def test_kanban_update_profile_model_success_receipt_and_allowed_keys(kanban_home):
    cfg = _write_profile_config(
        kanban_home,
        "coder",
        "model:\n  default: old-model\n  provider: old-provider\ntoolsets:\n  - kanban\n",
    )

    receipt = kb.kanban_update_profile_model("coder", "openai-codex", "gpt-5.5")
    parsed = yaml.safe_load(cfg.read_text(encoding="utf-8"))

    assert parsed["model"] == {"default": "gpt-5.5", "provider": "openai-codex"}
    assert parsed["toolsets"] == ["kanban"]
    assert receipt["profile"] == "coder"
    assert receipt["changed_file"] == str(cfg)
    assert Path(receipt["backup_path"]).read_text(encoding="utf-8").startswith("model:\n")
    assert receipt["pre_values"] == {
        "model.default": "old-model",
        "model.provider": "old-provider",
    }
    assert receipt["post_values"] == {
        "model.default": "gpt-5.5",
        "model.provider": "openai-codex",
    }
    assert receipt["changed_keys"] == ["model.default", "model.provider"]
    assert receipt["parse_status"] == {"pre": "ok", "post": "ok"}
    assert receipt["rollback_status"] == "not_needed"
    assert "no_gateway_restart" in receipt["non_actions"]


def test_kanban_update_profile_model_rejects_unknown_profile(kanban_home):
    with pytest.raises(ValueError, match="does not exist"):
        kb.kanban_update_profile_model("ghost", "openrouter", "model-x")


def test_kanban_update_profile_model_rejects_symlinked_config_path(kanban_home, tmp_path):
    profile_dir = kanban_home / "profiles" / "coder"
    profile_dir.mkdir(parents=True)
    outside = tmp_path / "outside-config.yaml"
    outside.write_text("model:\n  default: old\n  provider: old\n", encoding="utf-8")
    (profile_dir / "config.yaml").symlink_to(outside)

    with pytest.raises(ValueError, match="symlinked profile config"):
        kb.kanban_update_profile_model("coder", "openrouter", "new-model")
    assert outside.read_text(encoding="utf-8") == "model:\n  default: old\n  provider: old\n"


def test_kanban_update_profile_model_invalid_yaml_preserves_file_and_writes_backup(kanban_home):
    cfg = _write_profile_config(kanban_home, "coder", "model: [unterminated\n")

    with pytest.raises(ValueError, match="failed to parse YAML"):
        kb.kanban_update_profile_model("coder", "openrouter", "new-model")

    assert cfg.read_text(encoding="utf-8") == "model: [unterminated\n"
    backups = sorted(cfg.parent.glob("config.yaml.bak.*"))
    assert len(backups) == 1
    assert backups[0].read_text(encoding="utf-8") == "model: [unterminated\n"


def test_transactional_profile_config_update_rejects_unknown_key(kanban_home):
    _write_profile_config(
        kanban_home,
        "coder",
        "model:\n  default: old\n  provider: old\n",
    )

    with pytest.raises(ValueError, match="unsupported key"):
        kb._transactional_update_profile_config(
            "coder",
            {"model.temperature": 0.2},
            allowed_keys=kb.PROFILE_MODEL_CONFIG_KEYS,
        )


def test_kanban_update_profile_model_rolls_back_on_semantic_failure(kanban_home):
    cfg = _write_profile_config(
        kanban_home,
        "coder",
        "model:\n  default: old\n  provider: old\nextra: keep\n",
    )

    def fail_postcheck(_post_cfg):
        raise ValueError("forced semantic failure")

    with pytest.raises(RuntimeError, match="rolled_back"):
        kb.kanban_update_profile_model(
            "coder", "openrouter", "new-model", _postcheck=fail_postcheck
        )

    assert cfg.read_text(encoding="utf-8") == "model:\n  default: old\n  provider: old\nextra: keep\n"


def _hub_plan_spec(workflow_id="wf-kernel-gate"):
    return {
        "workflow_id": workflow_id,
        "source_role": "hub",
        "goal": "enforce coordinator handoff gate in kernel",
        "risk_class": "low",
        "scope_contract": {
            "version": 2,
            "allowed_systems": ["hermes-agent", "hermes-kanban"],
            "forbidden_systems": ["OpenClaw", "Atlas", "Mission-Control", "Telegram"],
        },
    }


def _approved_reviewer_metadata(workflow_id="wf-kernel-gate"):
    return {
        "workflow_id": workflow_id,
        "verdict": "APPROVED",
        "evidence_audited": ["hub_plan_spec", "tests"],
        "residual_risk": "none for hermes-only kernel gate test",
        "scope_attestation": True,
        "scope_contract_version": 2,
        "forbidden_actions_taken": 0,
    }


def _control_plane_gate(**overrides):
    hub_plan = _hub_plan_spec()
    gate = {
        "hub_plan_spec": hub_plan,
        "reviewer_metadata": _approved_reviewer_metadata(),
        "coordinator_plan_spec": {**hub_plan, "workflow_id": "wf-kernel-gate-coordinator"},
        "mechanical_fields": ["workflow_id"],
    }
    gate.update(overrides)
    return gate


def test_create_coordinator_task_blocks_without_control_plane_gate(kanban_home):
    with kb.connect() as conn, pytest.raises(ValueError, match="control_plane_gate"):
        kb.create_task(conn, title="coordinator bypass", assignee="coordinator")

    with kb.connect() as conn:
        assert kb.list_tasks(conn, assignee="coordinator") == []


def test_create_coordinator_task_blocks_without_approved_reviewer_metadata(kanban_home):
    rejected_gate = _control_plane_gate(
        reviewer_metadata={**_approved_reviewer_metadata(), "verdict": "NEEDS_REVISION"}
    )

    with kb.connect() as conn, pytest.raises(ValueError, match="reviewer_verdict_not_approved"):
        kb.create_task(
            conn,
            title="coordinator bypass",
            assignee="coordinator",
            control_plane_gate=rejected_gate,
        )

    with kb.connect() as conn:
        assert kb.list_tasks(conn, assignee="coordinator") == []


@pytest.mark.parametrize("field,value", [
    ("goal", "expanded goal"),
    ("risk_class", "medium"),
    ("scope_contract", {"version": 2, "forbidden_systems": ["OpenClaw"]}),
])
def test_create_coordinator_task_blocks_substantive_plan_change(kanban_home, field, value):
    changed = {**_hub_plan_spec(), field: value}

    with kb.connect() as conn, pytest.raises(ValueError, match=field):
        kb.create_task(
            conn,
            title="coordinator bypass",
            assignee="coordinator",
            control_plane_gate=_control_plane_gate(coordinator_plan_spec=changed),
        )

    with kb.connect() as conn:
        assert kb.list_tasks(conn, assignee="coordinator") == []


def test_create_coordinator_task_accepts_mechanical_normalization_and_records_diffs(kanban_home):
    with kb.connect() as conn:
        tid = kb.create_task(
            conn,
            title="coordinator gated",
            assignee="coordinator",
            control_plane_gate=_control_plane_gate(),
        )
        task = kb.get_task(conn, tid)
        comments = kb.list_comments(conn, tid)

    assert task is not None
    assert task.assignee == "coordinator"
    assert len(comments) == 1
    assert comments[0].author == "control-plane-gate"
    payload = json.loads(comments[0].body)
    assert payload["control_plane_gate"]["mechanical_diffs"] == {
        "workflow_id": {"from": "wf-kernel-gate", "to": "wf-kernel-gate-coordinator"}
    }


# ---------------------------------------------------------------------------
# Links + dependency resolution
# ---------------------------------------------------------------------------

def test_link_demotes_ready_child_to_todo_when_parent_not_done(kanban_home):
    with kb.connect() as conn:
        a = kb.create_task(conn, title="a")
        b = kb.create_task(conn, title="b")
        assert kb.get_task(conn, b).status == "ready"
        kb.link_tasks(conn, a, b)
        assert kb.get_task(conn, b).status == "todo"


def test_link_keeps_ready_child_when_parent_already_done(kanban_home):
    with kb.connect() as conn:
        a = kb.create_task(conn, title="a")
        kb.complete_task(conn, a)
        b = kb.create_task(conn, title="b")
        assert kb.get_task(conn, b).status == "ready"
        kb.link_tasks(conn, a, b)
        assert kb.get_task(conn, b).status == "ready"


def test_link_rejects_self_loop(kanban_home):
    with kb.connect() as conn:
        a = kb.create_task(conn, title="a")
        with pytest.raises(ValueError, match="itself"):
            kb.link_tasks(conn, a, a)


def test_link_detects_cycle(kanban_home):
    with kb.connect() as conn:
        a = kb.create_task(conn, title="a")
        b = kb.create_task(conn, title="b", parents=[a])
        c = kb.create_task(conn, title="c", parents=[b])
        with pytest.raises(ValueError, match="cycle"):
            kb.link_tasks(conn, c, a)
        with pytest.raises(ValueError, match="cycle"):
            kb.link_tasks(conn, b, a)


def test_recompute_ready_cascades_through_chain(kanban_home):
    with kb.connect() as conn:
        a = kb.create_task(conn, title="a")
        b = kb.create_task(conn, title="b", parents=[a])
        c = kb.create_task(conn, title="c", parents=[b])
        assert [kb.get_task(conn, x).status for x in (a, b, c)] == \
               ["ready", "todo", "todo"]
        kb.complete_task(conn, a)
        assert kb.get_task(conn, b).status == "ready"
        kb.complete_task(conn, b)
        assert kb.get_task(conn, c).status == "ready"


def test_recompute_ready_fan_in_waits_for_all_parents(kanban_home):
    with kb.connect() as conn:
        a = kb.create_task(conn, title="a")
        b = kb.create_task(conn, title="b")
        c = kb.create_task(conn, title="c", parents=[a, b])
        kb.complete_task(conn, a)
        assert kb.get_task(conn, c).status == "todo"
        kb.complete_task(conn, b)
        assert kb.get_task(conn, c).status == "ready"


# ---------------------------------------------------------------------------
# Atomic claim (CAS)
# ---------------------------------------------------------------------------

def test_claim_once_wins_second_loses(kanban_home):
    with kb.connect() as conn:
        t = kb.create_task(conn, title="x", assignee="a")
        first = kb.claim_task(conn, t, claimer="host:1")
        assert first is not None and first.status == "running"
        second = kb.claim_task(conn, t, claimer="host:2")
        assert second is None


def test_claim_fails_on_non_ready(kanban_home):
    with kb.connect() as conn:
        t = kb.create_task(conn, title="x")
        # Move to todo by introducing an unsatisfied parent.
        p = kb.create_task(conn, title="p")
        kb.link_tasks(conn, p, t)
        assert kb.get_task(conn, t).status == "todo"
        assert kb.claim_task(conn, t) is None


def test_stale_claim_reclaimed(kanban_home, monkeypatch):
    import signal
    import hermes_cli.kanban_db as _kb

    with kb.connect() as conn:
        t = kb.create_task(conn, title="x", assignee="a")
        host = _kb._claimer_id().split(":", 1)[0]
        kb.claim_task(conn, t, claimer=f"{host}:worker")
        killed: list[int] = []

        def _signal(_pid, sig):
            killed.append(sig)

        kb._set_worker_pid(conn, t, 12345)
        # Rewind claim_expires so it looks stale.
        conn.execute(
            "UPDATE tasks SET claim_expires = ? WHERE id = ?",
            (int(time.time()) - 3600, t),
        )
        # Worker PID has died — exactly the case ``release_stale_claims``
        # should still reclaim (post-#23025: live PIDs are now extended).
        monkeypatch.setattr(_kb, "_pid_alive", lambda _pid: False)
        reclaimed = kb.release_stale_claims(conn, signal_fn=_signal)
        assert reclaimed == 1
        assert kb.get_task(conn, t).status == "ready"
        assert killed == [signal.SIGTERM]


def test_stale_claim_with_live_pid_extends_instead_of_reclaiming(
    kanban_home, monkeypatch,
):
    """A stale-by-TTL claim whose worker PID is still alive should be
    extended, not reclaimed (#23025). Slow models can spend longer than
    ``DEFAULT_CLAIM_TTL_SECONDS`` inside a single tool-free LLM call;
    killing those healthy workers produces a respawn loop with zero
    progress."""
    import hermes_cli.kanban_db as _kb

    with kb.connect() as conn:
        t = kb.create_task(conn, title="x", assignee="a")
        host = _kb._claimer_id().split(":", 1)[0]
        kb.claim_task(conn, t, claimer=f"{host}:worker")
        kb._set_worker_pid(conn, t, 12345)

        old_expires = int(time.time()) - 60
        conn.execute(
            "UPDATE tasks SET claim_expires = ? WHERE id = ?",
            (old_expires, t),
        )

        monkeypatch.setattr(_kb, "_pid_alive", lambda _pid: True)
        killed: list[int] = []
        reclaimed = kb.release_stale_claims(
            conn, signal_fn=lambda _p, sig: killed.append(sig),
        )
        assert reclaimed == 0
        task = kb.get_task(conn, t)
        assert task.status == "running"
        assert task.claim_expires is not None
        assert task.claim_expires > old_expires
        assert killed == []  # live worker not killed

        kinds = [
            r["kind"] for r in conn.execute(
                "SELECT kind FROM task_events WHERE task_id = ?", (t,),
            ).fetchall()
        ]
        assert "claim_extended" in kinds
        assert "reclaimed" not in kinds


def test_stale_claim_reclaim_event_records_diagnostic_payload(
    kanban_home, monkeypatch,
):
    """``reclaimed`` events should carry claim_expires, last_heartbeat_at,
    and worker_pid so operators can diagnose why a claim went stale
    (#23025: previous payload only had ``stale_lock`` which gives no
    timing context)."""
    import json
    import hermes_cli.kanban_db as _kb

    with kb.connect() as conn:
        t = kb.create_task(conn, title="x", assignee="a")
        host = _kb._claimer_id().split(":", 1)[0]
        kb.claim_task(conn, t, claimer=f"{host}:worker")
        kb._set_worker_pid(conn, t, 12345)
        old_expires = int(time.time()) - 3600
        hb_at = int(time.time()) - 1800
        conn.execute(
            "UPDATE tasks SET claim_expires = ?, last_heartbeat_at = ? "
            "WHERE id = ?",
            (old_expires, hb_at, t),
        )

        monkeypatch.setattr(_kb, "_pid_alive", lambda _pid: False)
        kb.release_stale_claims(conn, signal_fn=lambda _p, _s: None)
        row = conn.execute(
            "SELECT payload FROM task_events "
            "WHERE task_id = ? AND kind = 'reclaimed'",
            (t,),
        ).fetchone()
        assert row is not None
        payload = json.loads(row["payload"])
        assert payload["claim_expires"] == old_expires
        assert payload["last_heartbeat_at"] == hb_at
        assert payload["worker_pid"] == 12345
        assert payload["host_local"] is True


def test_max_runtime_uses_current_run_start_after_retry(kanban_home):
    """A retry should get a fresh max-runtime window.

    ``tasks.started_at`` intentionally records the first time the task ever
    started. Runtime enforcement must therefore use the active
    ``task_runs.started_at`` row; otherwise every retry of an old task is
    immediately timed out again.
    """
    with kb.connect() as conn:
        host = kb._claimer_id().split(":", 1)[0]
        t = kb.create_task(
            conn, title="retry", assignee="a", max_runtime_seconds=10,
        )

        kb.claim_task(conn, t, claimer=f"{host}:first")
        first_run_id = kb.latest_run(conn, t).id
        old_started = int(time.time()) - 20
        conn.execute(
            "UPDATE tasks SET started_at = ?, worker_pid = ? WHERE id = ?",
            (old_started, 999999, t),
        )
        conn.execute(
            "UPDATE task_runs SET started_at = ?, worker_pid = ? WHERE id = ?",
            (old_started, 999999, first_run_id),
        )

        timed_out = kb.enforce_max_runtime(conn, signal_fn=lambda _pid, _sig: None)
        assert timed_out == [t]
        assert kb.get_task(conn, t).status == "ready"

        kb.claim_task(conn, t, claimer=f"{host}:retry")
        retry_run = kb.latest_run(conn, t)
        conn.execute(
            "UPDATE tasks SET worker_pid = ? WHERE id = ?",
            (999999, t),
        )
        conn.execute(
            "UPDATE task_runs SET worker_pid = ? WHERE id = ?",
            (999999, retry_run.id),
        )

        timed_out = kb.enforce_max_runtime(conn, signal_fn=lambda _pid, _sig: None)
        assert timed_out == []
        assert kb.get_task(conn, t).status == "running"


def test_heartbeat_extends_claim(kanban_home):
    with kb.connect() as conn:
        t = kb.create_task(conn, title="x", assignee="a")
        claimer = "host:hb"
        kb.claim_task(conn, t, claimer=claimer, ttl_seconds=60)
        original = kb.get_task(conn, t).claim_expires
        # Rewind then heartbeat.
        conn.execute("UPDATE tasks SET claim_expires = ? WHERE id = ?", (0, t))
        ok = kb.heartbeat_claim(conn, t, claimer=claimer, ttl_seconds=3600)
        assert ok
        new = kb.get_task(conn, t).claim_expires
        assert new > int(time.time()) + 3000


def test_concurrent_claims_only_one_wins(kanban_home):
    """Fire N threads claiming the same task; exactly one must win."""
    with kb.connect() as conn:
        t = kb.create_task(conn, title="race", assignee="a")

    def attempt(i):
        with kb.connect() as c:
            return kb.claim_task(c, t, claimer=f"host:{i}")

    n_workers = 8
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as ex:
        results = list(ex.map(attempt, range(n_workers)))
    winners = [r for r in results if r is not None]
    assert len(winners) == 1
    assert winners[0].status == "running"


# ---------------------------------------------------------------------------
# Complete / block / unblock / archive / assign
# ---------------------------------------------------------------------------

def test_complete_records_result(kanban_home):
    with kb.connect() as conn:
        t = kb.create_task(conn, title="x")
        assert kb.complete_task(conn, t, result="done and dusted")
        task = kb.get_task(conn, t)
    assert task.status == "done"
    assert task.result == "done and dusted"
    assert task.completed_at is not None


def test_block_then_unblock(kanban_home):
    with kb.connect() as conn:
        t = kb.create_task(conn, title="x", assignee="a")
        kb.claim_task(conn, t)
        assert kb.block_task(conn, t, reason="need input")
        assert kb.get_task(conn, t).status == "blocked"
        assert kb.unblock_task(conn, t)
        assert kb.get_task(conn, t).status == "ready"


# ---------------------------------------------------------------------------
# Parent-completion invariant at the claim gate (RCA t_a6acd07d)
# ---------------------------------------------------------------------------

def test_claim_rejects_when_parents_not_done(kanban_home):
    """claim_task must refuse ready->running if any parent isn't 'done'.

    Simulates the create-then-link race: a task gets status='ready' via a
    racy writer while it still has undone parents. The claim gate must
    detect the violation, demote the child back to 'todo', append a
    'claim_rejected' event, and return None. Covers Fix 1 of the RCA.
    """
    with kb.connect() as conn:
        parent = kb.create_task(conn, title="parent", assignee="a")
        child = kb.create_task(
            conn, title="child", assignee="a", parents=[parent],
        )
        # Child correctly starts 'todo' because parent is not 'done'.
        assert kb.get_task(conn, child).status == "todo"
        # Simulate the race: a racy writer force-promotes the child to
        # 'ready' while parent is still pending.
        conn.execute(
            "UPDATE tasks SET status='ready' WHERE id=?", (child,),
        )
        conn.commit()
        assert kb.get_task(conn, child).status == "ready"

        result = kb.claim_task(conn, child, claimer="host:1")

    assert result is None
    with kb.connect() as conn:
        assert kb.get_task(conn, child).status == "todo"
        events = conn.execute(
            "SELECT kind, payload FROM task_events "
            "WHERE task_id = ? ORDER BY id",
            (child,),
        ).fetchall()
    kinds = [e["kind"] for e in events]
    assert "claim_rejected" in kinds
    # No 'claimed' event was emitted for the blocked attempt.
    assert "claimed" not in kinds


def test_claim_succeeds_once_parents_done(kanban_home):
    """After parents complete, recompute_ready -> claim_task must succeed."""
    with kb.connect() as conn:
        parent = kb.create_task(conn, title="parent", assignee="a")
        child = kb.create_task(
            conn, title="child", assignee="a", parents=[parent],
        )
        kb.claim_task(conn, parent)
        assert kb.complete_task(conn, parent, result="ok")
        kb.recompute_ready(conn)
        assert kb.get_task(conn, child).status == "ready"
        claimed = kb.claim_task(conn, child, claimer="host:1")
    assert claimed is not None
    assert claimed.status == "running"


def test_create_with_parents_stays_todo_until_parents_done(kanban_home):
    """kanban_create(parents=[...]) must land in 'todo' and only promote on parent done."""
    with kb.connect() as conn:
        parent = kb.create_task(conn, title="parent", assignee="a")
        child = kb.create_task(
            conn, title="child", assignee="a", parents=[parent],
        )
        assert kb.get_task(conn, child).status == "todo"
        # Dispatcher tick between create and some later event must NOT
        # produce a winner for this child.
        promoted = kb.recompute_ready(conn)
        assert promoted == 0
        assert kb.get_task(conn, child).status == "todo"
        # Complete parent; complete_task internally runs recompute_ready,
        # which promotes the child to 'ready'.
        kb.claim_task(conn, parent)
        kb.complete_task(conn, parent, result="ok")
        assert kb.get_task(conn, child).status == "ready"


def test_unblock_with_pending_parents_goes_to_todo(kanban_home):
    """unblock_task must re-gate on parent completion (Fix 3).

    A task blocked while parents are still in progress must return to
    'todo' (not 'ready') on unblock. Otherwise the dispatcher will claim
    it immediately, repeating Bug 2 from the RCA.
    """
    with kb.connect() as conn:
        parent = kb.create_task(conn, title="parent", assignee="a")
        child = kb.create_task(
            conn, title="child", assignee="a", parents=[parent],
        )
        # Force child into 'blocked' regardless of parent progress
        # (simulates a worker that self-blocked, or an operator block).
        conn.execute(
            "UPDATE tasks SET status='blocked' WHERE id=?", (child,),
        )
        conn.commit()
        assert kb.unblock_task(conn, child)
        assert kb.get_task(conn, child).status == "todo"
        # After parent completes + recompute, the child is ready.
        kb.claim_task(conn, parent)
        kb.complete_task(conn, parent, result="ok")
        kb.recompute_ready(conn)
        assert kb.get_task(conn, child).status == "ready"


def test_unblock_without_parents_goes_to_ready(kanban_home):
    """Parent-free unblock still produces 'ready' (behavior preserved)."""
    with kb.connect() as conn:
        t = kb.create_task(conn, title="lone", assignee="a")
        kb.claim_task(conn, t)
        assert kb.block_task(conn, t, reason="need input")
        assert kb.unblock_task(conn, t)
        assert kb.get_task(conn, t).status == "ready"


def test_assign_refuses_while_running(kanban_home):
    with kb.connect() as conn:
        t = kb.create_task(conn, title="x", assignee="a")
        kb.claim_task(conn, t)
        with pytest.raises(RuntimeError, match="currently running"):
            kb.assign_task(conn, t, "b")


def test_assign_reassigns_when_not_running(kanban_home):
    with kb.connect() as conn:
        t = kb.create_task(conn, title="x", assignee="a")
        assert kb.assign_task(conn, t, "b")
        assert kb.get_task(conn, t).assignee == "b"


def test_assignee_normalized_to_lowercase_on_create_and_assign(kanban_home):
    """Dashboard/CLI may pass title-cased profile labels; DB + spawn use canonical id."""
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="cased", assignee="Jules")
        assert kb.get_task(conn, tid).assignee == "jules"
        assert kb.assign_task(conn, tid, "Librarian")
        assert kb.get_task(conn, tid).assignee == "librarian"


def test_list_tasks_assignee_filter_case_insensitive(kanban_home):
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="q", assignee="jules")
        found = kb.list_tasks(conn, assignee="Jules")
        assert len(found) == 1 and found[0].id == tid


def test_archive_hides_from_default_list(kanban_home):
    with kb.connect() as conn:
        t = kb.create_task(conn, title="x")
        kb.complete_task(conn, t)
        assert kb.archive_task(conn, t)
        assert len(kb.list_tasks(conn)) == 0
        assert len(kb.list_tasks(conn, include_archived=True)) == 1


# ---------------------------------------------------------------------------
# Comments / events / worker context
# ---------------------------------------------------------------------------

def test_comments_recorded_in_order(kanban_home):
    with kb.connect() as conn:
        t = kb.create_task(conn, title="x")
        kb.add_comment(conn, t, "user", "first")
        kb.add_comment(conn, t, "researcher", "second")
        comments = kb.list_comments(conn, t)
    assert [c.body for c in comments] == ["first", "second"]
    assert [c.author for c in comments] == ["user", "researcher"]


def test_empty_comment_rejected(kanban_home):
    with kb.connect() as conn:
        t = kb.create_task(conn, title="x")
        with pytest.raises(ValueError, match="body is required"):
            kb.add_comment(conn, t, "user", "")


def test_events_capture_lifecycle(kanban_home):
    with kb.connect() as conn:
        t = kb.create_task(conn, title="x", assignee="a")
        kb.claim_task(conn, t)
        kb.complete_task(conn, t, result="ok")
        events = kb.list_events(conn, t)
    kinds = [e.kind for e in events]
    assert "created" in kinds
    assert "claimed" in kinds
    assert "completed" in kinds


def test_block_event_can_link_context_comment(kanban_home):
    with kb.connect() as conn:
        t = kb.create_task(conn, title="blocked context", assignee="a")
        kb.claim_task(conn, t)
        comment_id = kb.add_comment(
            conn,
            t,
            "worker",
            "Detailed blocker context: checked inputs A and B; need Piet to choose C.",
        )

        assert kb.block_task(
            conn,
            t,
            reason="Need Piet to choose C; see context comment.",
            context_comment_id=comment_id,
        )

        blocked = [e for e in kb.list_events(conn, t) if e.kind == "blocked"][-1]

    assert blocked.payload["reason"] == "Need Piet to choose C; see context comment."
    assert blocked.payload["context_comment_id"] == comment_id
    assert blocked.payload["context_snippet"].startswith("Detailed blocker context")


def test_terminal_events_include_run_outcome_profile_and_handoff(kanban_home):
    with kb.connect() as conn:
        done_task = kb.create_task(conn, title="done", assignee="alice")
        kb.claim_task(conn, done_task)
        assert kb.complete_task(conn, done_task, summary="finished cleanly")
        completed = [e for e in kb.list_events(conn, done_task) if e.kind == "completed"][-1]

        blocked_task = kb.create_task(conn, title="blocked", assignee="bob")
        kb.claim_task(conn, blocked_task)
        assert kb.block_task(conn, blocked_task, reason="need decision")
        blocked = [e for e in kb.list_events(conn, blocked_task) if e.kind == "blocked"][-1]

    assert completed.run_id is not None
    assert completed.payload["run_id"] == completed.run_id
    assert completed.payload["outcome"] == "completed"
    assert completed.payload["profile"] == "alice"
    assert completed.payload["summary"] == "finished cleanly"
    assert completed.payload["ended_at"] is not None

    assert blocked.run_id is not None
    assert blocked.payload["run_id"] == blocked.run_id
    assert blocked.payload["outcome"] == "blocked"
    assert blocked.payload["profile"] == "bob"
    assert blocked.payload["summary"] == "need decision"
    assert blocked.payload["ended_at"] is not None


def test_spawn_failure_terminal_event_includes_error_and_outcome(kanban_home):
    with kb.connect() as conn:
        t = kb.create_task(conn, title="spawn failure", assignee="coder")
        kb.claim_task(conn, t)

        kb._record_spawn_failure(conn, t, "No Codex credentials stored", failure_limit=3)

        event = [e for e in kb.list_events(conn, t) if e.kind == "spawn_failed"][-1]

    assert event.run_id is not None
    assert event.payload["run_id"] == event.run_id
    assert event.payload["outcome"] == "spawn_failed"
    assert event.payload["profile"] == "coder"
    assert event.payload["error"] == "No Codex credentials stored"
    assert event.payload["ended_at"] is not None


def test_worker_context_includes_parent_results_and_comments(kanban_home):
    with kb.connect() as conn:
        p = kb.create_task(conn, title="p")
        kb.complete_task(conn, p, result="PARENT_RESULT_MARKER")
        c = kb.create_task(conn, title="child", parents=[p])
        kb.add_comment(conn, c, "user", "CLARIFICATION_MARKER")
        ctx = kb.build_worker_context(conn, c)
    assert "PARENT_RESULT_MARKER" in ctx
    assert "CLARIFICATION_MARKER" in ctx
    assert c in ctx
    assert "child" in ctx


def test_complete_task_requires_scope_attestation_metadata_when_policy_enabled(kanban_home):
    body = """
completion_policy:
  require_scope_attestation: true
"""
    with kb.connect() as conn:
        t = kb.create_task(conn, title="scoped", assignee="alice", body=body)
        with pytest.raises(kb.ScopeAttestationError, match="scope attestation"):
            kb.complete_task(conn, t, summary="done", metadata={})
        task = kb.get_task(conn, t)
        assert task.status == "ready"
        events = [e for e in kb.list_events(conn, t) if e.kind == "completion_blocked_scope_attestation"]
        assert events


def test_complete_task_accepts_valid_scope_attestation_metadata(kanban_home):
    body = """
completion_policy:
  require_scope_attestation: true
"""
    with kb.connect() as conn:
        t = kb.create_task(conn, title="scoped", assignee="alice", body=body)
        assert kb.complete_task(
            conn,
            t,
            summary="done",
            metadata={
                "scope_contract_version": 2,
                "scope_attestation": True,
                "forbidden_actions_taken": 0,
            },
        )
        assert kb.get_task(conn, t).status == "done"


def test_scope_attestation_policy_uses_outer_contract_after_embedded_child_template(
    kanban_home, all_assignees_spawnable
):
    body = """
workflow_id: wf-test

Reviewer task body template:
---REVIEWER_BODY_START---
scope_contract:
  version: 2
  assignee: reviewer
  allowed_tools: [kanban_show, kanban_complete, kanban_block, kanban_comment]
  forbidden_systems: [OpenClaw, Atlas, Mission-Control, Telegram]
completion_policy:
  require_scope_attestation: true
---REVIEWER_BODY_END---

scope_contract:
  version: 2
  assignee: coordinator
  allowed_tools: [kanban_show, kanban_create, kanban_complete, kanban_block, kanban_comment]
  forbidden_systems: [OpenClaw, Atlas, Mission-Control, Telegram]
completion_policy:
  require_scope_attestation: true
"""
    expected_tools = [
        "kanban_show",
        "kanban_create",
        "kanban_complete",
        "kanban_block",
        "kanban_comment",
    ]

    with kb.connect() as conn:
        t = kb.create_task(
            conn,
            title="coordinator with embedded reviewer",
            assignee="coordinator",
            body=body,
            internal_test_bypass_control_plane_gate=True,
        )
        res = kb.dispatch_once(conn, spawn_fn=lambda task, workspace: 43210)
        assert [item[0] for item in res.spawned] == [t]

        passed = [e for e in kb.list_events(conn, t) if e.kind == "dispatch_preflight_passed"]
        assert passed[-1].payload["effective_toolsets"] == expected_tools

        assert kb.complete_task(
            conn,
            t,
            summary="reviewer verdict collected",
            metadata={
                "scope_contract_version": 2,
                "scope_attestation": True,
                "forbidden_actions_taken": 0,
                "effective_toolsets": expected_tools,
                "reviewer_verdict": {"verdict": "APPROVED"},
            },
        )
        assert kb.get_task(conn, t).status == "done"


def test_scope_attestation_policy_accepts_yaml_frontmatter(kanban_home):
    body = """---
completion_policy:
  require_scope_attestation: true
---

Worker task body.
"""
    with kb.connect() as conn:
        t = kb.create_task(conn, title="frontmatter scoped", assignee="alice", body=body)
        with pytest.raises(kb.ScopeAttestationError):
            kb.complete_task(conn, t, summary="done", metadata={})
        assert kb.get_task(conn, t).status == "ready"


def test_scope_attestation_policy_accepts_fenced_yaml(kanban_home):
    body = """Task contract:

```yaml
completion_policy:
  require_scope_attestation: true
scope_contract:
  version: 2
```
"""
    with kb.connect() as conn:
        t = kb.create_task(conn, title="fenced scoped", assignee="alice", body=body)
        with pytest.raises(kb.ScopeAttestationError):
            kb.complete_task(conn, t, summary="done", metadata={})
        assert kb._task_has_scope_contract_v2(body)


def test_scope_attestation_policy_malformed_yaml_does_not_enforce_or_crash(kanban_home):
    body = """
completion_policy:
  require_scope_attestation: [true
"""
    with kb.connect() as conn:
        t = kb.create_task(conn, title="malformed scoped", assignee="alice", body=body)
        assert kb.complete_task(conn, t, summary="done", metadata={})
        assert kb.get_task(conn, t).status == "done"


def test_scope_attestation_policy_ignores_false_positive_inline_string(kanban_home):
    body = "Please write about completion_policy: require_scope_attestation: true in the report."
    with kb.connect() as conn:
        t = kb.create_task(conn, title="false positive", assignee="alice", body=body)
        assert kb.complete_task(conn, t, summary="done", metadata={})
        assert kb.get_task(conn, t).status == "done"


def test_scope_attestation_policy_respects_false_value(kanban_home):
    body = """
completion_policy:
  require_scope_attestation: false
"""
    with kb.connect() as conn:
        t = kb.create_task(conn, title="false policy", assignee="alice", body=body)
        assert kb.complete_task(conn, t, summary="done", metadata={})
        assert kb.get_task(conn, t).status == "done"


def test_block_task_allows_admin_block_for_todo_and_triage(kanban_home):
    with kb.connect() as conn:
        triage = kb.create_task(conn, title="triage fixture", assignee="alice", triage=True)
        todo = kb.create_task(conn, title="todo fixture", assignee="alice", parents=[triage])

        assert kb.block_task(conn, triage, reason="superseded fixture")
        assert kb.block_task(conn, todo, reason="superseded fixture")

        assert kb.get_task(conn, triage).status == "blocked"
        assert kb.get_task(conn, todo).status == "blocked"
        for task_id in (triage, todo):
            events = [e for e in kb.list_events(conn, task_id) if e.kind == "blocked"]
            assert events
            assert kb.latest_run(conn, task_id).outcome == "blocked"


def test_block_task_with_expected_run_id_still_rejects_unclaimed_todo(kanban_home):
    with kb.connect() as conn:
        triage = kb.create_task(conn, title="parent", assignee="alice", triage=True)
        todo = kb.create_task(conn, title="child", assignee="alice", parents=[triage])
        assert not kb.block_task(conn, todo, reason="worker block", expected_run_id=12345)
        assert kb.get_task(conn, todo).status == "todo"


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

def test_dispatch_dry_run_does_not_claim(kanban_home, all_assignees_spawnable):
    with kb.connect() as conn:
        t1 = kb.create_task(conn, title="a", assignee="alice")
        t2 = kb.create_task(conn, title="b", assignee="bob")
        res = kb.dispatch_once(conn, dry_run=True)
    assert {s[0] for s in res.spawned} == {t1, t2}
    with kb.connect() as conn:
        # Dry run must NOT mutate status.
        assert kb.get_task(conn, t1).status == "ready"
        assert kb.get_task(conn, t2).status == "ready"


def test_dispatch_skips_unassigned(kanban_home):
    with kb.connect() as conn:
        t = kb.create_task(conn, title="floater")
        res = kb.dispatch_once(conn, dry_run=True)
    assert t in res.skipped_unassigned
    assert t not in res.skipped_nonspawnable
    assert not res.spawned


def test_dispatch_skips_nonspawnable_into_separate_bucket(kanban_home, monkeypatch):
    """Tasks whose assignee fails profile_exists() must NOT land in
    ``skipped_unassigned`` (which is operator-actionable) — they go in
    the dedicated ``skipped_nonspawnable`` bucket so health telemetry
    can suppress false-positive "stuck" warnings."""
    from hermes_cli import profiles
    monkeypatch.setattr(profiles, "profile_exists", lambda name: False)
    with kb.connect() as conn:
        t = kb.create_task(conn, title="for-terminal", assignee="orion-cc")
        res = kb.dispatch_once(conn, dry_run=True)
    assert t in res.skipped_nonspawnable
    assert t not in res.skipped_unassigned
    assert not res.spawned


def test_has_spawnable_ready_false_when_only_terminal_lanes(kanban_home, monkeypatch):
    """``has_spawnable_ready`` returns False when every ready task is
    assigned to a control-plane lane — used by gateway/CLI dispatchers
    to silence the stuck-warn while terminals still have queued work."""
    from hermes_cli import profiles
    monkeypatch.setattr(profiles, "profile_exists", lambda name: False)
    with kb.connect() as conn:
        kb.create_task(conn, title="t1", assignee="orion-cc")
        kb.create_task(conn, title="t2", assignee="orion-research")
        assert kb.has_spawnable_ready(conn) is False


def test_has_spawnable_ready_true_when_real_profile_present(kanban_home, monkeypatch):
    """``has_spawnable_ready`` returns True as soon as ANY ready task
    has an assignee that maps to a real Hermes profile — preserves the
    real "stuck" signal when a daily/agent task is queued."""
    from hermes_cli import profiles
    monkeypatch.setattr(
        profiles, "profile_exists", lambda name: name == "daily"
    )
    with kb.connect() as conn:
        kb.create_task(conn, title="terminal-task", assignee="orion-cc")
        kb.create_task(conn, title="hermes-task", assignee="daily")
        assert kb.has_spawnable_ready(conn) is True


def test_has_spawnable_ready_false_on_empty_queue(kanban_home):
    """Empty queue is the trivial false case — no ready tasks at all."""
    with kb.connect() as conn:
        assert kb.has_spawnable_ready(conn) is False


def test_dispatch_promotes_ready_and_spawns(kanban_home, all_assignees_spawnable):
    spawns = []

    def fake_spawn(task, workspace):
        spawns.append((task.id, task.assignee, workspace))

    with kb.connect() as conn:
        p = kb.create_task(conn, title="p", assignee="alice")
        c = kb.create_task(conn, title="c", assignee="bob", parents=[p])
        # Finish parent outside dispatch; promotion happens inside.
        kb.complete_task(conn, p)
        res = kb.dispatch_once(conn, spawn_fn=fake_spawn)
    # Spawned c (a was already done when dispatch was called).
    assert len(spawns) == 1
    assert spawns[0][0] == c
    assert spawns[0][1] == "bob"
    # c is now running
    with kb.connect() as conn:
        assert kb.get_task(conn, c).status == "running"


def test_dispatch_auto_continues_iteration_budget_block_once(
    kanban_home, all_assignees_spawnable
):
    spawns = []

    def fake_spawn(task, workspace):
        spawns.append(task.id)

    with kb.connect() as conn:
        t = kb.create_task(conn, title="resume me", assignee="alice")
        kb.claim_task(conn, t)
        run = kb.active_run(conn, t)
        assert run is not None
        assert kb.block_task(
            conn,
            t,
            reason=(
                "Iteration budget exhausted (60/60) — task could not complete "
                "within the allowed iterations"
            ),
            expected_run_id=run.id,
        )

        res = kb.dispatch_once(conn, spawn_fn=fake_spawn)

        assert res.auto_continued == [t]
        assert spawns == [t]
        assert kb.get_task(conn, t).status == "running"

        comments = kb.list_comments(conn, t)
        assert len(comments) == 1
        assert "Iteration budget exhausted (60/60)" in comments[0].body
        assert "Resume from the previous worker checkpoint" in comments[0].body

        events = kb.list_events(conn, t)
        auto_event = next(
            e for e in events if e.kind == "dispatch_auto_continued_iteration_budget"
        )
        assert auto_event.run_id == run.id
        assert auto_event.payload["previous_run_id"] == run.id
        assert auto_event.payload["continuation_index"] == 1
        assert auto_event.payload["continuation_limit"] == 1
        assert auto_event.payload["checkpoint_comment_id"] == comments[0].id



def test_dispatch_creates_independent_reviewer_for_review_required_block(
    kanban_home, all_assignees_spawnable
):
    spawns = []

    def fake_spawn(task, workspace):
        spawns.append(task.id)
        return 0

    body = """
scope_contract:
  version: 2
  assignee: coder
  allowed_tools: [kanban_show, read_file, search_files, patch, write_file, kanban_comment, kanban_complete, kanban_block]
  allowed_paths:
    - /tmp/repo/**
  forbidden_systems: [OpenClaw, Atlas, Mission-Control, Telegram]
completion_policy:
  require_scope_attestation: true
"""

    with kb.connect() as conn:
        source = kb.create_task(
            conn,
            title="implement feature",
            assignee="coder",
            body=body,
            workspace_kind="dir",
            workspace_path="/tmp/repo",
        )
        context_comment_id = kb.add_comment(
            conn,
            source,
            "coder",
            "review handoff context with diff/test evidence",
        )
        kb.claim_task(conn, source)
        run = kb.active_run(conn, source)
        assert run is not None
        assert kb.block_task(
            conn,
            source,
            reason="review-required: feature shipped",
            expected_run_id=run.id,
            context_comment_id=context_comment_id,
        )

        res = kb.dispatch_once(conn, spawn_fn=fake_spawn)

        reviewers = [t for t in kb.list_tasks(conn, assignee="reviewer") if t.created_by == "dispatcher"]
        assert len(reviewers) == 1
        reviewer = reviewers[0]
        assert res.spawned == [(reviewer.id, "reviewer", "/tmp/repo")]
        assert spawns == [reviewer.id]
        assert reviewer.workspace_kind == "dir"
        assert reviewer.workspace_path == "/tmp/repo"
        assert kb.parent_ids(conn, reviewer.id) == []
        assert f"source_task: {source}" in (reviewer.body or "")
        assert f"source_run_id: {run.id}" in (reviewer.body or "")
        assert f"context_comment_id: {context_comment_id}" in (reviewer.body or "")
        assert "source remains blocked pending explicit Coordinator/Admin finalization" in (
            reviewer.body or ""
        )

        handoff_events = [
            e
            for e in kb.list_events(conn, source)
            if e.kind == "dispatch_review_required_handoff_created"
        ]
        assert len(handoff_events) == 1
        assert handoff_events[0].run_id == run.id
        assert handoff_events[0].payload["source_task"] == source
        assert handoff_events[0].payload["source_run_id"] == run.id
        assert handoff_events[0].payload["context_comment_id"] == context_comment_id
        assert handoff_events[0].payload["reviewer_task_id"] == reviewer.id
        assert handoff_events[0].payload["blocked_source_parent_edge"] is False

        source_comments = kb.list_comments(conn, source)
        assert source_comments[-1].author == "kanban-dispatcher"
        assert reviewer.id in source_comments[-1].body
        assert "no auto-completion" in source_comments[-1].body

        assert kb.complete_task(
            conn,
            reviewer.id,
            summary="verdict: APPROVED",
            metadata={
                "scope_contract_read": True,
                "scope_contract_version": 2,
                "scope_attestation": True,
                "forbidden_actions_taken": 0,
                "verdict": "APPROVED",
                "blocking_findings": [],
                "required_verification": [],
                "evidence_audited": [source, reviewer.id],
                "residual_risk": "local finalization remains explicit",
                "effective_toolsets": [
                    "skill_view",
                    "kanban_show",
                    "read_file",
                    "search_files",
                    "kanban_run_workspace_command",
                    "kanban_comment",
                    "kanban_complete",
                    "kanban_block",
                ],
            },
        )

        kb.dispatch_once(conn, spawn_fn=fake_spawn)
        assert kb.get_task(conn, source).status == "blocked"



def test_superseding_review_rewire_helper_is_explicit_and_audited(kanban_home):
    with kb.connect() as conn:
        source = kb.create_task(conn, title="source waiting on old review", assignee="coder")
        old_review = kb.create_task(conn, title="old review", assignee="reviewer")
        new_review = kb.create_task(conn, title="new review", assignee="reviewer")
        kb.link_tasks(conn, old_review, source)

        result = kb.rewire_superseding_review_parent(
            conn,
            source_task=source,
            old_review_task=old_review,
            new_review_task=new_review,
            reason="NEEDS_REVISION fixed and re-reviewed",
        )

        assert result == {
            "source_task": source,
            "old_review_task": old_review,
            "new_review_task": new_review,
            "old_parent_removed": True,
            "new_parent_added": True,
            "reason": "NEEDS_REVISION fixed and re-reviewed",
        }
        assert kb.parent_ids(conn, source) == [new_review]
        events = [
            e for e in kb.list_events(conn, source)
            if e.kind == "superseding_review_rewired"
        ]
        assert len(events) == 1
        assert events[0].payload == result


def test_superseding_review_rewire_is_noop_without_old_edge(kanban_home):
    with kb.connect() as conn:
        source = kb.create_task(conn, title="source", assignee="coder")
        old_review = kb.create_task(conn, title="old review", assignee="reviewer")
        new_review = kb.create_task(conn, title="new review", assignee="reviewer")

        result = kb.rewire_superseding_review_parent(
            conn,
            source_task=source,
            old_review_task=old_review,
            new_review_task=new_review,
            reason="operator requested audit-only check",
        )

        assert result["old_parent_removed"] is False
        assert result["new_parent_added"] is True
        assert kb.parent_ids(conn, source) == [new_review]
        event = [
            e for e in kb.list_events(conn, source)
            if e.kind == "superseding_review_rewired"
        ][-1]
        assert event.payload["old_parent_removed"] is False
        assert event.payload["new_parent_added"] is True


def test_needs_revision_fix_task_is_deterministic_idempotent_and_keeps_source_blocked(kanban_home):
    with kb.connect() as conn:
        source = kb.create_task(conn, title="implement lifecycle", assignee="coder")
        kb.claim_task(conn, source)
        run = kb.active_run(conn, source)
        assert run is not None
        assert kb.block_task(
            conn,
            source,
            reason="review-required: implementation ready for verdict",
            expected_run_id=run.id,
        )
        old_review = kb.create_task(conn, title="review implementation", assignee="reviewer")
        reviewer_metadata = {
            "verdict": "NEEDS_REVISION",
            "blocking_findings": ["missing supersedes relation"],
            "required_verification": ["pytest tests/hermes_cli/test_kanban_db.py -q"],
            "evidence_audited": [source, old_review],
            "residual_risk": "source must remain blocked until finalization gate",
        }

        first = kb.ensure_needs_revision_fix_task(
            conn,
            source_task=source,
            review_task=old_review,
            reviewer_metadata=reviewer_metadata,
            reason="Reviewer requested deterministic fix",
        )
        second = kb.ensure_needs_revision_fix_task(
            conn,
            source_task=source,
            review_task=old_review,
            reviewer_metadata=reviewer_metadata,
            reason="Reviewer requested deterministic fix",
        )

        assert second == first
        fix = kb.get_task(conn, first["fix_task"])
        assert fix is not None
        assert fix.assignee == "coder"
        assert fix.status == "ready"
        assert kb.parent_ids(conn, fix.id) == []
        assert "verdict: NEEDS_REVISION" in (fix.body or "")
        assert "source remains blocked" in (fix.body or "")
        assert kb.get_task(conn, source).status == "blocked"
        events = [
            e for e in kb.list_events(conn, source)
            if e.kind == "needs_revision_fix_task_ensured"
        ]
        assert len(events) == 1
        assert events[0].payload["source_task"] == source
        assert events[0].payload["review_task"] == old_review
        assert events[0].payload["fix_task"] == fix.id
        assert events[0].payload["created"] is True



def test_dispatch_iteration_budget_continuation_cap_blocks_loop(
    kanban_home, all_assignees_spawnable
):
    spawns = []

    def fake_spawn(task, workspace):
        spawns.append(task.id)

    with kb.connect() as conn:
        t = kb.create_task(conn, title="bounded continuation", assignee="alice")
        kb.claim_task(conn, t)
        first_run = kb.active_run(conn, t)
        assert first_run is not None
        assert kb.block_task(
            conn,
            t,
            reason=(
                "Iteration budget exhausted (60/60) — task could not complete "
                "within the allowed iterations"
            ),
            expected_run_id=first_run.id,
        )

        first_dispatch = kb.dispatch_once(conn, spawn_fn=fake_spawn)
        assert first_dispatch.auto_continued == [t]
        assert spawns == [t]

        second_run = kb.active_run(conn, t)
        assert second_run is not None
        assert second_run.id != first_run.id
        assert kb.block_task(
            conn,
            t,
            reason=(
                "Iteration budget exhausted (60/60) — task could not complete "
                "within the allowed iterations"
            ),
            expected_run_id=second_run.id,
        )

        second_dispatch = kb.dispatch_once(conn, spawn_fn=fake_spawn)
        assert second_dispatch.auto_continued == []
        assert second_dispatch.continuation_capped == [t]
        assert kb.get_task(conn, t).status == "blocked"
        assert spawns == [t]

        comments = kb.list_comments(conn, t)
        assert len(comments) == 2
        assert "Coordinator/Human review required" in comments[-1].body

        events = kb.list_events(conn, t)
        capped_event = next(
            e for e in events if e.kind == "dispatch_iteration_budget_continuation_capped"
        )
        assert capped_event.run_id == second_run.id
        assert capped_event.payload["previous_run_id"] == second_run.id
        assert capped_event.payload["continuation_limit"] == 1

        repeat_dispatch = kb.dispatch_once(conn, spawn_fn=fake_spawn)
        assert repeat_dispatch.continuation_capped == []
        assert len(kb.list_comments(conn, t)) == 2
        assert len(
            [
                e
                for e in kb.list_events(conn, t)
                if e.kind == "dispatch_iteration_budget_continuation_capped"
            ]
        ) == 1



def test_dispatch_ignores_non_budget_blocks(kanban_home, all_assignees_spawnable):
    with kb.connect() as conn:
        t = kb.create_task(conn, title="needs human", assignee="alice")
        kb.claim_task(conn, t)
        run = kb.active_run(conn, t)
        assert run is not None
        assert kb.block_task(
            conn,
            t,
            reason="Need a human decision on the rollout plan",
            expected_run_id=run.id,
        )

        res = kb.dispatch_once(conn, spawn_fn=lambda *_args: 123)

        assert res.auto_continued == []
        assert res.continuation_capped == []
        assert kb.get_task(conn, t).status == "blocked"
        assert kb.list_comments(conn, t) == []



def test_dispatch_does_not_auto_continue_substring_only_budget_mentions(
    kanban_home, all_assignees_spawnable
):
    with kb.connect() as conn:
        t = kb.create_task(conn, title="human triage", assignee="alice")
        kb.claim_task(conn, t)
        run = kb.active_run(conn, t)
        assert run is not None
        assert kb.block_task(
            conn,
            t,
            reason="Need triage: prior run said Iteration budget exhausted (60/60), but this block is for policy approval",
            expected_run_id=run.id,
        )

        res = kb.dispatch_once(conn, spawn_fn=lambda *_args: 123)

        assert res.auto_continued == []
        assert res.continuation_capped == []
        assert kb.get_task(conn, t).status == "blocked"
        assert kb.list_comments(conn, t) == []



def test_dispatch_does_not_auto_continue_canonical_budget_reason_with_non_budget_block_type(
    kanban_home, all_assignees_spawnable
):
    with kb.connect() as conn:
        t = kb.create_task(conn, title="canonical-non-budget-marker", assignee="alice")
        kb.claim_task(conn, t)
        run = kb.active_run(conn, t)
        assert run is not None
        assert kb.block_task(
            conn,
            t,
            reason=(
                "Iteration budget exhausted (60/60) — task could not complete "
                "within the allowed iterations"
            ),
            block_type="human_review_needed",
            expected_run_id=run.id,
        )

        blocked = [e for e in kb.list_events(conn, t) if e.kind == "blocked"][-1]
        assert blocked.payload.get("reason", "").startswith("Iteration budget exhausted")
        assert blocked.payload.get("block_type") == "human_review_needed"

        res = kb.dispatch_once(conn, spawn_fn=lambda *_args: 123)

        assert res.auto_continued == []
        assert res.continuation_capped == []
        assert kb.get_task(conn, t).status == "blocked"
        assert kb.list_comments(conn, t) == []



def test_dispatch_does_not_auto_continue_review_required_budget_phrase(
    kanban_home, all_assignees_spawnable
):
    with kb.connect() as conn:
        t = kb.create_task(conn, title="review-needed", assignee="coder")
        kb.claim_task(conn, t)
        run = kb.active_run(conn, t)
        assert run is not None
        assert kb.block_task(
            conn,
            t,
            reason=(
                "review-required: Iteration budget exhausted (60/60) — "
                "task could not complete within the allowed iterations"
            ),
            expected_run_id=run.id,
        )

        res = kb.dispatch_once(conn, spawn_fn=lambda *_args: 123)

        assert res.auto_continued == []
        assert t not in [task_id for task_id, _assignee, _workspace in res.spawned]
        assert [
            e
            for e in kb.list_events(conn, t)
            if e.kind == "dispatch_auto_continued_iteration_budget"
        ] == []
        assert not any(
            "dispatcher auto-continuation checkpoint" in c.body
            for c in kb.list_comments(conn, t)
        )


def test_dispatch_does_not_auto_continue_budget_warning_strings(
    kanban_home, all_assignees_spawnable
):
    with kb.connect() as conn:
        t = kb.create_task(conn, title="warning only", assignee="alice")
        kb.claim_task(conn, t)
        run = kb.active_run(conn, t)
        assert run is not None
        assert kb.block_task(
            conn,
            t,
            reason="Iteration budget warning (60/60)",
            expected_run_id=run.id,
        )

        res = kb.dispatch_once(conn, spawn_fn=lambda *_args: 123)

        assert res.auto_continued == []
        assert res.continuation_capped == []
        assert kb.get_task(conn, t).status == "blocked"
        assert kb.list_comments(conn, t) == []



def test_dispatch_does_not_reopen_budget_block_when_parents_are_undone(
    kanban_home, all_assignees_spawnable
):
    with kb.connect() as conn:
        parent = kb.create_task(conn, title="parent", assignee="alice")
        child = kb.create_task(
            conn,
            title="child",
            assignee="bob",
            parents=[parent],
        )
        archived = kb.create_task(conn, title="archived", assignee="carol")
        assert kb.archive_task(conn, archived)

        assert kb.get_task(conn, child).status == "todo"
        assert kb.block_task(
            conn,
            child,
            reason=(
                "Iteration budget exhausted (60/60) — task could not complete "
                "within the allowed iterations"
            ),
        )

        res = kb.dispatch_once(conn, spawn_fn=lambda *_args: 123)

        assert res.auto_continued == []
        assert res.continuation_capped == []
        assert kb.get_task(conn, child).status == "blocked"
        assert kb.list_comments(conn, child) == []
        assert kb.get_task(conn, archived).status == "archived"



def test_dispatch_ignores_non_review_required_blocks(kanban_home, all_assignees_spawnable):
    with kb.connect() as conn:
        t = kb.create_task(conn, title="needs human", assignee="coder")
        kb.claim_task(conn, t)
        run = kb.active_run(conn, t)
        assert run is not None
        assert kb.block_task(
            conn,
            t,
            reason="Need a human decision on the rollout plan",
            expected_run_id=run.id,
        )

        kb.dispatch_once(conn, spawn_fn=lambda *_args: 0)

        assert kb.list_tasks(conn, assignee="reviewer") == []
        assert [
            e for e in kb.list_events(conn, t) if e.kind == "dispatch_review_required_handoff_created"
        ] == []



def test_dispatch_review_required_handoff_is_idempotent_per_blocked_run(
    kanban_home, all_assignees_spawnable
):
    with kb.connect() as conn:
        t = kb.create_task(conn, title="ship patch", assignee="coder")
        kb.claim_task(conn, t)
        run = kb.active_run(conn, t)
        assert run is not None
        assert kb.block_task(
            conn,
            t,
            reason="review-required: patch shipped",
            expected_run_id=run.id,
        )

        first = kb.dispatch_once(conn, spawn_fn=lambda *_args: 0)
        reviewers_after_first = [task.id for task in kb.list_tasks(conn, assignee="reviewer")]
        comments_after_first = len(kb.list_comments(conn, t))

        second = kb.dispatch_once(conn, spawn_fn=lambda *_args: 0)
        reviewers_after_second = [task.id for task in kb.list_tasks(conn, assignee="reviewer")]
        handoff_events = [
            e for e in kb.list_events(conn, t) if e.kind == "dispatch_review_required_handoff_created"
        ]

        assert len(first.spawned) == 1
        assert second.spawned == []
        assert reviewers_after_first == reviewers_after_second
        assert len(reviewers_after_second) == 1
        assert len(handoff_events) == 1
        assert len(kb.list_comments(conn, t)) == comments_after_first



def test_dispatch_review_required_handoff_ignores_reviewer_loops_and_embedded_marker(
    kanban_home, all_assignees_spawnable
):
    with kb.connect() as conn:
        reviewer_source = kb.create_task(conn, title="reviewer task", assignee="reviewer")
        kb.claim_task(conn, reviewer_source)
        reviewer_run = kb.active_run(conn, reviewer_source)
        assert reviewer_run is not None
        assert kb.block_task(
            conn,
            reviewer_source,
            reason="review-required: reviewer should not recurse",
            expected_run_id=reviewer_run.id,
        )

        embedded = kb.create_task(conn, title="embedded marker", assignee="coder")
        kb.claim_task(conn, embedded)
        embedded_run = kb.active_run(conn, embedded)
        assert embedded_run is not None
        assert kb.block_task(
            conn,
            embedded,
            reason="Need human choice before review-required: later follow-up",
            expected_run_id=embedded_run.id,
        )

        kb.dispatch_once(conn, spawn_fn=lambda *_args: 0)

        reviewer_tasks = kb.list_tasks(conn, assignee="reviewer")
        assert len(reviewer_tasks) == 1
        assert reviewer_tasks[0].id == reviewer_source
        assert [
            e
            for e in kb.list_events(conn, reviewer_source)
            if e.kind == "dispatch_review_required_handoff_created"
        ] == []
        assert [
            e for e in kb.list_events(conn, embedded) if e.kind == "dispatch_review_required_handoff_created"
        ] == []



def test_dispatch_review_required_handoff_leaves_parent_gated_legacy_reviewer_inert(
    kanban_home, all_assignees_spawnable
):
    with kb.connect() as conn:
        source = kb.create_task(conn, title="coder task", assignee="coder")
        legacy = kb.create_task(
            conn,
            title="legacy reviewer child",
            assignee="reviewer",
            parents=[source],
        )
        assert kb.get_task(conn, legacy).status == "todo"

        kb.claim_task(conn, source)
        run = kb.active_run(conn, source)
        assert run is not None
        assert kb.block_task(
            conn,
            source,
            reason="review-required: ready for verdict",
            expected_run_id=run.id,
        )

        kb.dispatch_once(conn, spawn_fn=lambda *_args: 0)

        reviewer_tasks = [task for task in kb.list_tasks(conn, assignee="reviewer")]
        independent = next(task for task in reviewer_tasks if task.id != legacy)
        assert kb.get_task(conn, legacy).status == "todo"
        assert kb.parent_ids(conn, independent.id) == []
        handoff_event = [
            e for e in kb.list_events(conn, source) if e.kind == "dispatch_review_required_handoff_created"
        ][-1]
        assert handoff_event.payload["legacy_reviewer_children"] == [legacy]
        assert legacy in (independent.body or "")



def test_dispatch_spawn_failure_releases_claim(kanban_home, all_assignees_spawnable):
    def boom(task, workspace):
        raise RuntimeError("spawn failed")

    with kb.connect() as conn:
        t = kb.create_task(conn, title="boom", assignee="alice")
        kb.dispatch_once(conn, spawn_fn=boom)
        # Must return to ready so the next tick can retry.
        assert kb.get_task(conn, t).status == "ready"
        assert kb.get_task(conn, t).claim_lock is None


def test_dispatch_max_spawn_counts_existing_running_tasks(
    kanban_home, all_assignees_spawnable
):
    """max_spawn is a live concurrency cap, not a per-tick spawn cap.

    Without counting tasks already in ``running``, every dispatcher tick can
    launch up to ``max_spawn`` more workers while previous workers are still
    alive. Long-running boards then accumulate unbounded worker subprocesses.
    """
    spawns = []

    def fake_spawn(task, workspace):
        spawns.append(task.id)

    with kb.connect() as conn:
        running_a = kb.create_task(conn, title="running-a", assignee="alice")
        running_b = kb.create_task(conn, title="running-b", assignee="bob")
        ready = kb.create_task(conn, title="ready", assignee="carol")
        kb.claim_task(conn, running_a)
        kb.claim_task(conn, running_b)

        res = kb.dispatch_once(conn, spawn_fn=fake_spawn, max_spawn=2)

        assert res.spawned == []
        assert spawns == []
        assert kb.get_task(conn, ready).status == "ready"


def test_dispatch_max_spawn_fills_remaining_capacity(
    kanban_home, all_assignees_spawnable
):
    """When below cap, dispatch only fills available worker slots."""
    spawns = []

    def fake_spawn(task, workspace):
        spawns.append(task.id)

    with kb.connect() as conn:
        running = kb.create_task(conn, title="running", assignee="alice")
        ready_a = kb.create_task(conn, title="ready-a", assignee="bob")
        ready_b = kb.create_task(conn, title="ready-b", assignee="carol")
        kb.claim_task(conn, running)

        res = kb.dispatch_once(conn, spawn_fn=fake_spawn, max_spawn=2)

        assert len(res.spawned) == 1
        assert spawns == [ready_a]
        assert kb.get_task(conn, ready_a).status == "running"
        assert kb.get_task(conn, ready_b).status == "ready"


def test_dispatch_reclaims_stale_before_spawning(kanban_home):
    with kb.connect() as conn:
        t = kb.create_task(conn, title="x", assignee="alice")
        kb.claim_task(conn, t)
        conn.execute(
            "UPDATE tasks SET claim_expires = ? WHERE id = ?",
            (int(time.time()) - 1, t),
        )
        res = kb.dispatch_once(conn, dry_run=True)
    assert res.reclaimed == 1


def test_dispatch_blocks_unknown_force_loaded_skill_before_spawn(kanban_home, all_assignees_spawnable):
    with kb.connect() as conn:
        t = kb.create_task(
            conn,
            title="unknown skill",
            assignee="alice",
            skills=["definitely-missing-skill-for-test"],
        )
        res = kb.dispatch_once(conn, spawn_fn=lambda *_args: 123)
        assert t in res.preflight_blocked
        task = kb.get_task(conn, t)
        assert task.status == "blocked"
        assert "unknown force-loaded skill" in (task.result or "")


def test_dispatch_preflight_blocks_scope_policy_without_scope_contract_v2(kanban_home, all_assignees_spawnable):
    body = """
completion_policy:
  require_scope_attestation: true
"""
    with kb.connect() as conn:
        t = kb.create_task(conn, title="missing scope contract", assignee="alice", body=body)
        res = kb.dispatch_once(conn, spawn_fn=lambda *_args: 123)
        assert t in res.preflight_blocked
        assert kb.get_task(conn, t).status == "blocked"


def test_dispatch_preflight_blocks_scope_contract_v2_without_core_forbidden_systems(
    kanban_home, all_assignees_spawnable
):
    body = """
scope_contract:
  version: 2
  allowed_tools:
    - kanban_show
    - kanban_complete
completion_policy:
  require_scope_attestation: true
"""
    spawns = []
    with kb.connect() as conn:
        t = kb.create_task(conn, title="missing forbidden systems", assignee="alice", body=body)
        res = kb.dispatch_once(conn, spawn_fn=lambda task, workspace: spawns.append(task.id))
        assert not spawns
        assert t in res.preflight_blocked
        task = kb.get_task(conn, t)
        assert task.status == "blocked"
        assert "scope_contract.forbidden_systems is missing required entries" in (task.result or "")
        assert "OpenClaw" in (task.result or "")
        assert "Atlas" in (task.result or "")
        assert "Mission-Control" in (task.result or "")
        assert "Telegram" in (task.result or "")


def test_dispatch_preflight_blocks_partial_core_forbidden_systems(
    kanban_home, all_assignees_spawnable
):
    body = """
scope_contract:
  version: 2
  allowed_tools:
    - kanban_show
  forbidden_systems:
    - OpenClaw
    - Atlas
completion_policy:
  require_scope_attestation: true
"""
    with kb.connect() as conn:
        t = kb.create_task(conn, title="partial forbidden systems", assignee="alice", body=body)
        res = kb.dispatch_once(conn, spawn_fn=lambda *_args: 123)
        assert t in res.preflight_blocked
        task = kb.get_task(conn, t)
        assert task.status == "blocked"
        assert "Mission-Control" in (task.result or "")
        assert "Telegram" in (task.result or "")


def test_dispatch_allows_scope_contract_v2_task_to_spawn(kanban_home, all_assignees_spawnable):
    spawns = []
    body = """
scope_contract:
  version: 2
  allowed_tools:
    - kanban_show
    - kanban_complete
    - kanban_block
  forbidden_systems:
    - OpenClaw
    - Atlas
    - Mission-Control
    - Telegram
completion_policy:
  require_scope_attestation: true
"""
    def fake_spawn(task, workspace):
        spawns.append(task.id)
        return 123

    with kb.connect() as conn:
        t = kb.create_task(conn, title="scoped v2", assignee="alice", body=body)
        res = kb.dispatch_once(conn, spawn_fn=fake_spawn)
        assert t not in res.preflight_blocked
        assert spawns == [t]
        assert kb.get_task(conn, t).status == "running"
        events = [e for e in kb.list_events(conn, t) if e.kind == "dispatch_preflight_passed"]
        assert events
        assert events[-1].payload["effective_toolsets"] == [
            "kanban_show",
            "kanban_complete",
            "kanban_block",
        ]


def test_dispatch_preflight_blocks_when_required_lifecycle_tool_missing(
    kanban_home, all_assignees_spawnable
):
    body = """
scope_contract:
  version: 2
  allowed_tools:
    - kanban_show
    - kanban_complete
  forbidden_systems:
    - OpenClaw
    - Atlas
    - Mission-Control
    - Telegram
completion_policy:
  require_scope_attestation: true
"""
    spawns = []
    with kb.connect() as conn:
        t = kb.create_task(conn, title="missing lifecycle tool", assignee="alice", body=body)
        res = kb.dispatch_once(conn, spawn_fn=lambda task, workspace: spawns.append(task.id))
        assert not spawns
        assert t in res.preflight_blocked
        task = kb.get_task(conn, t)
        assert task.status == "blocked"
        assert "empty/incomplete effective runtime tools" in (task.result or "")
        assert "kanban_block" in (task.result or "")

        events = [e for e in kb.list_events(conn, t) if e.kind == "dispatch_preflight_empty_toolset"]
        assert events
        payload = events[-1].payload
        assert payload["task_id"] == t
        assert payload["effective_toolsets"] == ["kanban_show", "kanban_complete"]
        assert payload["required_lifecycle_tools"] == [
            "kanban_show",
            "kanban_complete",
            "kanban_block",
        ]
        assert payload["skills_requested"] == []
        assert payload["skill_resolution"] == {"status": "ok", "missing": []}
        assert "kanban_block" in payload["failure_reason"]


def test_dispatch_preflight_blocks_when_runtime_schema_missing_required_lifecycle_tool(
    kanban_home, all_assignees_spawnable
):
    body = """
scope_contract:
  version: 2
  allowed_tools:
    - kanban_show
    - kanban_complete
    - kanban_block
  forbidden_systems:
    - OpenClaw
    - Atlas
    - Mission-Control
    - Telegram
completion_policy:
  require_scope_attestation: true
"""
    from model_tools import _clear_tool_defs_cache
    from tools.registry import invalidate_check_fn_cache, registry

    entry = registry.get_entry("kanban_block")
    assert entry is not None
    original_check_fn = entry.check_fn
    entry.check_fn = lambda: False
    invalidate_check_fn_cache()
    _clear_tool_defs_cache()
    try:
        spawns = []
        with kb.connect() as conn:
            t = kb.create_task(conn, title="runtime schema missing lifecycle tool", assignee="alice", body=body)
            res = kb.dispatch_once(conn, spawn_fn=lambda task, workspace: spawns.append(task.id))
            assert not spawns
            assert t in res.preflight_blocked
            task = kb.get_task(conn, t)
            assert task.status == "blocked"
            assert "runtime tool schema" in (task.result or "")
            assert "kanban_block" in (task.result or "")

            events = [e for e in kb.list_events(conn, t) if e.kind == "dispatch_preflight_empty_toolset"]
            assert events
            payload = events[-1].payload
            assert payload["task_id"] == t
            assert payload["declared_allowed_tools"] == [
                "kanban_show",
                "kanban_complete",
                "kanban_block",
            ]
            assert payload["effective_toolsets"] == ["kanban_show", "kanban_complete"]
            assert payload["required_lifecycle_tools"] == [
                "kanban_show",
                "kanban_complete",
                "kanban_block",
            ]
            assert payload["skill_resolution"] == {"status": "ok", "missing": []}
    finally:
        entry.check_fn = original_check_fn
        invalidate_check_fn_cache()
        _clear_tool_defs_cache()


def test_dispatch_preflight_blocks_scope_contract_v2_without_allowed_tools(kanban_home, all_assignees_spawnable):
    body = """
scope_contract:
  version: 2
completion_policy:
  require_scope_attestation: true
"""
    with kb.connect() as conn:
        t = kb.create_task(conn, title="missing allowed tools", assignee="alice", body=body)
        res = kb.dispatch_once(conn, spawn_fn=lambda *_args: 123)
        assert t in res.preflight_blocked
        task = kb.get_task(conn, t)
        assert task.status == "blocked"
        assert "allowed_tools is required" in (task.result or "")


def test_dispatch_preflight_blocks_unknown_allowed_tool(kanban_home, all_assignees_spawnable):
    body = """
scope_contract:
  version: 2
  allowed_tools:
    - kanban_show
    - definitely_not_a_tool
completion_policy:
  require_scope_attestation: true
"""
    with kb.connect() as conn:
        t = kb.create_task(conn, title="unknown allowed tool", assignee="alice", body=body)
        res = kb.dispatch_once(conn, spawn_fn=lambda *_args: 123)
        assert t in res.preflight_blocked
        task = kb.get_task(conn, t)
        assert task.status == "blocked"
        assert "unknown allowed_tool: definitely_not_a_tool" in (task.result or "")


def test_dispatch_preflight_blocks_broad_allowed_tool(kanban_home, all_assignees_spawnable):
    body = """
scope_contract:
  version: 2
  allowed_tools:
    - all
completion_policy:
  require_scope_attestation: true
"""
    with kb.connect() as conn:
        t = kb.create_task(conn, title="broad allowed tool", assignee="alice", body=body)
        res = kb.dispatch_once(conn, spawn_fn=lambda *_args: 123)
        assert t in res.preflight_blocked
        task = kb.get_task(conn, t)
        assert task.status == "blocked"
        assert "allowed_tool is too broad: all" in (task.result or "")


def test_dispatch_preflight_blocks_toolset_like_allowed_tool_names(
    kanban_home, all_assignees_spawnable
):
    toolset_like_names = [
        "terminal",
        "file",
        "kanban",
        "mcp",
        "delegation",
        "code_execution",
        "memory",
        "clarify",
        "tools",
        "all_tools",
        "all",
        "any",
        "*",
    ]
    spawns = []

    with kb.connect() as conn:
        task_ids = []
        for name in toolset_like_names:
            body = f"""
scope_contract:
  version: 2
  allowed_tools:
    - "{name}"
completion_policy:
  require_scope_attestation: true
"""
            task_ids.append(
                kb.create_task(conn, title=f"broad allowed tool {name}", assignee="alice", body=body)
            )

        res = kb.dispatch_once(conn, spawn_fn=lambda task, workspace: spawns.append(task.id))

        assert not spawns
        assert set(task_ids) <= set(res.preflight_blocked)
        for task_id, name in zip(task_ids, toolset_like_names):
            task = kb.get_task(conn, task_id)
            assert task.status == "blocked"
            assert f"allowed_tool is too broad: {name}" in (task.result or "")


def test_dispatch_preflight_blocks_mixed_concrete_and_toolset_like_allowed_tools(
    kanban_home, all_assignees_spawnable
):
    body = """
scope_contract:
  version: 2
  allowed_tools:
    - kanban_show
    - terminal
completion_policy:
  require_scope_attestation: true
"""
    spawns = []
    with kb.connect() as conn:
        t = kb.create_task(conn, title="mixed concrete and broad", assignee="alice", body=body)
        res = kb.dispatch_once(conn, spawn_fn=lambda task, workspace: spawns.append(task.id))
        assert not spawns
        assert t in res.preflight_blocked
        task = kb.get_task(conn, t)
        assert task.status == "blocked"
        assert "allowed_tool is too broad: terminal" in (task.result or "")


def test_dispatch_preflight_accepts_kanban_minimal_concrete_tool_names_only(
    kanban_home, all_assignees_spawnable
):
    body = """
scope_contract:
  version: 2
  allowed_tools:
    - kanban_show
    - kanban_complete
    - kanban_block
    - kanban_comment
  forbidden_systems:
    - OpenClaw
    - Atlas
    - Mission-Control
    - Telegram
completion_policy:
  require_scope_attestation: true
"""
    spawns = []
    with kb.connect() as conn:
        t = kb.create_task(conn, title="minimal concrete allowed tools", assignee="alice", body=body)
        res = kb.dispatch_once(conn, spawn_fn=lambda task, workspace: spawns.append(task.id))
        assert t not in res.preflight_blocked
        assert spawns == [t]
        task = kb.get_task(conn, t)
        assert task.status == "running"
        events = [e for e in kb.list_events(conn, t) if e.kind == "dispatch_preflight_passed"]
        assert events[-1].payload["effective_toolsets"] == [
            "kanban_show",
            "kanban_complete",
            "kanban_block",
            "kanban_comment",
        ]


def test_dispatch_preflight_accepts_safe_workspace_runner_tool(
    kanban_home, all_assignees_spawnable
):
    body = """
scope_contract:
  version: 2
  allowed_tools:
    - kanban_show
    - write_file
    - read_file
    - kanban_run_workspace_command
    - kanban_complete
    - kanban_block
  forbidden_systems:
    - OpenClaw
    - Atlas
    - Mission-Control
    - Telegram
completion_policy:
  require_scope_attestation: true
"""
    spawns = []
    with kb.connect() as conn:
        t = kb.create_task(conn, title="safe runner", assignee="alice", body=body)
        res = kb.dispatch_once(conn, spawn_fn=lambda task, workspace: spawns.append(task.id))
        assert t not in res.preflight_blocked
        assert spawns == [t]
        task = kb.get_task(conn, t)
        assert task.status == "running"
        events = [e for e in kb.list_events(conn, t) if e.kind == "dispatch_preflight_passed"]
        assert events[-1].payload["effective_toolsets"] == [
            "kanban_show",
            "write_file",
            "read_file",
            "kanban_run_workspace_command",
            "kanban_complete",
            "kanban_block",
        ]


def test_validate_created_cards_dry_run_reports_verified_and_phantom_without_mutation(kanban_home):
    with kb.connect() as conn:
        parent = kb.create_task(conn, title="parent", assignee="alice")
        owned = kb.create_task(conn, title="owned", assignee="worker", created_by="alice")
        foreign = kb.create_task(conn, title="foreign", assignee="worker", created_by="bob")
        linked = kb.create_task(conn, title="linked", assignee="worker", created_by="dashboard")
        kb.link_tasks(conn, parent, linked)
        before = kb.get_task(conn, parent).status

        result = kb.validate_created_cards(
            conn,
            parent,
            [owned, foreign, linked, "t_deadbeef"],
        )
        after = kb.get_task(conn, parent).status
        events = kb.list_events(conn, parent)

    assert result == {
        "ok": False,
        "task_id": parent,
        "claimed_cards": [owned, foreign, linked, "t_deadbeef"],
        "verified_cards": [owned, linked],
        "phantom_cards": [foreign, "t_deadbeef"],
    }
    assert after == before == "ready"
    assert "completed" not in [e.kind for e in events]
    assert "completion_blocked_hallucination" not in [e.kind for e in events]
    assert "created_cards_validated" not in [e.kind for e in events]


def test_complete_task_reuses_created_cards_validation_gate(kanban_home):
    with kb.connect() as conn:
        parent = kb.create_task(conn, title="parent", assignee="alice")
        owned = kb.create_task(conn, title="owned", assignee="worker", created_by="alice")
        foreign = kb.create_task(conn, title="foreign", assignee="worker", created_by="bob")

        dry_run = kb.validate_created_cards(conn, parent, [owned, foreign])
        with pytest.raises(kb.HallucinatedCardsError) as err:
            kb.complete_task(conn, parent, summary="done", created_cards=[owned, foreign])
        parent_task = kb.get_task(conn, parent)
        blocked = [
            e for e in kb.list_events(conn, parent)
            if e.kind == "completion_blocked_hallucination"
        ][-1]

    assert dry_run["verified_cards"] == [owned]
    assert dry_run["phantom_cards"] == [foreign]
    assert err.value.phantom == [foreign]
    assert parent_task.status == "ready"
    assert blocked.payload["verified_cards"] == dry_run["verified_cards"]
    assert blocked.payload["phantom_cards"] == dry_run["phantom_cards"]


def test_completion_attestation_requires_effective_toolsets_when_allowed_tools_declared(kanban_home):
    body = """
scope_contract:
  version: 2
  allowed_tools:
    - kanban_show
completion_policy:
  require_scope_attestation: true
"""
    with kb.connect() as conn:
        t = kb.create_task(conn, title="effective toolsets required", assignee="alice", body=body)
        with pytest.raises(kb.ScopeAttestationError, match="effective_toolsets"):
            kb.complete_task(
                conn,
                t,
                summary="done",
                metadata={
                    "scope_contract_version": 2,
                    "scope_attestation": True,
                    "forbidden_actions_taken": 0,
                },
            )
        assert kb.complete_task(
            conn,
            t,
            summary="done",
            metadata={
                "scope_contract_version": 2,
                "scope_attestation": True,
                "forbidden_actions_taken": 0,
                "effective_toolsets": ["kanban_show"],
            },
        )


def test_completion_attestation_blocks_effective_toolsets_mismatch_after_dispatch_preflight(
    kanban_home, all_assignees_spawnable
):
    body = """
scope_contract:
  version: 2
  allowed_tools:
    - kanban_show
    - kanban_complete
    - kanban_block
  forbidden_systems:
    - OpenClaw
    - Atlas
    - Mission-Control
    - Telegram
completion_policy:
  require_scope_attestation: true
"""
    with kb.connect() as conn:
        t = kb.create_task(conn, title="mismatched completion toolsets", assignee="alice", body=body)
        res = kb.dispatch_once(conn, spawn_fn=lambda task, workspace: 123)
        assert res.spawned and res.spawned[0][0] == t
        preflight_events = [
            e for e in kb.list_events(conn, t) if e.kind == "dispatch_preflight_passed"
        ]
        assert preflight_events[-1].payload["effective_toolsets"] == [
            "kanban_show",
            "kanban_complete",
            "kanban_block",
        ]

        with pytest.raises(kb.ScopeAttestationError, match="effective_toolsets mismatch"):
            kb.complete_task(
                conn,
                t,
                summary="done",
                metadata={
                    "scope_contract_version": 2,
                    "scope_attestation": True,
                    "forbidden_actions_taken": 0,
                    "effective_toolsets": ["kanban_show"],
                },
            )
        task = kb.get_task(conn, t)
        assert task.status == "running"
        block_events = [
            e for e in kb.list_events(conn, t) if e.kind == "completion_blocked_scope_attestation"
        ]
        assert "effective_toolsets mismatch" in block_events[-1].payload["missing"]


def test_completion_attestation_accepts_effective_toolsets_matching_dispatch_preflight(
    kanban_home, all_assignees_spawnable
):
    body = """
scope_contract:
  version: 2
  allowed_tools:
    - kanban_show
    - kanban_complete
    - kanban_block
  forbidden_systems:
    - OpenClaw
    - Atlas
    - Mission-Control
    - Telegram
completion_policy:
  require_scope_attestation: true
"""
    with kb.connect() as conn:
        t = kb.create_task(conn, title="matching completion toolsets", assignee="alice", body=body)
        res = kb.dispatch_once(conn, spawn_fn=lambda task, workspace: 123)
        assert res.spawned and res.spawned[0][0] == t
        assert kb.complete_task(
            conn,
            t,
            summary="done",
            metadata={
                "scope_contract_version": 2,
                "scope_attestation": True,
                "forbidden_actions_taken": 0,
                "effective_toolsets": ["kanban_show", "kanban_complete", "kanban_block"],
            },
        )
        completed_events = [e for e in kb.list_events(conn, t) if e.kind == "completed"]
        assert completed_events
        completed_runs = [r for r in kb.list_runs(conn, t) if r.outcome == "completed"]
        assert completed_runs[-1].metadata["effective_toolsets"] == [
            "kanban_show",
            "kanban_complete",
            "kanban_block",
        ]


# ---------------------------------------------------------------------------
# Workspace resolution
# ---------------------------------------------------------------------------

def test_scratch_workspace_created_under_hermes_home(kanban_home):
    with kb.connect() as conn:
        t = kb.create_task(conn, title="x")
        task = kb.get_task(conn, t)
        ws = kb.resolve_workspace(task)
    assert ws.exists()
    assert ws.is_dir()
    assert "kanban" in str(ws)


def test_dir_workspace_honors_given_path(kanban_home, tmp_path):
    target = tmp_path / "my-vault"
    with kb.connect() as conn:
        t = kb.create_task(
            conn, title="biz", workspace_kind="dir", workspace_path=str(target)
        )
        task = kb.get_task(conn, t)
        ws = kb.resolve_workspace(task)
    assert ws == target
    assert ws.exists()


def test_worktree_workspace_returns_intended_path(kanban_home, tmp_path):
    target = str(tmp_path / ".worktrees" / "my-task")
    with kb.connect() as conn:
        t = kb.create_task(
            conn, title="ship", workspace_kind="worktree", workspace_path=target
        )
        task = kb.get_task(conn, t)
        ws = kb.resolve_workspace(task)
    # We do NOT auto-create worktrees; the worker's skill handles that.
    assert str(ws) == target


# ---------------------------------------------------------------------------
# Tenancy
# ---------------------------------------------------------------------------

def test_tenant_column_filters_listings(kanban_home):
    with kb.connect() as conn:
        kb.create_task(conn, title="a1", tenant="biz-a")
        kb.create_task(conn, title="b1", tenant="biz-b")
        kb.create_task(conn, title="shared")  # no tenant
        biz_a = kb.list_tasks(conn, tenant="biz-a")
        biz_b = kb.list_tasks(conn, tenant="biz-b")
    assert [t.title for t in biz_a] == ["a1"]
    assert [t.title for t in biz_b] == ["b1"]


def test_tenant_propagates_to_events(kanban_home):
    with kb.connect() as conn:
        t = kb.create_task(conn, title="tenant-task", tenant="biz-a")
        events = kb.list_events(conn, t)
    # The "created" event should have tenant in its payload.
    created = [e for e in events if e.kind == "created"]
    assert created and created[0].payload.get("tenant") == "biz-a"


# ---------------------------------------------------------------------------
# Shared-board path resolution (issue #19348)
#
# The kanban board is a cross-profile coordination primitive: a worker
# spawned with `hermes -p <profile>` must read/write the same kanban.db
# as the dispatcher that claimed the task. These tests exercise the
# path-resolution layer directly and would have caught the regression
# where `kanban_db_path()` resolved to the active profile's HERMES_HOME.
# ---------------------------------------------------------------------------

class TestPathResolutionAndWorkerEnv:
    """`kanban_home`/`kanban_db_path`/`workspaces_root`/`worker_log_path`
    must anchor at the **shared root**, not the active profile's HERMES_HOME."""

    def _set_home(self, monkeypatch, tmp_path, hermes_home):
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        monkeypatch.delenv("HERMES_KANBAN_HOME", raising=False)

    def test_default_install_anchors_at_home_dot_hermes(
        self, tmp_path, monkeypatch
    ):
        # Standard install: HERMES_HOME == ~/.hermes, no profile active.
        default_home = tmp_path / ".hermes"
        default_home.mkdir()
        self._set_home(monkeypatch, tmp_path, default_home)

        assert kb.kanban_home() == default_home
        assert kb.kanban_db_path() == default_home / "kanban.db"
        assert kb.workspaces_root() == default_home / "kanban" / "workspaces"
        assert (
            kb.worker_log_path("t_demo")
            == default_home / "kanban" / "logs" / "t_demo.log"
        )

    def test_profile_worker_resolves_to_shared_root(
        self, tmp_path, monkeypatch
    ):
        # Reproduces the bug: dispatcher uses ~/.hermes/kanban.db,
        # worker spawned with -p <profile> previously resolved to
        # ~/.hermes/profiles/<profile>/kanban.db. After the fix both
        # converge on ~/.hermes/kanban.db.
        default_home = tmp_path / ".hermes"
        default_home.mkdir()
        profile_home = default_home / "profiles" / "nehemiahkanban"
        profile_home.mkdir(parents=True)
        self._set_home(monkeypatch, tmp_path, profile_home)

        # All four resolvers must anchor at the shared root, not the
        # profile-local HERMES_HOME.
        assert kb.kanban_home() == default_home
        assert kb.kanban_db_path() == default_home / "kanban.db"
        assert kb.workspaces_root() == default_home / "kanban" / "workspaces"
        assert (
            kb.worker_log_path("t_0d214f19")
            == default_home / "kanban" / "logs" / "t_0d214f19.log"
        )

        # Sanity: the profile-local path that used to be returned is
        # explicitly NOT what we resolve to anymore.
        assert kb.kanban_db_path() != profile_home / "kanban.db"

    def test_dispatcher_and_profile_worker_converge(
        self, tmp_path, monkeypatch
    ):
        # End-to-end convergence: resolve the path under each side's
        # HERMES_HOME and confirm equality. This is the property the
        # dispatcher/worker handoff actually depends on.
        default_home = tmp_path / ".hermes"
        default_home.mkdir()
        profile_home = default_home / "profiles" / "coder"
        profile_home.mkdir(parents=True)

        # Dispatcher's perspective.
        self._set_home(monkeypatch, tmp_path, default_home)
        dispatcher_db = kb.kanban_db_path()
        dispatcher_ws = kb.workspaces_root()
        dispatcher_log = kb.worker_log_path("t_handoff")

        # Worker's perspective (profile activated by `hermes -p coder`).
        monkeypatch.setenv("HERMES_HOME", str(profile_home))
        worker_db = kb.kanban_db_path()
        worker_ws = kb.workspaces_root()
        worker_log = kb.worker_log_path("t_handoff")

        assert dispatcher_db == worker_db
        assert dispatcher_ws == worker_ws
        assert dispatcher_log == worker_log

    def test_docker_custom_hermes_home_uses_env_path_directly(
        self, tmp_path, monkeypatch
    ):
        # Docker / custom deployment: HERMES_HOME points outside ~/.hermes.
        # `get_default_hermes_root()` returns env_home directly when it
        # is not a `<root>/profiles/<name>` shape and not under
        # `Path.home() / ".hermes"`.
        custom_root = tmp_path / "opt" / "hermes"
        custom_root.mkdir(parents=True)
        self._set_home(monkeypatch, tmp_path, custom_root)

        assert kb.kanban_home() == custom_root
        assert kb.kanban_db_path() == custom_root / "kanban.db"

    def test_docker_profile_layout_uses_grandparent(
        self, tmp_path, monkeypatch
    ):
        # Docker profile shape: HERMES_HOME=/opt/hermes/profiles/coder;
        # `get_default_hermes_root()` walks up to /opt/hermes because
        # the immediate parent dir is named "profiles".
        custom_root = tmp_path / "opt" / "hermes"
        profile = custom_root / "profiles" / "coder"
        profile.mkdir(parents=True)
        self._set_home(monkeypatch, tmp_path, profile)

        assert kb.kanban_home() == custom_root
        assert kb.kanban_db_path() == custom_root / "kanban.db"

    def test_explicit_override_via_hermes_kanban_home(
        self, tmp_path, monkeypatch
    ):
        # Explicit override: HERMES_KANBAN_HOME beats every other
        # resolution rule.
        default_home = tmp_path / ".hermes"
        profile_home = default_home / "profiles" / "any"
        profile_home.mkdir(parents=True)
        override = tmp_path / "shared-board"
        override.mkdir()

        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        monkeypatch.setenv("HERMES_HOME", str(profile_home))
        monkeypatch.setenv("HERMES_KANBAN_HOME", str(override))

        assert kb.kanban_home() == override
        assert kb.kanban_db_path() == override / "kanban.db"
        assert kb.workspaces_root() == override / "kanban" / "workspaces"

    def test_empty_override_falls_through(self, tmp_path, monkeypatch):
        # Empty/whitespace override is treated as unset.
        default_home = tmp_path / ".hermes"
        default_home.mkdir()
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        monkeypatch.setenv("HERMES_HOME", str(default_home))
        monkeypatch.setenv("HERMES_KANBAN_HOME", "   ")

        assert kb.kanban_home() == default_home

    def test_dispatcher_and_worker_share_a_real_database(
        self, tmp_path, monkeypatch
    ):
        # Belt-and-suspenders: round-trip a task across the two
        # HERMES_HOME perspectives via a real SQLite file. Without the
        # fix the worker would open a different file and see no rows.
        default_home = tmp_path / ".hermes"
        default_home.mkdir()
        profile_home = default_home / "profiles" / "nehemiahkanban"
        profile_home.mkdir(parents=True)

        # Dispatcher creates the board and a task.
        self._set_home(monkeypatch, tmp_path, default_home)
        kb.init_db()
        with kb.connect() as conn:
            task_id = kb.create_task(conn, title="cross-profile")

        # Worker switches to the profile HERMES_HOME and reads.
        monkeypatch.setenv("HERMES_HOME", str(profile_home))
        with kb.connect() as conn:
            task = kb.get_task(conn, task_id)
        assert task is not None
        assert task.title == "cross-profile"

    def test_hermes_kanban_db_pin_beats_kanban_home(
        self, tmp_path, monkeypatch
    ):
        # HERMES_KANBAN_DB pins the file path directly and beats both
        # HERMES_KANBAN_HOME and the `get_default_hermes_root()` path.
        # This is the env the dispatcher injects into workers.
        default_home = tmp_path / ".hermes"
        default_home.mkdir()
        umbrella = tmp_path / "umbrella"
        umbrella.mkdir()
        pinned_db = tmp_path / "pinned" / "board.db"
        pinned_db.parent.mkdir()

        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        monkeypatch.setenv("HERMES_HOME", str(default_home))
        monkeypatch.setenv("HERMES_KANBAN_HOME", str(umbrella))
        monkeypatch.setenv("HERMES_KANBAN_DB", str(pinned_db))

        assert kb.kanban_db_path() == pinned_db
        # workspaces_root still follows HERMES_KANBAN_HOME -- the pins
        # are independent.
        assert kb.workspaces_root() == umbrella / "kanban" / "workspaces"

    def test_hermes_kanban_workspaces_root_pin_beats_kanban_home(
        self, tmp_path, monkeypatch
    ):
        # HERMES_KANBAN_WORKSPACES_ROOT pins the workspaces root directly.
        default_home = tmp_path / ".hermes"
        default_home.mkdir()
        umbrella = tmp_path / "umbrella"
        umbrella.mkdir()
        pinned_ws = tmp_path / "pinned-workspaces"
        pinned_ws.mkdir()

        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        monkeypatch.setenv("HERMES_HOME", str(default_home))
        monkeypatch.setenv("HERMES_KANBAN_HOME", str(umbrella))
        monkeypatch.setenv("HERMES_KANBAN_WORKSPACES_ROOT", str(pinned_ws))

        assert kb.workspaces_root() == pinned_ws
        # kanban_db_path still follows HERMES_KANBAN_HOME.
        assert kb.kanban_db_path() == umbrella / "kanban.db"

    def test_empty_per_path_overrides_fall_through(
        self, tmp_path, monkeypatch
    ):
        # Empty/whitespace pins are treated as unset, same as
        # HERMES_KANBAN_HOME.
        default_home = tmp_path / ".hermes"
        default_home.mkdir()
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        monkeypatch.setenv("HERMES_HOME", str(default_home))
        monkeypatch.setenv("HERMES_KANBAN_DB", "   ")
        monkeypatch.setenv("HERMES_KANBAN_WORKSPACES_ROOT", "")

        assert kb.kanban_db_path() == default_home / "kanban.db"
        assert kb.workspaces_root() == default_home / "kanban" / "workspaces"

    def test_dispatcher_spawn_injects_kanban_db_and_workspaces_root(
        self, tmp_path, monkeypatch
    ):
        # The dispatcher's `_default_spawn` must inject HERMES_KANBAN_DB
        # and HERMES_KANBAN_WORKSPACES_ROOT into the worker env so the
        # worker converges on the dispatcher's paths even when the
        # `-p <profile>` flag rewrites HERMES_HOME.
        default_home = tmp_path / ".hermes"
        default_home.mkdir()
        self._set_home(monkeypatch, tmp_path, default_home)

        captured = {}

        class _FakePopen:
            def __init__(self, cmd, **kwargs):
                captured["cmd"] = cmd
                captured["env"] = kwargs.get("env", {})
                self.pid = 4242

        monkeypatch.setattr("subprocess.Popen", _FakePopen)

        task = kb.Task(
            id="t_dispatch_env",
            title="x",
            body=None,
            assignee="coder",
            status="ready",
            priority=0,
            created_by=None,
            created_at=0,
            started_at=None,
            completed_at=None,
            workspace_kind="scratch",
            workspace_path=None,
            claim_lock=None,
            claim_expires=None,
            tenant=None,
        )
        kb._default_spawn(task, str(tmp_path / "ws"))

        env = captured["env"]
        assert env["HERMES_KANBAN_DB"] == str(default_home / "kanban.db")
        assert env["HERMES_KANBAN_WORKSPACES_ROOT"] == str(
            default_home / "kanban" / "workspaces"
        )
        assert env["HERMES_KANBAN_TASK"] == "t_dispatch_env"
        assert env["HERMES_KANBAN_WORKSPACE_KIND"] == "scratch"

    def test_default_spawn_inherits_schema_audit_env_for_one_dispatch(
        self, tmp_path, monkeypatch
    ):
        default_home = tmp_path / ".hermes"
        default_home.mkdir()
        self._set_home(monkeypatch, tmp_path, default_home)
        monkeypatch.setenv("HERMES_KANBAN_SCHEMA_AUDIT", "1")

        captured = {}

        class _FakePopen:
            def __init__(self, cmd, **kwargs):
                captured["cmd"] = cmd
                captured["env"] = kwargs.get("env", {})
                self.pid = 4242

        monkeypatch.setattr("subprocess.Popen", _FakePopen)

        task = kb.Task(
            id="t_schema_audit_env",
            title="x",
            body=None,
            assignee="coder",
            status="ready",
            priority=0,
            created_by=None,
            created_at=0,
            started_at=None,
            completed_at=None,
            workspace_kind="scratch",
            workspace_path=None,
            claim_lock=None,
            claim_expires=None,
            tenant=None,
        )
        kb._default_spawn(task, str(tmp_path / "ws"))

        assert captured["env"]["HERMES_KANBAN_SCHEMA_AUDIT"] == "1"


# ---------------------------------------------------------------------------
# latest_summary / latest_summaries — surface task_runs.summary handoffs
# ---------------------------------------------------------------------------

def test_latest_summary_returns_none_when_no_runs(kanban_home):
    """A freshly-created task has no runs and therefore no summary."""
    with kb.connect() as conn:
        t = kb.create_task(conn, title="fresh", assignee="alice")
        assert kb.latest_summary(conn, t) is None


def test_latest_summary_returns_summary_after_complete(kanban_home):
    """``complete_task(summary=...)`` is the canonical kanban-worker
    handoff; ``latest_summary`` must surface it so dashboards/CLI can
    render what the worker actually did."""
    handoff = "shipped 3 files, ran tests, opened PR #42"
    with kb.connect() as conn:
        t = kb.create_task(conn, title="work", assignee="alice")
        kb.complete_task(conn, t, summary=handoff)
        assert kb.latest_summary(conn, t) == handoff


def test_latest_summary_picks_newest_when_multiple_runs(kanban_home):
    """When a task has been re-run (block → unblock → complete), the
    newest run's summary wins. We unblock to take the task back to
    ``ready``, then complete a second time and verify the second
    summary surfaces."""
    with kb.connect() as conn:
        t = kb.create_task(conn, title="retry", assignee="alice")
        kb.complete_task(conn, t, summary="first attempt")
        # Move back to ready by direct SQL — block_task / unblock_task
        # paths require an active claim, but we just want a second run
        # row to exist with a later ended_at.
        conn.execute(
            "UPDATE tasks SET status='ready', completed_at=NULL WHERE id=?",
            (t,),
        )
        # Sleep 1s so the second run's ended_at is provably later than
        # the first (complete_task uses int(time.time())).
        time.sleep(1.05)
        kb.complete_task(conn, t, summary="second attempt — final")
        assert kb.latest_summary(conn, t) == "second attempt — final"


def test_latest_summary_skips_empty_string(kanban_home):
    """A run with an empty-string summary should not mask an earlier
    populated one — empty strings carry no information."""
    with kb.connect() as conn:
        t = kb.create_task(conn, title="t", assignee="alice")
        kb.complete_task(conn, t, summary="real handoff")
        # Inject a later run with empty summary directly. Workers
        # writing "" instead of None is a real shape we want to ignore.
        conn.execute(
            "INSERT INTO task_runs (task_id, status, started_at, ended_at, "
            "outcome, summary) VALUES (?, 'done', ?, ?, 'completed', ?)",
            (t, int(time.time()) + 1, int(time.time()) + 2, ""),
        )
        conn.commit()
        assert kb.latest_summary(conn, t) == "real handoff"


def test_latest_summaries_batch_omits_tasks_without_summary(kanban_home):
    """``latest_summaries`` is the dashboard's N+1 escape hatch — it
    must return only entries for tasks that actually have a summary,
    keep the per-task latest, and accept an empty input gracefully."""
    with kb.connect() as conn:
        t1 = kb.create_task(conn, title="a", assignee="alice")
        t2 = kb.create_task(conn, title="b", assignee="bob")
        t3 = kb.create_task(conn, title="c", assignee="carol")
        kb.complete_task(conn, t1, summary="alpha")
        kb.complete_task(conn, t3, summary="charlie")
        out = kb.latest_summaries(conn, [t1, t2, t3])
        assert out == {t1: "alpha", t3: "charlie"}
        # Empty input → empty dict, no SQL syntax error from "IN ()".
        assert kb.latest_summaries(conn, []) == {}



# ---------------------------------------------------------------------------
# NFS / network-filesystem fallback (see hermes_state.apply_wal_with_fallback)
# ---------------------------------------------------------------------------

def test_connect_falls_back_to_delete_on_locking_protocol(kanban_home, caplog):
    """kanban_db.connect() must handle ``locking protocol`` on NFS/SMB.

    Without this fallback, the gateway's kanban dispatcher crashes every
    60s and the kanban migration (``consecutive_failures`` ADD COLUMN) is
    retried forever — which is what the real-world user report shows
    (see hermes-agent issue #22032).
    """
    import sqlite3 as _sqlite3
    from unittest.mock import patch as _patch

    # Clear module cache so a fresh connect() is attempted
    kb._INITIALIZED_PATHS.clear()

    real_connect = _sqlite3.connect

    class _WalBlockingConnection(_sqlite3.Connection):
        def execute(self, sql, *args, **kwargs):  # type: ignore[override]
            if "journal_mode=wal" in sql.lower().replace(" ", ""):
                raise _sqlite3.OperationalError("locking protocol")
            return super().execute(sql, *args, **kwargs)

    def wal_blocking_connect(*args, **kwargs):
        return real_connect(
            *args, factory=_WalBlockingConnection, **kwargs
        )

    with _patch("hermes_cli.kanban_db.sqlite3.connect", side_effect=wal_blocking_connect):
        with caplog.at_level("WARNING", logger="hermes_state"):
            conn = kb.connect()

    # One fallback warning, naming kanban.db
    warnings = [
        r for r in caplog.records
        if r.levelname == "WARNING" and "kanban.db" in r.getMessage()
    ]
    assert len(warnings) >= 1, (
        f"Expected a kanban.db WARNING, got: {[r.getMessage() for r in caplog.records]}"
    )

    # DB still usable end-to-end — create + list a task
    t = kb.create_task(conn, title="post-fallback task")
    tasks = kb.list_tasks(conn)
    assert any(row.id == t for row in tasks)
    conn.close()


def test_unlink_tasks_triggers_recompute_ready(kanban_home):
    """Regression test for issue #22459.

    Removing a dependency via unlink_tasks must immediately promote the child
    to ready when all remaining parents are done — same contract as
    complete_task and unblock_task.

    Before the fix, child stayed 'todo' indefinitely after unlink; only the
    next dispatcher tick or a manual 'hermes kanban recompute' would promote it.
    """
    with kb.connect() as conn:
        # A is done.
        a = kb.create_task(conn, title="parent-done")
        kb.complete_task(conn, a)

        # C is running (not done) — blocks child B.
        c = kb.create_task(conn, title="parent-running")
        kb.claim_task(conn, c, claimer="worker:1")

        # B depends on both A (done) and C (running) → stays todo.
        b = kb.create_task(conn, title="child", parents=[a, c])
        assert kb.get_task(conn, b).status == "todo"

        # Remove the blocking dependency C → B.
        removed = kb.unlink_tasks(conn, c, b)
        assert removed is True

        # B's only remaining parent is A (done) → must be ready immediately.
        assert kb.get_task(conn, b).status == "ready", (
            "child should promote to ready immediately after unlink_tasks "
            "removes its last blocking dependency"
        )
# ---------------------------------------------------------------------------
# _add_column_if_missing / _migrate_add_optional_columns idempotency (#21708)
# ---------------------------------------------------------------------------

def test_add_column_if_missing_is_idempotent_on_race(kanban_home):
    """``_add_column_if_missing`` must swallow 'duplicate column name' errors.

    Regression for #21708: the kanban dispatcher opens the DB twice per tick
    (once via _tick_once_for_board, once via init_db's discard-and-reconnect
    path).  A second concurrent connection runs _migrate_add_optional_columns
    before the first one commits, so ALTER TABLE raises OperationalError with
    'duplicate column name: consecutive_failures'.  Without the idempotency
    guard that crashes the dispatcher on the first tick after every restart.
    """
    import sqlite3

    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute(
        "CREATE TABLE tasks (id INTEGER PRIMARY KEY, title TEXT NOT NULL)"
    )

    # First call adds the column — returns True.
    added = kb._add_column_if_missing(conn, "tasks", "extra_col", "extra_col TEXT")
    assert added is True
    cols = {row["name"] for row in conn.execute("PRAGMA table_info(tasks)")}
    assert "extra_col" in cols

    # Second call on same connection — column already exists — must return
    # False without raising, simulating the race the dispatcher hits.
    added_again = kb._add_column_if_missing(
        conn, "tasks", "extra_col", "extra_col TEXT"
    )
    assert added_again is False

    conn.close()


def test_migrate_add_optional_columns_tolerates_concurrent_migration(kanban_home):
    """Full _migrate_add_optional_columns must not raise when columns already
    exist (issue #21708 race window — two connections migrate concurrently)."""
    import sqlite3

    # Schema already in fully-migrated state (all optional columns present).
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute(
        """
        CREATE TABLE tasks (
            id INTEGER PRIMARY KEY,
            title TEXT NOT NULL,
            tenant TEXT,
            result TEXT,
            idempotency_key TEXT,
            consecutive_failures INTEGER NOT NULL DEFAULT 0,
            worker_pid INTEGER,
            last_failure_error TEXT,
            max_runtime_seconds INTEGER,
            last_heartbeat_at INTEGER,
            current_run_id INTEGER,
            workflow_template_id TEXT,
            current_step_key TEXT,
            skills TEXT,
            max_retries INTEGER
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE task_events (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            task_id    TEXT NOT NULL DEFAULT '',
            run_id     INTEGER,
            kind       TEXT NOT NULL DEFAULT '',
            payload    TEXT,
            created_at INTEGER NOT NULL DEFAULT 0
        )
        """
    )

    # Running migration on an already-migrated schema must not raise.
    kb._migrate_add_optional_columns(conn)
    conn.close()


# ---------------------------------------------------------------------------
# Dispatcher spawn invocation — _resolve_hermes_argv()
#
# Workers spawned by the dispatcher must use a `hermes` invocation that does
# not depend on PATH being set up correctly. cron jobs, systemd User= services,
# launchd jobs, and other detached processes routinely run with a stripped
# $PATH that doesn't include the venv's bin/, so a bare `["hermes", ...]`
# spawn fails with FileNotFoundError and the task gets stuck. The resolver
# prefers the PATH shim (familiar `ps` output) but falls back to the module
# form so the spawn keeps working when PATH is missing the shim.
# ---------------------------------------------------------------------------


def test_resolve_hermes_argv_prefers_path_shim(monkeypatch):
    """When `hermes` is on PATH, use the shim — preserves familiar ps output."""
    import shutil
    import hermes_cli.kanban_db as kb

    monkeypatch.setattr(shutil, "which", lambda name: "/usr/local/bin/hermes")
    argv = kb._resolve_hermes_argv()
    assert argv == ["/usr/local/bin/hermes"]


def test_resolve_hermes_argv_falls_back_to_module_form_when_no_path_shim(monkeypatch):
    """When the shim is not on PATH, fall back to `python -m hermes_cli.main`.

    Pins the correct module name (NOT `hermes` — there is no top-level
    `hermes` package). Regression for #23198: the original PR shipped
    `python -m hermes` which fails with `No module named hermes` on every
    invocation.
    """
    import shutil
    import sys
    import hermes_cli.kanban_db as kb

    monkeypatch.setattr(shutil, "which", lambda name: None)
    argv = kb._resolve_hermes_argv()
    assert argv == [sys.executable, "-m", "hermes_cli.main"]


def test_resolve_hermes_argv_module_actually_runs():
    """The fallback module name must be importable + runnable.

    A unit test that pins the literal string is necessary but not
    sufficient — if `hermes_cli.main` ever loses `if __name__ == "__main__"`
    handling or its argparse setup, `python -m hermes_cli.main --version`
    would fail and so would every dispatcher spawn that hits the fallback.
    Run it as a real subprocess to catch that regression.
    """
    import subprocess
    import sys
    import hermes_cli.kanban_db as kb
    import shutil
    import unittest.mock as mock

    with mock.patch.object(shutil, "which", return_value=None):
        argv = kb._resolve_hermes_argv()
    r = subprocess.run(argv + ["--version"], capture_output=True, text=True, timeout=30)
    assert r.returncode == 0, (
        f"`{' '.join(argv)} --version` failed (rc={r.returncode}); "
        f"stderr={r.stderr[:200]!r}"
    )
    assert "Hermes Agent" in r.stdout, f"unexpected output: {r.stdout[:200]!r}"


# ---------------------------------------------------------------------------
# task_age — guard against corrupt timestamp values
#
# The Task dataclass declares ``created_at: int`` but rows come from sqlite
# without coercion at the boundary. A row that ever held a non-int (e.g. an
# unsubstituted ``'%s'`` from a logged format string, ``None``, an arbitrary
# string, or a float-as-string) used to crash ``task_age`` with ``ValueError``
# and turn ``GET /api/plugins/kanban/board`` into a 500 because the dashboard
# calls ``task_age`` unguarded for every task in the response.
#
# After the fix, ``_safe_int`` returns ``None`` on bad input and ``task_age``
# degrades gracefully (per-field ``None`` rather than a hard crash).
# ---------------------------------------------------------------------------


def _make_task(**overrides) -> "kb.Task":
    """Minimal Task with all required fields filled in. Override anything."""
    defaults = dict(
        id="t_age",
        title="x",
        body=None,
        assignee=None,
        status="ready",
        priority=0,
        created_by=None,
        created_at=0,
        started_at=None,
        completed_at=None,
        workspace_kind="scratch",
        workspace_path=None,
        claim_lock=None,
        claim_expires=None,
        tenant=None,
    )
    defaults.update(overrides)
    return kb.Task(**defaults)


def test_safe_int_accepts_int_and_int_string():
    """Sanity: well-typed values pass through."""
    assert kb._safe_int(0) == 0
    assert kb._safe_int(1700000000) == 1700000000
    assert kb._safe_int("1700000000") == 1700000000


def test_safe_int_returns_none_on_corrupt_inputs():
    """All the failure modes that used to crash task_age."""
    # None — common when the column was never written
    assert kb._safe_int(None) is None
    # Unsubstituted format string — the literal case the PR title cites
    assert kb._safe_int("%s") is None
    # Arbitrary non-numeric strings
    assert kb._safe_int("abc") is None
    assert kb._safe_int("") is None
    # Float-ish strings: int("1.5") raises ValueError too — caller wants None.
    assert kb._safe_int("1.5") is None
    # Random object — covered by TypeError branch
    assert kb._safe_int(object()) is None


def test_task_age_handles_corrupt_created_at():
    """Pre-fix this raised ValueError and 500'd /api/plugins/kanban/board."""
    t = _make_task(created_at="%s")
    age = kb.task_age(t)
    assert age["created_age_seconds"] is None
    assert age["started_age_seconds"] is None
    assert age["time_to_complete_seconds"] is None


def test_task_age_handles_corrupt_started_and_completed():
    """All three timestamp fields share the same _safe_int treatment."""
    t = _make_task(
        created_at=1700000000,
        started_at="garbage",
        completed_at=None,
    )
    age = kb.task_age(t)
    assert isinstance(age["created_age_seconds"], int)
    assert age["started_age_seconds"] is None
    assert age["time_to_complete_seconds"] is None


def test_task_age_well_formed_task():
    """Regression: the safe-int path must not change behavior for normal data."""
    import time
    now = int(time.time())
    t = _make_task(
        created_at=now - 60,
        started_at=now - 30,
        completed_at=now,
    )
    age = kb.task_age(t)
    assert 55 <= age["created_age_seconds"] <= 65
    assert 25 <= age["started_age_seconds"] <= 35
    assert 25 <= age["time_to_complete_seconds"] <= 35


def test_task_dict_survives_corrupt_created_at(tmp_path, monkeypatch):
    """Defense in depth: even if task_age ever raised, plugin_api must not 500.

    The PR also added a try/except around the task_age call in
    `plugins/kanban/dashboard/plugin_api.py::_task_dict`. Verify a single
    corrupt row doesn't turn the whole board response into an error.
    """
    # Set up an isolated kanban home so we can write a corrupt created_at.
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    kb._INITIALIZED_PATHS.clear()
    kb.init_db()

    # Insert a row with a non-int created_at (simulates the historical
    # bug that produced corrupt rows).
    conn = kb.connect()
    try:
        good_id = kb.create_task(conn, title="good")
        # Now write a row with corrupt created_at directly.
        conn.execute(
            "UPDATE tasks SET created_at = ? WHERE id = ?",
            ("%s", good_id),
        )
    finally:
        conn.close()

    # Re-read and pass through task_age — must not raise.
    conn = kb.connect()
    try:
        task = kb.get_task(conn, good_id)
    finally:
        conn.close()
    age = kb.task_age(task)
    assert age["created_age_seconds"] is None


def test_dispatch_preflight_blocks_scope_v2_with_unknown_assignee(kanban_home, monkeypatch):
    """Scope v2 task with unknown assignee must be blocked, not silently skipped.

    Regression for autonomy-sprint 2026-05-08 finding: planner produced child
    cards with assignees ``researcher``/``analyst``/``reviewer`` outside the
    OS-spec ``{default|coder|admin}`` set. Old behavior silently skipped them,
    leaving the chain stuck in ``ready`` forever.
    """
    from hermes_cli import profiles
    monkeypatch.setattr(profiles, "profile_exists", lambda name: name in {"default", "admin", "coder"})
    body = """
scope_contract:
  version: 2
  allowed_tools:
    - kanban_show
    - kanban_complete
  forbidden_systems:
    - OpenClaw
    - Atlas
    - Mission-Control
    - Telegram
completion_policy:
  require_scope_attestation: true
"""
    spawns = []
    with kb.connect() as conn:
        t = kb.create_task(conn, title="planner-bug task", assignee="researcher", body=body)
        res = kb.dispatch_once(conn, spawn_fn=lambda task, workspace: spawns.append(task.id))
        assert not spawns
        assert t in res.preflight_blocked
        assert t not in res.skipped_nonspawnable
        task = kb.get_task(conn, t)
        assert task.status == "blocked"
        assert "researcher" in (task.result or "")
        assert "not a known Hermes profile" in (task.result or "")
        events = [r["kind"] for r in conn.execute(
            "SELECT kind FROM task_events WHERE task_id = ? ORDER BY id", (t,)
        ).fetchall()]
        assert "dispatch_preflight_invalid_assignee" in events


def test_dispatch_preflight_skips_unknown_assignee_without_scope_contract(kanban_home, monkeypatch):
    """Tasks WITHOUT scope_contract keep human-lane skip behavior (regression guard).

    Interactive Claude Code terminals (e.g. ``orion-cc``) and other
    human-pulled lanes deliberately use assignees that don't map to
    Hermes profiles. They must continue to be bucketed as
    ``skipped_nonspawnable`` rather than blocked.
    """
    from hermes_cli import profiles
    monkeypatch.setattr(profiles, "profile_exists", lambda name: name in {"default", "admin", "coder"})
    body = "Plain task body with no scope contract."
    spawns = []
    with kb.connect() as conn:
        t = kb.create_task(conn, title="orion-cc lane task", assignee="orion-cc", body=body)
        res = kb.dispatch_once(conn, spawn_fn=lambda task, workspace: spawns.append(task.id))
        assert not spawns
        assert t in res.skipped_nonspawnable
        assert t not in res.preflight_blocked
        task = kb.get_task(conn, t)
        assert task.status == "ready"  # still ready, not blocked


def test_dispatch_preflight_allows_known_profile_with_scope_v2(kanban_home, monkeypatch):
    """Scope v2 task with valid assignee passes preflight and spawns."""
    from hermes_cli import profiles
    monkeypatch.setattr(profiles, "profile_exists", lambda name: name in {"default", "admin", "coder"})
    body = """
scope_contract:
  version: 2
  allowed_tools:
    - kanban_show
    - kanban_complete
    - kanban_block
  forbidden_systems:
    - OpenClaw
    - Atlas
    - Mission-Control
    - Telegram
completion_policy:
  require_scope_attestation: true
"""
    spawns = []
    with kb.connect() as conn:
        t = kb.create_task(conn, title="valid-assignee task", assignee="admin", body=body)
        res = kb.dispatch_once(conn, spawn_fn=lambda task, workspace: spawns.append(task.id))
        assert spawns == [t]
        assert t not in res.preflight_blocked
        assert t not in res.skipped_nonspawnable


# ---------------------------------------------------------------------------
# Review lane classification
# ---------------------------------------------------------------------------

class TestClassifyKanbanReviewLane:
    """Tests for classify_kanban_review_lane()."""

    def test_fastlane_default_for_plain_kanban_task(self):
        """No critical/standard triggers → FASTLANE_KANBAN."""
        result = kb.classify_kanban_review_lane(
            title="Polish error messages in hub response",
            body="Tweak wording in Hub responses.\nNo system-level changes.",
        )
        assert result["lane"] == "FASTLANE_KANBAN"
        assert result["risk"] == "low"
        assert result["hub_coordinator_evidence_check_required"] is True
        assert result["reviewer_a_required"] is False
        assert result["reviewer_b_required"] is False
        assert any("default" in reason for reason in result["reasons"])

    def test_fastlane_explicit_request_kept(self):
        """Explicit request keeps FASTLANE when no triggers fire."""
        result = kb.classify_kanban_review_lane(
            title="Refactor kanban tool handlers",
            body="Extract shared logic from _handle_* functions.",
            requested_lane="FASTLANE_KANBAN",
        )
        assert result["lane"] == "FASTLANE_KANBAN"

    def test_forbidden_systems_do_not_escalate_fastlane_scope_contract(self):
        """Negative scope declarations must not trip critical text matching."""
        body = """
review_lane: FASTLANE_KANBAN
scope_contract:
  version: 2
  allowed_systems: [hermes-agent, hermes-kanban]
  forbidden_systems: [OpenClaw, Atlas, Mission-Control, Telegram]
  anti_scope:
    - keine OpenClaw-Touches
    - keine Mission-Control-Mutation
"""
        result = kb.classify_kanban_review_lane(
            title="Write scratchpad note",
            body=body,
        )
        assert result["lane"] == "FASTLANE_KANBAN"
        assert "critical_allowed_system_or_text" not in result["escalation_triggers"]

    def test_forbidden_systems_do_not_hide_standard_review_trigger(self):
        """Structured negative scope keeps STANDARD triggers observable."""
        body = """
review_lane: STANDARD_REVIEW
Diese Probe testet task-lifecycle semantics.
scope_contract:
  version: 2
  allowed_systems: [hermes-agent, hermes-kanban]
  forbidden_systems: [OpenClaw, Atlas, Mission-Control, Telegram]
  anti_scope: [keine Mission-Control-Mutation]
"""
        result = kb.classify_kanban_review_lane(
            title="Lane validation standard probe",
            body=body,
        )
        assert result["lane"] == "STANDARD_REVIEW"
        assert "critical_allowed_system_or_text" not in result["escalation_triggers"]

    def test_standard_review_escalates_on_lifecycle_terms(self):
        """Lifecycle / dispatcher semantics → STANDARD_REVIEW."""
        result = kb.classify_kanban_review_lane(
            title="Extend task-lifecycle policy with retry semantics",
            body="Add retry_after_seconds to completion_policy.\nDispatcher should handle retry dispatch.",
        )
        assert result["lane"] == "STANDARD_REVIEW"
        assert result["risk"] == "medium"
        assert result["hub_coordinator_evidence_check_required"] is False
        assert result["reviewer_a_required"] is False
        assert result["reviewer_b_required"] is True

    def test_standard_review_escalates_on_kanban_db_semantics(self):
        """kanban-db-semantics in body → STANDARD_REVIEW."""
        result = kb.classify_kanban_review_lane(
            title="Add computed status field",
            body="Add a computed read-only field to the task schema.\nkanban-db-semantics change.",
        )
        assert result["lane"] == "STANDARD_REVIEW"

    def test_standard_review_explicit_request_escalates(self):
        """Explicit STANDARD_REVIEW request is honored."""
        result = kb.classify_kanban_review_lane(
            title="Minor tweak to review trigger logic",
            body="Small refactor in hub.",
            requested_lane="STANDARD_REVIEW",
        )
        assert result["lane"] == "STANDARD_REVIEW"

    def test_critical_review_on_runtime_activation(self):
        """Runtime activation term → CRITICAL_REVIEW."""
        result = kb.classify_kanban_review_lane(
            title="Activate gateway runtime on startup",
            body="Add gateway-runtime activation in startup sequence.",
        )
        assert result["lane"] == "CRITICAL_REVIEW"
        assert result["risk"] == "high"
        assert result["hub_coordinator_evidence_check_required"] is False
        assert result["reviewer_a_required"] is True
        assert result["reviewer_b_required"] is True

    def test_critical_review_on_openclaw_path(self):
        """OpenClaw path marker → CRITICAL_REVIEW."""
        result = kb.classify_kanban_review_lane(
            title="Update openclaw model catalog",
            body="Add new model entry.",
            changed_paths=["/home/piet/.openclaw/openclaw.json"],
        )
        assert result["lane"] == "CRITICAL_REVIEW"
        assert result["risk"] == "high"

    def test_critical_review_on_secrets_path(self):
        """Secrets path → CRITICAL_REVIEW."""
        result = kb.classify_kanban_review_lane(
            title="Rotate Discord webhook token",
            body="Update stored token.",
            changed_paths=["/home/piet/.hermes/auth.json"],
        )
        assert result["lane"] == "CRITICAL_REVIEW"
        assert result["risk"] == "high"

    def test_critical_review_on_restart_deploy(self):
        """Restart/deploy terms → CRITICAL_REVIEW."""
        result = kb.classify_kanban_review_lane(
            title="Restart openclaw-gateway after config change",
            body="Restart the gateway service and smoke-test.",
        )
        assert result["lane"] == "CRITICAL_REVIEW"

    def test_critical_review_on_mission_control_path(self):
        """Mission Control path → CRITICAL_REVIEW."""
        result = kb.classify_kanban_review_lane(
            title="Update MC dispatch priority",
            body="Change priority field in MC.",
            changed_paths=["/home/piet/.openclaw/workspace/mission-control/src/dispatch.py"],
        )
        assert result["lane"] == "CRITICAL_REVIEW"

    def test_critical_review_on_telegram(self):
        """Telegram term → CRITICAL_REVIEW."""
        result = kb.classify_kanban_review_lane(
            title="Add Telegram notification channel",
            body="Wire up Telegram bot for alert delivery.",
        )
        assert result["lane"] == "CRITICAL_REVIEW"

    def test_critical_review_on_atlas(self):
        """Atlas term → CRITICAL_REVIEW."""
        result = kb.classify_kanban_review_lane(
            title="Hook Atlas executor into dispatch loop",
            body="Add Atlas call in dispatcher.",
        )
        assert result["lane"] == "CRITICAL_REVIEW"

    def test_explicit_lane_can_escalate_not_downgrade(self):
        """Explicit FASTLANE cannot downgrade CRITICAL trigger to FASTLANE."""
        result = kb.classify_kanban_review_lane(
            title="Restart gateway with new config",
            body="Restart openclaw-gateway.service",
            requested_lane="FASTLANE_KANBAN",
            changed_paths=["/home/piet/.openclaw/openclaw.json"],
        )
        assert result["lane"] == "CRITICAL_REVIEW"  # escalation wins

    def test_standard_alias_normalized(self):
        """STANDARD alias → STANDARD_REVIEW."""
        result = kb.classify_kanban_review_lane(
            title="Update dispatcher lifecycle",
            body="Add dispatcher retry logic.",
            requested_lane="STANDARD",
        )
        assert result["lane"] == "STANDARD_REVIEW"

    def test_fastlane_alias_normalized(self):
        """FASTLANE alias → FASTLANE_KANBAN."""
        result = kb.classify_kanban_review_lane(
            title="Tweak error wording",
            body="No system changes.",
            requested_lane="FASTLANE",
        )
        assert result["lane"] == "FASTLANE_KANBAN"

    def test_invalid_requested_lane_falls_back_to_computed(self):
        """Invalid lane name → falls back to computed lane."""
        result = kb.classify_kanban_review_lane(
            title="Refactor",
            body="Small refactor.",
            requested_lane="INVALID_LANE",
        )
        assert result["lane"] == "FASTLANE_KANBAN"  # computed default

    def test_fastlane_with_only_kanban_tool_changes(self):
        """Kanban-only tool changes → FASTLANE_KANBAN."""
        result = kb.classify_kanban_review_lane(
            title="Add kanban_history tool",
            body="Add new kanban tool for history browsing.\nallowed_tools: kanban_show, kanban_history.",
            changed_paths=["/home/piet/.hermes/hermes-agent/tools/kanban_tools.py"],
        )
        # Tool changes in hermes-agent (not system paths) → not auto-escalated
        assert result["lane"] == "FASTLANE_KANBAN"

    def test_critical_trumps_standard_term(self):
        """Critical triggers override standard terms."""
        result = kb.classify_kanban_review_lane(
            title="Restart Atlas after dispatcher change",
            body="Dispatcher lifecycle update then restart Atlas.",
        )
        assert result["lane"] == "CRITICAL_REVIEW"

    def test_plain_restart_or_deploy_words_do_not_force_critical(self):
        """Plain words like restart/deploy in notes stay FASTLANE absent hard triggers."""
        result = kb.classify_kanban_review_lane(
            title="Document deploy notes for kanban workers",
            body="Add restart checklist wording to docs; no service or runtime changes.",
        )
        assert result["lane"] == "FASTLANE_KANBAN"
        assert result["reviewer_b_required"] is False
        assert result["hub_coordinator_evidence_check_required"] is True

    def test_policy_review_lane_standard_requires_reviewer_b(self):
        """Structured body policy can request STANDARD_REVIEW."""
        result = kb.classify_kanban_review_lane(
            title="Small wording tweak",
            body="""
review_lane: STANDARD_REVIEW
summary: Small wording tweak, but request one reviewer.
""",
        )
        assert result["lane"] == "STANDARD_REVIEW"
        assert result["reviewer_a_required"] is False
        assert result["reviewer_b_required"] is True
        assert result["hub_coordinator_evidence_check_required"] is False

    def test_policy_review_lane_critical_requires_reviewer_a_and_b(self):
        """Structured body policy can request CRITICAL_REVIEW."""
        result = kb.classify_kanban_review_lane(
            title="Small wording tweak",
            body="""
review_lane: CRITICAL_REVIEW
summary: Explicitly request critical review.
""",
        )
        assert result["lane"] == "CRITICAL_REVIEW"
        assert result["reviewer_a_required"] is True
        assert result["reviewer_b_required"] is True
        assert result["hub_coordinator_evidence_check_required"] is False

    def test_scope_contract_allowed_systems_restart_is_critical(self):
        """Structured allowed_systems hard-trigger CRITICAL_REVIEW."""
        result = kb.classify_kanban_review_lane(
            title="Scoped restart action",
            body="""
scope_contract:
  allowed_systems:
    - restart
""",
        )
        assert result["lane"] == "CRITICAL_REVIEW"
        assert result["reviewer_a_required"] is True
        assert result["reviewer_b_required"] is True

    def test_critical_alias_normalized(self):
        """CRITICAL alias → CRITICAL_REVIEW."""
        result = kb.classify_kanban_review_lane(
            title="Force strict review",
            body="Small refactor.",
            requested_lane="CRITICAL",
        )
        assert result["lane"] == "CRITICAL_REVIEW"
        assert result["reviewer_a_required"] is True
        assert result["reviewer_b_required"] is True

    def test_explicit_fastlane_cannot_downgrade_standard_trigger(self):
        """Explicit FASTLANE cannot downgrade STANDARD lifecycle semantics."""
        result = kb.classify_kanban_review_lane(
            title="Update dispatcher lifecycle",
            body="Change lifecycle behavior for task-links.",
            requested_lane="FASTLANE_KANBAN",
        )
        assert result["lane"] == "STANDARD_REVIEW"
        assert result["reviewer_b_required"] is True

    def test_empty_title_and_body_defaults_to_fastlane(self):
        """No title/body → FASTLANE_KANBAN."""
        result = kb.classify_kanban_review_lane(title="", body="")
        assert result["lane"] == "FASTLANE_KANBAN"
        assert result["risk"] == "low"
