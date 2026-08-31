"""Behavior tests for deterministic Kanban worker-health supervision."""

from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import projects_db as pdb
from hermes_cli.kanban_worker_watchdog import (
    WatchdogConfig,
    WatchdogTickResult,
    config_from_runtime_config,
    detect_log_finding,
    run_watchdog_tick,
)


def _config() -> WatchdogConfig:
    return WatchdogConfig(
        enabled=True,
        grace_seconds=0,
        repeat_threshold=3,
        compaction_threshold=3,
        reasoning_repeat_threshold=3,
        min_reasoning_chars=60,
    )


@pytest.fixture
def kanban_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _running_task(conn, workspace: Path, *, body: str = "bounded work"):
    task_id = kb.create_task(
        conn,
        title="Original worker task",
        body=body,
        assignee="implementation-worker",
        workspace_kind="dir",
        workspace_path=str(workspace),
    )
    claimed = kb.claim_task(conn, task_id)
    assert claimed is not None
    kb._set_worker_pid(conn, task_id, 424242)
    return kb.get_task(conn, task_id)


def _verified_termination(_pid, _claim_lock):
    return {
        "prev_pid": 424242,
        "host_local": True,
        "termination_attempted": True,
        "terminated": True,
        "sigkill": False,
    }


def test_repeated_failed_tool_call_is_detected() -> None:
    """Removing failed-call normalization would let identical failures spin."""
    log = "\n".join([
        "┊ 💻 $ rg missing.file  0.1s [exit 2]",
        "I will try that exact lookup again.",
        "┊ 💻 $ rg missing.file  0.1s [exit 2]",
        "Perhaps the same command will work now.",
        "┊ 💻 $ rg missing.file  0.2s [exit 2]",
    ])

    finding = detect_log_finding(log, _config())

    assert finding is not None
    assert finding.category == "tool_failure_loop"
    assert finding.count == 3
    assert len(finding.fingerprint) == 16


def test_repeated_context_compression_is_detected() -> None:
    """Ignoring compression count would allow context churn to run forever."""
    log = "\n".join([
        "📦 Pre-API compression: ~51,000 tokens near the limit.",
        "🗜️ Compacting context — summarizing earlier conversation so I can continue...",
        "useful tool progress",
        "📦 Pre-API compression: ~52,000 tokens near the limit.",
        "🗜️ Compacting context — summarizing earlier conversation so I can continue...",
        "more output",
        "📦 Pre-API compression: ~49,500 tokens near the limit.",
        "🗜️ Compacting context — summarizing earlier conversation so I can continue...",
    ])

    finding = detect_log_finding(log, _config())

    assert finding is not None
    assert finding.category == "compaction_loop"
    assert finding.count == 3


def test_repeated_provider_stall_is_detected() -> None:
    """Dropping provider-stall recognition would leave unresponsive calls alive."""
    log = "\n".join([
        "⏳ waiting on qwen3.5:4b — 30s with no output yet",
        "⏳ waiting on qwen3.5:4b — 60s with no output yet",
        "Provider has been unresponsive for 2 consecutive stale attempts",
    ])

    finding = detect_log_finding(log, _config())

    assert finding is not None
    assert finding.category == "provider_stall_loop"
    assert finding.count == 3


def test_repeated_long_reasoning_without_tool_progress_is_detected() -> None:
    """Removing reasoning fingerprints would miss semantic no-progress loops."""
    paragraph = (
        "I need to compare the same two equivalent strategies again before choosing; "
        "both approaches preserve the same behavior and neither changes the worktree."
    )
    log = f"{paragraph}\n\n{paragraph}\n\n{paragraph}"

    finding = detect_log_finding(log, _config())

    assert finding is not None
    assert finding.category == "reasoning_loop"
    assert finding.count == 3


def test_unique_progress_and_single_compression_are_healthy() -> None:
    """Over-broad matching would block a healthy worker doing distinct work."""
    log = "\n".join([
        "📦 Pre-API compression: ~51,000 tokens near the limit.",
        "🗜️ Compacting context — summarizing earlier conversation so I can continue...",
        "┊ 💻 $ rg -n target module_a.py  0.1s",
        "I found the owner and will patch the bounded function.",
        "┊ 🩹 patched module_a.py",
        "┊ 💻 $ scripts/run_tests.sh tests/test_a.py -q  2.0s",
        "All focused tests passed and the worktree contains one intended change.",
    ])

    assert detect_log_finding(log, _config()) is None


def test_watchdog_ignores_failure_loop_from_a_previous_task_run(
    kanban_home: Path, tmp_path: Path
) -> None:
    """An append-only task log must not make a repaired run inherit old failures."""
    config = WatchdogConfig(
        enabled=True,
        grace_seconds=0,
        repeat_threshold=3,
        repair_profiles={"tool_failure_loop": "tooling-repair"},
    )

    with kb.connect() as conn:
        current = _running_task(conn, tmp_path)
        stale_failures = "\n".join(
            ["┊ 💻 $ python /tmp/probe.py 0.1s [exit 1]"] * 3
        )
        log = "\n".join(
            [
                stale_failures,
                f"Query: work kanban task {current.id}",
                "Initializing agent...",
                "┊ 💻 $ python /tmp/probe.py 0.4s",
                "Focused verification passed.",
            ]
        )

        result = run_watchdog_tick(
            conn,
            config=config,
            now=current.started_at + 1,
            read_log_fn=lambda *_args, **_kwargs: log,
            terminate_fn=_verified_termination,
        )

        assert result.blocked == []
        assert kb.get_task(conn, current.id).status == "running"


def test_watchdog_blocks_worker_and_creates_one_linked_repair(
    kanban_home: Path, tmp_path: Path
) -> None:
    """Removing idempotent repair linkage would fan out duplicate repairs."""
    config = WatchdogConfig(
        enabled=True,
        grace_seconds=0,
        repeat_threshold=3,
        repair_profiles={"tool_failure_loop": "tooling-repair"},
    )
    unhealthy_log = "\n".join(["┊ 💻 $ rg missing 0.1s [exit 2]"] * 3)

    with kb.connect() as conn:
        original = _running_task(conn, tmp_path)
        first = run_watchdog_tick(
            conn,
            config=config,
            now=original.started_at + 1,
            read_log_fn=lambda *_args, **_kwargs: unhealthy_log,
            terminate_fn=_verified_termination,
        )
        second = run_watchdog_tick(
            conn,
            config=config,
            now=original.started_at + 2,
            read_log_fn=lambda *_args, **_kwargs: unhealthy_log,
            terminate_fn=_verified_termination,
        )

        assert first.blocked == [original.id]
        assert second.blocked == []
        assert kb.get_task(conn, original.id).status == "blocked"
        repairs = conn.execute(
            "SELECT id, assignee, status FROM tasks "
            "WHERE created_by = 'worker-health-watchdog'"
        ).fetchall()
        assert [(row["assignee"], row["status"]) for row in repairs] == [
            ("tooling-repair", "ready")
        ]
        link = conn.execute("SELECT parent_id, child_id FROM task_links").fetchone()
        assert (link["parent_id"], link["child_id"]) == (repairs[0]["id"], original.id)


def test_repair_borrows_original_workspace_without_owning_cleanup(
    kanban_home: Path, tmp_path: Path
) -> None:
    """Completing a repair must never delete the original task's workspace."""
    config = WatchdogConfig(
        enabled=True,
        grace_seconds=0,
        repeat_threshold=3,
        repair_profiles={"tool_failure_loop": "tooling-repair"},
    )
    unhealthy_log = "\n".join(["┊ 💻 $ rg missing 0.1s [exit 2]"] * 3)
    scratch = tmp_path / "original-scratch"
    scratch.mkdir()

    with kb.connect() as conn:
        task_id = kb.create_task(
            conn,
            title="Original scratch task",
            body="bounded work",
            assignee="implementation-worker",
            workspace_kind="scratch",
            workspace_path=str(scratch),
        )
        original = kb.claim_task(conn, task_id)
        assert original is not None
        kb._set_worker_pid(conn, task_id, 424242)

        run_watchdog_tick(
            conn,
            config=config,
            now=original.started_at + 1,
            read_log_fn=lambda *_args, **_kwargs: unhealthy_log,
            terminate_fn=_verified_termination,
        )

        repair = conn.execute(
            "SELECT workspace_kind, workspace_path, branch_name FROM tasks "
            "WHERE created_by = 'worker-health-watchdog'"
        ).fetchone()
        assert repair["workspace_kind"] == "dir"
        assert repair["workspace_path"] == str(scratch)
        assert repair["branch_name"] is None


def test_compaction_repair_uses_clean_scratch_not_conflicted_original_workspace(
    kanban_home: Path, tmp_path: Path
) -> None:
    """Infrastructure repair must not confuse source conflicts with its own failure."""
    config = WatchdogConfig(
        enabled=True,
        grace_seconds=0,
        compaction_threshold=3,
        repair_profiles={"compaction_loop": "task-orchestrator"},
    )
    unhealthy_log = "\n".join(["Compacting context — summarizing"] * 3)

    with kb.connect() as conn:
        original = _running_task(conn, tmp_path)
        run_watchdog_tick(
            conn,
            config=config,
            now=original.started_at + 1,
            read_log_fn=lambda *_args, **_kwargs: unhealthy_log,
            terminate_fn=_verified_termination,
        )

        repair = conn.execute(
            "SELECT workspace_kind, workspace_path FROM tasks "
            "WHERE created_by = 'worker-health-watchdog'"
        ).fetchone()
        assert repair["workspace_kind"] == "scratch"
        assert repair["workspace_path"] is None


def test_provider_stall_repair_borrows_project_workspace_for_route_diagnosis(
    kanban_home: Path, tmp_path: Path
) -> None:
    """Provider recovery needs the target checkout, but must never own it."""
    config = WatchdogConfig(
        enabled=True,
        grace_seconds=0,
        repeat_threshold=3,
        repair_profiles={"provider_stall_loop": "route-repair"},
    )
    unhealthy_log = "\n".join([
        "Provider has been unresponsive (no response received)",
    ] * 3)
    workspace = tmp_path / "project-worktree"
    workspace.mkdir()
    with pdb.connect_closing() as project_conn:
        project_id = pdb.create_project(
            project_conn,
            name="Watchdog Project",
            primary_path=str(workspace),
        )

    with kb.connect() as conn:
        task_id = kb.create_task(
            conn,
            title="Original project task",
            body="bounded work",
            assignee="implementation-worker",
            workspace_kind="worktree",
            workspace_path=str(workspace),
            project_id=project_id,
        )
        original = kb.claim_task(conn, task_id)
        assert original is not None
        kb._set_worker_pid(conn, task_id, 424242)

        run_watchdog_tick(
            conn,
            config=config,
            now=original.started_at + 1,
            read_log_fn=lambda *_args, **_kwargs: unhealthy_log,
            terminate_fn=_verified_termination,
        )

        repair = conn.execute(
            "SELECT workspace_kind, workspace_path, project_id, branch_name FROM tasks "
            "WHERE created_by = 'worker-health-watchdog'"
        ).fetchone()
        assert repair["workspace_kind"] == "dir"
        assert repair["workspace_path"] == str(workspace)
        assert repair["project_id"] == project_id
        assert repair["branch_name"] is None


def test_watchdog_waits_for_repair_then_restarts_original(
    kanban_home: Path, tmp_path: Path
) -> None:
    """Restarting before a done repair receipt would reproduce the same loop."""
    config = WatchdogConfig(
        enabled=True,
        grace_seconds=0,
        repeat_threshold=3,
        repair_profiles={"tool_failure_loop": "tooling-repair"},
    )
    unhealthy_log = "\n".join(["┊ 💻 $ rg missing 0.1s [exit 2]"] * 3)

    with kb.connect() as conn:
        original = _running_task(conn, tmp_path)
        run_watchdog_tick(
            conn,
            config=config,
            now=original.started_at + 1,
            read_log_fn=lambda *_args, **_kwargs: unhealthy_log,
            terminate_fn=_verified_termination,
        )
        repair_id = conn.execute(
            "SELECT id FROM tasks WHERE created_by = 'worker-health-watchdog'"
        ).fetchone()["id"]

        waiting = run_watchdog_tick(conn, config=config, now=original.started_at + 2)
        assert waiting.restarted == []
        assert kb.get_task(conn, original.id).status == "blocked"

        repair = kb.claim_task(conn, repair_id)
        assert repair is not None
        assert kb.complete_task(
            conn,
            repair_id,
            summary="tool invocation fixed and verified",
            expected_run_id=repair.current_run_id,
        )
        resumed = run_watchdog_tick(conn, config=config, now=original.started_at + 3)

        assert resumed.restarted == [original.id]
        assert kb.get_task(conn, original.id).status == "ready"


def test_watchdog_does_not_restart_a_newer_non_watchdog_block(
    kanban_home: Path, tmp_path: Path
) -> None:
    """A historical repair must not override a later operator safety block."""
    config = WatchdogConfig(
        enabled=True,
        grace_seconds=0,
        repeat_threshold=3,
        repair_profiles={"tool_failure_loop": "tooling-repair"},
    )
    unhealthy_log = "\n".join(["┊ 💻 $ rg missing 0.1s [exit 2]"] * 3)

    with kb.connect() as conn:
        original = _running_task(conn, tmp_path)
        run_watchdog_tick(
            conn,
            config=config,
            now=original.started_at + 1,
            read_log_fn=lambda *_args, **_kwargs: unhealthy_log,
            terminate_fn=_verified_termination,
        )
        repair_id = conn.execute(
            "SELECT id FROM tasks WHERE created_by = 'worker-health-watchdog'"
        ).fetchone()["id"]
        repair = kb.claim_task(conn, repair_id)
        assert repair is not None
        assert kb.complete_task(
            conn,
            repair_id,
            summary="historical tool repair completed",
            expected_run_id=repair.current_run_id,
        )

        assert kb.unblock_task(conn, original.id)
        retried = kb.claim_task(conn, original.id)
        assert retried is not None
        assert kb.block_task(
            conn,
            original.id,
            reason="new workspace collision safety block",
            kind="capability",
            expected_run_id=retried.current_run_id,
        )

        result = run_watchdog_tick(conn, config=config, now=original.started_at + 2)

        assert result.restarted == []
        assert result.needs_operator == [original.id]
        assert kb.get_task(conn, original.id).status == "blocked"


def test_watchdog_keeps_original_blocked_when_repair_fails(
    kanban_home: Path, tmp_path: Path
) -> None:
    """Treating a blocked repair as success would restart broken work."""
    config = WatchdogConfig(
        enabled=True,
        grace_seconds=0,
        repeat_threshold=3,
        repair_profiles={"tool_failure_loop": "tooling-repair"},
    )
    unhealthy_log = "\n".join(["┊ 💻 $ rg missing 0.1s [exit 2]"] * 3)

    with kb.connect() as conn:
        original = _running_task(conn, tmp_path)
        run_watchdog_tick(
            conn,
            config=config,
            now=original.started_at + 1,
            read_log_fn=lambda *_args, **_kwargs: unhealthy_log,
            terminate_fn=_verified_termination,
        )
        repair_id = conn.execute(
            "SELECT id FROM tasks WHERE created_by = 'worker-health-watchdog'"
        ).fetchone()["id"]
        repair = kb.claim_task(conn, repair_id)
        assert repair is not None
        assert kb.block_task(
            conn,
            repair_id,
            reason="repair verification failed",
            expected_run_id=repair.current_run_id,
        )

        result = run_watchdog_tick(conn, config=config, now=original.started_at + 2)

        assert result.needs_operator == [original.id]
        assert kb.get_task(conn, original.id).status == "blocked"


def test_watchdog_refuses_to_release_unterminated_worker(
    kanban_home: Path, tmp_path: Path
) -> None:
    """Releasing a surviving worker would spawn a duplicate beside it."""
    config = WatchdogConfig(
        enabled=True,
        grace_seconds=0,
        repeat_threshold=3,
        repair_profiles={"tool_failure_loop": "tooling-repair"},
    )
    unhealthy_log = "\n".join(["┊ 💻 $ rg missing 0.1s [exit 2]"] * 3)
    survived = {
        "prev_pid": 424242,
        "host_local": True,
        "termination_attempted": True,
        "terminated": False,
        "sigkill": True,
    }

    with kb.connect() as conn:
        original = _running_task(conn, tmp_path)
        result = run_watchdog_tick(
            conn,
            config=config,
            now=original.started_at + 1,
            read_log_fn=lambda *_args, **_kwargs: unhealthy_log,
            terminate_fn=lambda *_args: survived,
        )

        current = kb.get_task(conn, original.id)
        assert result.blocked == []
        assert result.needs_operator == [original.id]
        assert current.status == "running"
        assert current.claim_lock is not None
        assert current.worker_pid == 424242


def test_watchdog_stops_after_recovery_attempt_limit(
    kanban_home: Path, tmp_path: Path
) -> None:
    """Ignoring the recovery limit would move the loop into repair tasks."""
    config = WatchdogConfig(
        enabled=True,
        grace_seconds=0,
        repeat_threshold=3,
        max_recovery_attempts=1,
        repair_profiles={"tool_failure_loop": "tooling-repair"},
    )
    unhealthy_log = "\n".join(["┊ 💻 $ rg missing 0.1s [exit 2]"] * 3)

    with kb.connect() as conn:
        original = _running_task(conn, tmp_path)
        with kb.write_txn(conn):
            kb._append_event(
                conn,
                original.id,
                "watchdog_blocked",
                {"category": "tool_failure_loop", "fingerprint": "prior"},
                run_id=original.current_run_id,
            )

        result = run_watchdog_tick(
            conn,
            config=config,
            now=original.started_at + 1,
            read_log_fn=lambda *_args, **_kwargs: unhealthy_log,
            terminate_fn=_verified_termination,
        )

        assert result.blocked == [original.id]
        assert result.needs_operator == [original.id]
        assert kb.get_task(conn, original.id).status == "blocked"
        assert (
            conn.execute(
                "SELECT COUNT(*) FROM tasks WHERE created_by = 'worker-health-watchdog'"
            ).fetchone()[0]
            == 0
        )


def test_disabled_watchdog_does_not_read_worker_logs(kanban_home: Path) -> None:
    """Ignoring the feature gate would mutate existing boards on upgrade."""
    reads = []

    with kb.connect() as conn:
        result = run_watchdog_tick(
            conn,
            config=WatchdogConfig(enabled=False),
            read_log_fn=lambda *_args, **_kwargs: reads.append(True),
        )

    assert result == WatchdogTickResult()
    assert reads == []


def test_runtime_config_is_bounded_and_routes_each_finding() -> None:
    """Unbounded or dropped config values would make supervision unsafe."""
    config = config_from_runtime_config({
        "kanban": {
            "worker_watchdog": {
                "enabled": True,
                "grace_seconds": "900",
                "log_tail_bytes": 524288,
                "repeat_threshold": 4,
                "compaction_threshold": 5,
                "reasoning_repeat_threshold": 6,
                "max_recovery_attempts": 3,
                "repair_max_runtime_seconds": 1800,
                "repair_profiles": {
                    "tool_failure_loop": "terminal-repair",
                    "compaction_loop": "context-repair",
                    "provider_stall_loop": "provider-repair",
                    "reasoning_loop": "orchestration-repair",
                    "unknown": "must-be-ignored",
                },
            }
        }
    })

    assert config.enabled is True
    assert config.grace_seconds == 900
    assert config.log_tail_bytes == 524288
    assert config.repeat_threshold == 4
    assert config.compaction_threshold == 5
    assert config.reasoning_repeat_threshold == 6
    assert config.max_recovery_attempts == 3
    assert config.repair_max_runtime_seconds == 1800
    assert config.repair_profiles == {
        "tool_failure_loop": "terminal-repair",
        "compaction_loop": "context-repair",
        "provider_stall_loop": "provider-repair",
        "reasoning_loop": "orchestration-repair",
    }


def test_runtime_config_defaults_to_active_profile_for_unmapped_repairs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Enabled-by-default supervision must also have a runnable repair lane."""
    from hermes_cli import profiles

    monkeypatch.setattr(profiles, "get_active_profile_name", lambda: "maintainer")

    config = config_from_runtime_config({"kanban": {"worker_watchdog": {}}})

    assert config.enabled is True
    assert config.repair_profiles == {
        "tool_failure_loop": "maintainer",
        "compaction_loop": "maintainer",
        "provider_stall_loop": "maintainer",
        "reasoning_loop": "maintainer",
    }


def test_shipped_watchdog_default_is_enabled() -> None:
    """Fresh installs should supervise Kanban workers without an opt-in."""
    from hermes_cli.config_defaults import DEFAULT_CONFIG

    assert DEFAULT_CONFIG["kanban"]["worker_watchdog"]["enabled"] is True


def test_dispatch_result_surfaces_watchdog_progress(
    kanban_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Dropping tick results would make automatic recovery invisible."""
    import hermes_cli.kanban_worker_watchdog as watchdog

    monkeypatch.setattr(
        watchdog,
        "load_watchdog_config",
        lambda: WatchdogConfig(enabled=True),
    )
    monkeypatch.setattr(
        watchdog,
        "run_watchdog_tick",
        lambda *_args, **_kwargs: WatchdogTickResult(
            blocked=["t_blocked"],
            restarted=["t_restarted"],
            needs_operator=["t_operator"],
        ),
    )

    with kb.connect() as conn:
        result = kb.dispatch_once(conn, max_spawn=0)

    assert result.watchdog_blocked == ["t_blocked"]
    assert result.watchdog_restarted == ["t_restarted"]
    assert result.watchdog_needs_operator == ["t_operator"]
