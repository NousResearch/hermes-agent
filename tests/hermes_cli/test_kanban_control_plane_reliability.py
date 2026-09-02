"""Control-plane reliability regressions for Kanban dispatcher safety.

Covers the incident classes from the 2026-08/09 dead-worker / failover event:

A. One-shot dispatch / create without a persistent dispatcher must warn that
   later worker death will not be autonomously reconciled.
B. A host-local dead PID wins over a still-valid claim lease and a fresh
   heartbeat once launch grace has elapsed.
C. Crash/reconcile/retry must not mutate a dirty task workspace.
D. Classified transient provider failures must use the rate-limit sentinel
   (no circuit-breaker burn) without string-matching arbitrary errors.
E. Reassignment must retain model/provider overrides and make the effective
   execution identity unmistakable.
F. Explicit human authorization of an execution-policy exception is distinct
   from operational mutation events; workers cannot self-authorize.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_cli import kanban as kc
from hermes_cli import kanban_db as kb


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr(
        "hermes_cli.profiles.profile_exists",
        lambda name: bool(str(name or "").strip()),
    )
    kb._INITIALIZED_PATHS.discard(str(kb.kanban_db_path(board="default").resolve()))
    kb.init_db()
    return home


def _absent_gateway(monkeypatch):
    """Force the create/dispatch presence probe to observe no live gateway."""

    class _Liveness:
        pid = None
        probe_error = None

    monkeypatch.setattr(
        "gateway.status.resolve_gateway_liveness",
        lambda **_kwargs: _Liveness(),
    )


def _create_noncritical(conn, **kwargs):
    kwargs.setdefault("routing_criticality", "noncritical")
    kwargs.setdefault("routing_role", "noncritical")
    kwargs.setdefault("assignee", "worker")
    kwargs.setdefault("title", "cp-reliability")
    return kb.create_task(conn, **kwargs)


# ---------------------------------------------------------------------------
# A. Persistent dispatcher / dead-worker safety diagnostic
# ---------------------------------------------------------------------------


def test_create_warns_that_worker_death_will_not_be_reconciled_without_dispatcher(
    kanban_home, monkeypatch, capsys,
):
    """Create of a ready+assigned task with no gateway must warn about
    unreconciled later worker death, not merely that the card sits in ready."""
    _absent_gateway(monkeypatch)
    rc = kc.run_slash(
        "create 'ready assigned' --assignee worker --criticality noncritical --role noncritical"
    )
    assert "Created" in rc or "created" in rc.lower() or rc
    combined = rc.lower()
    assert "worker death" in combined or "dead worker" in combined, (
        "operator must be told later worker death will not be autonomously "
        f"reconciled; got: {rc!r}"
    )
    assert "reconcil" in combined, (
        "warning must mention autonomous reconciliation, got: {rc!r}".format(rc=rc)
    )


def test_one_shot_dispatch_warns_when_no_persistent_dispatcher(
    kanban_home, monkeypatch,
):
    """`hermes kanban dispatch` is one-shot. If no persistent dispatcher is
    available, the operator must be told spawned workers will not be reaped
    after this command exits."""
    _absent_gateway(monkeypatch)
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {"kanban": {"dispatch_in_gateway": False}},
    )
    args = argparse.Namespace(
        dry_run=True, max=None, failure_limit=2, json=False,
    )
    # Capture via redirect because _cmd_dispatch prints to stdout/stderr.
    import io
    import contextlib

    buf_out, buf_err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(buf_out), contextlib.redirect_stderr(buf_err):
        rc = kc._cmd_dispatch(args)
    assert rc == 0
    combined = (buf_out.getvalue() + "\n" + buf_err.getvalue()).lower()
    assert "worker death" in combined or "dead worker" in combined, (
        "one-shot dispatch must warn that later worker death will not be "
        f"autonomously reconciled; got stdout={buf_out.getvalue()!r} "
        f"stderr={buf_err.getvalue()!r}"
    )


def test_dispatcher_presence_message_names_dead_worker_reconciliation(
    kanban_home, monkeypatch,
):
    _absent_gateway(monkeypatch)
    running, message = kc._check_dispatcher_presence()
    assert running is False
    lowered = message.lower()
    assert "reconcil" in lowered
    assert "worker" in lowered and ("death" in lowered or "dead" in lowered)


# ---------------------------------------------------------------------------
# B. Dead local PID wins over lease age + fresh heartbeat
# ---------------------------------------------------------------------------


def test_dead_local_pid_reclaims_before_lease_or_heartbeat_ttl(
    kanban_home, monkeypatch,
):
    """A host-local running task whose worker PID is dead must be reaped on
    the next dispatch tick even when the claim lease is still valid and a
    heartbeat is younger than the 1h stale gap. Launch grace has elapsed.
    """
    monkeypatch.setattr(kb, "_pid_alive", lambda _pid: False)
    # Do NOT zero the crash grace: the fixture leaves the production default
    # (30s). started_at is rewound past that window.
    now = int(time.time())
    host = kb._claimer_id().split(":", 1)[0]
    dead_pid = 424242

    with kb.connect() as conn:
        tid = _create_noncritical(conn, title="dead-pid-wins")
        task = kb.get_task(conn, tid)
        workspace = kb.resolve_workspace(task)
        kb.set_workspace_path(conn, tid, str(workspace))
        marker = workspace / "authorized-work.txt"
        marker.write_text("keep me", encoding="utf-8")

        claimed = kb.claim_task(conn, tid, claimer=f"{host}:w1")
        assert claimed is not None
        kb._set_worker_pid(conn, tid, dead_pid)
        conn.execute(
            "UPDATE tasks SET claim_expires=?, last_heartbeat_at=?, "
            "started_at=? WHERE id=?",
            (now + 3600, now, now - 120, tid),
        )
        conn.execute(
            "UPDATE task_runs SET claim_expires=?, last_heartbeat_at=?, "
            "started_at=? WHERE id=?",
            (
                now + 3600,
                now,
                now - 120,
                kb.get_task(conn, tid).current_run_id,
            ),
        )
        conn.commit()

        mutations: list[str] = []

        def _forbid_rmtree(*_a, **_k):
            mutations.append("rmtree")
            raise AssertionError("workspace rmtree must not run on crash reclaim")

        monkeypatch.setattr(shutil, "rmtree", _forbid_rmtree)

        spawned: list[str] = []

        def _spawn(task, workspace_path, **_kwargs):
            spawned.append(task.id)
            return None

        result = kb.dispatch_once(conn, spawn_fn=_spawn)
        assert tid in result.crashed, (
            f"dead pid must be reaped immediately, not deferred to TTL/heartbeat; "
            f"crashed={result.crashed!r} reclaimed={result.reclaimed!r} "
            f"stale={result.stale!r}"
        )

        task = kb.get_task(conn, tid)
        assert task.worker_pid != dead_pid, (
            "dead pid must not remain the live worker_pid after the tick"
        )
        old_run = None
        for run in kb.list_runs(conn, tid):
            if run.outcome == "crashed" and run.ended_at is not None:
                old_run = run
                break
        assert old_run is not None, "old run must become terminal under crash semantics"
        assert old_run.outcome == "crashed"
        # Bounded retry may re-claim on the same tick; that is a NEW run.
        if task.status == "running":
            assert task.current_run_id != old_run.id
            assert spawned == [tid]
        else:
            assert task.status in {"ready", "todo", "blocked"}

        assert marker.read_text(encoding="utf-8") == "keep me"
        assert mutations == []
        assert task.consecutive_failures >= 1


# ---------------------------------------------------------------------------
# C. Dirty-workspace continuation
# ---------------------------------------------------------------------------


def test_crash_retry_preserves_dirty_workspace(kanban_home, monkeypatch, tmp_path):
    """A crash/reconciliation/retry operates on lifecycle state only and must
    not reset, checkout, stash, clean, delete, or recreate the workspace."""
    monkeypatch.setattr(kb, "_pid_alive", lambda _pid: False)
    monkeypatch.setenv("HERMES_KANBAN_CRASH_GRACE_SECONDS", "0")

    repo = tmp_path / "task-repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-b", "main", str(repo)], check=True, capture_output=True)
    subprocess.run(
        ["git", "-C", str(repo), "config", "user.email", "kanban@example.com"],
        check=True, capture_output=True,
    )
    subprocess.run(
        ["git", "-C", str(repo), "config", "user.name", "Kanban Test"],
        check=True, capture_output=True,
    )
    (repo / "tracked.txt").write_text("base\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(repo), "add", "tracked.txt"], check=True, capture_output=True)
    subprocess.run(
        ["git", "-C", str(repo), "commit", "-m", "init"],
        check=True, capture_output=True,
    )
    (repo / "tracked.txt").write_text("dirty authorized work\n", encoding="utf-8")
    (repo / "untracked.txt").write_text("also keep\n", encoding="utf-8")

    host = kb._claimer_id().split(":", 1)[0]
    forbidden = ("reset", "checkout", "stash", "clean")
    real_run = subprocess.run

    def _guarded_run(argv, *args, **kwargs):
        tokens = [str(x) for x in (argv or [])]
        if tokens and Path(str(tokens[0])).name == "git":
            if any(tok in forbidden for tok in tokens[1:6]):
                raise AssertionError(f"crash retry must not mutate git workspace: {tokens}")
        return real_run(argv, *args, **kwargs)

    monkeypatch.setattr(subprocess, "run", _guarded_run)

    with kb.connect() as conn:
        tid = _create_noncritical(
            conn,
            title="dirty-ws",
            workspace_kind="dir",
            workspace_path=str(repo),
        )
        kb.claim_task(conn, tid, claimer=f"{host}:w1")
        kb._set_worker_pid(conn, tid, 525252)
        conn.execute(
            "UPDATE tasks SET started_at = started_at - 120 WHERE id=?",
            (tid,),
        )
        conn.commit()

        rmtree_calls: list[str] = []
        monkeypatch.setattr(
            shutil, "rmtree",
            lambda *a, **k: rmtree_calls.append("rmtree"),
        )

        retry_workspaces: list[str] = []

        def _spawn(task, workspace_path, **_kwargs):
            retry_workspaces.append(str(workspace_path))
            return None

        result = kb.dispatch_once(conn, spawn_fn=_spawn)
        assert tid in result.crashed

        assert (repo / "tracked.txt").read_text(encoding="utf-8") == "dirty authorized work\n"
        assert (repo / "untracked.txt").read_text(encoding="utf-8") == "also keep\n"
        assert rmtree_calls == []
        assert retry_workspaces, "retry spawn must receive the preserved workspace"
        assert Path(retry_workspaces[0]).resolve() == repo.resolve()


# ---------------------------------------------------------------------------
# D. Transient provider failure semantics
# ---------------------------------------------------------------------------


def test_classified_overloaded_uses_tempfail_sentinel_not_breaker():
    """FailoverReason.overloaded is a provider/service failure, not an
    implementation-quality failure. The worker exit helper must map it to
    the existing EX_TEMPFAIL sentinel so detect_crashed_workers requeues
    without counting a failure."""
    fn = getattr(kb, "kanban_worker_exit_code_for_result", None)
    assert fn is not None, (
        "kanban_worker_exit_code_for_result must exist so cli.py and tests "
        "share one classified-reason mapping"
    )
    code = fn({
        "failed": True,
        "failure_reason": "overloaded",
        "error": "Our servers are currently overloaded. Please try again later.",
    })
    assert code == kb.KANBAN_RATE_LIMIT_EXIT_CODE


@pytest.mark.parametrize(
    "reason",
    ["rate_limit", "billing", "server_error", "timeout", "upstream_rate_limit"],
)
def test_classified_transient_provider_reasons_use_tempfail_sentinel(reason):
    fn = getattr(kb, "kanban_worker_exit_code_for_result", None)
    assert fn is not None
    assert fn({"failed": True, "failure_reason": reason}) == kb.KANBAN_RATE_LIMIT_EXIT_CODE


def test_unclassified_error_text_is_not_string_matched_as_provider_failure():
    """Broad matching of 'overloaded' in free-form error text is unsafe —
    only a classified failure_reason may take the tempfail path."""
    fn = getattr(kb, "kanban_worker_exit_code_for_result", None)
    assert fn is not None
    code = fn({
        "failed": True,
        "error": "Our servers are currently overloaded. Please try again later.",
    })
    assert code == 1
    assert fn({"failed": True, "failure_reason": "unknown"}) == 1
    assert fn({"failed": True, "failure_reason": "auth"}) == 1
    assert fn({"failed": False}) == 0
    assert fn(None) == 0


def test_overloaded_sentinel_exit_does_not_consume_failure_budget(
    kanban_home, monkeypatch,
):
    monkeypatch.setattr(kb, "_pid_alive", lambda _pid: False)
    monkeypatch.setenv("HERMES_KANBAN_CRASH_GRACE_SECONDS", "0")
    host = kb._claimer_id().split(":", 1)[0]

    with kb.connect() as conn:
        tid = _create_noncritical(conn, title="overloaded-requeue")
        pid = 616161
        kb.claim_task(conn, tid, claimer=f"{host}:w1")
        conn.execute("UPDATE tasks SET worker_pid=? WHERE id=?", (pid, tid))
        conn.commit()
        kb._record_worker_exit(pid, kb.KANBAN_RATE_LIMIT_EXIT_CODE << 8)

        crashed = kb.detect_crashed_workers(conn)
        assert tid not in crashed
        rl = getattr(kb.detect_crashed_workers, "_last_rate_limited", [])
        assert tid in rl
        task = kb.get_task(conn, tid)
        assert task.status == "ready"
        assert task.consecutive_failures == 0


# ---------------------------------------------------------------------------
# E. Effective execution identity / failover safety
# ---------------------------------------------------------------------------


def test_reassign_retains_overrides_and_cannot_be_mistaken_for_profile_backend(
    kanban_home, capsys,
):
    """assignee engineer-sol + Sol/OpenAI override → assign engineer-grok
    must keep the override visible and warn that execution is still Sol/Codex.
    """
    with kb.connect() as conn:
        tid = _create_noncritical(
            conn,
            title="failover-identity",
            assignee="engineer-sol",
            model_override="gpt-5.6-sol",
            provider_override="openai-codex",
        )

    rc = kc._cmd_assign(argparse.Namespace(task_id=tid, profile="engineer-grok"))
    assert rc == 0
    captured = capsys.readouterr()
    combined = captured.out + "\n" + captured.err
    with kb.connect() as conn:
        task = kb.get_task(conn, tid)
    assert task.assignee == "engineer-grok"
    assert task.model_override == "gpt-5.6-sol"
    assert task.provider_override == "openai-codex"
    assert "gpt-5.6-sol" in combined
    assert "openai-codex" in combined
    lowered = combined.lower()
    assert "override" in lowered
    assert "effective" in lowered
    # Must not present the assignee change as a Grok/xAI backend switch.
    assert "xai" not in lowered


def test_show_renders_effective_execution_identity(kanban_home):
    with kb.connect() as conn:
        tid = _create_noncritical(
            conn,
            title="show-pinned-backend",
            assignee="engineer-grok",
            model_override="gpt-5.6-sol",
            provider_override="openai-codex",
        )
    output = kc.run_slash(f"show {tid}")
    assert "engineer-grok" in output
    assert "gpt-5.6-sol" in output
    assert "openai-codex" in output
    lowered = output.lower()
    assert "effective execution" in lowered
    assert "override" in lowered
    assert "xai" not in lowered


def test_assign_does_not_silently_clear_overrides(kanban_home):
    with kb.connect() as conn:
        tid = _create_noncritical(
            conn,
            title="keep-pin",
            assignee="engineer-sol",
            model_override="gpt-5.6-sol",
            provider_override="openai-codex",
        )
        kb.assign_task(conn, tid, "engineer-grok")
        task = kb.get_task(conn, tid)
        assert task.model_override == "gpt-5.6-sol"
        assert task.provider_override == "openai-codex"
        kinds = [e.kind for e in kb.list_events(conn, tid)]
        assert "model_override_set" not in kinds or task.model_override is not None


# ---------------------------------------------------------------------------
# F. Durable human authorization provenance
# ---------------------------------------------------------------------------


def test_assignee_or_model_mutation_is_not_authorization(kanban_home):
    """Operational mutation events are not human authorization of a
    no-substitution exception."""
    fn = getattr(kb, "has_human_execution_policy_authorization", None)
    assert fn is not None
    with kb.connect() as conn:
        tid = _create_noncritical(
            conn,
            title="mutation-is-not-auth",
            assignee="engineer-sol",
            model_override="gpt-5.6-sol",
            provider_override="openai-codex",
        )
        kb.assign_task(conn, tid, "engineer-grok")
        kb.set_model_override(conn, tid, None)
        assert fn(conn, tid) is False
        assert fn(conn, tid, kind="executor_substitution") is False


def test_explicit_human_authorization_is_durable_and_verifiable(kanban_home):
    authorize = getattr(kb, "authorize_execution_policy_exception", None)
    query = getattr(kb, "has_human_execution_policy_authorization", None)
    assert authorize is not None
    assert query is not None
    with kb.connect() as conn:
        tid = _create_noncritical(
            conn,
            title="explicit-auth",
            assignee="engineer-sol",
            model_override="gpt-5.6-sol",
            provider_override="openai-codex",
        )
        event_id = authorize(
            conn,
            tid,
            author="bruce",
            kind="executor_substitution",
            summary="Bruce authorized Sol -> Grok failover interactively",
            from_assignee="engineer-sol",
            to_assignee="engineer-grok",
            from_model="gpt-5.6-sol",
            from_provider="openai-codex",
        )
        assert int(event_id) > 0
        assert query(conn, tid, kind="executor_substitution") is True
        kinds = [e.kind for e in kb.list_events(conn, tid)]
        assert "execution_policy_authorized" in kinds
        payload = next(
            e.payload for e in kb.list_events(conn, tid)
            if e.kind == "execution_policy_authorized"
        )
        assert payload["author"] == "bruce"
        assert payload["kind"] == "executor_substitution"
        assert "authorized" in payload["summary"].lower()


def test_worker_cannot_self_authorize_execution_policy_exception(
    kanban_home, monkeypatch,
):
    authorize = getattr(kb, "authorize_execution_policy_exception", None)
    assert authorize is not None
    with kb.connect() as conn:
        tid = _create_noncritical(conn, title="no-self-auth", assignee="engineer-grok")
        monkeypatch.setenv("HERMES_KANBAN_TASK", tid)
        with pytest.raises((PermissionError, RuntimeError, ValueError)):
            authorize(
                conn,
                tid,
                author="engineer-grok",
                kind="executor_substitution",
                summary="worker self-auth",
            )
        query = getattr(kb, "has_human_execution_policy_authorization")
        assert query(conn, tid) is False


def test_authorize_exception_cli_records_provenance(kanban_home, monkeypatch):
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    monkeypatch.setenv("HERMES_PROFILE", "bruce")
    with kb.connect() as conn:
        tid = _create_noncritical(
            conn,
            title="cli-auth",
            assignee="engineer-sol",
            model_override="gpt-5.6-sol",
            provider_override="openai-codex",
        )
    output = kc.run_slash(
        f"authorize-exception {tid} --kind executor_substitution "
        "--summary 'Bruce authorized Sol to Grok failover'"
    )
    assert "error" not in output.lower() or "authorized" in output.lower()
    with kb.connect() as conn:
        assert kb.has_human_execution_policy_authorization(
            conn, tid, kind="executor_substitution",
        )


def test_delegated_child_cannot_authorize_via_cli(kanban_home, monkeypatch):
    with kb.connect() as conn:
        tid = _create_noncritical(conn, title="child-auth")
    monkeypatch.setenv("HERMES_DELEGATED_CHILD_CONTEXT", "1")
    output = kc.run_slash(
        f"authorize-exception {tid} --kind executor_substitution --summary nope"
    )
    assert "cannot mutate" in output.lower() or "cannot authorize" in output.lower()
    monkeypatch.delenv("HERMES_DELEGATED_CHILD_CONTEXT", raising=False)
    with kb.connect() as conn:
        query = getattr(kb, "has_human_execution_policy_authorization", lambda *_a, **_k: False)
        assert query(conn, tid) is False
