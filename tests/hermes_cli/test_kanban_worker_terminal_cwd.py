"""Tests: kanban worker spawn pins TERMINAL_CWD to the task workspace.

Regression coverage for #34619 and #41312 (same root cause): ``_default_spawn``
launched the worker subprocess with ``cwd=workspace`` and set
``HERMES_KANBAN_WORKSPACE``, but did NOT set ``TERMINAL_CWD``. Because
``TERMINAL_CWD`` takes precedence over the process cwd in both
``tools/file_tools.py::_resolve_base_dir`` (relative ``write_file`` paths) and
``agent_init``'s context-file loader (``AGENTS.md`` discovery), workers inherited
the dispatching gateway's cwd — relative writes landed in the gateway user's
home (#41312) and the wrong profile's ``AGENTS.md`` was loaded (#34619).
Pinning ``TERMINAL_CWD`` to the workspace fixes both.
"""

from __future__ import annotations

import subprocess


def _make_task(kb, *, assignee: str = "w", tenant: str | None = None):
    return kb.Task(
        id="t_cwd",
        title="cwd pin",
        body=None,
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
        tenant=tenant,
        current_run_id=1,
    )


def _capture_spawn_env(kb, monkeypatch, workspace: str) -> dict:
    monkeypatch.setattr(kb, "_resolve_hermes_argv", lambda: ["hermes"])

    captured: dict = {}

    class FakeProc:
        pid = 4242

    def fake_popen(cmd, *args, **kwargs):
        captured["cmd"] = list(cmd)
        captured["env"] = dict(kwargs.get("env") or {})
        captured["cwd"] = kwargs.get("cwd")
        return FakeProc()

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    kb._default_spawn(_make_task(kb), workspace)
    return captured


def test_terminal_cwd_pinned_to_workspace(monkeypatch, tmp_path):
    """A real, absolute workspace dir is pinned as TERMINAL_CWD."""
    root = tmp_path / ".hermes"
    (root / "profiles" / "w").mkdir(parents=True)
    (root / "profiles" / "w" / "config.yaml").write_text("toolsets:\n  - kanban\n", encoding="utf-8")
    root.joinpath("config.yaml").write_text("toolsets:\n  - kanban\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(root))

    from hermes_cli import kanban_db as kb

    workspace = tmp_path / "ws"
    workspace.mkdir()

    captured = _capture_spawn_env(kb, monkeypatch, str(workspace))

    assert captured["env"]["TERMINAL_CWD"] == str(workspace)
    # The subprocess cwd and TERMINAL_CWD must agree — both anchor the workspace.
    assert captured["cwd"] == str(workspace)
    assert captured["env"]["HERMES_KANBAN_WORKSPACE"] == str(workspace)


def test_default_spawn_pins_stable_kanban_worker_env(monkeypatch, tmp_path):
    """Dispatcher-spawned workers carry the full Kanban observer contract env."""
    root = tmp_path / ".hermes"
    (root / "profiles" / "w").mkdir(parents=True)
    (root / "profiles" / "w" / "config.yaml").write_text("toolsets:\n  - kanban\n", encoding="utf-8")
    root.joinpath("config.yaml").write_text("toolsets:\n  - kanban\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(root))

    from hermes_cli import kanban_db as kb

    workspace = tmp_path / "ws"
    workspace.mkdir()

    captured = _capture_spawn_env(kb, monkeypatch, str(workspace))
    env = captured["env"]

    assert env["HERMES_KANBAN_TASK"] == "t_cwd"
    assert env["HERMES_KANBAN_RUN_ID"] == "1"
    assert env["HERMES_KANBAN_BOARD"] == "default"
    assert env["HERMES_KANBAN_DB"] == str(root / "kanban.db")
    assert env["HERMES_KANBAN_WORKSPACES_ROOT"] == str(root / "kanban" / "workspaces")
    assert env["HERMES_KANBAN_WORKSPACE"] == str(workspace)
    assert env["HERMES_PROFILE"] == "w"
    assert env["TERMINAL_CWD"] == str(workspace)


def test_default_spawn_injects_tenant_when_present(monkeypatch, tmp_path):
    """Tenant is optional, but when present workers inherit it exactly."""
    root = tmp_path / ".hermes"
    (root / "profiles" / "w").mkdir(parents=True)
    (root / "profiles" / "w" / "config.yaml").write_text("toolsets:\n  - kanban\n", encoding="utf-8")
    root.joinpath("config.yaml").write_text("toolsets:\n  - kanban\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(root))

    from hermes_cli import kanban_db as kb

    workspace = tmp_path / "ws"
    workspace.mkdir()
    monkeypatch.setattr(kb, "_resolve_hermes_argv", lambda: ["hermes"])

    captured: dict = {}

    class FakeProc:
        pid = 4242

    def fake_popen(cmd, *args, **kwargs):
        captured["env"] = dict(kwargs.get("env") or {})
        return FakeProc()

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    kb._default_spawn(_make_task(kb, tenant="tenant-a"), str(workspace))

    assert captured["env"]["HERMES_TENANT"] == "tenant-a"


def test_terminal_cwd_not_pinned_for_nonexistent_workspace(monkeypatch, tmp_path):
    """A non-directory workspace must NOT clobber the inherited TERMINAL_CWD.

    file_tools rejects relative / sentinel TERMINAL_CWD values, so writing a
    meaningless (nonexistent) path would be worse than leaving the inherited
    one. The guard requires an existing absolute dir.
    """
    root = tmp_path / ".hermes"
    (root / "profiles" / "w").mkdir(parents=True)
    (root / "profiles" / "w" / "config.yaml").write_text("toolsets:\n  - kanban\n", encoding="utf-8")
    root.joinpath("config.yaml").write_text("toolsets:\n  - kanban\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(root))
    monkeypatch.setenv("TERMINAL_CWD", "/pre/existing/anchor")

    from hermes_cli import kanban_db as kb

    missing = tmp_path / "does-not-exist"

    captured = _capture_spawn_env(kb, monkeypatch, str(missing))

    # Inherited value is preserved (not overwritten with a bogus path).
    assert captured["env"]["TERMINAL_CWD"] == "/pre/existing/anchor"


def test_kanban_trace_context_propagates_to_parallel_children_and_worker_env(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    traceparent = "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"
    tracestate = "rojo=00f067aa0ba902b7"
    monkeypatch.setenv("HERMES_KANBAN_TRACEPARENT", traceparent)
    monkeypatch.setenv("HERMES_KANBAN_TRACESTATE", tracestate)

    from hermes_cli import kanban_db as kb

    with kb.connect() as conn:
        parent = kb.create_task(conn, title="parent", assignee="planner")
        child_a = kb.create_task(conn, title="child a", assignee="worker", parents=[parent])
        child_b = kb.create_task(conn, title="child b", assignee="worker", parents=[parent])

        for tid in (parent, child_a, child_b):
            task = kb.get_task(conn, tid)
            assert task is not None
            assert task.traceparent == traceparent
            assert task.tracestate == tracestate

        kb.complete_task(conn, parent, summary="done")
        claimed = kb.claim_task(conn, child_a)
        assert claimed is not None

        workspace = tmp_path / "ws"
        workspace.mkdir()
        monkeypatch.setattr(kb, "_resolve_hermes_argv", lambda: ["hermes"])
        captured: dict = {}

        class FakeProc:
            pid = 1234

        def fake_popen(cmd, *args, **kwargs):
            captured.update(dict(kwargs.get("env") or {}))
            return FakeProc()

        monkeypatch.setattr(subprocess, "Popen", fake_popen)
        kb._default_spawn(claimed, str(workspace))

    assert captured["HERMES_KANBAN_TRACEPARENT"] == traceparent
    assert captured["HERMES_KANBAN_TRACESTATE"] == tracestate
    assert not any("SECRET" in key or "TOKEN" in key for key in captured)


def test_linking_existing_tasks_backfills_missing_child_trace_context(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    traceparent = "00-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa-bbbbbbbbbbbbbbbb-01"
    monkeypatch.setenv("HERMES_KANBAN_TRACEPARENT", traceparent)

    from hermes_cli import kanban_db as kb

    with kb.connect() as conn:
        parent = kb.create_task(conn, title="parent", assignee="planner")
        monkeypatch.delenv("HERMES_KANBAN_TRACEPARENT", raising=False)
        child = kb.create_task(conn, title="child", assignee="worker")
        child_task = kb.get_task(conn, child)
        assert child_task is not None
        assert child_task.traceparent is None

        kb.link_tasks(conn, parent, child)

        child_task = kb.get_task(conn, child)
        assert child_task is not None
        assert child_task.traceparent == traceparent


def test_create_task_ignores_invalid_trace_context_and_secret_env(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    monkeypatch.setenv("HERMES_KANBAN_TRACEPARENT", "not-a-traceparent")
    monkeypatch.setenv("HERMES_KANBAN_TRACESTATE", "x" * 513)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-sho...leak")

    from hermes_cli import kanban_db as kb

    with kb.connect() as conn:
        tid = kb.create_task(conn, title="task", assignee="worker")
        task = kb.get_task(conn, tid)

    assert task is not None
    assert task.traceparent is None
    assert task.tracestate is None
