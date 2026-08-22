from __future__ import annotations

import subprocess
from pathlib import Path

import pytest


def _make_task(kb, *, assignee: str):
    return kb.Task(
        id="t_spawn_tools",
        title="spawn tools",
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
        tenant=None,
        current_run_id=7,
    )


def test_default_spawn_pins_assignee_profile_cli_toolsets(monkeypatch, tmp_path):
    """Manual profile assignment should keep that profile's CLI tools.

    Regression guard for dispatcher-spawned workers that boot with
    HERMES_KANBAN_TASK: the worker must not collapse to only kanban lifecycle
    tools when the assigned profile's top-level ``toolsets`` is the default
    composite. The spawned CLI gets an explicit --toolsets pin resolved from
    platform_toolsets.cli; model_tools appends task-scoped kanban tools later.
    """
    root = tmp_path / ".hermes"
    profile = root / "profiles" / "elias"
    profile.mkdir(parents=True)
    profile.joinpath("config.yaml").write_text(
        """
platform_toolsets:
  cli:
    - clarify
    - code_execution
    - delegation
    - file
    - memory
    - session_search
    - skills
    - terminal
    - web
toolsets:
  - hermes-cli
agent:
  disabled_toolsets: []
""".lstrip(),
        encoding="utf-8",
    )
    root.joinpath("config.yaml").write_text("toolsets:\n  - kanban\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(root))

    from hermes_cli import kanban_db as kb

    monkeypatch.setattr(kb, "_resolve_hermes_argv", lambda: ["hermes"])

    captured = {}

    class FakeProc:
        pid = 4242

    def fake_popen(cmd, *args, **kwargs):
        captured["cmd"] = list(cmd)
        captured["env"] = dict(kwargs.get("env") or {})
        captured["cwd"] = kwargs.get("cwd")
        return FakeProc()

    monkeypatch.setattr(subprocess, "Popen", fake_popen)

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    pid = kb._default_spawn(_make_task(kb, assignee="elias"), str(workspace))

    assert pid == 4242
    assert captured["env"]["HERMES_HOME"] == str(profile)
    assert captured["env"]["HERMES_KANBAN_TASK"] == "t_spawn_tools"
    assert "--toolsets" in captured["cmd"]
    pinned = captured["cmd"][captured["cmd"].index("--toolsets") + 1].split(",")
    for required in ("terminal", "web", "file", "skills", "code_execution", "delegation"):
        assert required in pinned


def test_default_spawn_model_override_survives_real_cli_parse(monkeypatch, tmp_path):
    """The dispatcher's pre-``chat`` model flag must reach ``args.model``.

    This is an integration contract between Kanban's worker argv builder and
    the real CLI parser. A parser default once erased the explicit override,
    silently sending the worker to its profile default or fallback instead.
    """
    root = tmp_path / ".hermes"
    (root / "profiles" / "elias").mkdir(parents=True)
    root.joinpath("config.yaml").write_text("{}\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(root))

    from hermes_cli import kanban_db as kb
    from hermes_cli._parser import build_top_level_parser

    monkeypatch.setattr(kb, "_resolve_hermes_argv", lambda: ["hermes"])
    captured = {}

    class FakeProc:
        pid = 4244

    def fake_popen(cmd, *args, **kwargs):
        captured["cmd"] = list(cmd)
        return FakeProc()

    monkeypatch.setattr(subprocess, "Popen", fake_popen)

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    task = _make_task(kb, assignee="elias")
    task.model_override = "gpt-5.6-sol"
    kb._default_spawn(task, str(workspace))

    parser, _subparsers, _chat_parser = build_top_level_parser()
    # Profile selection is attached by the outer CLI bootstrap rather than
    # build_top_level_parser(); remove that already-validated prefix and parse
    # the worker flags/subcommand through the real shared parser.
    assert captured["cmd"][1:3] == ["-p", "elias"]
    args = parser.parse_args(captured["cmd"][3:])

    assert args.command == "chat"
    assert args.model == "gpt-5.6-sol"
    assert args.query == "work kanban task t_spawn_tools"


def test_resolve_worker_cli_toolsets_uses_profile_home_not_parent_config(monkeypatch, tmp_path):
    root = tmp_path / ".hermes"
    profile = root / "profiles" / "elias"
    profile.mkdir(parents=True)
    root.joinpath("config.yaml").write_text("platform_toolsets:\n  cli:\n    - kanban\n", encoding="utf-8")
    profile.joinpath("config.yaml").write_text(
        """
platform_toolsets:
  cli:
    - terminal
    - web
toolsets:
  - hermes-cli
""".lstrip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(root))

    from hermes_cli import kanban_db as kb

    resolved = kb._resolve_worker_cli_toolsets(str(profile))

    assert resolved is not None
    assert "terminal" in resolved
    assert "web" in resolved
    assert "kanban" in resolved  # recovered worker lifecycle surface
    assert resolved != ["kanban"]


# ---------------------------------------------------------------------------
# Worker behavioral constraints (#46582): forbidden_tools enforcement rides
# the pre-tool-call block-directive pipeline. The dispatcher persists the
# constraint on the task row; the WORKER session installs it from the board
# DB at boot (no env propagation). These tests exercise that worker-session
# path with a real board DB and the same HERMES_KANBAN_* env pins
# _default_spawn sets.
# ---------------------------------------------------------------------------


@pytest.fixture
def kanban_board_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME with an initialized default-board kanban DB."""
    home = tmp_path / ".hermes"
    home.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    from hermes_cli import kanban_db as kb

    kb.init_db()
    return home


def _create_constrained_task(kb, conn, *, forbidden, require_summary=False):
    return kb.create_task(
        conn,
        title="hard-constrained card",
        assignee="elias",
        forbidden_tools=forbidden,
        require_summary=require_summary,
    )


def test_worker_session_installs_forbidden_tools_guard(kanban_board_home, monkeypatch):
    """A spawned-worker session refuses constrained tool calls — mechanically.

    Mirrors the worker bootstrap: the task row carries forbidden_tools, the
    worker process has only the spawn env pins, and
    ``install_task_constraint_guard()`` reads the board and installs the deny
    directive. The constrained call is refused with an error naming the
    constraint and the task; every other tool flows. Clearing restores
    today's behaviour.
    """
    from hermes_cli import kanban_db as kb
    from tools.kanban_tools import (
        install_task_constraint_guard,
        clear_task_constraint_guard,
    )
    from hermes_cli.plugins import get_pre_tool_call_directive

    conn = kb.connect()
    try:
        tid = _create_constrained_task(
            kb, conn, forbidden=["delegate_task", "kanban_create"]
        )
    finally:
        conn.close()

    # The only worker-side signal is the env pin the dispatcher sets.
    monkeypatch.setenv("HERMES_KANBAN_TASK", tid)

    installed = install_task_constraint_guard()
    try:
        assert installed == ["delegate_task", "kanban_create"]

        directive, message = get_pre_tool_call_directive("delegate_task", {})
        assert directive == "block"
        assert "delegate_task" in message
        assert tid in message

        directive, message = get_pre_tool_call_directive(
            "kanban_create", {"title": "x"}
        )
        assert directive == "block"

        # Everything else — terminal included — passes through untouched.
        assert get_pre_tool_call_directive("terminal", {}) == (None, None)
        assert get_pre_tool_call_directive("write_file", {}) == (None, None)
    finally:
        clear_task_constraint_guard()

    assert get_pre_tool_call_directive("delegate_task", {}) == (None, None)


def test_worker_session_without_constraints_is_unconstrained(kanban_board_home, monkeypatch):
    """Tasks created without forbidden_tools leave workers exactly as before."""
    from hermes_cli import kanban_db as kb
    from tools.kanban_tools import install_task_constraint_guard
    from hermes_cli.plugins import get_pre_tool_call_directive

    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="free card", assignee="elias")
    finally:
        conn.close()
    monkeypatch.setenv("HERMES_KANBAN_TASK", tid)

    assert install_task_constraint_guard() is None
    assert get_pre_tool_call_directive("terminal", {}) == (None, None)


def test_worker_session_guard_survives_unknown_task_gracefully(kanban_board_home, monkeypatch):
    """A missing task / broken board must degrade to no guard, never wedge
    worker startup (dispatcher claim-TTL is the backstop)."""
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_nonexistent")

    from tools.kanban_tools import install_task_constraint_guard
    from hermes_cli.plugins import get_pre_tool_call_directive

    assert install_task_constraint_guard() is None
    assert get_pre_tool_call_directive("terminal", {}) == (None, None)
