from __future__ import annotations

import pytest
import subprocess


def _make_task(kb, *, assignee: str, created_by="test", session_id=None):
    return kb.Task(
        id="t_spawn_tools",
        title="spawn tools",
        body=None,
        assignee=assignee,
        status="running",
        priority=0,
        created_by=created_by,
        created_at=1,
        started_at=None,
        completed_at=None,
        workspace_kind="dir",
        workspace_path=None,
        claim_lock="lock",
        claim_expires=None,
        tenant=None,
        current_run_id=7,
        session_id=session_id,
    )


def test_default_spawn_pins_assignee_profile_cli_toolsets(monkeypatch, tmp_path):
    """The transport client reaches the assigned profile with its exact run claim."""
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
    from hermes_cli import kanban_db_dispatch as kbd

    monkeypatch.setattr(kbd, "_resolve_hermes_argv", lambda: ["hermes"])

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
    pid = kbd._default_spawn(_make_task(kb, assignee="elias"), str(workspace))

    assert pid == 4242
    assert captured["env"]["HERMES_HOME"] == str(profile)
    assert captured["env"]["HERMES_KANBAN_TASK"] == "t_spawn_tools"
    assert captured["cmd"][-2:] == ["-m", "hermes_cli.kanban_worker_client"]
    assert captured["env"]["HERMES_KANBAN_RUN_ID"] == "7"
    assert captured["env"]["HERMES_KANBAN_CLAIM_LOCK"] == "lock"


def test_default_spawn_does_not_pass_mutable_launch_overrides(monkeypatch, tmp_path):
    """Owner policy reads overrides from the claimed card, not executable argv."""
    root = tmp_path / ".hermes"
    (root / "profiles" / "elias").mkdir(parents=True)
    root.joinpath("config.yaml").write_text("{}\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(root))

    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_dispatch as kbd

    monkeypatch.setattr(kbd, "_resolve_hermes_argv", lambda: ["hermes"])
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
    kbd._default_spawn(task, str(workspace))

    # Mutable model strings are not launch authority; the owner reads the claimed card.
    assert captured["cmd"][-2:] == ["-m", "hermes_cli.kanban_worker_client"]
    assert task.model_override not in captured["cmd"]

def test_issue22_worker_resumes_bound_session_and_marks_busy_refusal(monkeypatch, tmp_path):
    root = tmp_path / ".hermes"
    (root / "profiles" / "issue22-dispatch").mkdir(parents=True)
    root.joinpath("config.yaml").write_text("{}\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(root))

    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_dispatch as kbd

    monkeypatch.setattr(kbd, "_resolve_hermes_argv", lambda: ["hermes"])
    captured = {}

    class FakeProc:
        pid = 4245

    def fake_popen(cmd, *args, **kwargs):
        captured["cmd"] = list(cmd)
        captured["env"] = dict(kwargs.get("env") or {})
        return FakeProc()

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    task = _make_task(
        kb, assignee="issue22-dispatch",
        created_by="github:ISVGitHub/autonomous-entity#22", session_id="session-22",
    )
    assert kbd._default_spawn(task, str(workspace)) == 4245
    assert captured["cmd"][captured["cmd"].index("--resume") + 1] == "session-22"
    assert captured["env"]["HERMES_KANBAN_TASK"] == "t_spawn_tools"
    from hermes_cli._parser import build_top_level_parser
    parser, _subparsers, _chat_parser = build_top_level_parser()
    args = parser.parse_args(captured["cmd"][3:])
    assert args.resume == "session-22"
    assert args.query == "work kanban task t_spawn_tools"


def test_default_spawn_resolves_env_passthrough_under_multiplex(monkeypatch, tmp_path):
    """Under multiplex a worker spawn with ``terminal.env_passthrough`` configured must
    forward the ASSIGNEE profile's own value, never crash on an unscoped read or leak the
    dispatcher's ambient os.environ (#109494).
    """
    root = tmp_path / ".hermes"
    profile = root / "profiles" / "elias"
    profile.mkdir(parents=True)
    root.joinpath("config.yaml").write_text(
        "terminal:\n  env_passthrough:\n    - MY_PASSTHROUGH_VAR\n", encoding="utf-8")
    profile.joinpath("config.yaml").write_text("{}\n", encoding="utf-8")
    profile.joinpath(".env").write_text("MY_PASSTHROUGH_VAR=elias-value\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(root))
    monkeypatch.setenv("MY_PASSTHROUGH_VAR", "dispatcher-value")

    from agent.secret_scope import set_multiplex_active
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_dispatch as kbd

    monkeypatch.setattr(kbd, "_resolve_hermes_argv", lambda: ["hermes"])

    captured = {}

    class FakeProc:
        pid = 4243

    def fake_popen(cmd, *args, **kwargs):
        captured["env"] = dict(kwargs.get("env") or {})
        return FakeProc()

    monkeypatch.setattr(subprocess, "Popen", fake_popen)

    workspace = tmp_path / "workspace"
    workspace.mkdir()

    set_multiplex_active(True)
    try:
        pid = kbd._default_spawn(_make_task(kb, assignee="elias"), str(workspace))
    finally:
        set_multiplex_active(False)

    assert pid == 4243
    # The assignee's own scoped value, not the dispatcher's ambient os.environ one.
    assert captured["env"].get("MY_PASSTHROUGH_VAR") == "elias-value"


def test_worker_toolsets_come_from_the_assignee_profile_not_the_parent_config(tmp_path, monkeypatch):
    """The worker's toolsets are frozen by the assignee profile's own gateway (its config), never
    by the dispatcher's root/active profile: a root config that only lists ``kanban`` must not
    reach a worker whose profile grants ``terminal`` and ``web``."""
    import json
    from contextlib import closing
    from types import SimpleNamespace

    from hermes_cli import kanban_db as kb
    from hermes_cli.kanban_db_connect import connect
    from gateway.session_contract import Principal
    from gateway.session_kanban import build_kanban_policy

    profile = tmp_path / "profiles" / "elias"
    profile.mkdir(parents=True)
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_HOME", str(profile))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.delenv("HERMES_KANBAN_DB", raising=False)
    monkeypatch.setattr("hermes_cli.profiles.resolve_profile_env", lambda name: str(profile))
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    kb.create_board("owned")  # explicit creation; connect() never creates a board (#43243)
    with closing(connect(board="owned")) as conn:
        task_id = kb.create_task(conn, title="t", body="b", assignee="elias",
                                 workspace_kind="dir", workspace_path=str(workspace))
        kb.recompute_ready(conn)
        task = kb.claim_task(conn, task_id)
    actor = Principal("owner", str(profile), frozenset({"session:create"}), "transport")
    connection = SimpleNamespace(authority=SimpleNamespace(profile_id=str(profile)), actor=actor, native_owner=True)
    params = dict(board="owned", task_id=task.id, run_id=task.current_run_id, claim_lock=task.claim_lock)

    assignee_config = {"model": {"default": "m", "provider": "custom"},
                       "platform_toolsets": {"cli": ["terminal", "web"]}, "toolsets": ["hermes-cli"]}
    policy, _ = build_kanban_policy(connection, params, assignee_config)
    assert {"terminal", "web", "kanban"} <= set(policy.toolsets)

    root_only_kanban = {"model": {"default": "m", "provider": "custom"}, "platform_toolsets": {"cli": ["kanban"]}}
    other_profile = tmp_path
    with pytest.raises(Exception):
        # A dispatcher acting for another profile cannot mint this worker's policy at all.
        build_kanban_policy(SimpleNamespace(authority=SimpleNamespace(profile_id=str(other_profile)),
                                            actor=actor, native_owner=True), params, root_only_kanban)
    assert json.loads(policy.kanban_json)["profile"] == "elias"
