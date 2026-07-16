from __future__ import annotations

import subprocess


def _write_skill(root, name: str) -> None:
    skill_dir = root / "skills" / name
    skill_dir.mkdir(parents=True)
    skill_dir.joinpath("SKILL.md").write_text(
        f"---\nname: {name}\ndescription: Test skill\n---\n\n# {name}\n",
        encoding="utf-8",
    )


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


def test_dispatch_blocks_card_with_skill_missing_from_worker_profile(
    monkeypatch, tmp_path
):
    root = tmp_path / ".hermes"
    profile = root / "profiles" / "elias"
    profile.mkdir(parents=True)
    root.joinpath("config.yaml").write_text("", encoding="utf-8")
    profile.joinpath("config.yaml").write_text("", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(root))

    from hermes_cli import kanban_db as kb

    kb.init_db()
    spawned = []

    def fake_spawn(task, workspace):
        spawned.append(task.id)
        return 123

    with kb.connect_closing() as conn:
        tid = kb.create_task(
            conn,
            title="needs unavailable skill",
            assignee="elias",
            skills=["missing-card-skill"],
        )
        result = kb.dispatch_once(conn, spawn_fn=fake_spawn)
        task = kb.get_task(conn, tid)
        runs = kb.list_runs(conn, tid)

    assert spawned == []
    assert tid in result.auto_blocked
    assert task.status == "blocked"
    assert task.block_kind == "capability"
    assert "missing-card-skill" in (runs[-1].summary or "")
    assert "elias" in (runs[-1].summary or "")


def test_dispatch_allows_card_skill_present_in_worker_profile(
    monkeypatch, tmp_path
):
    root = tmp_path / ".hermes"
    profile = root / "profiles" / "elias"
    profile.mkdir(parents=True)
    root.joinpath("config.yaml").write_text("", encoding="utf-8")
    profile.joinpath("config.yaml").write_text("", encoding="utf-8")
    _write_skill(profile, "card-skill")
    monkeypatch.setenv("HERMES_HOME", str(root))

    from hermes_cli import kanban_db as kb

    kb.init_db()
    spawned = []

    def fake_spawn(task, workspace):
        spawned.append((task.id, task.skills))
        return 456

    with kb.connect_closing() as conn:
        tid = kb.create_task(
            conn,
            title="has profile skill",
            assignee="elias",
            skills=["card-skill"],
        )
        result = kb.dispatch_once(conn, spawn_fn=fake_spawn)
        task = kb.get_task(conn, tid)

    assert spawned == [(tid, ["card-skill"])]
    assert result.spawned and result.spawned[0][0] == tid
    assert task.status == "running"


def test_dispatch_dry_run_missing_card_skill_reports_block_without_mutating(
    monkeypatch, tmp_path
):
    """Dry-run must surface the capability block, not a false spawn.

    Real dispatch blocks cards whose requested skills are unavailable on the
    assignee profile. Dry-run is documented to show that outcome without
    mutating the DB — so missing skills belong in ``auto_blocked``, not
    ``spawned``, and the card stays ``ready``.
    """
    root = tmp_path / ".hermes"
    profile = root / "profiles" / "elias"
    profile.mkdir(parents=True)
    root.joinpath("config.yaml").write_text("", encoding="utf-8")
    profile.joinpath("config.yaml").write_text("", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(root))

    from hermes_cli import kanban_db as kb

    kb.init_db()
    spawned = []

    def fake_spawn(task, workspace):
        spawned.append(task.id)
        return 789

    with kb.connect_closing() as conn:
        tid = kb.create_task(
            conn,
            title="dry-run missing skill",
            assignee="elias",
            skills=["missing-card-skill"],
        )
        result = kb.dispatch_once(conn, spawn_fn=fake_spawn, dry_run=True)
        task = kb.get_task(conn, tid)

    assert spawned == []
    assert tid in result.auto_blocked
    assert result.spawned == []
    assert task.status == "ready"
    assert task.block_kind is None
