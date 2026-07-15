from __future__ import annotations

import subprocess


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


def test_default_spawn_inherits_session_contextvars_into_worker_env(monkeypatch, tmp_path):
    """Worker subprocess env should carry the originating platform/chat/thread.

    ContextVars are task-local and don't cross the subprocess boundary on
    their own; _default_spawn bridges them via build_session_subprocess_env
    so the worker can auto-target send_message to the originating thread.
    """
    from gateway.session_context import set_session_vars, clear_session_vars

    from hermes_cli import kanban_db as kb

    monkeypatch.setattr(kb, "_resolve_hermes_argv", lambda: ["hermes"])
    captured = {}

    class FakeProc:
        pid = 4243

    def fake_popen(cmd, *args, **kwargs):
        captured["env"] = dict(kwargs.get("env") or {})
        return FakeProc()

    monkeypatch.setattr(subprocess, "Popen", fake_popen)

    workspace = tmp_path / "workspace"
    workspace.mkdir()

    from gateway.session_context import reset_session_vars

    tokens = set_session_vars(
        platform="telegram", chat_id="-1001", thread_id="17585"
    )
    try:
        kb._default_spawn(_make_task(kb, assignee="elias"), str(workspace))
    finally:
        clear_session_vars(tokens)
        # Restore _UNSET (not clear_session_vars' "" sentinel) so later tests
        # in the same worker process see "never bound" and get the
        # os.environ fallback instead of an inherited empty string.
        reset_session_vars()

    assert captured["env"]["HERMES_SESSION_PLATFORM"] == "telegram"
    assert captured["env"]["HERMES_SESSION_CHAT_ID"] == "-1001"
    assert captured["env"]["HERMES_SESSION_THREAD_ID"] == "17585"


def test_default_spawn_falls_back_when_gateway_module_missing(monkeypatch, tmp_path):
    """A genuinely absent gateway package (ModuleNotFoundError) should fall
    back to a plain os.environ copy rather than crashing the spawn."""
    import builtins

    from hermes_cli import kanban_db as kb

    monkeypatch.setattr(kb, "_resolve_hermes_argv", lambda: ["hermes"])
    captured = {}

    class FakeProc:
        pid = 4245

    def fake_popen(cmd, *args, **kwargs):
        captured["env"] = dict(kwargs.get("env") or {})
        return FakeProc()

    monkeypatch.setattr(subprocess, "Popen", fake_popen)

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "gateway.session_context":
            raise ModuleNotFoundError("No module named 'gateway.session_context'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    pid = kb._default_spawn(_make_task(kb, assignee="elias"), str(workspace))

    assert pid == 4245
    assert "env" in captured


def test_default_spawn_propagates_real_import_error_from_gateway_module(monkeypatch, tmp_path):
    """A real ImportError raised from inside gateway.session_context (e.g. a
    regression, not a missing package) must propagate, not be swallowed.

    Regression guard: the except clause was narrowed from ImportError to
    ModuleNotFoundError specifically so this class of bug surfaces loudly.
    """
    import builtins

    from hermes_cli import kanban_db as kb

    monkeypatch.setattr(kb, "_resolve_hermes_argv", lambda: ["hermes"])

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "gateway.session_context":
            raise ImportError("cannot import name 'build_session_subprocess_env'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    workspace = tmp_path / "workspace"
    workspace.mkdir()

    import pytest

    with pytest.raises(ImportError, match="build_session_subprocess_env"):
        kb._default_spawn(_make_task(kb, assignee="elias"), str(workspace))
