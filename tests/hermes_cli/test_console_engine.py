from __future__ import annotations

import io
import sys
from pathlib import Path

import pytest

from hermes_cli.console_engine import HermesConsoleEngine, run_console_repl


EXPECTED_CONSOLE_COMMANDS = {
    ("status",),
    ("doctor",),
    ("logs",),
    ("version",),
    ("dump",),
    ("debug", "share"),
    ("debug", "delete"),
    ("prompt-size",),
    ("insights",),
    ("security", "audit"),
    ("portal", "info"),
    ("portal", "tools"),
    ("backup",),
    ("import",),
    ("send",),
    ("config", "show"),
    ("config", "path"),
    ("config", "env-path"),
    ("config", "check"),
    ("config", "migrate"),
    ("config", "set"),
    ("sessions", "list"),
    ("sessions", "stats"),
    ("sessions", "export"),
    ("sessions", "rename"),
    ("sessions", "optimize"),
    ("sessions", "repair"),
    ("cron", "list"),
    ("cron", "status"),
    ("cron", "create"),
    ("cron", "edit"),
    ("cron", "pause"),
    ("cron", "resume"),
    ("cron", "run"),
    ("cron", "remove"),
    ("cron", "tick"),
    ("profile",),
    ("profile", "list"),
    ("profile", "show"),
    ("profile", "info"),
    ("profile", "create"),
    ("profile", "use"),
    ("profile", "describe"),
    ("profile", "rename"),
    ("profile", "delete"),
    ("profile", "export"),
    ("profile", "import"),
    ("profile", "install"),
    ("profile", "update"),
    ("tools", "list"),
    ("tools", "enable"),
    ("tools", "disable"),
    ("tools", "post-setup"),
    ("plugins", "list"),
    ("plugins", "enable"),
    ("plugins", "disable"),
    ("plugins", "install"),
    ("plugins", "update"),
    ("plugins", "remove"),
    ("skills", "browse"),
    ("skills", "search"),
    ("skills", "inspect"),
    ("skills", "list"),
    ("skills", "check"),
    ("skills", "list-modified"),
    ("skills", "diff"),
    ("skills", "install"),
    ("skills", "update"),
    ("skills", "audit"),
    ("skills", "uninstall"),
    ("skills", "reset"),
    ("skills", "opt-in"),
    ("skills", "opt-out"),
    ("skills", "repair-official"),
    ("skills", "snapshot", "export"),
    ("skills", "snapshot", "import"),
    ("skills", "tap", "list"),
    ("skills", "tap", "add"),
    ("skills", "tap", "remove"),
    ("mcp", "list"),
    ("mcp", "catalog"),
    ("mcp", "test"),
    ("mcp", "add"),
    ("mcp", "remove"),
    ("mcp", "install"),
    ("mcp", "login"),
    ("mcp", "reauth"),
    ("mcp", "configure"),
    ("mcp", "picker"),
    ("memory", "status"),
    ("memory", "off"),
    ("memory", "reset"),
    ("auth", "list"),
    ("auth", "status"),
    ("auth", "reset"),
    ("auth", "add"),
    ("auth", "remove"),
    ("auth", "logout"),
    ("auth", "spotify", "status"),
    ("auth", "spotify", "login"),
    ("auth", "spotify", "logout"),
    ("pairing", "list"),
    ("pairing", "approve"),
    ("pairing", "revoke"),
    ("pairing", "clear-pending"),
    ("webhook", "list"),
    ("webhook", "subscribe"),
    ("webhook", "remove"),
    ("webhook", "test"),
    ("hooks", "list"),
    ("hooks", "test"),
    ("hooks", "doctor"),
    ("hooks", "revoke"),
    ("slack", "manifest"),
    ("project", "list"),
    ("project", "show"),
    ("project", "create"),
    ("project", "add-folder"),
    ("project", "remove-folder"),
    ("project", "rename"),
    ("project", "set-primary"),
    ("project", "use"),
    ("project", "archive"),
    ("project", "restore"),
    ("project", "bind-board"),
    ("kanban", "init"),
    ("kanban", "boards", "list"),
    ("kanban", "boards", "create"),
    ("kanban", "boards", "rm"),
    ("kanban", "boards", "switch"),
    ("kanban", "boards", "current"),
    ("kanban", "boards", "rename"),
    ("kanban", "boards", "set-workdir"),
    ("kanban", "create"),
    ("kanban", "list"),
    ("kanban", "show"),
    ("kanban", "assign"),
    ("kanban", "reclaim"),
    ("kanban", "reassign"),
    ("kanban", "diagnose"),
    ("kanban", "link"),
    ("kanban", "unlink"),
    ("kanban", "claim"),
    ("kanban", "comment"),
    ("kanban", "complete"),
    ("kanban", "edit"),
    ("kanban", "block"),
    ("kanban", "schedule"),
    ("kanban", "unblock"),
    ("kanban", "promote"),
    ("kanban", "archive"),
    ("kanban", "stats"),
    ("kanban", "runs"),
    ("kanban", "heartbeat"),
    ("kanban", "assignments"),
    ("kanban", "context"),
    ("bundles", "list"),
    ("bundles", "show"),
    ("bundles", "create"),
    ("bundles", "delete"),
    ("bundles", "reload"),
    ("checkpoints", "status"),
    ("checkpoints", "list"),
    ("checkpoints", "prune"),
    ("checkpoints", "clear"),
    ("checkpoints", "clear-legacy"),
    ("curator", "status"),
    ("curator", "run"),
    ("curator", "pause"),
    ("curator", "resume"),
    ("curator", "pin"),
    ("curator", "unpin"),
    ("curator", "restore"),
    ("curator", "list-archived"),
    ("curator", "archive"),
    ("curator", "prune"),
    ("curator", "backup"),
    ("curator", "rollback"),
    ("pets", "list"),
    ("pets", "install"),
    ("pets", "select"),
    ("pets", "show"),
    ("pets", "off"),
    ("pets", "scale"),
    ("pets", "remove"),
    ("pets", "doctor"),
}


MUTATING_CONFIRMATION_SMOKE_COMMANDS = [
    "config set console.test true",
    "config migrate",
    "sessions rename abc123 new title",
    "sessions optimize",
    "cron create 'every 1h' 'say hello'",
    "cron remove abc123",
    "profile create tester --no-alias --no-skills",
    "profile delete tester",
    "tools disable web",
    "plugins install owner/repo --no-enable",
    "skills install openai/skills/example",
    "mcp add demo --url https://example.com/sse",
    "mcp configure github",
    "mcp picker",
    "backup --quick -o /tmp/hermes-console-test.zip",
    "import /tmp/hermes-console-test.zip",
    "send --to telegram hello",
    "memory reset --target memory",
    "auth remove openrouter 1",
    "pairing approve abc123",
    "webhook subscribe test --prompt hello",
    "hooks test pre_tool_call",
    "project create demo",
    "kanban create 'demo task'",
    "bundles create demo --skill skill-a",
    "checkpoints prune",
    "curator pause",
    "pets install cat",
]












def test_sessions_list_and_stats_use_isolated_session_store(_isolate_hermes_home):
    from hermes_state import SessionDB

    db = SessionDB()
    try:
        db.create_session("chat-session", source="cli", model="test/model")
        db.create_session("tool-session", source="tool", model="test/model")
    finally:
        db.close()

    engine = HermesConsoleEngine()
    listed = engine.execute("sessions list --limit 10")
    stats = engine.execute("sessions stats")

    assert listed.status == "ok"
    assert "chat-session" in listed.output
    assert "tool-session" not in listed.output
    assert "Total sessions: 2" in stats.output
    assert "Listable sessions: 1" in stats.output


def test_sessions_export_rejects_oversized_single_before_touching_output(
    _isolate_hermes_home,
    monkeypatch,
    tmp_path,
):
    import hermes_state
    from hermes_state import SessionDB

    db = SessionDB()
    try:
        db.create_session("too-large", source="cli")
        db.append_messages_batch(
            "too-large",
            [{"role": "user", "content": f"message-{i}"} for i in range(3)],
        )
    finally:
        db.close()

    monkeypatch.setattr(hermes_state, "resolved_max_export_messages", lambda: 2)
    materialized = []
    original_export_session = SessionDB.export_session

    def tracked_export_session(self, session_id):
        materialized.append(session_id)
        return original_export_session(self, session_id)

    monkeypatch.setattr(SessionDB, "export_session", tracked_export_session)
    output = tmp_path / "sessions.jsonl"
    output.write_text("keep me\n", encoding="utf-8")

    result = HermesConsoleEngine().execute(
        f"sessions export {output} --session-id too-large",
        confirmed=True,
    )

    assert result.status == "error"
    assert "too-large" in result.output
    assert "streaming Export" in result.output
    assert "resume" not in result.output.lower()
    assert materialized == []
    assert output.read_text(encoding="utf-8") == "keep me\n"


def test_sessions_export_all_uses_per_session_budget(
    _isolate_hermes_home,
    monkeypatch,
    tmp_path,
):
    """N small sessions export fine; ONE oversized session still rejects.

    The budget is per session, not cumulative across the export set —
    a cumulative budget broke full-DB backups of many small sessions.
    """
    import json

    import hermes_state
    from hermes_state import SessionDB

    db = SessionDB()
    try:
        for name in ("first-safe", "second-safe", "third-safe"):
            db.create_session(name, source="cli")
            db.append_messages_batch(
                name,
                [{"role": "user", "content": f"{name}-{i}"} for i in range(2)],
            )
    finally:
        db.close()

    monkeypatch.setattr(hermes_state, "resolved_max_export_messages", lambda: 3)
    output = tmp_path / "all-sessions.jsonl"

    # 3 sessions x 2 messages = 6 total > 3, but each session is under the
    # per-session limit, so the full-DB export succeeds.
    result = HermesConsoleEngine().execute(
        f"sessions export {output}",
        confirmed=True,
    )
    assert result.status == "ok"
    exported = [
        json.loads(line)
        for line in output.read_text(encoding="utf-8").splitlines()
        if line
    ]
    assert {row["id"] for row in exported} == {
        "first-safe",
        "second-safe",
        "third-safe",
    }


def test_sessions_export_all_rejects_single_oversized_session(
    _isolate_hermes_home,
    monkeypatch,
    tmp_path,
):
    import hermes_state
    from hermes_state import SessionDB

    db = SessionDB()
    try:
        db.create_session("small", source="cli")
        db.append_messages_batch(
            "small",
            [{"role": "user", "content": f"small-{i}"} for i in range(2)],
        )
        db.create_session("runaway", source="cli")
        db.append_messages_batch(
            "runaway",
            [{"role": "user", "content": f"runaway-{i}"} for i in range(4)],
        )
    finally:
        db.close()

    monkeypatch.setattr(hermes_state, "resolved_max_export_messages", lambda: 3)
    export_all_calls = []

    def tracked_export_all(self, source=None):
        export_all_calls.append(source)
        raise AssertionError("export_all must not run before every guard passes")

    monkeypatch.setattr(SessionDB, "export_all", tracked_export_all)
    output = tmp_path / "all-sessions.jsonl"

    result = HermesConsoleEngine().execute(
        f"sessions export {output}",
        confirmed=True,
    )

    assert result.status == "error"
    assert "runaway" in result.output
    assert "more than 3 active" in result.output
    assert "streaming Export" in result.output
    assert "max_export_messages" in result.output
    assert export_all_calls == []
    assert not output.exists()


def test_sessions_export_zero_limit_disables_guard(
    _isolate_hermes_home,
    monkeypatch,
    tmp_path,
):
    import hermes_state
    from hermes_state import SessionDB

    db = SessionDB()
    try:
        db.create_session("huge", source="cli")
        db.append_messages_batch(
            "huge",
            [{"role": "user", "content": f"huge-{i}"} for i in range(5)],
        )
    finally:
        db.close()

    monkeypatch.setattr(hermes_state, "resolved_max_export_messages", lambda: 0)
    output = tmp_path / "huge.jsonl"

    result = HermesConsoleEngine().execute(
        f"sessions export {output} --session-id huge",
        confirmed=True,
    )
    assert result.status == "ok"
    assert output.exists()


def test_cron_pause_resume_and_run_require_confirmation(_isolate_hermes_home):
    from cron.jobs import create_job, get_job

    job = create_job(prompt="say hello", schedule="every 1h", name="alpha")
    engine = HermesConsoleEngine()

    pending = engine.execute(f"cron pause {job['id']}")
    assert pending.status == "confirm_required"
    stored = get_job(job["id"])
    assert stored is not None
    assert stored["state"] == "scheduled"

    paused = engine.execute(f"cron pause {job['id']}", confirmed=True)
    assert paused.status == "ok"
    stored = get_job(job["id"])
    assert stored is not None
    assert stored["state"] == "paused"

    resumed = engine.execute("cron resume alpha", confirmed=True)
    assert resumed.status == "ok"
    stored = get_job(job["id"])
    assert stored is not None
    assert stored["state"] == "scheduled"

    triggered = engine.execute("cron run alpha", confirmed=True)
    assert triggered.status == "ok"
    assert "Triggered job" in triggered.output


def test_repl_runs_non_interactive_lines_without_prompts(_isolate_hermes_home):
    stdin = io.StringIO("help\nexit\n")
    stdout = io.StringIO()
    stderr = io.StringIO()

    code = run_console_repl(
        stdin=stdin,
        stdout=stdout,
        stderr=stderr,
        interactive=False,
    )

    assert code == 0
    assert "Hermes Console" in stdout.getvalue()
    assert "hermes>" not in stdout.getvalue()
    assert stderr.getvalue() == ""


def test_capture_output_surfaces_string_exit_code_as_command_error():
    from hermes_cli.console_engine import ConsoleCommandError, _capture_output

    def _boom():
        sys.exit("No credential matching \"nope\".")

    with pytest.raises(ConsoleCommandError) as exc_info:
        _capture_output(_boom)

    assert "No credential matching" in str(exc_info.value)


def test_capture_output_preserves_integer_exit_code_message():
    from hermes_cli.console_engine import ConsoleCommandError, _capture_output

    with pytest.raises(ConsoleCommandError) as exc_info:
        _capture_output(lambda: sys.exit(3))

    assert "status 3" in str(exc_info.value)


def test_execute_handler_string_exit_returns_error_not_crash(_isolate_hermes_home):
    result = HermesConsoleEngine().execute(
        "auth remove openrouter __no_such_credential__", confirmed=True
    )

    assert result.status == "error"
    assert result.output
