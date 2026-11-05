from __future__ import annotations

import io
import sys

import pytest

from hermes_cli.console_engine import HermesConsoleEngine, run_console_repl


def _track_maintenance(monkeypatch, name, calls, *, raise_on_call=False):
    """Record every ``maintenance.<name>`` call the console's store makes (raising instead when asked)."""
    from contextlib import contextmanager
    from types import SimpleNamespace

    from hermes_cli import console_engine

    real_session_db = console_engine._session_db

    @contextmanager
    def tracked_session_db(*, read_only=True):
        with real_session_db(read_only=read_only) as store:
            maintenance = store.maintenance

            def tracked(*args, **kwargs):
                calls.append(args[0] if args else kwargs.get("source"))
                if raise_on_call:
                    raise AssertionError(f"{name} must not run before every guard passes")
                return getattr(maintenance, name)(*args, **kwargs)

            yield SimpleNamespace(sessions=store.sessions, maintenance=SimpleNamespace(
                assert_export_safe=maintenance.assert_export_safe, **{name: tracked}))

    monkeypatch.setattr(console_engine, "_session_db", tracked_session_db)


def test_sessions_optimize_accepts_the_force_override_it_advertises(_isolate_hermes_home):
    """The held-store refusal this command prints points at `sessions optimize --force`; if the
    console rejected the flag, the command could only ever refuse whenever a gateway is running."""
    result = HermesConsoleEngine().execute("sessions optimize --force", confirmed=True)

    assert result.status == "ok", result.output
    assert "Usage:" not in result.output


def test_sessions_list_and_stats_use_isolated_session_store(_isolate_hermes_home):
    from persistence_assembly import open_persistence, profile_scope
    from tests.fixtures.storage_seed import seed_session

    with open_persistence(profile_scope()) as store:
        seed_session(store, "chat-session", "cli", model="test/model")
        seed_session(store, "tool-session", "tool", model="test/model")

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
    from hermes_cli import session_export
    from persistence_assembly import open_persistence, profile_scope
    from tests.fixtures.storage_seed import seed_messages, seed_session

    with open_persistence(profile_scope()) as store:
        seed_session(store, "too-large", "cli")
        seed_messages(store, "too-large", *[{"role": "user", "content": f"message-{i}"} for i in range(3)])

    monkeypatch.setattr(session_export, "resolved_max_export_messages", lambda: 2)
    materialized = []
    _track_maintenance(monkeypatch, "export_session", materialized)
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

    from hermes_cli import session_export
    from persistence_assembly import open_persistence, profile_scope
    from tests.fixtures.storage_seed import seed_messages, seed_session

    with open_persistence(profile_scope()) as store:
        for name in ("first-safe", "second-safe", "third-safe"):
            seed_session(store, name, "cli")
            seed_messages(store, name, *[{"role": "user", "content": f"{name}-{i}"} for i in range(2)])

    monkeypatch.setattr(session_export, "resolved_max_export_messages", lambda: 3)
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
    from hermes_cli import session_export
    from persistence_assembly import open_persistence, profile_scope
    from tests.fixtures.storage_seed import seed_messages, seed_session

    with open_persistence(profile_scope()) as store:
        seed_session(store, "small", "cli")
        seed_messages(store, "small", *[{"role": "user", "content": f"small-{i}"} for i in range(2)])
        seed_session(store, "runaway", "cli")
        seed_messages(store, "runaway", *[{"role": "user", "content": f"runaway-{i}"} for i in range(4)])

    monkeypatch.setattr(session_export, "resolved_max_export_messages", lambda: 3)
    export_all_calls = []
    _track_maintenance(monkeypatch, "export_all", export_all_calls, raise_on_call=True)
    output = tmp_path / "all-sessions.jsonl"

    result = HermesConsoleEngine().execute(
        f"sessions export {output}",
        confirmed=True,
    )

    assert result.status == "error"
    assert "runaway" in result.output
    assert "more than 3 exportable" in result.output
    assert "streaming Export" in result.output
    assert "max_export_messages" in result.output
    assert export_all_calls == []
    assert not output.exists()


def test_sessions_export_zero_limit_disables_guard(
    _isolate_hermes_home,
    monkeypatch,
    tmp_path,
):
    from hermes_cli import session_export
    from persistence_assembly import open_persistence, profile_scope
    from tests.fixtures.storage_seed import seed_messages, seed_session

    with open_persistence(profile_scope()) as store:
        seed_session(store, "huge", "cli")
        seed_messages(store, "huge", *[{"role": "user", "content": f"huge-{i}"} for i in range(5)])

    monkeypatch.setattr(session_export, "resolved_max_export_messages", lambda: 0)
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




def test_execute_handler_string_exit_returns_error_not_crash(_isolate_hermes_home):
    result = HermesConsoleEngine().execute(
        "auth remove openrouter __no_such_credential__", confirmed=True
    )

    assert result.status == "error"
    assert result.output


_ORPHAN_STORE_STATUS = {
    "projects": [
        {"hash": "abc123", "workdir": "/gone/v2-project", "exists": False, "commits": 4},
    ],
    "pre_v2_projects": [],
}


def _patch_checkpoint_manager(monkeypatch, prune_calls: list) -> None:
    """Report one orphan project and record the resulting prune call."""
    import tools.checkpoint_maintenance as ckpt_mgr

    monkeypatch.setattr(ckpt_mgr, "store_status", lambda *a, **k: _ORPHAN_STORE_STATUS)

    def _fake_prune(**kwargs):
        prune_calls.append(kwargs)
        return {
            "scanned": 1,
            "deleted_orphan": 1,
            "deleted_stale": 0,
            "errors": 0,
            "bytes_freed": 0,
        }

    monkeypatch.setattr(ckpt_mgr, "prune_checkpoints", _fake_prune)


def test_console_checkpoints_prune_does_not_reprompt_for_orphans(
    _isolate_hermes_home, monkeypatch
):
    """`checkpoints prune` is console-mutating, so the nested prompt must be skipped.

    The console asks for confirmation itself before dispatching any command in the
    `checkpoints` mutating set, and `_apply_confirmed_defaults` exists to keep the
    CLI layer from asking a second time. `clear` and `clear-legacy` are force
    defaulted; `prune` was not, so its orphan confirmation still called `input()`.
    """
    prune_calls: list = []
    _patch_checkpoint_manager(monkeypatch, prune_calls)

    def _unexpected_input(_prompt):
        raise AssertionError(
            "input() must not be called: the console already confirmed `checkpoints prune`"
        )

    monkeypatch.setattr("builtins.input", _unexpected_input)

    result = HermesConsoleEngine().execute("checkpoints prune", confirmed=True)

    assert result.status == "ok"
    assert len(prune_calls) == 1
    assert prune_calls[0]["delete_orphans"] is True
    # No preview was shown, so there is nothing to bind the deletion to — the
    # documented `--force` case for `orphan_allowlist`.
    assert prune_calls[0]["orphan_allowlist"] is None




def test_config_set_on_unparseable_yaml_reports_error_not_crash(tmp_path, monkeypatch):
    """The fail-closed config write guard raises RuntimeError; the console must
    surface it as a command error, not let it escape execute() and kill the
    REPL / dashboard websocket session (regression for PR #71385 follow-up)."""
    config_path = tmp_path / "config.yaml"
    original = "model:\n  default: keep\nbroken: [unterminated\n"
    config_path.write_text(original, encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    result = HermesConsoleEngine().execute(
        "config set model.default gpt-4o", confirmed=True
    )

    assert result.status == "error"
    assert "formatting error" in (result.output or "")
    # The broken-but-recoverable file must survive untouched.
    assert config_path.read_text(encoding="utf-8") == original
