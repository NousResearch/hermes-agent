"""Worker diagnostics must belong to the latest attempt, not an earlier profile."""

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_dispatch as dispatch
from hermes_cli.kanban_worker_diagnostics import _exit_summary_marker
from hermes_cli.quiet_single_query import KANBAN_WORKER_EXIT_TRAILER


@pytest.mark.parametrize("new_output", ["", "No usable credentials found for provider 'xai'.\n"])
def test_new_attempt_without_exit_trailer_cannot_inherit_old_result(tmp_path, monkeypatch, new_output):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    task = kb.Task(
        id="t_fixture", title="fixture", body=None, assignee="ads", status="running",
        priority=0, created_by=None, created_at=0, started_at=0, completed_at=None,
        workspace_kind="none", workspace_path=None, claim_lock=None, claim_expires=None,
        tenant=None, current_run_id=302,
    )
    with dispatch._open_worker_log(task, "diagnostics") as log:
        log.write(f"previous profile output\n{KANBAN_WORKER_EXIT_TRAILER}0\n".encode())
    task.current_run_id = 303
    with dispatch._open_worker_log(task, "diagnostics") as log:
        # The start boundary must already be durable before child stdout is written.
        raw = kb.read_worker_log(task.id, board="diagnostics")
        assert raw is not None
        assert raw.endswith("[hermes-kanban-worker-start] run_id=303\n")
        log.write(new_output.encode())

    assert dispatch._worker_final_output(task.id, board="diagnostics") == new_output.strip()
    assert dispatch._worker_log_exit_code(task.id, board="diagnostics") is None


@pytest.mark.parametrize("footer", ["quiet", "summary"])
@pytest.mark.parametrize("current_trailer", [False, True])
def test_legacy_footer_keeps_new_startup_failure(tmp_path, monkeypatch, footer, current_trailer):
    from agent.i18n import t

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    path = kb.worker_log_path("t_fixture", board="diagnostics")
    path.parent.mkdir(parents=True, exist_ok=True)
    old_footer = "session_id: old\n" if footer == "quiet" else (
        f"{_exit_summary_marker()}\n  hermes --resume old\n"
        + t("cli.session.exit_label_messages", count=2, user=1, tool_calls=0) + "\n"
    )
    path.write_text(
        "old OAuth org rejected\n" + old_footer
        + "No usable credentials found for provider 'xai'.\n"
        + (f"{KANBAN_WORKER_EXIT_TRAILER}78\n" if current_trailer else ""),
        encoding="utf-8",
    )

    assert dispatch._worker_final_output("t_fixture", board="diagnostics") == (
        "No usable credentials found for provider 'xai'."
    )
    assert dispatch._worker_log_exit_code("t_fixture", board="diagnostics") == (
        78 if current_trailer else None
    )


@pytest.mark.parametrize("previous_code", [0, 78])
@pytest.mark.parametrize("current_code", [1, 78])
def test_pre_summary_failure_does_not_return_previous_attempt(tmp_path, monkeypatch, previous_code, current_code):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    path = kb.worker_log_path("t_fixture", board="diagnostics")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "previous profile output\n"
        f"{_exit_summary_marker()}\n  hermes --resume previous\n"
        f"{KANBAN_WORKER_EXIT_TRAILER}{previous_code}\n"
        "No usable credentials found for provider 'xai'.\n"
        f"{KANBAN_WORKER_EXIT_TRAILER}{current_code}\n",
        encoding="utf-8",
    )

    assert dispatch._worker_final_output("t_fixture", board="diagnostics") == (
        "No usable credentials found for provider 'xai'."
    )
    assert dispatch._worker_log_exit_code("t_fixture", board="diagnostics") == current_code
    monkeypatch.setattr(dispatch, "_classify_worker_exit", lambda pid: ("unknown", None))
    dead = dispatch._classify_dead_worker(12345, "ads", task_id="t_fixture", board="diagnostics")
    assert dead.code == current_code
    assert dead.terminal_provider is (current_code == 78)
    assert dead.event_payload["worker_output"] == "No usable credentials found for provider 'xai'."
    assert "previous profile output" not in dead.error_text


@pytest.mark.parametrize("footer_kind", ["quiet", "summary"])
def test_legacy_completed_attempt_retains_only_current_output(tmp_path, monkeypatch, footer_kind):
    from agent.i18n import t

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    path = kb.worker_log_path("t_fixture", board="diagnostics")
    path.parent.mkdir(parents=True, exist_ok=True)
    footer = "session_id: fixture\n" if footer_kind == "quiet" else (
        f"{_exit_summary_marker()}\n  hermes --resume fixture\n"
        + t("cli.session.exit_label_messages", count=2, user=1, tool_calls=0)
        + "\nsession_id: fixture\n"
    )
    path.write_text("old output\n" + footer + "current output\n" + footer, encoding="utf-8")

    assert dispatch._worker_final_output("t_fixture", board="diagnostics") == "current output"
