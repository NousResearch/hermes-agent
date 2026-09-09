"""Phase-C read-only task/project status command contract."""

from __future__ import annotations

import json
from pathlib import Path
import sqlite3
from unittest.mock import MagicMock

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_notify as kbn
from hermes_cli import projects_db as pdb


@pytest.fixture
def status_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    for name in ("HERMES_KANBAN_BOARD", "HERMES_KANBAN_DB", "HERMES_KANBAN_HOME"):
        monkeypatch.delenv(name, raising=False)
    kb._INITIALIZED_PATHS.clear()
    return home


def _create_task(board: str, title: str, monkeypatch: pytest.MonkeyPatch, task_id: str) -> str:
    kb.create_board(board)
    monkeypatch.setattr(kb, "_new_task_id", lambda: task_id)
    with kbc.connect(board=board) as conn:
        return kb.create_task(conn, title=title, assignee="rozmilo-codex", board=board)


def test_resolver_prefers_exact_task_id_then_unambiguous_reference(
    status_home: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from hermes_cli.kanban_status import resolve_status_reference

    task_id = _create_task("alpha", "Checkout hardening", monkeypatch, "t_a1b2c3d4")

    exact = resolve_status_reference(task_id)
    by_title = resolve_status_reference("checkout hardening")

    assert exact.ok is True
    assert (exact.scope, exact.task_id, exact.board) == ("task", task_id, "alpha")
    assert by_title.ok is True
    assert (by_title.scope, by_title.task_id, by_title.board) == ("task", task_id, "alpha")


def test_ambiguous_reference_fails_closed_with_candidates(
    status_home: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from hermes_cli.kanban_status import resolve_status_reference

    first = _create_task("alpha", "Checkout", monkeypatch, "t_11111111")
    second = _create_task("beta", "Checkout", monkeypatch, "t_22222222")

    resolved = resolve_status_reference("checkout")

    assert resolved.ok is False
    assert resolved.error == "ambiguous reference 'checkout'"
    assert [(c.id, c.name, c.board) for c in resolved.candidates] == [
        (first, "Checkout", "alpha"),
        (second, "Checkout", "beta"),
    ]


def test_explicit_board_takes_precedence_for_reference_resolution(
    status_home: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from hermes_cli.kanban_status import resolve_status_reference

    _create_task("alpha", "Checkout", monkeypatch, "t_11111111")
    expected = _create_task("beta", "Checkout", monkeypatch, "t_22222222")

    resolved = resolve_status_reference("checkout", board="beta")

    assert resolved.ok is True
    assert (resolved.task_id, resolved.board) == (expected, "beta")


def test_exact_task_id_precedes_an_explicit_board_filter(
    status_home: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from hermes_cli.kanban_status import resolve_status_reference

    expected = _create_task("alpha", "Exact identity", monkeypatch, "t_aaaa1111")
    _create_task("beta", "Other", monkeypatch, "t_bbbb2222")

    resolved = resolve_status_reference(expected, board="beta")

    assert resolved.ok is True
    assert (resolved.task_id, resolved.board) == (expected, "alpha")


def test_archived_tasks_remain_excluded_by_default_but_resolve_when_opted_in(
    status_home: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from hermes_cli.kanban_status import resolve_status_reference

    task_id = _create_task("alpha", "Archived checkout", monkeypatch, "t_archived1")
    with kbc.connect(board="alpha") as conn:
        conn.execute("UPDATE tasks SET status = 'archived' WHERE id = ?", (task_id,))
        conn.commit()

    default = resolve_status_reference(task_id)
    opted_in = resolve_status_reference(task_id, include_archived=True)
    by_title = resolve_status_reference("archived checkout", include_archived=True)

    assert default.ok is False
    assert opted_in.ok is True
    assert (opted_in.scope, opted_in.task_id, opted_in.board) == ("task", task_id, "alpha")
    assert (by_title.scope, by_title.task_id, by_title.board) == ("task", task_id, "alpha")


def test_archived_reference_opt_in_preserves_ambiguity_fail_closed(
    status_home: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from hermes_cli.kanban_status import resolve_status_reference

    first = _create_task("alpha", "Archived checkout", monkeypatch, "t_archived2")
    second = _create_task("beta", "Archived checkout", monkeypatch, "t_archived3")
    for board, task_id in (("alpha", first), ("beta", second)):
        with kbc.connect(board=board) as conn:
            conn.execute("UPDATE tasks SET status = 'archived' WHERE id = ?", (task_id,))
            conn.commit()

    resolved = resolve_status_reference("archived checkout", include_archived=True)

    assert resolved.ok is False
    assert resolved.error == "ambiguous reference 'archived checkout'"


def test_project_default_board_is_used_when_authoritative(status_home: Path) -> None:
    from hermes_cli.kanban_status import resolve_status_reference

    kb.create_board("alpha")
    with pdb.connect_closing() as conn:
        project_id = pdb.create_project(conn, name="Web Store", board_slug="alpha")

    resolved = resolve_status_reference("Web Store")

    assert resolved.ok is True
    assert (resolved.scope, resolved.project_id, resolved.board) == (
        "project", project_id, "alpha"
    )


def test_explicit_board_disambiguates_project_name(status_home: Path) -> None:
    from hermes_cli.kanban_status import resolve_status_reference

    with pdb.connect_closing() as conn:
        alpha_project = pdb.create_project(conn, name="Shared Name", board_slug="alpha")
        pdb.create_project(conn, name="Shared Name", board_slug="beta")
    kb.create_board("alpha", project_id=alpha_project)
    kb.create_board("beta")

    resolved = resolve_status_reference("Shared Name", board="alpha")

    assert resolved.ok is True
    assert (resolved.scope, resolved.project_id, resolved.board) == (
        "project", alpha_project, "alpha"
    )


def test_exact_project_slug_precedes_a_task_title_collision(
    status_home: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from hermes_cli.kanban_status import resolve_status_reference

    _create_task("alpha", "project-key", monkeypatch, "t_99999999")
    with pdb.connect_closing() as conn:
        project_id = pdb.create_project(conn, name="Other Name", slug="project-key")

    resolved = resolve_status_reference("project-key")

    assert resolved.ok is True
    assert (resolved.scope, resolved.project_id) == ("project", project_id)


def test_unreadable_board_fails_closed_instead_of_reporting_no_match(
    status_home: Path,
) -> None:
    from hermes_cli.kanban_status import resolve_status_reference

    kb.create_board("alpha")
    kb.kanban_db_path(board="alpha").write_bytes(b"not a sqlite database")

    resolved = resolve_status_reference("anything")

    assert resolved.ok is False
    assert resolved.candidates == ()
    assert resolved.error.startswith("status read failed:")


@pytest.mark.parametrize(
    "state",
    ("triage", "todo", "ready", "running", "review", "blocked", "scheduled", "done"),
)
def test_all_supported_states_have_concise_telegram_rendering(state: str) -> None:
    from hermes_cli.kanban_status import ProjectStatusResult, render_project_status

    result = ProjectStatusResult(
        ok=True,
        scope="task",
        task_id="t_a1b2c3d4",
        board="alpha",
        state=state,
        profile="rozmilo-codex",
        failure_loop_count=2,
        next_action="inspect",
    )

    rendered = render_project_status(result)

    assert rendered.splitlines()[0] == f"t_a1b2c3d4 — {state}"
    assert "Profile: rozmilo-codex" in rendered
    assert "Failures: 2" in rendered
    assert "Production: not inferred" in rendered
    assert len(rendered) < 1000


def _table_snapshot(path: Path) -> dict[str, list[tuple]]:
    conn = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        return {
            "tasks": conn.execute("SELECT * FROM tasks ORDER BY rowid").fetchall(),
            "task_events": conn.execute(
                "SELECT * FROM task_events ORDER BY rowid"
            ).fetchall(),
            "task_runs": conn.execute("SELECT * FROM task_runs ORDER BY rowid").fetchall(),
            "kanban_notify_subs": conn.execute(
                "SELECT * FROM kanban_notify_subs ORDER BY rowid"
            ).fetchall(),
        }
    finally:
        conn.close()


def test_status_is_read_only_idempotent_and_profile_null_is_advisory(
    status_home: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from hermes_cli.kanban_status import get_project_status

    task_id = _create_task("alpha", "Legacy route", monkeypatch, "t_33333333")
    with kbc.connect(board="alpha") as conn:
        assert kb.assign_task(conn, task_id, None) is True
        kbn.add_notify_sub(
            conn,
            task_id=task_id,
            platform="telegram",
            chat_id="legacy-chat",
            notifier_profile=None,
        )
    db_path = kb.kanban_db_path(board="alpha")
    before = _table_snapshot(db_path)

    first = get_project_status(task_id)
    second = get_project_status(task_id)

    assert first == second
    assert first.ok is True
    assert first.read_only is True
    assert first.profile is None
    assert first.notification_route == {"advisory_unowned": True, "owned_profiles": []}
    assert _table_snapshot(db_path) == before


def test_structured_snapshot_bounds_lifecycle_payloads(
    status_home: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from hermes_cli.kanban_status import get_project_status

    task_id = _create_task("alpha", "Bounded status", monkeypatch, "t_77777777")
    with kbc.connect(board="alpha") as conn:
        assert kb.block_task(conn, task_id, reason="x" * 20_000)

    result = get_project_status(task_id)

    assert result.ok is True
    assert len(result.latest_lifecycle_event["payload"]["reason"]) <= 1001
    assert len(json.dumps(result.to_dict())) < 10_000


def test_archived_parent_satisfies_dependency_gate(
    status_home: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from hermes_cli.kanban_status import get_project_status

    parent_id = _create_task("alpha", "Parent", monkeypatch, "t_77777778")
    child_id = _create_task("alpha", "Child", monkeypatch, "t_77777779")
    with kbc.connect(board="alpha") as conn:
        kb.link_tasks(conn, parent_id, child_id)
        assert kb.archive_task(conn, parent_id)

    result = get_project_status(child_id, board="alpha")

    assert result.ok is True
    assert result.dependency_state["satisfied"] is True


def test_cleared_block_does_not_report_stale_policy_reason(
    status_home: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from hermes_cli.kanban_status import get_project_status

    task_id = _create_task("alpha", "Recovered task", monkeypatch, "t_77777780")
    with kbc.connect(board="alpha") as conn:
        assert kb.block_task(conn, task_id, reason="temporary policy hold")
        assert kb.unblock_task(conn, task_id)

    result = get_project_status(task_id, board="alpha")

    assert result.ok is True
    assert result.policy_block_reason is None


def test_interactive_alias_parses_reference(
    status_home: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from hermes_cli.kanban_status import run_project_status_slash

    task_id = _create_task("alpha", "Alias target", monkeypatch, "t_77777781")

    rendered = run_project_status_slash(f"/project_status {task_id} --board alpha")

    assert rendered.splitlines()[0] == f"{task_id} — ready"


def test_done_never_implies_merge_deployment_or_production(
    status_home: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from hermes_cli.kanban_status import get_project_status, render_project_status

    task_id = _create_task("alpha", "Finished workflow", monkeypatch, "t_44444444")
    with kbc.connect(board="alpha") as conn:
        claimed = kb.claim_task(conn, task_id, claimer="worker")
        assert claimed is not None
        assert kb.complete_task(
            conn,
            task_id,
            result="Kanban complete",
            expected_run_id=claimed.current_run_id,
        )

    result = get_project_status(task_id)

    assert result.state == "done"
    assert result.github == {"pr": None, "merge_state": None}
    assert result.deployment == {"deployment_state": None, "production_state": None, "inferred": False}
    assert "Production: not inferred" in render_project_status(result)


def _gateway_event(text: str):
    from gateway.config import Platform
    from gateway.platforms.base import MessageEvent
    from gateway.session import SessionSource

    return MessageEvent(
        text=text,
        source=SessionSource(platform=Platform.TELEGRAM, user_id="operator", chat_id="chat"),
    )


@pytest.mark.asyncio
async def test_feature_flag_disabled_preserves_prior_behavior_without_reading_status(
    status_home: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from gateway.run import GatewayRunner
    from hermes_cli import kanban_status
    from hermes_cli.commands import gateway_help_lines, resolve_command
    from hermes_cli.config_defaults import DEFAULT_CONFIG

    called = False

    def unexpected_read(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("disabled command must not read status")

    monkeypatch.setattr(kanban_status, "get_project_status", unexpected_read)
    result = await GatewayRunner._handle_project_status_command(
        object.__new__(GatewayRunner), _gateway_event("/project-status t_12345678")
    )

    command = resolve_command("project-status")
    assert command is not None
    assert command.gateway_config_gate == "kanban.project_status_command"
    assert DEFAULT_CONFIG["kanban"]["project_status_command"] is False
    assert "not enabled" in result.lower()
    assert called is False
    assert "`/project-status" not in "\n".join(gateway_help_lines())


@pytest.mark.asyncio
async def test_enabled_gateway_command_renders_status(
    status_home: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from gateway.run import GatewayRunner

    task_id = _create_task("alpha", "Gateway status", monkeypatch, "t_55555555")
    (status_home / "config.yaml").write_text(
        "kanban:\n  project_status_command: true\n", encoding="utf-8"
    )

    result = await GatewayRunner._handle_project_status_command(
        object.__new__(GatewayRunner),
        _gateway_event(f"/project_status {task_id} --board alpha"),
    )

    assert result.splitlines()[0] == f"{task_id} — ready"
    assert "Production: not inferred" in result


@pytest.mark.asyncio
async def test_existing_kanban_gateway_handler_remains_compatible(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from gateway.run import GatewayRunner
    from hermes_cli import kanban

    monkeypatch.setattr(kanban, "run_slash", MagicMock(return_value="existing kanban output"))

    result = await GatewayRunner._handle_kanban_command(
        object.__new__(GatewayRunner), _gateway_event("/kanban list")
    )

    assert result == "existing kanban output"
    kanban.run_slash.assert_called_once_with("list")
