"""Regression coverage for review routing under per-profile caps."""
from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb


@pytest.fixture
def kanban_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _add_profile(home: Path, name: str) -> None:
    (home / "profiles" / name).mkdir(parents=True)


def test_implicit_reviewer_avoids_busy_implementer_and_dispatches(
    kanban_home: Path,
) -> None:
    """A review without reviewer= must not remain behind its implementer's cap."""
    _add_profile(kanban_home, "builder")
    spawned: list[tuple[str, str]] = []

    def spawn(task, _workspace):
        spawned.append((task.id, task.assignee or ""))
        return 12345

    with kb.connect() as conn:
        busy_id = kb.create_task(conn, title="long implementation", assignee="builder")
        assert kb.claim_task(conn, busy_id) is not None

        review_id = kb.create_task(conn, title="finished implementation", assignee="builder")
        implementation = kb.claim_task(conn, review_id)
        assert implementation is not None
        assert kb.request_review(
            conn,
            review_id,
            summary="ready",
            expected_run_id=implementation.current_run_id,
        )

        review = kb.get_task(conn, review_id)
        assert review is not None
        assert review.assignee == "default"

        first = kb.dispatch_once(
            conn,
            spawn_fn=spawn,
            max_in_progress=2,
            max_in_progress_per_profile=1,
        )
        assert first.spawned == [(review_id, "default", first.spawned[0][2])]
        assert not first.skipped_per_profile_capped
        assert spawned == [(review_id, "default")]


def test_explicit_reviewer_is_honored_verbatim(kanban_home: Path) -> None:
    _add_profile(kanban_home, "builder")
    _add_profile(kanban_home, "verifier")

    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="explicit handoff", assignee="builder")
        implementation = kb.claim_task(conn, task_id)
        assert implementation is not None
        assert kb.request_review(
            conn,
            task_id,
            reviewer="Verifier",
            expected_run_id=implementation.current_run_id,
        )
        task = kb.get_task(conn, task_id)
        assert task is not None
        assert task.assignee == "verifier"


def test_configured_default_reviewer_wins_over_other_profiles(
    kanban_home: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _add_profile(kanban_home, "builder")
    _add_profile(kanban_home, "auditor")
    _add_profile(kanban_home, "verifier")
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {"kanban": {"default_reviewer": "Verifier"}},
    )

    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="configured handoff", assignee="builder")
        implementation = kb.claim_task(conn, task_id)
        assert implementation is not None
        assert kb.request_review(
            conn,
            task_id,
            expected_run_id=implementation.current_run_id,
        )
        task = kb.get_task(conn, task_id)
        assert task is not None
        assert task.assignee == "verifier"


def test_ready_lane_still_enforces_per_profile_cap(kanban_home: Path) -> None:
    _add_profile(kanban_home, "builder")

    with kb.connect() as conn:
        running_id = kb.create_task(conn, title="running", assignee="builder")
        assert kb.claim_task(conn, running_id) is not None
        ready_id = kb.create_task(conn, title="queued", assignee="builder")

        result = kb.dispatch_once(
            conn,
            dry_run=True,
            max_in_progress=2,
            max_in_progress_per_profile=1,
        )

        assert ready_id not in [task_id for task_id, _who, _ws in result.spawned]
        assert result.skipped_per_profile_capped == [(ready_id, "builder", 1)]


def test_single_profile_self_review_has_distinct_cap_diagnostic(
    kanban_home: Path,
) -> None:
    with kb.connect() as conn:
        busy_id = kb.create_task(conn, title="running", assignee="default")
        assert kb.claim_task(conn, busy_id) is not None
        review_id = kb.create_task(conn, title="review", assignee="default")
        implementation = kb.claim_task(conn, review_id)
        assert implementation is not None
        assert kb.request_review(
            conn,
            review_id,
            expected_run_id=implementation.current_run_id,
        )

        result = kb.dispatch_once(
            conn,
            dry_run=True,
            max_in_progress=2,
            max_in_progress_per_profile=1,
        )

        assert result.skipped_review_self_capped == [(review_id, "default", 1)]
        assert not result.skipped_per_profile_capped
