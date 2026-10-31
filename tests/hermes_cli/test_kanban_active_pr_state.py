"""Regression coverage for PR-state-aware Kanban respawn guards."""

from collections.abc import Iterator
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as dispatch
from hermes_cli import kanban_pr_acceptance as acceptance


@pytest.fixture
def kanban_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    acceptance._pr_state_cache.clear()
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    yield home
    acceptance._pr_state_cache.clear()


@pytest.mark.parametrize(
    ("number", "response", "expected"),
    [
        (44, {"state": "closed", "merged": False}, None),
        (45, {"state": "closed", "merged": True}, None),
        (46, {"state": "open", "merged": False}, "active_pr"),
    ],
    ids=["closed", "merged", "open"],
)
def test_active_pr_guard_follows_referenced_pr_state(
    kanban_home: Path,
    monkeypatch: pytest.MonkeyPatch,
    number: int,
    response: dict,
    expected: str | None,
) -> None:
    calls: list[tuple[str, str | None]] = []

    def fake_api(endpoint: str, **kwargs):
        calls.append((endpoint, kwargs.get("profile_home")))
        return response

    monkeypatch.setattr(acceptance, "_api", fake_api)

    with kbc.connect() as conn:
        task_id = kb.create_task(conn, title="finish PR", assignee="qe")
        kb.add_comment(
            conn,
            task_id,
            author="qe",
            body=f"PR https://github.com/example/project/pull/{number}",
        )

        assert dispatch.check_respawn_guard(conn, task_id) == expected

    assert calls == [(f"repos/example/project/pulls/{number}", None)]


@pytest.mark.parametrize(
    ("failure", "number"),
    [(OSError, 47), (acceptance._GateAuthError, 48)],
)
def test_active_pr_guard_fails_closed_when_pr_state_is_unavailable(
    kanban_home: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: type[Exception],
    number: int,
) -> None:
    def unavailable(*_args, **_kwargs):
        raise failure("gh unavailable")

    monkeypatch.setattr(acceptance, "_api", unavailable)

    with kbc.connect() as conn:
        task_id = kb.create_task(conn, title="finish PR", assignee="qe")
        kb.add_comment(
            conn,
            task_id,
            author="qe",
            body=f"PR https://github.com/example/project/pull/{number}",
        )

        assert dispatch.check_respawn_guard(conn, task_id) == "active_pr"
