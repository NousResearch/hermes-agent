"""Regression coverage for dispatcher-local profile ownership (#110995)."""

from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd


@pytest.fixture()
def kanban_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _spawned_ids(result: kbd.DispatchResult) -> list[str]:
    return [task_id for task_id, _assignee, _workspace in result.spawned]


def test_dispatch_only_claims_profiles_owned_by_its_current_home(
    kanban_home: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A shared board may be read by many homes, but only its owner may spawn."""
    import hermes_cli.profiles as profiles

    def profile_exists(name: str) -> bool:
        return name in {"default", "builder"}

    monkeypatch.setattr(profiles, "profile_exists", profile_exists)
    matches_current_home = {"default": False, "builder": True}
    monkeypatch.setattr(
        profiles,
        "profile_matches_home",
        lambda name: matches_current_home.get(name, False),
    )

    with kbc.connect() as conn:
        default_task = kb.create_task(conn, title="default card", assignee="default")
        named_task = kb.create_task(conn, title="named card", assignee="builder")
        control_task = kb.create_task(conn, title="control lane", assignee="orion-cc")

        non_default_result = kbd.dispatch_once(conn, dry_run=True)
        assert default_task not in _spawned_ids(non_default_result)
        assert default_task in non_default_result.skipped_nonspawnable
        assert named_task in _spawned_ids(non_default_result)
        assert control_task not in _spawned_ids(non_default_result)
        assert control_task in non_default_result.skipped_nonspawnable

        matches_current_home["default"] = True
        default_home_result = kbd.dispatch_once(conn, dry_run=True)
        assert default_task in _spawned_ids(default_home_result)
