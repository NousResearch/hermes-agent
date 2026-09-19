"""CLI contracts for pruning a frozen helper selection exactly."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

from hermes_cli.session_prune_selection import load_exact_prune_selection
from hermes_state import SessionDB


def _stamp_plan_identity(plan: dict) -> None:
    authority = {
        key: plan[key]
        for key in (
            "format",
            "source",
            "selection",
            "provenance",
            "candidates",
            "exclusions",
            "exclusion_predicates",
        )
    }
    encoded = json.dumps(
        authority,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    plan["plan_identity"] = hashlib.sha256(encoded).hexdigest()


def _write_plan(path: Path, database: Path, session_ids: list[str]) -> dict:
    plan = {
        "format": "hermes-session-plan/v2",
        "source": {
            "database": str(database.absolute()),
            "changed_during_read": False,
        },
        "selection": {"physical_session_ids": list(session_ids)},
        "provenance": {
            "helper_revision": "helper-test-revision",
            "native_revision": "native-test-revision",
            "installed_command_provenance": "test-source-checkout",
        },
        "candidates": [
            {
                "id": session_ids[-1],
                "physical_session_ids": session_ids,
            }
        ],
        "exclusions": {},
        "exclusion_predicates": {},
    }
    _stamp_plan_identity(plan)
    path.write_text(json.dumps(plan), encoding="utf-8")
    return plan


def _run_cli(monkeypatch, capsys, argv: list[str]) -> tuple[int, str]:
    import hermes_cli.main as main_mod

    monkeypatch.setattr(sys, "argv", ["hermes", "sessions", "prune", *argv])
    try:
        result = main_mod.main()
        code = int(result or 0)
    except SystemExit as exc:
        code = int(exc.code or 0)
    return code, capsys.readouterr().out


def test_exact_selection_plan_rejects_non_string_session_id(tmp_path: Path) -> None:
    database = tmp_path / "state.db"
    SessionDB(db_path=database).close()
    plan_file = tmp_path / "plan.json"
    plan = _write_plan(plan_file, database, ["selected"])
    plan["selection"]["physical_session_ids"] = [123]
    plan["candidates"] = [{"id": "selected", "physical_session_ids": [123]}]
    _stamp_plan_identity(plan)
    plan_file.write_text(json.dumps(plan), encoding="utf-8")

    with pytest.raises(ValueError, match="non-empty string list"):
        load_exact_prune_selection(plan_file, expected_database=database)


def test_selection_file_conflicts_with_each_normal_prune_filter(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    database = tmp_path / "state.db"
    db = SessionDB(db_path=database)
    db.create_session("selected", source="cron")
    db.end_session("selected", end_reason="done")
    db.set_session_archived("selected", True)
    db.close()
    plan_file = tmp_path / "plan.json"
    _write_plan(plan_file, database, ["selected"])
    import hermes_state

    monkeypatch.setattr(hermes_state, "SessionDB", lambda: SessionDB(db_path=database))
    for filter_flag in (
        ("--include-archived",),
        ("--include-pinned",),
        ("--never-active",),
    ):
        code, output = _run_cli(
            monkeypatch,
            capsys,
            ["--selection-file", str(plan_file), *filter_flag, "--yes"],
        )
        assert code == 1, output
        assert "cannot be combined" in output
        check = SessionDB(db_path=database)
        try:
            assert check.get_session("selected") is not None
        finally:
            check.close()


def test_selection_file_accepts_yes_interactive_confirmation(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    database = tmp_path / "state.db"
    db = SessionDB(db_path=database)
    db.create_session("selected", source="cron")
    db.end_session("selected", end_reason="done")
    db.set_session_archived("selected", True)
    db.close()
    plan_file = tmp_path / "plan.json"
    _write_plan(plan_file, database, ["selected"])
    import hermes_state

    monkeypatch.setattr(hermes_state, "SessionDB", lambda: SessionDB(db_path=database))
    monkeypatch.setattr("builtins.input", lambda _prompt="": "y")
    code, output = _run_cli(monkeypatch, capsys, ["--selection-file", str(plan_file)])

    check = SessionDB(db_path=database)
    try:
        assert code == 0, output
        assert check.get_session("selected") is None
    finally:
        check.close()


def test_selection_file_deletes_each_of_multiple_frozen_ids(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    database = tmp_path / "state.db"
    db = SessionDB(db_path=database)
    for session_id in ("one", "two"):
        db.create_session(session_id, source="cron")
        db.end_session(session_id, end_reason="done")
        db.set_session_archived(session_id, True)
    db.create_session("keeper", source="cron")
    db.end_session("keeper", end_reason="done")
    db.close()
    plan_file = tmp_path / "plan.json"
    _write_plan(plan_file, database, ["one", "two"])
    import hermes_state

    monkeypatch.setattr(hermes_state, "SessionDB", lambda: SessionDB(db_path=database))
    code, output = _run_cli(
        monkeypatch,
        capsys,
        ["--selection-file", str(plan_file), "--yes"],
    )

    check = SessionDB(db_path=database)
    try:
        assert code == 0, output
        assert check.get_session("one") is None
        assert check.get_session("two") is None
        assert check.get_session("keeper") is not None
    finally:
        check.close()


def test_exact_selection_plan_dry_run_lists_ids_without_deleting(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    database = tmp_path / "state.db"
    db = SessionDB(db_path=database)
    db.create_session("selected", source="cron")
    db.end_session("selected", end_reason="done")
    db.set_session_archived("selected", True)
    db.create_session("keeper", source="cron")
    db.end_session("keeper", end_reason="done")
    db.close()
    plan_file = tmp_path / "plan.json"
    _write_plan(plan_file, database, ["selected"])
    import hermes_state

    monkeypatch.setattr(hermes_state, "SessionDB", lambda: SessionDB(db_path=database))

    code, output = _run_cli(
        monkeypatch,
        capsys,
        ["--selection-file", str(plan_file), "--dry-run"],
    )

    check = SessionDB(db_path=database)
    try:
        assert code == 0, output
        assert "selected" in output
        assert "Dry run" in output
        assert check.get_session("selected") is not None
        assert check.get_session("keeper") is not None
    finally:
        check.close()


def test_exact_selection_plan_yes_deletes_only_frozen_ids(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    database = tmp_path / "state.db"
    db = SessionDB(db_path=database)
    db.create_session("selected", source="cron")
    db.end_session("selected", end_reason="done")
    db.set_session_archived("selected", True)
    db.create_session("keeper", source="cron")
    db.end_session("keeper", end_reason="done")
    db.close()
    plan_file = tmp_path / "plan.json"
    _write_plan(plan_file, database, ["selected"])
    import hermes_state

    monkeypatch.setattr(hermes_state, "SessionDB", lambda: SessionDB(db_path=database))

    code, output = _run_cli(
        monkeypatch,
        capsys,
        ["--selection-file", str(plan_file), "--yes"],
    )

    check = SessionDB(db_path=database)
    try:
        assert code == 0, output
        assert "Pruned 1 session" in output
        assert check.get_session("selected") is None
        assert check.get_session("keeper") is not None
    finally:
        check.close()


def test_exact_selection_plan_requires_nonempty_provenance(tmp_path: Path) -> None:
    database = tmp_path / "state.db"
    SessionDB(db_path=database).close()
    plan_file = tmp_path / "plan.json"
    plan = _write_plan(plan_file, database, ["selected"])
    plan["provenance"]["native_revision"] = ""
    _stamp_plan_identity(plan)
    plan_file.write_text(json.dumps(plan), encoding="utf-8")

    with pytest.raises(ValueError, match="provenance"):
        load_exact_prune_selection(plan_file, expected_database=database)


def test_exact_selection_plan_rejects_tampered_authority(tmp_path: Path) -> None:
    database = tmp_path / "state.db"
    SessionDB(db_path=database).close()
    plan_file = tmp_path / "plan.json"
    plan = _write_plan(plan_file, database, ["selected"])
    plan["selection"]["physical_session_ids"].append("not-reviewed")
    plan_file.write_text(json.dumps(plan), encoding="utf-8")

    with pytest.raises(ValueError, match="identity"):
        load_exact_prune_selection(plan_file, expected_database=database)


def test_exact_selection_plan_rejects_candidate_selection_mismatch(tmp_path: Path) -> None:
    database = tmp_path / "state.db"
    SessionDB(db_path=database).close()
    plan_file = tmp_path / "plan.json"
    plan = _write_plan(plan_file, database, ["selected"])
    plan["selection"]["physical_session_ids"].append("not-in-candidates")
    _stamp_plan_identity(plan)
    plan_file.write_text(json.dumps(plan), encoding="utf-8")

    with pytest.raises(ValueError, match="do not match"):
        load_exact_prune_selection(plan_file, expected_database=database)


def test_exact_selection_plan_rejects_malformed_candidate_entry(tmp_path: Path) -> None:
    database = tmp_path / "state.db"
    SessionDB(db_path=database).close()
    plan_file = tmp_path / "plan.json"
    plan = _write_plan(plan_file, database, ["selected"])
    plan["candidates"] = [{}]
    _stamp_plan_identity(plan)
    plan_file.write_text(json.dumps(plan), encoding="utf-8")

    with pytest.raises(ValueError, match="candidate physical_session_ids"):
        load_exact_prune_selection(plan_file, expected_database=database)


def test_exact_selection_plan_rejects_a_different_database(tmp_path: Path) -> None:
    database = tmp_path / "state.db"
    other_database = tmp_path / "other.db"
    SessionDB(db_path=database).close()
    SessionDB(db_path=other_database).close()
    plan_file = tmp_path / "plan.json"
    _write_plan(plan_file, other_database, ["selected"])

    with pytest.raises(ValueError, match="different session database"):
        load_exact_prune_selection(plan_file, expected_database=database)


def test_exact_selection_plan_rejects_source_drift_during_read(tmp_path: Path) -> None:
    database = tmp_path / "state.db"
    SessionDB(db_path=database).close()
    plan_file = tmp_path / "plan.json"
    plan = _write_plan(plan_file, database, ["selected"])
    plan["source"]["changed_during_read"] = True
    _stamp_plan_identity(plan)
    plan_file.write_text(json.dumps(plan), encoding="utf-8")

    with pytest.raises(ValueError, match="source drift"):
        load_exact_prune_selection(plan_file, expected_database=database)


def test_selection_file_conflicting_with_prune_filters_is_refused(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    database = tmp_path / "state.db"
    db = SessionDB(db_path=database)
    db.create_session("selected", source="cron")
    db.end_session("selected", end_reason="done")
    db.set_session_archived("selected", True)
    db.close()
    plan_file = tmp_path / "plan.json"
    _write_plan(plan_file, database, ["selected"])
    import hermes_state

    monkeypatch.setattr(hermes_state, "SessionDB", lambda: SessionDB(db_path=database))

    code, output = _run_cli(
        monkeypatch,
        capsys,
        ["--selection-file", str(plan_file), "--include-pinned", "--yes"],
    )

    check = SessionDB(db_path=database)
    try:
        assert code == 1, output
        assert "cannot be combined" in output
        assert check.get_session("selected") is not None
    finally:
        check.close()


def test_selection_file_declined_confirmation_deletes_nothing(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    database = tmp_path / "state.db"
    db = SessionDB(db_path=database)
    db.create_session("selected", source="cron")
    db.end_session("selected", end_reason="done")
    db.set_session_archived("selected", True)
    db.close()
    plan_file = tmp_path / "plan.json"
    _write_plan(plan_file, database, ["selected"])
    import hermes_state

    monkeypatch.setattr(hermes_state, "SessionDB", lambda: SessionDB(db_path=database))
    monkeypatch.setattr("builtins.input", lambda _prompt="": "n")

    code, output = _run_cli(
        monkeypatch,
        capsys,
        ["--selection-file", str(plan_file)],
    )

    check = SessionDB(db_path=database)
    try:
        assert "Cancelled" in output
        assert check.get_session("selected") is not None
    finally:
        check.close()


def test_selection_file_revalidates_archived_eligibility_at_prune_time(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    database = tmp_path / "state.db"
    db = SessionDB(db_path=database)
    db.create_session("selected", source="cron")
    db.end_session("selected", end_reason="done")
    db.set_session_archived("selected", True)
    db.close()
    plan_file = tmp_path / "plan.json"
    _write_plan(plan_file, database, ["selected"])

    # The helper-reviewed selection remains syntactically valid, but native
    # Prune must check the mutable destructive preconditions in its transaction.
    db = SessionDB(db_path=database)
    try:
        db.set_session_archived("selected", False)
    finally:
        db.close()
    import hermes_state

    monkeypatch.setattr(hermes_state, "SessionDB", lambda: SessionDB(db_path=database))
    code, output = _run_cli(
        monkeypatch,
        capsys,
        ["--selection-file", str(plan_file), "--yes"],
    )

    check = SessionDB(db_path=database)
    try:
        assert code == 1, output
        assert "not archived" in output
        assert check.get_session("selected") is not None
    finally:
        check.close()
