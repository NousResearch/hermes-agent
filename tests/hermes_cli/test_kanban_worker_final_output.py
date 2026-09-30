"""Dead-worker diagnostics are bounded to the current invocation of an append-only log."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd


@pytest.fixture
def kanban_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Isolated HERMES_HOME with an empty kanban DB."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home




def test_worker_final_output_ignores_old_success_before_run_header(
    kanban_home: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An append-only log with an OLD success, a resume footer, a run header for
    the CURRENT invocation, and a NEW startup error must report the new error —
    not the old success (the exact defect: old text misattributed to a new crash)."""
    fixture_log = (
        "OLD SUCCESS: review requested\n"
        "Resume this session with:\n"
        "  hermes --resume old\n"
        f"{kbd._run_header_line(SimpleNamespace(id='t_fixture', current_run_id=2)).rstrip()}\n"
        "Query: work kanban task t_fixture\n"
        "Initializing agent...\n"
        "Error: Unknown skill(s): sdlc-review\n"
    )
    monkeypatch.setattr(kbd._kb, "read_worker_log", lambda *_a, **_k: fixture_log)
    reported = kbd._worker_final_output("t_fixture", board="default")
    assert "Unknown skill(s): sdlc-review" in reported
    assert "OLD SUCCESS" not in reported


def test_worker_final_output_legacy_log_without_header_uses_latest_query_segment(
    kanban_home: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A legacy log (no run header) still bounds to the latest ``Query:`` segment
    instead of the whole append-only tail, so an old success before an old resume
    footer does not leak into the current segment's diagnostic."""
    fixture_log = (
        "Query: work kanban task t_fixture\n"
        "OLD SUCCESS: review requested\n"
        "Resume this session with:\n"
        "  hermes --resume old\n"
        "Query: work kanban task t_fixture\n"
        "Initializing agent...\n"
        "Error: Unknown skill(s): sdlc-review\n"
    )
    monkeypatch.setattr(kbd._kb, "read_worker_log", lambda *_a, **_k: fixture_log)
    reported = kbd._worker_final_output("t_fixture", board="default")
    assert "Unknown skill(s): sdlc-review" in reported
    assert "OLD SUCCESS" not in reported


def test_worker_final_output_empty_current_invocation_does_not_fall_back(
    kanban_home: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An empty remainder after the current run's header must report "" — never
    fall back to a predecessor's success text."""
    fixture_log = (
        "OLD SUCCESS: review requested\n"
        "Resume this session with:\n"
        "  hermes --resume old\n"
        f"{kbd._run_header_line(SimpleNamespace(id='t_fixture', current_run_id=2)).rstrip()}\n"
    )
    monkeypatch.setattr(kbd._kb, "read_worker_log", lambda *_a, **_k: fixture_log)
    reported = kbd._worker_final_output("t_fixture", board="default")
    assert reported == ""


def test_worker_final_output_missing_log_returns_empty(
    kanban_home: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(kbd._kb, "read_worker_log", lambda *_a, **_k: "")
    assert kbd._worker_final_output("t_missing", board="default") == ""
