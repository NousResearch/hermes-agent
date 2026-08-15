"""Tests for `hermes sessions activity` — the aggregate interactive-session read.

Interactive sessions are a different subject from kanban rows, and this command
is the only place the two meet a reader. Its whole value is that it can be
handed to a dashboard, so what it must NOT carry (ids, titles, previews, paths)
is as load-bearing as what it counts.
"""

import argparse
import hashlib
import json
import time

from hermes_cli.sessions_cmd import SESSIONS_ACTIVITY_CONTRACT, _sessions_activity


WORKSPACE = "/repo/alpha"
OTHER_WORKSPACE = "/repo/beta"


def _digest(path):
    return hashlib.sha256(path.encode("utf-8")).hexdigest()


class _FakeDB:
    """Stands in for SessionDB, recording how the command queried it."""

    def __init__(self, rows):
        self._rows = rows
        self.calls = []

    def list_sessions_rich(self, **kwargs):
        self.calls.append(kwargs)
        rows = sorted(self._rows, key=lambda row: -(row.get("last_active") or 0))
        return rows[: kwargs.get("limit") or len(rows)]


def _row(*, last_active, ended_at=None, workspace=WORKSPACE, **extra):
    row = {
        "id": "20260813_120000_abcdef",
        "source": "desktop",
        "model": "test/model",
        "title": "private chat title",
        "preview": "private prompt text",
        "cwd": workspace,
        "last_active": last_active,
        "started_at": last_active - 60,
        "ended_at": ended_at,
    }
    row.update(extra)
    return row


def _run(rows, capsys, monkeypatch, **argv):
    monkeypatch.setattr(
        "hermes_state.workspace_key", lambda row: row.get("cwd") or "", raising=False
    )
    args = argparse.Namespace(
        sessions_action="activity",
        source=argv.get("source"),
        window=argv.get("window", 300),
        json=True,
    )
    db = _FakeDB(rows)
    assert _sessions_activity(db, args) == 0
    return json.loads(capsys.readouterr().out), db


def test_open_recent_session_is_grouped_under_its_workspace_digest(capsys, monkeypatch):
    now = int(time.time())
    payload, _ = _run([_row(last_active=now - 5)], capsys, monkeypatch)

    assert payload["contract_version"] == SESSIONS_ACTIVITY_CONTRACT
    assert payload["totals"] == {"open_sessions": 1, "recently_active_sessions": 1}
    assert payload["workspaces"] == [
        {
            "workspace_sha256": _digest(WORKSPACE),
            "open_sessions": 1,
            "recently_active_sessions": 1,
            "last_activity_at": now - 5,
        }
    ]


def test_no_identifier_path_title_or_preview_survives_the_aggregate(
    capsys, monkeypatch
):
    now = int(time.time())
    payload, _ = _run([_row(last_active=now - 5)], capsys, monkeypatch)

    rendered = json.dumps(payload)
    for leaked in (
        WORKSPACE,
        "private chat title",
        "private prompt text",
        "20260813_120000_abcdef",
        "test/model",
    ):
        assert leaked not in rendered, leaked


def test_an_ended_session_is_not_open_work(capsys, monkeypatch):
    now = int(time.time())
    payload, _ = _run(
        [_row(last_active=now - 5, ended_at=now - 1)], capsys, monkeypatch
    )

    assert payload["totals"] == {"open_sessions": 0, "recently_active_sessions": 0}
    assert payload["workspaces"] == []


def test_an_open_but_quiet_session_is_open_without_being_recently_active(
    capsys, monkeypatch
):
    now = int(time.time())
    payload, _ = _run([_row(last_active=now - 4000)], capsys, monkeypatch)

    # Open and recent are separate facts: silence is not closure, and an old
    # open session must never be counted as current work.
    assert payload["totals"] == {"open_sessions": 1, "recently_active_sessions": 0}
    assert payload["workspaces"][0]["recently_active_sessions"] == 0
    assert payload["workspaces"][0]["last_activity_at"] == now - 4000


def test_a_session_bound_to_no_workspace_is_counted_but_never_attributed(
    capsys, monkeypatch
):
    now = int(time.time())
    payload, _ = _run(
        [_row(last_active=now - 5, workspace=""), _row(last_active=now - 6)],
        capsys,
        monkeypatch,
    )

    assert payload["totals"]["open_sessions"] == 2
    assert payload["unresolved_workspace"] == {
        "open_sessions": 1,
        "recently_active_sessions": 1,
    }
    # Exactly one workspace bucket: the unbound session is in the totals and in
    # `unresolved_workspace`, never invented into a digest.
    assert len(payload["workspaces"]) == 1


def test_workspace_counts_reconcile_with_the_totals(capsys, monkeypatch):
    now = int(time.time())
    payload, _ = _run(
        [
            _row(last_active=now - 5),
            _row(last_active=now - 9),
            _row(last_active=now - 7, workspace=OTHER_WORKSPACE),
            _row(last_active=now - 5000, workspace=OTHER_WORKSPACE),
            _row(last_active=now - 8, workspace=""),
        ],
        capsys,
        monkeypatch,
    )

    for field in ("open_sessions", "recently_active_sessions"):
        summed = sum(entry[field] for entry in payload["workspaces"])
        assert (
            summed + payload["unresolved_workspace"][field] == payload["totals"][field]
        )


def test_the_window_is_honoured_and_reported(capsys, monkeypatch):
    now = int(time.time())
    payload, _ = _run([_row(last_active=now - 600)], capsys, monkeypatch, window=900)

    assert payload["window_seconds"] == 900
    assert payload["totals"]["recently_active_sessions"] == 1


def test_the_scan_is_bounded_and_ordered_by_activity(capsys, monkeypatch):
    now = int(time.time())
    payload, db = _run(
        [_row(last_active=now - index) for index in range(5)], capsys, monkeypatch
    )

    # Newest-activity-first ordering is what makes `recently_active_sessions`
    # exact under truncation, so the command must request it explicitly.
    assert db.calls[0]["order_by_last_active"] is True
    assert db.calls[0]["limit"] == payload["scan_limit"]
    assert payload["scan_truncated"] is False


def test_a_truncated_scan_says_so(capsys, monkeypatch):
    now = int(time.time())
    payload, _ = _run(
        [_row(last_active=now - index) for index in range(400)], capsys, monkeypatch
    )

    assert payload["scan_truncated"] is True
    assert payload["totals"]["open_sessions"] == payload["scan_limit"]


def test_an_explicit_source_reaches_the_query_and_lifts_the_tool_exclusion(
    capsys, monkeypatch
):
    now = int(time.time())
    _, db = _run([_row(last_active=now - 5)], capsys, monkeypatch, source="desktop")

    assert db.calls[0]["source"] == "desktop"
    assert db.calls[0]["exclude_sources"] is None


def test_third_party_tool_sessions_stay_excluded_by_default(capsys, monkeypatch):
    now = int(time.time())
    _, db = _run([_row(last_active=now - 5)], capsys, monkeypatch)

    assert db.calls[0]["source"] is None
    assert db.calls[0]["exclude_sources"] == ["tool"]
