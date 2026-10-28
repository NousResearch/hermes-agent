"""Explicit-``local-only`` regression tests for the dispatcher's ``active_pr`` guard.

``validate_contract`` normalises an *omitted* ``completion_contract`` to
``local-only`` before ``create_task`` persists it, so the ``tasks`` column
alone cannot distinguish a card that *declared* the report/notify lane from
one that never said anything. The guard read every stored ``local-only`` row
as the exemption, so an omitted-contract implementer card with an open PR was
re-spawned against that very PR — the regression these tests pin down:

* the ``created`` event records the RAW creation argument (omitted →
  ``None``), making it the authority on whether ``local-only`` was explicit;
* the guard exempts a stored ``local-only`` row ONLY when the created event
  says the card explicitly declared ``local-only`` — and that path still
  costs zero ``gh`` lookups;
* everything else fails closed to guarded: an omitted argument, a legacy NULL
  column (even with a marker present), a created event without the marker
  (decompose sub-cards / pre-fix rows), a corrupt payload;
* a ``published_pr`` run still overrides the declaration, and the
  MERGED/CLOSED/unresolved PR-state matrix is untouched.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Iterator

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd

# A repo slug is the canonical PR-bearing declaration ("this card publishes a PR").
_SLUG_CONTRACT = "example/repo"

_PR_COMMENT = "Opened https://github.com/example/repo/pull/44 for review."


@pytest.fixture(autouse=True)
def _no_gh_pr_lookup(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """No test in this file may shell out to ``gh``.

    CI has no GitHub auth and each lookup carries a 10s timeout. Tests that
    need a state patch ``kbd._resolve_pr_state`` themselves; the cache is
    cleared on both sides so module state cannot leak between tests.
    """
    kbd._reset_pr_state_cache()
    monkeypatch.setattr(kbd, "_resolve_pr_state", lambda _url: None)
    yield
    kbd._reset_pr_state_cache()


@pytest.fixture
def kanban_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Isolated HERMES_HOME with an empty kanban DB."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _created_payload(conn, tid: int):
    """The card's earliest ``created`` event payload (dict), or ``None``."""
    row = conn.execute(
        "SELECT payload FROM task_events WHERE task_id = ? AND kind = 'created' "
        "ORDER BY id LIMIT 1",
        (tid,),
    ).fetchone()
    if row is None or not row["payload"]:
        return None
    return json.loads(row["payload"])


def _contract_column(conn, tid: int):
    return conn.execute(
        "SELECT completion_contract FROM tasks WHERE id = ?", (tid,),
    ).fetchone()["completion_contract"]


def _open_pr_comment(conn, tid: int) -> None:
    kb.add_comment(conn, tid, author="dev", body=_PR_COMMENT)


def _crashed_run(conn, tid: int) -> None:
    """A crashed attempt whose metadata recorded nothing (the worker died first)."""
    now = int(time.time())
    with kb.write_txn(conn):
        conn.execute(
            "INSERT INTO task_runs (task_id, profile, status, outcome, "
            "started_at, ended_at, metadata) VALUES (?, 'dev', 'crashed', "
            "'crashed', ?, ?, ?)",
            (tid, now - 60, now - 30, json.dumps({})),
        )


def _published_pr_run(conn, tid: int) -> None:
    """A completed run that recorded a published PR, backdated beyond the 1h
    ``recent_success`` window so tests isolate the active_pr step."""
    now = int(time.time())
    with kb.write_txn(conn):
        conn.execute(
            "INSERT INTO task_runs (task_id, profile, status, outcome, "
            "started_at, ended_at, metadata) VALUES (?, 'dev', 'completed', "
            "'completed', ?, ?, ?)",
            (tid, now - 7200, now - 7100,
             json.dumps({"published_pr": "https://github.com/example/repo/pull/9"})),
        )


def _dispatch_ready(conn, monkeypatch: pytest.MonkeyPatch):
    """One dry-run dispatch tick with every profile spawnable and default config."""
    import hermes_cli.config as cfgmod
    import hermes_cli.profiles as profmod

    monkeypatch.setattr(profmod, "profile_exists", lambda _name: True)
    monkeypatch.setattr(cfgmod, "load_config", lambda *a, **k: {})
    return kbd.dispatch_once(conn, dry_run=True)


# ---------------------------------------------------------------------------
# persistence contract: the created event records the RAW contract argument
# ---------------------------------------------------------------------------


def test_created_event_records_raw_contract_omitted_is_none(
    kanban_home: Path,
) -> None:
    """An omitted ``completion_contract`` is recorded as ``None`` on the
    created event — the raw argument, before ``validate_contract`` normalises
    it — while the stored column keeps the normalised ``local-only``."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="omitted contract", assignee="dev")
        assert _contract_column(conn, tid) == "local-only"
        assert _created_payload(conn, tid)["completion_contract"] is None


def test_created_event_records_raw_contract_explicit_values(
    kanban_home: Path,
) -> None:
    """An explicit declaration rides the created event verbatim — both the
    ``local-only`` exemption and a PR-bearing repo slug."""
    with kbc.connect() as conn:
        for declared in ("local-only", _SLUG_CONTRACT):
            tid = kb.create_task(
                conn, title=f"declared {declared}", assignee="dev",
                completion_contract=declared,
            )
            assert _created_payload(conn, tid)["completion_contract"] == declared


# ---------------------------------------------------------------------------
# T1: omitted contract + crashed run + OPEN PR → guarded (the regression)
# ---------------------------------------------------------------------------


def test_omitted_contract_card_with_open_pr_stays_guarded(
    kanban_home: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An implementer card whose contract was merely OMITTED is not a report
    card. Its stored ``local-only`` came from normalisation; with a crashed
    run and an open PR the card must stay ``active_pr``-guarded, not be
    re-spawned against the same PR."""
    monkeypatch.setattr(kbd, "_resolve_pr_state", lambda _url: "OPEN")
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="implement a feature", assignee="dev")
        _crashed_run(conn, tid)
        _open_pr_comment(conn, tid)

        assert kbd.check_respawn_guard(conn, tid) == "active_pr"
        res = _dispatch_ready(conn, monkeypatch)
        assert tid not in [row[0] for row in res.spawned]
        assert dict(res.respawn_guarded).get(tid) == "active_pr"


# ---------------------------------------------------------------------------
# T2: explicit local-only keeps the exemption — and costs no gh lookup
# ---------------------------------------------------------------------------


def test_explicit_local_only_card_is_exempt_and_costs_no_lookup(
    kanban_home: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A card that *declared* ``local-only`` keeps the report/notify
    exemption, and lifting it must still cost zero ``gh`` lookups (the guard
    steps out before PR-state parsing)."""
    calls: list[str] = []

    def _resolve(url: str):
        calls.append(url)
        return "OPEN"

    monkeypatch.setattr(kbd, "_resolve_pr_state", _resolve)
    with kbc.connect() as conn:
        tid = kb.create_task(
            conn, title="notify operator the PR is ready", assignee="chief",
            completion_contract="local-only",
        )
        _open_pr_comment(conn, tid)

        assert kbd.check_respawn_guard(conn, tid) is None
        assert calls == []
        res = _dispatch_ready(conn, monkeypatch)
        assert tid in [row[0] for row in res.spawned]
        assert calls == []


# ---------------------------------------------------------------------------
# T3: legacy NULL column stays guarded, even with an exemption marker present
# ---------------------------------------------------------------------------


def test_legacy_null_contract_card_stays_guarded_despite_created_marker(
    kanban_home: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A legacy card whose ``completion_contract`` column is NULL (it predates
    the column's meaning) is PR-bearing whatever the created event says — an
    exemption marker must not outlive the declaration it came from."""
    monkeypatch.setattr(kbd, "_resolve_pr_state", lambda _url: "OPEN")
    with kbc.connect() as conn:
        tid = kb.create_task(
            conn, title="legacy card", assignee="dev",
            completion_contract="local-only",
        )
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE tasks SET completion_contract = NULL WHERE id = ?", (tid,),
            )
        _open_pr_comment(conn, tid)

        assert kbd.check_respawn_guard(conn, tid) == "active_pr"


# ---------------------------------------------------------------------------
# T4: a published_pr run still overrides an explicit local-only declaration
# ---------------------------------------------------------------------------


def test_explicit_local_only_card_that_published_a_pr_stays_guarded(
    kanban_home: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A recorded ``published_pr`` run outranks an explicit ``local-only``
    declaration — what the card DID matters more than what it declared."""
    monkeypatch.setattr(kbd, "_resolve_pr_state", lambda _url: "OPEN")
    with kbc.connect() as conn:
        tid = kb.create_task(
            conn, title="report card that PRed", assignee="dev",
            completion_contract="local-only",
        )
        _open_pr_comment(conn, tid)
        _published_pr_run(conn, tid)
        assert kbd.check_respawn_guard(conn, tid) == "active_pr"


# ---------------------------------------------------------------------------
# T5: the MERGED/CLOSED/unresolved matrix for a declared slug card is untouched
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("state", "guarded"),
    [("MERGED", False), ("CLOSED", False), (None, True)],
    ids=["merged-spawns", "closed-spawns", "unresolved-fails-closed"],
)
def test_pr_state_matrix_unchanged_for_declared_slug_card(
    kanban_home: Path, monkeypatch: pytest.MonkeyPatch,
    state: str | None, guarded: bool,
) -> None:
    """A slug-declared card spawns once its PR merged/closed and stays guarded
    on an unresolved lookup — the explicitness fix must not move this matrix."""
    monkeypatch.setattr(kbd, "_resolve_pr_state", lambda _url: state)
    with kbc.connect() as conn:
        tid = kb.create_task(
            conn, title="slug card", assignee="dev",
            completion_contract=_SLUG_CONTRACT,
        )
        _open_pr_comment(conn, tid)

        reason = kbd.check_respawn_guard(conn, tid)
        if guarded:
            assert reason == "active_pr"
        else:
            assert reason is None
        res = _dispatch_ready(conn, monkeypatch)
        spawned = tid in [row[0] for row in res.spawned]
        assert spawned != guarded


# ---------------------------------------------------------------------------
# contract rule 4: a created event without the marker fails closed
# ---------------------------------------------------------------------------


def test_created_event_without_marker_fails_closed(
    kanban_home: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A stored ``local-only`` row whose created event carries no
    ``completion_contract`` key (decompose sub-cards, pre-fix rows) is
    guarded — absence of evidence is not exemption."""
    monkeypatch.setattr(kbd, "_resolve_pr_state", lambda _url: "OPEN")
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="pre-marker card", assignee="dev")
        # Rewrite the created payload into the decompose sub-card shape: a
        # ``created`` event with no ``completion_contract`` key at all.
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE task_events SET payload = ? "
                "WHERE task_id = ? AND kind = 'created'",
                (json.dumps({"by": "decomposer",
                             "from_decompose_of": "t_rootdeadbeef"}), tid),
            )
        _open_pr_comment(conn, tid)

        assert kbd.check_respawn_guard(conn, tid) == "active_pr"
