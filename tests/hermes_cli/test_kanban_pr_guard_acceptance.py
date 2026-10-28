"""Red-team acceptance tests for the ``active_pr`` respawn guard's
``completion_contract`` explicitness contract (PR 145849 regression fix).

Contract under test (behaviour only, no white-box implementation asserts):

1. ``contract`` not ``local-only`` (NULL/empty/slug/URL) -> guarded.
2. ``local-only`` + any run metadata ``published_pr`` -> guarded (history).
3. ``local-only`` + created-event payload ``completion_contract == "local-only"``
   -> exempt, and the exemption costs zero ``gh`` lookups (it happens before
   PR-state resolution).
4. Every other ``local-only`` row (legacy NULL normalisation, omission window,
   missing/corrupt created payload) fails closed -> guarded.

PR states are injected through the ``_resolve_pr_state`` seam (counted); no
test here may shell out to ``gh``.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Iterator

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd

PR_URL = "https://github.com/acme/repo/pull/7"


@pytest.fixture
def kanban_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Isolated HERMES_HOME with an empty kanban DB (real schema)."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


@pytest.fixture
def pr_seam(monkeypatch: pytest.MonkeyPatch) -> Iterator[SimpleNamespace]:
    """Counted PR-state seam: ``seam.state`` in OPEN/MERGED/CLOSED/None."""
    seam = SimpleNamespace(state="OPEN", calls=0)

    def _stub(_url: str):
        seam.calls += 1
        return seam.state

    kbd._reset_pr_state_cache()
    monkeypatch.setattr(kbd, "_resolve_pr_state", _stub)
    yield seam
    kbd._reset_pr_state_cache()


@pytest.fixture(autouse=True)
def _spawnable_profiles(monkeypatch: pytest.MonkeyPatch) -> None:
    """Dry-run dispatch must consider every assignee lane spawnable with a
    default config (same seam stubs the suite's established harness uses);
    otherwise cards are skipped before the guard is ever consulted."""
    import hermes_cli.config as cfgmod
    import hermes_cli.profiles as profmod

    monkeypatch.setattr(profmod, "profile_exists", lambda _name: True)
    monkeypatch.setattr(cfgmod, "load_config", lambda *a, **k: {})


def _set_state(seam: SimpleNamespace, state) -> None:
    kbd._reset_pr_state_cache()  # the gh lookup cache must not mask the new state
    seam.state = state


def _impl_card(conn, *, explicit_local_only: bool = False) -> str:
    kwargs = dict(title="impl: ship the feature", assignee="worker")
    if explicit_local_only:
        kwargs["completion_contract"] = "local-only"
    return kb.create_task(conn, **kwargs)


def _crashed_run(conn, tid: str, metadata: dict) -> None:
    """Real claim then a crashed run carrying ``metadata``; card back to ready."""
    assert kb.claim_task(conn, tid)
    run_id = kb.get_task(conn, tid).current_run_id
    assert run_id is not None
    conn.execute(
        "UPDATE task_runs SET status='crashed', outcome='crashed', ended_at=?, "
        "metadata=? WHERE id=?",
        (int(time.time()), json.dumps(metadata), run_id),
    )
    conn.execute(
        "UPDATE tasks SET status='ready', current_run_id=NULL, "
        "claim_lock=NULL, claim_expires=NULL, worker_pid=NULL WHERE id=?",
        (tid,),
    )


def _pr_comment(conn, tid: str) -> None:
    kb.add_comment(conn, tid, "reviewer", f"Tracking PR: {PR_URL}")


def _created_payload(conn, tid: str) -> dict:
    row = conn.execute(
        "SELECT payload FROM task_events WHERE task_id=? AND kind='created' "
        "ORDER BY id LIMIT 1",
        (tid,),
    ).fetchone()
    return json.loads(row["payload"]) if row and row["payload"] else {}


def _rewrite_created_payload(conn, tid: str, payload) -> None:
    row = conn.execute(
        "SELECT id FROM task_events WHERE task_id=? AND kind='created' "
        "ORDER BY id LIMIT 1",
        (tid,),
    ).fetchone()
    assert row is not None
    body = None if payload is None else json.dumps(payload)
    conn.execute("UPDATE task_events SET payload=? WHERE id=?", (body, row["id"]))


def _dispatch_view(conn, tid: str):
    """One dry-run tick: returns ``(spawned_ids, guarded_reasons)`` for tid."""
    res = kbd.dispatch_once(conn, dry_run=True)
    spawned = [t for t, _assignee, _ws in res.spawned if t == tid]
    guarded = [reason for t, reason in res.respawn_guarded if t == tid]
    return spawned, guarded


# ---------------------------------------------------------------------------
# A1 — the regression counter-example: omitted contract must stay guarded.
# ---------------------------------------------------------------------------


def test_acceptance_a1_omitted_contract_open_pr_crashed_run_guarded(
    kanban_home, pr_seam
) -> None:
    with kbc.connect() as conn:
        tid = _impl_card(conn)  # completion_contract omitted at creation
        _crashed_run(conn, tid, metadata={})
        _pr_comment(conn, tid)

        assert kbd.check_respawn_guard(conn, tid) == "active_pr"
        spawned, guarded = _dispatch_view(conn, tid)
        assert spawned == []
        assert "active_pr" in guarded
        assert pr_seam.calls >= 1  # PR-bearing lane actually resolved PR state


# ---------------------------------------------------------------------------
# A2 — explicit local-only stays exempt and costs zero gh lookups.
# ---------------------------------------------------------------------------


def test_acceptance_a2_explicit_local_only_exempt_zero_lookup(
    kanban_home, pr_seam
) -> None:
    with kbc.connect() as conn:
        tid = _impl_card(conn, explicit_local_only=True)
        _crashed_run(conn, tid, metadata={})
        _pr_comment(conn, tid)

        assert kbd.check_respawn_guard(conn, tid) is None
        spawned, guarded = _dispatch_view(conn, tid)
        assert guarded == []
        assert spawned, "explicit local-only card must remain spawnable"
        assert pr_seam.calls == 0  # exemption precedes PR-state resolution


# ---------------------------------------------------------------------------
# A3 — legacy NULL column value (historical normalised row) fails closed.
# ---------------------------------------------------------------------------


def test_acceptance_a3_legacy_null_contract_guarded(kanban_home, pr_seam) -> None:
    with kbc.connect() as conn:
        tid = _impl_card(conn)
        conn.execute(
            "UPDATE tasks SET completion_contract=NULL WHERE id=?", (tid,)
        )
        _crashed_run(conn, tid, metadata={})
        _pr_comment(conn, tid)

        assert kbd.check_respawn_guard(conn, tid) == "active_pr"
        assert pr_seam.calls >= 1


# ---------------------------------------------------------------------------
# A4 — local-only column but no explicitness marker in the created payload.
# ---------------------------------------------------------------------------


def test_acceptance_a4_local_only_without_created_marker_guarded(
    kanban_home, pr_seam
) -> None:
    with kbc.connect() as conn:
        tid = _impl_card(conn, explicit_local_only=True)
        payload = _created_payload(conn, tid)
        payload.pop("completion_contract", None)  # simulate a pre-fix row
        _rewrite_created_payload(conn, tid, payload)
        _crashed_run(conn, tid, metadata={})
        _pr_comment(conn, tid)

        assert kbd.check_respawn_guard(conn, tid) == "active_pr"


# ---------------------------------------------------------------------------
# A5 — published_pr history behaviour unchanged (OPEN guarded / MERGED lifted).
# ---------------------------------------------------------------------------


def test_acceptance_a5_published_pr_history_unchanged(kanban_home, pr_seam) -> None:
    with kbc.connect() as conn:
        tid = _impl_card(conn)  # omitted contract, but the run published a PR
        _crashed_run(conn, tid, metadata={"published_pr": PR_URL})
        _pr_comment(conn, tid)

        assert kbd.check_respawn_guard(conn, tid) == "active_pr"
        spawned, guarded = _dispatch_view(conn, tid)
        assert spawned == [] and "active_pr" in guarded

        _set_state(pr_seam, "MERGED")
        assert kbd.check_respawn_guard(conn, tid) is None
        spawned, guarded = _dispatch_view(conn, tid)
        assert spawned and guarded == []


# ---------------------------------------------------------------------------
# A6 — the PR-state matrix does not regress: MERGED/CLOSED lift, unresolved stays.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("state,guarded_expected", [
    ("MERGED", False),
    ("CLOSED", False),
    (None, True),  # unresolved lookup fails closed
])
def test_acceptance_a6_pr_state_matrix_no_regression(
    kanban_home, pr_seam, state, guarded_expected
) -> None:
    with kbc.connect() as conn:
        tid = _impl_card(conn)
        _crashed_run(conn, tid, metadata={})
        _pr_comment(conn, tid)

        _set_state(pr_seam, state)
        guard = kbd.check_respawn_guard(conn, tid)
        spawned, guarded = _dispatch_view(conn, tid)
        if guarded_expected:
            assert guard == "active_pr"
            assert spawned == [] and "active_pr" in guarded
        else:
            assert guard is None
            assert spawned and guarded == []


# ---------------------------------------------------------------------------
# A7 — explicitness is recorded on the created event and observable in behaviour.
# ---------------------------------------------------------------------------


def test_acceptance_a7_created_payload_distinguishes_explicitness(
    kanban_home, pr_seam
) -> None:
    with kbc.connect() as conn:
        omitted = _impl_card(conn)
        explicit = _impl_card(conn, explicit_local_only=True)

        assert _created_payload(conn, explicit).get("completion_contract") == "local-only"
        assert _created_payload(conn, omitted).get("completion_contract") is None

        for tid in (omitted, explicit):
            _crashed_run(conn, tid, metadata={})
            _pr_comment(conn, tid)

        assert kbd.check_respawn_guard(conn, omitted) == "active_pr"
        assert kbd.check_respawn_guard(conn, explicit) is None


# ---------------------------------------------------------------------------
# A8 — a corrupt created payload must fail closed, not exempt.
# ---------------------------------------------------------------------------


def test_acceptance_a8_corrupt_created_payload_guarded(kanban_home, pr_seam) -> None:
    with kbc.connect() as conn:
        tid = _impl_card(conn, explicit_local_only=True)
        _rewrite_created_payload(conn, tid, None)
        conn.execute(
            "UPDATE task_events SET payload='{broken json' WHERE id=("
            "SELECT id FROM task_events WHERE task_id=? AND kind='created' "
            "ORDER BY id LIMIT 1)",
            (tid,),
        )
        _crashed_run(conn, tid, metadata={})
        _pr_comment(conn, tid)

        assert kbd.check_respawn_guard(conn, tid) == "active_pr"
