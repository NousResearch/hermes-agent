"""Regression tests for the decompose RECORD guard (defcon card t_9cb7f0b9).

The observed defect: the auto-decomposer claimed a review-blocked card out of triage and
minted four children off the root BODY alone. The root's newest comments said, in as many
words, "the CODE HALF IS APPROVED at 3ce158d8" and "Do NOT decompose or re-implement this
card" — work a reviewer had already ruled on, invented a second time by an LLM that never
read the thread.

These tests drive the REAL entry points (``kanban_decompose.decompose_task``, which is what
the gateway's auto-decompose tick and ``hermes kanban decompose`` call, and
``kanban_db_graph.decompose_triage_task``, which is what the dashboard calls) against a
throwaway board. Both directions are covered — a decided card refuses AND an ordinary triage
card still decomposes — so the guard cannot pass by simply breaking decomposition.

Deliberately NO module-level import of the new symbols: the pre-card tree must still
COLLECT this file, so the "before" run fails on these tests' behaviour (the LLM is asked to
invent work for a decided card; no refusal is recorded) rather than on an import error.
The event kind is a contract string and is spelled out here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_graph as kbg
from hermes_cli import kanban_decompose as decomp
from hermes_cli import kanban_specify as specify
from hermes_cli.kanban_db import BLOCK_RECURRENCE_LIMIT

# The contract string the kernel writes and the operator greps for. Spelled out on purpose
# (see the module docstring).
REFUSAL_EVENT_KIND = "decompose_refused"

# ---------------------------------------------------------------------------------------
# The real card, verbatim where it matters. Line excerpts from defcon t_d0520071, the card
# the decomposer fanned out at 06:16 on 2026-09-27: comment 27 = the review verdict,
# comment 28 = the triage note written one second before the fan-out. Nothing about the
# markers below has been paraphrased.
# ---------------------------------------------------------------------------------------
INCIDENT_APPROVAL_COMMENT = (
    "REVIEW (platform-stl, round 1 — artifact lens, 0 prior `changes_requested` runs).\n"
    "\n"
    "VERDICT: the CODE HALF IS APPROVED at `3ce158d8` — DoD items 1–4 verified "
    "independently. The CARD IS NOT COMPLETED: its own Evidence line is unmet and ruling 15 "
    "§3 forbids this card from claiming done, so the disposition is an escalation, not a "
    "`complete`.\n"
    "\n"
    "ARTIFACT IDENTITY\n"
    "* tip `3ce158d8e8908706b869a8a0fb61fab3068eaefc`, branch "
    "`wt/t_d0520071-lane-scoped-lockdown`, worktree clean (0 porcelain lines).\n"
)
INCIDENT_TRIAGE_COMMENT = (
    "TRIAGE NOTE (review run 11 closed as `blocked`; card auto-moved to triage by "
    "`block_loop_detected` — this is the task's second block cycle, the first being run 3).\n"
    "\n"
    "Do NOT decompose or re-implement this card. The code half is approved at "
    "`3ce158d8`; the change is unmerged, undeployed and unpushed by design.\n"
)


@pytest.fixture
def kanban_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _children() -> list[dict]:
    return [
        {"title": "child a", "body": "do a", "assignee": "worker", "parents": []},
        {"title": "child b", "body": "do b", "assignee": "worker", "parents": [0]},
    ]


def _triage_card(conn, *, title: str = "decided card") -> str:
    tid = kb.create_task(conn, title=title, triage=True)
    assert _status(conn, tid) == "triage"
    return tid


def _status(conn, tid: str) -> str:
    """The card's status, asserting it still exists (no silent ``None`` in an assertion)."""
    task = kb.get_task(conn, tid)
    assert task is not None, f"no task {tid} on this board"
    return task.status


def _comment(conn, tid: str, body: str, author: str = "platform-stl") -> int:
    return kb.add_comment(conn, tid, author, body)


def _events(conn, tid: str, kind: str) -> list:
    return [e for e in kb.list_events(conn, tid) if e.kind == kind]


def _refusals(conn, tid: str) -> list:
    return _events(conn, tid, REFUSAL_EVENT_KIND)


def _set_status(conn, tid: str, status: str) -> None:
    """The board column write (dashboard drag / ``hermes kanban``), as the operator does it."""
    with kb.write_txn(conn):
        conn.execute("UPDATE tasks SET status=? WHERE id=?", (status, tid))


def _park_in_triage(conn, *, kind: str = "needs_input", title: str = "looping card") -> str:
    """Drive a card through the REAL block ladder until the breaker parks it in ``triage``."""
    tid = kb.create_task(conn, title=title, assignee="worker")
    assert kb.claim_task(conn, tid, claimer="worker") is not None
    for _ in range(BLOCK_RECURRENCE_LIMIT + 1):
        kb.block_task(conn, tid, reason="same cause", kind=kind)
        if _status(conn, tid) == "triage":
            return tid
        assert kb.unblock_task(conn, tid)
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET status='ready' WHERE id=?", (tid,))
        assert kb.claim_task(conn, tid, claimer="worker") is not None
    raise AssertionError("the breaker never parked the card in triage")


def _decompose_without_llm(tid: str) -> tuple[Any, MagicMock]:
    """Call the production decomposer with the aux client stubbed and its calls counted.

    ``_call_aux`` returns ``(raw, reason)``; the stub answers the shape the caller unpacks,
    so an unrefused card fails on the recorded reply rather than on a TypeError.
    """
    aux = MagicMock(return_value=(None, "aux client unavailable"))
    with patch("hermes_cli.kanban_decompose._call_aux", aux):
        outcome = decomp.decompose_task(tid, author="decomposer")
    return outcome, aux


# ---------------------------------------------------------------------------------------
# The defect: an APPROVED card is fanned out anyway
# ---------------------------------------------------------------------------------------

def test_decompose_task_refuses_a_card_whose_newest_comment_carries_an_approval(
    kanban_home: Path,
) -> None:
    """The LLM must never be asked to invent work for a card the record has decided."""
    with kbc.connect_closing() as conn:
        tid = _triage_card(conn)
        comment_id = _comment(conn, tid, "VERDICT: the CODE HALF IS APPROVED at `3ce158d8`.")
        before = len(kb.list_tasks(conn, limit=500))

    outcome, aux = _decompose_without_llm(tid)

    assert aux.call_count == 0, "an APPROVED card must not reach the LLM"
    assert not outcome.ok, "the decomposer must refuse a card the record already decided"
    assert "refused auto-decomposition" in outcome.reason
    assert "approved" in outcome.reason

    with kbc.connect_closing() as conn:
        assert _status(conn, tid) == "triage", "the root stays the escalation"
        assert len(kb.list_tasks(conn, limit=500)) == before, "children were minted"
        assert not _events(conn, tid, "decomposed")
        refusals = _refusals(conn, tid)
        assert len(refusals) == 1, "the refusal must be RECORDED, with its reason"
        assert refusals[0].payload["causes"] == ["approved"]
        match = refusals[0].payload["matches"][0]
        assert match["comment_id"] == comment_id
        assert match["author"] == "platform-stl"
        assert "CODE HALF IS APPROVED" in match["line"]


def test_the_incident_card_is_refused_on_every_cause_its_thread_carries(
    kanban_home: Path,
) -> None:
    """Replay of defcon t_d0520071: approval + do-not-decompose + a live branch.

    The two comments below are the card's real thread (excerpts, verbatim lines) in the
    order the decomposer would have read them.
    """
    with kbc.connect_closing() as conn:
        tid = _triage_card(conn, title="lane-scoped lockdown (code half)")
        _comment(conn, tid, INCIDENT_APPROVAL_COMMENT)
        _comment(conn, tid, INCIDENT_TRIAGE_COMMENT)

    outcome, aux = _decompose_without_llm(tid)

    assert aux.call_count == 0
    assert not outcome.ok
    assert "refused auto-decomposition" in outcome.reason

    with kbc.connect_closing() as conn:
        payload = _refusals(conn, tid)[0].payload
        assert payload["causes"] == ["superseded", "approved", "live_artifact"], (
            "every cause the thread carries is named, newest comment first"
        )
        assert {m["comment_id"] for m in payload["matches"]} == {1, 2}
        assert "Do NOT decompose" in next(
            m["line"] for m in payload["matches"] if m["cause"] == "superseded"
        )
        assert "wt/t_d0520071-lane-scoped-lockdown" in next(
            m["line"] for m in payload["matches"] if m["cause"] == "live_artifact"
        )
        assert not _events(conn, tid, "decomposed")
        assert len(kb.list_tasks(conn, limit=500)) == 1


def test_decompose_triage_task_refuses_a_decided_card(kanban_home: Path) -> None:
    """The dashboard / direct-DB path is guarded too, inside its own write txn."""
    with kbc.connect_closing() as conn:
        tid = _triage_card(conn)
        _comment(conn, tid, INCIDENT_APPROVAL_COMMENT)
        before = len(kb.list_tasks(conn, limit=500))

        refusal = kbg.decompose_triage_task(
            conn, tid, root_assignee="orchestrator", children=_children(), author="decomposer",
        )

        assert not refusal, "the fan-out must be refused"
        assert not isinstance(refusal, list)
        assert not isinstance(refusal, kb.TriageEscalationRefusal), (
            "this is the RECORD refusal, not the escalation park"
        )
        assert refusal.task_id == tid
        assert "approved" in refusal.causes
        assert "refused auto-decomposition" in refusal.detail

        assert _status(conn, tid) == "triage"
        assert len(kb.list_tasks(conn, limit=500)) == before
        assert not _events(conn, tid, "decomposed")
        assert len(_refusals(conn, tid)) == 1


# ---------------------------------------------------------------------------------------
# The other refusal conditions, one at a time
# ---------------------------------------------------------------------------------------

@pytest.mark.parametrize(
    "note",
    [
        "Do NOT decompose or re-implement this card.",
        "Do not implement this: it was superseded by the lane-scoped lockdown.",
        "SUPERSEDED — no work performed; the successor carries it.",
        "This ask is WITHDRAWN by the ops head.",
    ],
)
def test_a_superseded_or_do_not_implement_note_refuses(kanban_home: Path, note: str) -> None:
    with kbc.connect_closing() as conn:
        tid = _triage_card(conn)
        _comment(conn, tid, f"NOTE\n\n{note}\n")

    outcome, aux = _decompose_without_llm(tid)

    assert aux.call_count == 0
    assert not outcome.ok
    with kbc.connect_closing() as conn:
        assert _refusals(conn, tid)[0].payload["causes"] == ["superseded"]


@pytest.mark.parametrize(
    "artifact",
    [
        "branch `wt/t_d0520071-lane-scoped-lockdown` @ 3ce158d8",
        "the fix lives on `fix/t_9cb7f0b9-decompose-record-guard`",
        "the work rides `hermes-agent/t_9cb7f0b9-decompose-record-guard`",
        "https://github.com/NousResearch/hermes-agent/pull/123456 is open",
    ],
)
def test_a_live_branch_or_pr_refuses(kanban_home: Path, artifact: str) -> None:
    with kbc.connect_closing() as conn:
        tid = _triage_card(conn)
        _comment(conn, tid, f"STATUS\n\n{artifact}\n")

    outcome, aux = _decompose_without_llm(tid)

    assert aux.call_count == 0
    assert not outcome.ok
    with kbc.connect_closing() as conn:
        assert _refusals(conn, tid)[0].payload["causes"] == ["live_artifact"]


def test_a_card_in_the_review_column_is_refused(kanban_home: Path) -> None:
    """The predicate never calls a card in the review column decomposable.

    Both promotion paths screen the COLUMN first (``not in triage``), so this leg is the
    predicate's own answer for a caller reading the record directly — and the race-safe
    answer for a status write that lands between a caller's read and its fan-out.
    """
    with kbc.connect_closing() as conn:
        tid = kb.create_task(conn, title="in review", assignee="worker")
        assert _status(conn, tid) in {"ready", "todo"}
        assert kb.claim_task(conn, tid, claimer="worker") is not None
        assert kb.request_review(conn, tid, summary="review me", reviewer="reviewer", force=True)
        assert _status(conn, tid) == "review"
        before = len(kb.list_tasks(conn, limit=500))

        refusal = kb.decompose_refusal_guard(conn, tid, author="decomposer")

        assert refusal is not None and not isinstance(refusal, kb.TriageEscalationRefusal)
        assert refusal.causes == ["in_review"]
        assert len(_refusals(conn, tid)) == 1
        assert len(kb.list_tasks(conn, limit=500)) == before


def test_a_review_handoff_with_no_decision_is_refused(kanban_home: Path) -> None:
    """A card hand-moved back into triage with its review still outstanding."""
    with kbc.connect_closing() as conn:
        tid = kb.create_task(conn, title="handed to a reviewer", assignee="worker")
        assert kb.claim_task(conn, tid, claimer="worker") is not None
        assert kb.request_review(conn, tid, summary="review me", reviewer="reviewer", force=True)
        assert _status(conn, tid) == "review"
        _set_status(conn, tid, "triage")

    outcome, aux = _decompose_without_llm(tid)

    assert aux.call_count == 0
    assert not outcome.ok
    with kbc.connect_closing() as conn:
        payload = _refusals(conn, tid)[0].payload
        assert payload["causes"] == ["in_review"]
        assert payload["matches"][0]["event_kind"] == "review_requested"


def test_a_card_a_live_run_owns_is_refused(kanban_home: Path) -> None:
    """A card hand-moved into triage while a worker still holds its claim."""
    with kbc.connect_closing() as conn:
        tid = kb.create_task(conn, title="claimed card", assignee="worker")
        assert kb.claim_task(conn, tid, claimer="worker") is not None
        _set_status(conn, tid, "triage")

    outcome, aux = _decompose_without_llm(tid)

    assert aux.call_count == 0
    assert not outcome.ok
    with kbc.connect_closing() as conn:
        payload = _refusals(conn, tid)[0].payload
        assert payload["causes"] == ["live_run"]
        assert payload["matches"][0]["run_id"] is not None


# ---------------------------------------------------------------------------------------
# The refusal's own words, and its idempotence
# ---------------------------------------------------------------------------------------

def test_the_refusal_names_the_evidence_and_a_real_disposal(kanban_home: Path) -> None:
    with kbc.connect_closing() as conn:
        tid = _triage_card(conn)
        _comment(conn, tid, INCIDENT_APPROVAL_COMMENT)
        refusal = kb.decompose_refusal_guard(conn, tid, author="decomposer")
        assert refusal is not None

    assert "refused auto-decomposition" in refusal.detail
    assert "archive" in refusal.detail, "archive is a disposal"
    assert "move it out of the triage column" in refusal.detail
    assert "unblock / complete / reassign do not clear this refusal" in refusal.detail
    assert "comment 1" in refusal.detail, "the evidence names the row it read"


def test_a_second_attempt_records_no_second_refusal(kanban_home: Path) -> None:
    """The tick retries and the CLI sweeps: the record must not grow a row per attempt."""
    with kbc.connect_closing() as conn:
        tid = _triage_card(conn)
        _comment(conn, tid, INCIDENT_APPROVAL_COMMENT)

    for _ in range(3):
        outcome, aux = _decompose_without_llm(tid)
        assert not outcome.ok
        assert aux.call_count == 0

    with kbc.connect_closing() as conn:
        assert len(_refusals(conn, tid)) == 1


# ---------------------------------------------------------------------------------------
# The other direction: the guard must not refuse ordinary triage work
# ---------------------------------------------------------------------------------------

def test_an_ordinary_triage_card_still_decomposes(kanban_home: Path) -> None:
    with kbc.connect_closing() as conn:
        tid = _triage_card(conn, title="vague idea")
        child_ids = kbg.decompose_triage_task(
            conn, tid, root_assignee="orchestrator", children=_children(), author="decomposer",
        )
        assert isinstance(child_ids, list) and len(child_ids) == 2
        assert _status(conn, tid) == "todo"
        assert not _refusals(conn, tid)


def test_prose_about_approval_does_not_refuse_a_card(kanban_home: Path) -> None:
    """The marker is the verdict, not the word: an OPEN card discussing approval still runs.

    Asserts the DECISION — the predicate declines and the fan-out still happens — rather
    than that the card reaches the aux client: routing reads the ambient config, which is
    not this guard's business and differs between an isolated run and a full-suite run.
    """
    with kbc.connect_closing() as conn:
        tid = _triage_card(conn)
        _comment(
            conn, tid,
            "Waiting on the ops head: this needs an approval before\n"
            "we spend the token budget. Nothing is approved yet.\n",
        )
        assert kb.decompose_refusal(conn, tid) is None, "lower-case prose must not refuse the card"
        child_ids = kbg.decompose_triage_task(
            conn, tid, root_assignee="orchestrator", children=_children(), author="decomposer",
        )
        assert isinstance(child_ids, list) and len(child_ids) == 2, (
            "an open card must still fan out"
        )
        assert not _refusals(conn, tid), "a promoted card carries no refusal"


def test_a_marker_from_long_ago_does_not_refuse_a_settled_card(kanban_home: Path) -> None:
    """The scan is bounded to the tail: a stale branch mention is not the card's state."""
    with kbc.connect_closing() as conn:
        tid = _triage_card(conn)
        _comment(conn, tid, f"First attempt ran on `fix/{tid}-old-approach`.")
        for i in range(6):
            _comment(conn, tid, f"progress note {i}", author="worker")
        child_ids = kbg.decompose_triage_task(
            conn, tid, root_assignee="orchestrator", children=_children(), author="decomposer",
        )
        assert isinstance(child_ids, list), "the stale marker is out of the scanned window"
        assert not _refusals(conn, tid)


def test_the_escalation_park_still_refuses_without_a_record_refusal(kanban_home: Path) -> None:
    """The pre-existing breaker guard keeps its shape: its own event carries the reason."""
    with kbc.connect_closing() as conn:
        tid = _park_in_triage(conn)

    outcome, aux = _decompose_without_llm(tid)

    assert not outcome.ok
    assert aux.call_count == 0
    assert "block_loop_detected" in outcome.reason
    assert tid in outcome.reason
    with kbc.connect_closing() as conn:
        assert not _refusals(conn, tid), (
            "the escalation is recorded once, on its own event — not duplicated"
        )


# ---------------------------------------------------------------------------------------
# The predicate, at the layer the promotion paths call
# ---------------------------------------------------------------------------------------

def test_the_record_predicate_is_a_first_class_kernel_guard(kanban_home: Path) -> None:
    predicate = getattr(kb, "decompose_refusal", None)
    assert predicate is not None, "kanban_db must expose the record predicate"

    with kbc.connect_closing() as conn:
        plain = _triage_card(conn, title="clean")
        assert predicate(conn, plain) is None, "a clean triage card is not refused"
        assert kb.DECOMPOSE_REFUSAL_EVENT_KIND == REFUSAL_EVENT_KIND
        assert not kb.DecomposeRefusal("t_x", []), "the refusal types stay falsy"

        decided = _triage_card(conn, title="decided")
        _comment(conn, decided, INCIDENT_APPROVAL_COMMENT)
        _comment(conn, decided, INCIDENT_TRIAGE_COMMENT)
        refusal = predicate(conn, decided)
        assert refusal is not None
        assert refusal.causes == ["superseded", "approved", "live_artifact"]
        assert refusal.task_id == decided


# ---------------------------------------------------------------------------------------
# F1 — the branch-name SUPERSET, and its false-positive bound
# ---------------------------------------------------------------------------------------

def test_a_project_slug_branch_refuses(kanban_home: Path) -> None:
    """``projects_db.branch_name_for`` mints ``<project.slug>/<task_id>``.

    None of the six prefixes the old marker enumerated covers a USER-CHOSEN project slug,
    so a project-linked card with live work was fanned out. The stable discriminator is the
    card-id segment, whichever namespace precedes it.
    """
    with kbc.connect_closing() as conn:
        tid = _triage_card(conn)
        _comment(
            conn, tid,
            "STATUS\n\nthe work rides `hermes-agent/t_9cb7f0b9-decompose-record-guard`\n",
        )

    outcome, aux = _decompose_without_llm(tid)

    assert aux.call_count == 0
    assert not outcome.ok
    with kbc.connect_closing() as conn:
        assert _refusals(conn, tid)[0].payload["causes"] == ["live_artifact"]


def test_a_bare_card_id_mention_does_not_refuse(kanban_home: Path) -> None:
    """The marker is a BRANCH reference: a bare id in prose is not a live artifact.

    The superset must not swallow every ``t_<id>`` token — a slash-prefixed one is a branch,
    a bare one is a citation, and a card mentioning a sibling id must still fan out.
    """
    with kbc.connect_closing() as conn:
        tid = _triage_card(conn)
        _comment(conn, tid, "context: t_9cb7f0b9 is the sibling card; nothing built yet.")
        assert kb.decompose_refusal(conn, tid) is None, "a bare id is not a branch marker"
        child_ids = kbg.decompose_triage_task(
            conn, tid, root_assignee="orchestrator", children=_children(), author="decomposer",
        )
        assert isinstance(child_ids, list) and len(child_ids) == 2
        assert not _refusals(conn, tid)


# ---------------------------------------------------------------------------------------
# F2 — a durable decision is found beyond the perishable tail window
# ---------------------------------------------------------------------------------------

@pytest.mark.parametrize(
    "note,cause",
    [
        ("VERDICT: the CODE HALF IS APPROVED at `3ce158d8`.", "approved"),
        ("SUPERSEDED — no work performed; the successor carries it.", "superseded"),
    ],
)
def test_a_durable_verdict_is_found_beyond_the_tail_window(
    kanban_home: Path, note: str, cause: str,
) -> None:
    """A decision does not expire: it is read over the WHOLE thread, not the newest five.

    The old flat ``LIMIT 5`` hid the verdict behind any five later comments, so a decided
    card decomposed freely — the exact defect this guard exists to stop.
    """
    with kbc.connect_closing() as conn:
        tid = _triage_card(conn)
        _comment(conn, tid, f"NOTE\n\n{note}\n")
        for i in range(6):
            _comment(conn, tid, f"progress note {i}", author="worker")

    outcome, aux = _decompose_without_llm(tid)

    assert aux.call_count == 0, "a decided card must not reach the LLM"
    assert not outcome.ok
    with kbc.connect_closing() as conn:
        assert _refusals(conn, tid)[0].payload["causes"] == [cause]


# ---------------------------------------------------------------------------------------
# F3 — the SPECIFY path is guarded too (the record guard was decompose-only)
# ---------------------------------------------------------------------------------------

def test_specify_triage_task_refuses_a_decided_card(kanban_home: Path) -> None:
    """The kernel verb returns the RECORD refusal, recorded, and a falsy — not a bare False."""
    with kbc.connect_closing() as conn:
        tid = _triage_card(conn)
        _comment(conn, tid, INCIDENT_APPROVAL_COMMENT)
        task = kb.get_task(conn, tid)
        assert task is not None
        before_body = task.body

        refusal = kb.specify_triage_task(
            conn, tid, title="new title", body="new body", author="specifier",
        )

        assert not refusal, "a decided card must not be promoted"
        assert not isinstance(refusal, bool), "the refusal must carry the record reason"
        assert isinstance(refusal, kb.DecomposeRefusal)
        assert "approved" in refusal.causes
        assert _status(conn, tid) == "triage"
        after = kb.get_task(conn, tid)
        assert after is not None
        assert after.body == before_body, "the pinned body must be untouched"
        assert not _events(conn, tid, "specified")
        assert not _events(conn, tid, "promoted")
        assert len(_refusals(conn, tid)) == 1, "the refusal is recorded by the kernel verb"


def test_specify_task_refuses_before_the_aux_call(kanban_home: Path) -> None:
    """The pre-aux check refuses AND records, so the LLM round-trip is never spent."""
    with kbc.connect_closing() as conn:
        tid = _triage_card(conn)
        _comment(conn, tid, INCIDENT_APPROVAL_COMMENT)

    aux = MagicMock(return_value=('{"title": "invented", "body": "invented"}', ""))
    with patch("hermes_cli.kanban_specify._call_aux", aux):
        outcome = specify.specify_task(tid, author="specifier")

    assert aux.call_count == 0, "the pre-check must refuse before the aux round trip"
    assert outcome.ok is False
    assert "refused auto-decomposition" in outcome.reason, (
        "the reason must be the RECORD refusal, not the specified-race message"
    )
    with kbc.connect_closing() as conn:
        assert _status(conn, tid) == "triage"
        assert len(_refusals(conn, tid)) == 1, "only the pre-aux leg runs, and it records"


def test_the_escalation_park_still_refuses_the_specify_path(kanban_home: Path) -> None:
    """No regression: the breaker's park refuses the kernel verb AND the specify verb."""
    with kbc.connect_closing() as conn:
        tid = _park_in_triage(conn)
        refusal = kb.specify_triage_task(conn, tid, title="t", body="b", author="specifier")
        assert not refusal
        assert isinstance(refusal, kb.TriageEscalationRefusal), "the park, not a record refusal"
        assert _status(conn, tid) == "triage"
        assert not _refusals(conn, tid), "an escalation park records no decompose refusal"

    aux = MagicMock(return_value=(None, "aux client unavailable"))
    with patch("hermes_cli.kanban_specify._call_aux", aux):
        outcome = specify.specify_task(tid)
    assert outcome.ok is False
    assert aux.call_count == 0


def test_an_ordinary_triage_card_still_specifies_and_promotes(kanban_home: Path) -> None:
    """No over-refusal: the specifier path still promotes a plain triage card."""
    with kbc.connect_closing() as conn:
        tid = _triage_card(conn, title="vague idea")
        ok = kb.specify_triage_task(conn, tid, title="sharp idea", body="do it", author="specifier")
        assert ok is True
        assert _status(conn, tid) in {"todo", "ready"}, "promoted out of triage"
        assert not _refusals(conn, tid)
