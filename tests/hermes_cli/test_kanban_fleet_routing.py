"""Fleet routing at dispatch: Q2 (right agent) beside Q1 (can it start), and the card's class.

Criterion this file pins, one test per acceptance condition:

* a review card on a profile that cannot resolve ``sdlc-review`` is refused with that identifier
  named — Q1 is mechanical and rule-independent;
* a card classed ``code-review`` on a profile restricted to security work is refused while the
  same profile is allowed ``security-review`` — the rule binds, it is not deny-all;
* a profile absent from the rule is refused: deny by default;
* a card that satisfies both questions still spawns;
* with the rule still PROPOSED nothing is refused on policy — enforcement is switched on by the
  record, not by shipping this module;
* a card's ``task_class`` survives the row round trip and the full create → comment → complete →
  archive lifecycle (the double-pass regression that broke ``kanban complete``).

The rule, the profiles and their skills all live under an isolated ``HERMES_HOME``; no live board
or profile is touched.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_dispatch as kbd
from hermes_cli import kanban_fleet_routing as fr
from hermes_cli import kanban_skill_preflight as preflight
from hermes_cli.kanban_db_connect import connect

RULE = {
    "schema": "fleet-routing-rule/v1",
    "status": "ADOPTED - in force by decision test-adoption",
    "config_version": 52,
    "enforcement": {
        "enabled": True,
        "status": "ADOPTED",
        "decision_id": "test-adoption",
        "config_version": 52,
    },
    "task_classes": ["execution", "code-review", "security-review", "design-review"],
    "profiles": {
        "probe": {
            "may_serve": ["execution"],
            "must_not_serve": ["code-review", "security-review"],
            "review_allowed": False,
            "review_classes": [],
            "decided_by": "decision-probe",
        },
        "reviewer": {
            "may_serve": ["execution", "code-review", "security-review"],
            "must_not_serve": [],
            "review_allowed": True,
            "review_classes": ["code-review"],
            # t_d90101d2: reviewing a work-class is a RECORDED grant, not implied by may_serve.
            "review_competence": ["execution"],
            "review_competence_decided_by": "test-grant",
            "decided_by": None,
        },
        "generalist": {
            "may_serve": ["execution", "code-review"],
            "must_not_serve": [],
            "review_allowed": True,
            "review_classes": ["code-review"],
            "decided_by": None,
        },
        "retired": {
            "may_serve": [],
            "must_not_serve": ["*"],
            "review_allowed": False,
            "review_classes": [],
            "decided_by": None,
        },
    },
}


def _write_profile(home: Path, name: str, skills: tuple[str, ...]) -> None:
    profile = home / "profiles" / name
    profile.mkdir(parents=True, exist_ok=True)
    (profile / "config.yaml").write_text("model: {}\n", encoding="utf-8")
    for skill in skills:
        skill_dir = profile / "skills" / skill
        skill_dir.mkdir(parents=True, exist_ok=True)
        (skill_dir / "SKILL.md").write_text(
            f"---\nname: {skill}\ndescription: {skill}\n---\n\nBody.\n", encoding="utf-8")


@pytest.fixture
def fleet(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict:
    """Isolated fleet: two provisioned profiles and a rule file on the env override."""
    home = tmp_path / ".hermes"
    home.mkdir()
    # `probe` cannot start a review turn (no sdlc-review); `reviewer` can.
    _write_profile(home, "probe", ("demo",))
    _write_profile(home, "reviewer", ("demo", "sdlc-review"))
    _write_profile(home, "generalist", ("demo", "sdlc-review"))
    rule_path = tmp_path / "fleet-routing-rule.json"
    rule_path.write_text(json.dumps(RULE), encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv(fr.RULE_PATH_ENV, str(rule_path))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return {"home": home, "rule_path": rule_path}


# --- the rule record ---------------------------------------------------------------------------

def test_enforcement_is_switched_on_by_the_record(fleet):
    rule, path = fr.load_rule()
    assert path == fleet["rule_path"]
    enabled, status, decision = fr.enforcement_state(rule)
    assert (enabled, status, decision) == (True, "ADOPTED", "test-adoption")
    assert fr.rule_config_version(rule) == 52


def test_a_proposed_rule_is_evaluated_but_refuses_nothing(fleet):
    """Negative control for the adoption gate: shipping this code must not switch the rule on."""
    proposed = dict(RULE, status="PROPOSED", enforcement={
        "enabled": False, "status": "PROPOSED", "decision_id": None, "config_version": 52})
    assert fr.enforcement_state(proposed)[0] is False
    verdict = fr.assessment("ghost-profile", "code-review", lane="ready", rule=proposed,
                            path=str(fleet["rule_path"]))
    assert verdict.allowed is True and verdict.enforced is False
    # …while Q1, which is not policy, still refuses a profile that cannot start.
    assert fr.assessment("probe", None, lane="review", identifiers=["sdlc-review"],
                         rule=proposed).refusal.question == "Q1"


# --- Q1 ----------------------------------------------------------------------------------------

def test_q1_names_the_missing_identifier_not_just_the_profile(fleet):
    refusal = fr.assignment_refusal("probe", None, lane="review", identifiers=["sdlc-review"])
    assert refusal is not None and refusal.question == "Q1"
    assert refusal.missing == ("sdlc-review",)
    text = refusal.as_text()
    assert "sdlc-review" in text and "probe" in text
    assert fr.ROUTING_REFUSAL_MARKER in text
    # Traceable to the record it enforces.
    assert "fleet-routing-rule.json" in text
    assert "config_version=52" in text


def test_q1_is_the_same_predicate_the_worker_uses(fleet):
    """One resolved identifier is enough: Q1 must not refuse a partial miss."""
    assert preflight.unsatisfiable_preload("reviewer", ["demo", "absent"]) is None
    assert fr.assignment_refusal("reviewer", None, lane="review",
                                 identifiers=["demo", "sdlc-review"]) is None


def test_q1_does_not_fire_on_the_execute_lane(fleet):
    # An execute-lane worker is not force-loaded with sdlc-review, so nothing can go missing.
    identifiers = kb.worker_skill_identifiers([], review_lane=False)
    assert identifiers == []
    assert fr.assignment_refusal("probe", "execution", lane="ready", identifiers=identifiers) is None


# --- Q2 ----------------------------------------------------------------------------------------

def test_the_rule_binds_without_being_deny_all(fleet):
    # `design-review` is neither a review class nor work `reviewer` may serve.
    denied = fr.assignment_refusal("reviewer", "design-review", lane="review",
                                   identifiers=["sdlc-review"])
    allowed = fr.assignment_refusal("reviewer", "code-review", lane="review",
                                    identifiers=["sdlc-review"])
    assert denied is not None and denied.question == "Q2"
    assert "design-review" in denied.as_text()
    assert denied.capable_profiles == (), "no profile in this rule may review design work"
    assert allowed is None, "same profile, permitted class: the rule must not be deny-all"


def test_review_competence_covers_work_the_profile_may_serve(fleet):
    """t_2e0cd7e4: an execution-class defect card handed to a reviewer was refused forever.

    A card's class names the WORK, so a profile trusted to do ``execution`` is trusted to review
    it; the old review-lane check compared the class against ``review_classes`` alone, refused,
    re-refused after each unblock, and after three rounds escalated to triage and decomposed into
    duplicate children — one bug, three concurrent edits to the same file.
    """
    verdict = fr.assessment("reviewer", "execution", lane="review", identifiers=["sdlc-review"])
    assert verdict.allowed is True and verdict.refusal is None
    # Round-1 review (t_d90101d2): competence is a RECORDED grant, not `may_serve` in disguise.
    # The same profile may serve `security-review` as work and is still refused as its reviewer,
    # because the record does not grant that lane — otherwise a code predicate would silently
    # widen every profile's review duty past its own written note.
    refused = fr.assessment("reviewer", "security-review", lane="review",
                            identifiers=["sdlc-review"])
    assert refused.refusal is not None and refused.refusal.question == "Q2"
    # …and an explicit prohibition still fails closed.
    probe = fr.assignment_refusal("probe", "execution", lane="review", identifiers=[])
    assert probe is not None and probe.question == "Q2"


def test_a_narrow_audit_profile_stays_out_of_general_review(fleet):
    narrow = dict(RULE, profiles=dict(RULE["profiles"], narrow={
        "may_serve": ["execution"], "must_not_serve": [], "review_allowed": True,
        "review_classes": [], "excluded_from_general_review": True,
    }))
    verdict = fr.assessment("narrow", "execution", lane="review", rule=narrow,
                            path=str(fleet["rule_path"]))
    assert verdict.refusal is not None and verdict.refusal.question == "Q2"
    assert "review_classes" in verdict.refusal.as_text()
    # …while the same profile is still routed execution work on the ready lane.
    assert fr.assessment("narrow", "execution", lane="ready", rule=narrow,
                         path=str(fleet["rule_path"])).allowed is True


def test_a_profile_absent_from_the_rule_is_refused(fleet):
    refusal = fr.assignment_refusal("ghost-profile", "execution", lane="ready")
    assert refusal is not None and refusal.question == "Q2"
    assert "not in the routing rule" in refusal.as_text()
    assert "deny by default" in refusal.as_text()


def test_a_retired_profile_cannot_be_routed(fleet):
    refusal = fr.assignment_refusal("retired", "execution", lane="ready")
    assert refusal is not None and "retired from dispatch" in refusal.as_text()


def test_a_profile_without_review_duty_cannot_be_handed_a_review(fleet):
    # `probe` holds no review duty at all. Classless so Q1 cannot fire first: this is the pure
    # Q2 verdict for the review lane.
    verdict = fr.assessment("probe", None, lane="review", identifiers=[])
    assert verdict.refusal is not None and verdict.refusal.question == "Q2"
    assert "review_allowed" in verdict.refusal.as_text()
    # A profile that MAY review code is not caught by the same clause.
    assert fr.assignment_refusal("generalist", "code-review", lane="review",
                                 identifiers=["sdlc-review"]) is None


def test_an_unclassified_card_is_admitted_and_flagged(fleet):
    verdict = fr.assessment("reviewer", None, lane="review", identifiers=["sdlc-review"])
    assert verdict.allowed is True
    assert verdict.unclassified is True, "class fit must be reported, never guessed"
    assert fr.assessment("reviewer", "code-review", lane="review",
                         identifiers=["sdlc-review"]).unclassified is False


def test_an_unknown_class_is_refused_rather_than_guessed(fleet):
    with pytest.raises(ValueError) as excinfo:
        fr.check_task_class("code-reviewish")
    assert "code-reviewish" in str(excinfo.value)
    assert "execution" in str(excinfo.value), "the error must name the vocabulary"
    assert fr.check_task_class("  code-review  ") == "code-review"
    assert fr.check_task_class("") is None and fr.check_task_class(None) is None


# --- the card carries its class ----------------------------------------------------------------

def test_task_from_row_carries_task_class(fleet):
    """Regression: task_class must not be passed twice into ``Task(**...)``."""
    with connect() as conn:
        tid = kb.create_task(conn, title="classed", assignee="probe", task_class="execution")
        row = conn.execute("SELECT * FROM tasks WHERE id = ?", (tid,)).fetchone()
        task = kb.Task.from_row(row)
        assert task.task_class == "execution"
        assert kb.get_task(conn, tid).task_class == "execution"
        # An unclassified card reads back as None, never ''.
        plain = kb.create_task(conn, title="unclassified", assignee="probe")
        assert kb.get_task(conn, plain).task_class is None
        listed = {t.id: t.task_class for t in kb.list_tasks(conn)}
        assert listed[tid] == "execution" and listed[plain] is None


def test_board_lifecycle_survives_a_classed_card(fleet):
    """create → comment → complete → archive, the path a broken Task row kills last."""
    with connect() as conn:
        tid = kb.create_task(conn, title="lifecycle", assignee="probe", task_class="execution")
        kb.add_comment(conn, tid, "smoke", "lifecycle check")
        assert kb.complete_task(conn, tid, summary="done") is True
        assert kb.get_task(conn, tid).status == "done"
        assert kb.archive_task(conn, tid) is True
        assert kb.get_task(conn, tid).status == "archived"


def test_request_review_stamps_the_class(fleet):
    with connect() as conn:
        tid = kb.create_task(conn, title="handoff", assignee="probe")
        ok, why = kb.request_review(conn, tid, summary="ready", task_class="code-review",
                                    with_reason=True)
        assert ok, why
        task = kb.get_task(conn, tid)
        assert (task.status, task.task_class) == ("review", "code-review")


# --- the review return restores the work class ------------------------------------------------


def _code_review_round_trip(conn, tid):
    """Claim -> hand to review as ``code-review`` -> return with changes.

    Returns the card after ``request_changes``. The implementer ``probe`` may serve
    ``execution`` but not ``code-review``, so any leftover review class refuses it.
    """
    kb.claim_task(conn, tid)
    assert kb.request_review(
        conn, tid, summary="ready", reviewer="reviewer",
        expected_run_id=kb.get_task(conn, tid).current_run_id,
        task_class="code-review",
    ) is True
    assert kb.get_task(conn, tid).task_class == "code-review"
    review = kb.claim_review_task(conn, tid, claimer="reviewer:1")
    assert review is not None
    assert kb.request_changes(
        conn, tid, reason="found 3 issues", expected_run_id=review.current_run_id,
    ) == (True, "probe")
    return kb.get_task(conn, tid)


def test_request_changes_restores_the_implementers_work_class(fleet):
    """t_fe130245: returning a reviewed card to execution must restore the card's work class,
    not leave the review class that refuses the implementer on the ready lane."""
    with connect() as conn:
        tid = kb.create_task(conn, title="defect to review", assignee="probe",
                             task_class="execution")
        rework = _code_review_round_trip(conn, tid)
        assert (rework.status, rework.assignee, rework.task_class) == ("ready", "probe", "execution")
        # The implementer is dispatchable under its restored class on the ready lane.
        assert fr.assessment("probe", rework.task_class, lane="ready").allowed is True
        # ...while the review class it left would have refused it.
        assert fr.assessment("probe", "code-review", lane="ready").allowed is False


def test_request_changes_round_trip_is_dispatchable_to_the_implementer(fleet, no_profile_gate):
    """The auto-router must not re-hand a changes-requested card to the reviewer: the card's
    restored class routes to the implementer, and no reroute event is written."""
    with connect() as conn:
        tid = kb.create_task(conn, title="defect to review", assignee="probe",
                             task_class="execution")
        _code_review_round_trip(conn, tid)

    spawned: list[str] = []

    def spawn(task, workspace, board=None):
        spawned.append(task.id)
        return 4242

    with connect() as conn:
        res = kbd.dispatch_once(conn, spawn_fn=spawn, failure_limit=3)
    assert spawned == [tid], "the implementer must be dispatched, not re-routed to the reviewer"
    with connect() as conn:
        assert [e for e in kb.list_events(conn, tid) if e.kind == "rerouted"] == []
        assert [e for e in kb.list_events(conn, tid) if e.kind == "routing_refused"] == []


def test_reopen_review_restores_the_implementers_work_class(fleet):
    """t_fe130245: the reopen-review return path must restore the work class too."""
    with connect() as conn:
        tid = kb.create_task(conn, title="reopen me", assignee="probe",
                             task_class="execution")
        kb.claim_task(conn, tid)
        assert kb.request_review(
            conn, tid, summary="v1", reviewer="reviewer",
            expected_run_id=kb.get_task(conn, tid).current_run_id,
            task_class="code-review",
        ) is True
        assert kb.get_task(conn, tid).task_class == "code-review"
        assert kb.reopen_review_task(conn, tid) is True
        reopened = kb.get_task(conn, tid)
        assert (reopened.status, reopened.assignee, reopened.task_class) == (
            "ready", "probe", "execution")
        assert fr.assessment("probe", reopened.task_class, lane="ready").allowed is True


def test_request_changes_clears_a_legacy_review_class_with_no_recorded_prior(fleet):
    """A card that entered review before the fix carries a review class but no recorded work
    class; the return path clears it to NULL so the implementer is reachable, instead of
    leaving a class that refuses it."""
    with connect() as conn:
        tid = kb.create_task(conn, title="legacy", assignee="probe", task_class="code-review")
        kb.claim_task(conn, tid)
        assert kb.request_review(
            conn, tid, summary="v1", reviewer="reviewer",
            expected_run_id=kb.get_task(conn, tid).current_run_id,
            task_class="code-review",
        ) is True
        # Rewrite the handoff payload to the pre-fix shape: review class, no prior recorded.
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE task_events SET payload = ? "
                "WHERE task_id = ? AND kind = 'review_requested'",
                (json.dumps({"summary": "v1", "implementer": "probe",
                             "reviewer": "reviewer", "task_class": "code-review"}), tid),
            )
        review = kb.claim_review_task(conn, tid, claimer="reviewer:1")
        assert review is not None
        assert kb.request_changes(
            conn, tid, reason="fix", expected_run_id=review.current_run_id,
        ) == (True, "probe")
        assert kb.get_task(conn, tid).task_class is None
        assert fr.assessment("probe", None, lane="ready").allowed is True


def test_request_review_without_a_class_leaves_the_work_class_on_return(fleet):
    """A handoff that never overwrites the class must not touch it on return."""
    with connect() as conn:
        tid = kb.create_task(conn, title="no class", assignee="probe", task_class="execution")
        kb.claim_task(conn, tid)
        assert kb.request_review(
            conn, tid, summary="v1", reviewer="reviewer",
            expected_run_id=kb.get_task(conn, tid).current_run_id,
        ) is True
        assert kb.get_task(conn, tid).task_class == "execution"  # untouched by a classless handoff
        review = kb.claim_review_task(conn, tid, claimer="reviewer:1")
        assert review is not None
        assert kb.request_changes(
            conn, tid, reason="fix", expected_run_id=review.current_run_id,
        ) == (True, "probe")
        assert kb.get_task(conn, tid).task_class == "execution"


def test_create_task_refuses_a_class_outside_the_vocabulary(fleet):
    with connect() as conn:
        with pytest.raises(ValueError):
            kb.create_task(conn, title="bad class", assignee="probe", task_class="made-up-work")


# --- the dispatch gate -------------------------------------------------------------------------

@pytest.fixture
def no_profile_gate(monkeypatch: pytest.MonkeyPatch):
    """Every assignee counts as installed; the routing gate is what is under test."""
    monkeypatch.setattr(kbd, "_profile_exists_fn", lambda: None)


def _dispatch(conn, spawn_fn):
    return kbd.dispatch_once(conn, spawn_fn=spawn_fn, failure_limit=3)


def test_dispatch_refuses_q1_and_names_the_identifier(fleet, no_profile_gate):
    spawned: list[str] = []

    def spawn(task, workspace, board=None):
        spawned.append(task.id)
        return 4242

    with connect() as conn:
        tid = kb.create_task(conn, title="review me", assignee="probe")
        kb.request_review(conn, tid, summary="ready")  # no class: Q1 still applies
        res = kbd.dispatch_once(conn, spawn_fn=spawn, failure_limit=3)
        task = kb.get_task(conn, tid)
        events = [e for e in kb.list_events(conn, tid) if e.kind == "routing_refused"]
        blocked = [e for e in kb.list_events(conn, tid) if e.kind == "blocked"]
        run = kb.latest_run(conn, tid)

    assert spawned == [], "a card whose worker cannot start must not be spawned"
    assert [entry[0] for entry in res.routing_refused] == [tid]
    assert task is not None and task.status == "blocked" and task.block_kind == "capability"
    # The reason names the identifier, so the board alone says what to install.
    assert "sdlc-review" in (run.summary or "")
    assert blocked and "sdlc-review" in str(blocked[-1].payload)
    assert events and events[0].payload["question"] == "Q1"
    assert events[0].payload["missing"] == ["sdlc-review"]
    assert events[0].payload["rule_path"].endswith("fleet-routing-rule.json")
    assert events[0].payload["config_version"] == 52


def test_dispatch_refuses_q2_by_class_and_allows_the_permitted_class(fleet, no_profile_gate):
    spawned: list[str] = []

    def spawn(task, workspace, board=None):
        spawned.append(task.id)
        return 4242

    with connect() as conn:
        # Class permitted to this reviewer → must spawn.
        allowed = kb.create_task(conn, title="code review", assignee="reviewer",
                                 task_class="code-review")
        kb.request_review(conn, allowed, summary="ready", task_class="code-review")
        # Execution work the same reviewer may serve → must spawn too (t_2e0cd7e4).
        defect = kb.create_task(conn, title="defect raised by review", assignee="reviewer",
                                task_class="execution")
        kb.request_review(conn, defect, summary="ready", task_class="execution")
        # Class neither reviewable nor served by the reviewer → must be refused.
        denied = kb.create_task(conn, title="design review", assignee="reviewer",
                                task_class="design-review")
        kb.request_review(conn, denied, summary="ready", task_class="design-review")
        res = kbd.dispatch_once(conn, spawn_fn=spawn, failure_limit=3)
        allowed_task = kb.get_task(conn, allowed)
        defect_task = kb.get_task(conn, defect)
        denied_task = kb.get_task(conn, denied)
        denied_events = [e for e in kb.list_events(conn, denied) if e.kind == "routing_refused"]
        defect_events = [e for e in kb.list_events(conn, defect) if e.kind == "routing_refused"]

    assert spawned == [allowed, defect], "both permitted classes must be dispatched"
    assert defect_events == [], "a reviewer's execution-class defect card must not be refused"
    assert defect_task is not None and defect_task.status in ("running", "review")
    assert [entry[0] for entry in res.routing_refused] == [denied]
    assert allowed_task is not None and allowed_task.status in ("running", "review")
    assert denied_task is not None and denied_task.status == "blocked"
    assert denied_task.block_kind == "needs_input"
    assert denied_events and denied_events[0].payload is not None
    payload = denied_events[0].payload
    assert payload["question"] == "Q2"
    assert payload["profile"] == "reviewer" and payload["task_class"] == "design-review"
    assert payload["decision_id"] == "test-adoption"
    assert payload["rule_status"] == "ADOPTED"


def test_dispatch_refuses_an_unlisted_profile(fleet, no_profile_gate):
    spawned: list[str] = []

    def spawn(task, workspace, board=None):
        spawned.append(task.id)
        return 4242

    with connect() as conn:
        tid = kb.create_task(conn, title="orphan", assignee="ghost-profile", task_class="execution")
        res = kbd.dispatch_once(conn, spawn_fn=spawn, failure_limit=3)
        task = kb.get_task(conn, tid)

    assert spawned == []
    assert [entry[0] for entry in res.routing_refused] == [tid]
    assert task is not None and task.status == "blocked"


def test_a_proposed_rule_does_not_block_dispatch(fleet, no_profile_gate, monkeypatch):
    """The live switch is the record: with it off, a card the rule would deny still spawns."""
    proposed = dict(RULE, status="PROPOSED", enforcement={
        "enabled": False, "status": "PROPOSED", "decision_id": None, "config_version": 52})
    path = fleet["rule_path"]
    path.write_text(json.dumps(proposed), encoding="utf-8")
    spawned: list[str] = []

    def spawn(task, workspace, board=None):
        spawned.append(task.id)
        return 4242

    with connect() as conn:
        tid = kb.create_task(conn, title="orphan", assignee="ghost-profile", task_class="execution")
        res = kbd.dispatch_once(conn, spawn_fn=spawn, failure_limit=3)

    assert spawned == [tid]
    assert res.routing_refused == []


RECORD_COPY = Path(__file__).with_name("data") / "fleet-routing-rule.copy.json"


def _admission_rule() -> tuple[dict, Path]:
    """The fleet record as adopted, byte-for-byte copied into ``tests/hermes_cli/data``.

    Admission can only be judged against the *record*, not against a fixture invented here — that is
    precisely the round-1 defect, where a code predicate widened four profiles past their own notes.
    The suite refuses to read the live home (``tests/home_io_guard`` makes I/O against it a test
    bug), so the record is pinned instead: ``fleet-routing-rule.copy.json`` is a verbatim copy of
    ``$HERMES_HOME/audit/fleet-routing/fleet-routing-rule.json`` at ``config_version`` 54, taken
    when this grant was decided. Refresh it with::

        cp "$HERMES_HOME/audit/fleet-routing/fleet-routing-rule.json" \\
           tests/hermes_cli/data/fleet-routing-rule.copy.json

    and re-run this file — if a later edit grants a lane the canon does not record, that shows up
    as a failure here and in the drift check in the audit note.
    """
    return fr.load_rule(str(RECORD_COPY))


def test_review_competence_on_the_recorded_rule_is_only_the_recorded_grant():
    """t_d90101d2 round 1: a code predicate silently widened review lanes the canon never granted.

    Measured on the recorded rule, the old ``review_classes | may_serve`` union admitted four
    profiles past their own notes — ``hottr-bot`` ("Never a general review lane"), ``researcher``
    ("Reviews evidence claims only"), ``ux-ui-audit-agent`` ("Review class stays design-review").
    Only ``engineering-lead``/``execution`` was intended, because that is the combination the
    t_2e0cd7e4 defect card needs. The grant is now recorded data, so this test pins both halves:
    the one intended grant, and every profile whose note forbids the lane.
    """
    rule, path = _admission_rule()

    def verdict(profile: str, task_class: str):
        return fr.assessment(profile, task_class, lane="review", rule=rule,
                             path=str(path), identifiers=())

    # The defect card from t_2e0cd7e4: execution work handed to the reviewing profile.
    assert verdict("engineering-lead", "execution").allowed is True

    for profile, task_class in (
        ("hottr-bot", "execution"),
        ("hottr-bot", "portfolio-brief"),
        ("researcher", "research"),
        ("ux-ui-audit-agent", "evidence-review"),
    ):
        assessment = verdict(profile, task_class)
        assert assessment.allowed is False, (
            f"{profile!r} is admitted to review {task_class!r} on the live rule, "
            "but the record does not grant that lane")
        assert assessment.refusal is not None and assessment.refusal.question == "Q2", (
            "a lane the record withholds is a routing refusal (Q2), not a capability claim")

    # Refusal text drives reassignment advice, so it must not point at a brand profile.
    for task_class in ("execution", "portfolio-brief", "research"):
        assert "hottr-bot" not in fr.capable_profiles(rule, task_class, lane="review")
    assert fr.capable_profiles(rule, "execution", lane="review") == ("engineering-lead",)

    # The grant is recorded, with a decision and a stated bound — not inferred from may_serve.
    entry = rule["profiles"]["engineering-lead"]
    assert "execution" in entry.get("review_competence", ())
    assert entry.get("review_competence_decided_by") == "t_d90101d2"
    assert "execution" in entry.get("may_serve", ()), (
        "a competence grant may never name a class the profile may not serve")
    assert "execution" not in entry.get("review_classes", ()), (
        "the grant must not be laundered into the profile's standing review duty")


# --- auto-route instead of parking (t_e148930c) -----------------------------
# A refused card that the rule itself can place was parked and re-promoted on
# every tick: one unclassified review card burned 3.8 hours and 120 refusals on
# 2026-11-08. The refusal names where it could go, so dispatch sends it there.

def test_general_reviewer_names_a_recorded_general_reviewer(fleet):
    rule, _path = fr.load_rule()
    assert fr.general_reviewer(rule) == "reviewer"
    assert fr.general_reviewer(rule, exclude=("reviewer",)) == "generalist"
    assert fr.general_reviewer(rule, exclude=("reviewer", "generalist")) is None, (
        "a retired or review-less profile must never be offered")


def test_dispatch_reroutes_an_unclassified_review_card_to_a_general_reviewer(fleet, no_profile_gate):
    """A profile absent from the rule cannot review, but the card need not park.

    ``outsider`` can start a review turn (so Q1 passes) and holds no recorded routing
    permission (Q2 refuses). With no class on the card there is no class to match, so the
    fallback is a general reviewer rather than a human.
    """
    _write_profile(fleet["home"], "outsider", ("demo", "sdlc-review"))
    spawned: list[str] = []

    with connect() as conn:
        tid = kb.create_task(conn, title="review this handoff", assignee="outsider")
        kb.request_review(conn, tid, summary="ready")
        res = _dispatch(conn, lambda task, workspace, board=None: spawned.append(task.id) or 4242)
        task = kb.get_task(conn, tid)
        kinds = [e.kind for e in kb.list_events(conn, tid)]

    assert [entry[0] for entry in res.routing_refused] == [tid]
    assert task is not None and task.assignee == "reviewer"
    assert task.status in ("review", "ready"), "the card waits for the next tick under its new assignee"
    assert "rerouted" in kinds and "blocked" not in kinds


def test_a_reroute_happens_once_per_class(fleet):
    """Second hop is the ping-pong the park exists to stop."""
    from types import SimpleNamespace

    refusal = SimpleNamespace(
        question="Q2", profile="reviewer", task_class="execution", lane="review",
        capable_profiles=("generalist",),
        as_event=lambda: {"question": "Q2", "profile": "reviewer", "task_class": "execution"},
        as_text=lambda: "fleet routing refusal (Q2) profile='reviewer'",
    )
    with connect() as conn:
        tid = kb.create_task(conn, title="defect", assignee="reviewer",
                             task_class="execution")
        assert kbd._reroute_target(conn, tid, refusal) == "generalist"
        assert kbd._reroute_refused_card(conn, tid, refusal, "generalist")
        assert kb.get_task(conn, tid).assignee == "generalist"
        assert kbd._reroute_target(conn, tid, refusal) is None, (
            "a card already rerouted for this class must park, not bounce again")
