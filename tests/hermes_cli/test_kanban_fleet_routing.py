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
    denied = fr.assignment_refusal("reviewer", "security-review", lane="review",
                                   identifiers=["sdlc-review"])
    allowed = fr.assignment_refusal("reviewer", "code-review", lane="review",
                                    identifiers=["sdlc-review"])
    assert denied is not None and denied.question == "Q2"
    assert "security-review" in denied.as_text()
    assert allowed is None, "same profile, permitted class: the rule must not be deny-all"


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
        # Class forbidden to the same reviewer → must be refused.
        denied = kb.create_task(conn, title="security review", assignee="reviewer",
                                task_class="security-review")
        kb.request_review(conn, denied, summary="ready", task_class="security-review")
        res = kbd.dispatch_once(conn, spawn_fn=spawn, failure_limit=3)
        allowed_task = kb.get_task(conn, allowed)
        denied_task = kb.get_task(conn, denied)
        denied_events = [e for e in kb.list_events(conn, denied) if e.kind == "routing_refused"]

    assert spawned == [allowed], "the permitted class must still be dispatched"
    assert [entry[0] for entry in res.routing_refused] == [denied]
    assert allowed_task is not None and allowed_task.status in ("running", "review")
    assert denied_task is not None and denied_task.status == "blocked"
    assert denied_task.block_kind == "needs_input"
    payload = denied_events[0].payload
    assert payload["question"] == "Q2"
    assert payload["profile"] == "reviewer" and payload["task_class"] == "security-review"
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
