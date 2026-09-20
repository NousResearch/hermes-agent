"""TDD contract for the Mossy read-only review-feedback loop.

All data is fixture JSON and all writes are redirected to temporary HERMES_HOME.
No test invokes GitHub or a real git mutation.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
MODULE_PATH = REPO / "scripts" / "moss" / "moss_review_feedback.py"
WATCH_PATH = REPO / "scripts" / "moss" / "moss-review-feedback-watch.py"
CONTEXT_PATH = REPO / "scripts" / "moss" / "moss-review-feedback-context.py"
FIXTURE = REPO / "tests" / "fixtures" / "moss_review_feedback.json"


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def fake_home(tmp_path, monkeypatch):
    home = tmp_path / "hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    return home


@pytest.fixture
def feedback():
    return json.loads(FIXTURE.read_text())


def test_normalise_all_review_surfaces_rejects_self_and_bot_authors(feedback):
    mod = load_module(MODULE_PATH, "moss_review_feedback_normalise")

    records = mod.normalise_feedback(feedback)

    assert [(r["surface"], r["key"], r["author"]) for r in records] == [
        ("review", "review:NousResearch/hermes-agent#42:R-1", "alice"),
        ("inline", "inline:NousResearch/hermes-agent#42:I-1", "bob"),
        ("conversation", "conversation:NousResearch/hermes-agent#42:C-1", "carol"),
        ("conversation", "conversation:NousResearch/hermes-agent#42:C-2", "Teknium"),
    ]
    assert records[1]["path"] == "scripts/watcher.py"
    assert [(r["repo"], r["pr_number"], r["source_item_id"], r["surface"]) for r in records] == [
        ("NousResearch/hermes-agent", 42, "R-1", "review"),
        ("NousResearch/hermes-agent", 42, "I-1", "inline"),
        ("NousResearch/hermes-agent", 42, "C-1", "conversation"),
        ("NousResearch/hermes-agent", 42, "C-2", "conversation"),
    ]


@pytest.mark.parametrize(
    ("author", "body", "surface", "expected"),
    [
        ("alice", "Please add a regression test.", "inline", "routine_patch"),
        ("alice", "Thanks, this looks good.", "conversation", "reply_only"),
        ("alice", "Can you clarify why this is safe?", "conversation", "clarification"),
        ("alice", "This may cause data loss in production.", "review", "sahil_escalation"),
        # Teknium has priority, but still receives the same safety classification.
        ("Teknium", "nit: typo", "inline", "routine_patch"),
    ],
)
def test_classifier_matrix_reviews_teknium_without_a_blind_bypass(author, body, surface, expected):
    mod = load_module(MODULE_PATH, "moss_review_feedback_classifier")

    assert mod.classify_feedback({"author": author, "body": body, "surface": surface}) == expected


def test_atomic_paths_are_distinct_and_dedupe_is_namespaced(fake_home, feedback):
    mod = load_module(MODULE_PATH, "moss_review_feedback_paths")
    paths = mod.feedback_paths()
    assert len(set(paths.values())) == len(paths)

    records = mod.normalise_feedback(feedback)
    state = mod.load_json(paths["state"], {"seen": []})
    new_records = mod.dedupe_records(records, state)
    mod.atomic_write_json(paths["state"], {"seen": [r["key"] for r in new_records]})
    mod.atomic_write_json(paths["queue"], {"pending": new_records})
    mod.atomic_write_json(paths["ledger"], {"events": []})
    mod.atomic_write_json(paths["rules"], {"rules": []})

    assert mod.dedupe_records(records, mod.load_json(paths["state"], {})) == []
    assert {p.name for p in paths.values()} == {
        "moss-review-feedback-state.json", "moss-review-feedback-queue.json",
        "moss-review-feedback-ledger.json", "moss-review-feedback-rules.json",
    }
    assert not list(paths["state"].parent.glob("*.tmp"))


def test_promotion_rule_selection_and_pre_push_report(fake_home, feedback):
    mod = load_module(MODULE_PATH, "moss_review_feedback_promotion")
    records = mod.normalise_feedback(feedback)
    # Add a non-maintainer question so the report covers every matrix bucket.
    # Teknium's question is also a clarification: priority never bypasses review.
    classified = mod.classify_records(records + [{
        "author": "alice", "body": "Can you clarify why this is safe?", "surface": "conversation"
    }])
    promoted = mod.promote_records(classified)
    report = mod.build_pre_push_report(classified, promoted, {"routine_patch": "tests-first"})

    # The formal review and inline review are independently namespaced and
    # therefore both remain actionable routine patches.
    assert [r["classification"] for r in promoted] == ["routine_patch", "routine_patch"]
    assert report["blocked"] is False
    assert report["rule_selection"]["routine_patch"] == "tests-first"
    assert report["counts"] == {"routine_patch": 2, "reply_only": 1, "clarification": 2}


def test_dry_run_runner_denies_mutating_gh_and_git_commands():
    mod = load_module(MODULE_PATH, "moss_review_feedback_runner")
    invoked = []

    def runner(command, **kwargs):
        invoked.append(command)
        return "executed"

    assert mod.run_read_only(["gh", "pr", "comment", "42", "--body", "no"], runner=runner)["status"] == "denied"
    assert mod.run_read_only(["git", "push", "origin", "main"], runner=runner)["status"] == "denied"
    assert mod.run_read_only(["gh", "pr", "view", "42"], runner=runner)["status"] == "dry_run"
    assert invoked == []


def test_watch_and_context_scripts_use_fixture_only_paths(fake_home, feedback):
    watch = load_module(WATCH_PATH, "moss_review_feedback_watch")
    context = load_module(CONTEXT_PATH, "moss_review_feedback_context")

    result = watch.process_fixture(feedback)
    report = context.render_context(result["classified"], result["promoted"])

    assert result["new_count"] == 4
    assert "Mossy review feedback" in report
    assert "Teknium" in report


def test_action_plan_is_deterministic_and_teknium_priority_does_not_change_safety():
    mod = load_module(MODULE_PATH, "moss_review_feedback_action_plan")
    records = [
        {"key": "a", "repo": "owner/repo", "pr_number": 9, "source_item_id": "a", "surface": "inline", "author": "alice", "body": "Please fix the bug and add a test."},
        {"key": "b", "repo": "owner/repo", "pr_number": 9, "source_item_id": "b", "surface": "conversation", "author": "alice", "body": "Thanks, this looks good."},
        {"key": "c", "repo": "owner/repo", "pr_number": 9, "source_item_id": "c", "surface": "conversation", "author": "alice", "body": "Can you clarify why this is safe?"},
        {"key": "d", "repo": "owner/repo", "pr_number": 9, "source_item_id": "d", "surface": "review", "author": "Teknium", "body": "This is a security issue."},
    ]

    plan = mod.build_action_plan(mod.classify_records(records))

    assert [(item["key"], item["action"], item["priority"]) for item in plan] == [
        ("a", "routine_patch", "normal"),
        ("b", "reply_only", "normal"),
        ("c", "clarification", "normal"),
        ("d", "sahil_escalation", "maintainer"),
    ]
    assert plan[0]["requires"] == ["root_cause", "changed_paths", "test_evidence", "prevention"]
    assert plan[3]["requires"] == ["sahil_decision"]
    assert {field: plan[0][field] for field in ("repo", "pr_number", "source_item_id", "surface")} == {
        "repo": "owner/repo", "pr_number": 9, "source_item_id": "a", "surface": "inline",
    }


def test_resolved_actionable_feedback_creates_complete_ledger_record(fake_home):
    mod = load_module(MODULE_PATH, "moss_review_feedback_ledger")
    record = {
        "key": "inline:repo#1:7", "surface": "inline", "author": "alice",
        "body": "This bug regresses parsing.", "path": "scripts/parser.py",
        "classification": "routine_patch",
    }

    ledger = mod.record_resolution(record, {
        "root_cause": "empty token was accepted", "changed_paths": ["scripts/parser.py", "tests/test_parser.py"],
        "test_evidence": ["pytest tests/test_parser.py -q"], "prevention": "regression test",
    })

    assert ledger == {
        "source": "inline", "thread": "inline:repo#1:7", "reviewer": "alice",
        "root_cause": "empty token was accepted", "changed_paths": ["scripts/parser.py", "tests/test_parser.py"],
        "test_evidence": ["pytest tests/test_parser.py -q"], "prevention": "regression test",
    }
    with pytest.raises(ValueError, match="test_evidence"):
        mod.record_resolution(record, {"root_cause": "x", "changed_paths": ["x"], "prevention": "test"})

    persisted = mod.persist_resolution(record, {
        "root_cause": "empty token was accepted", "changed_paths": ["scripts/parser.py"],
        "test_evidence": ["pytest tests/test_parser.py -q"], "prevention": "regression test",
    })
    assert persisted["ledger"]["events"] == [persisted["record"]]
    assert persisted["rules"]["rules"][0]["kind"] == "regression_test"


@pytest.mark.parametrize(
    ("body", "prior_events", "expected"),
    [
        ("This bug permits an invalid token.", [], "regression_test"),
        ("CI fails because lint is not run.", [], "pre_check"),
        ("Please use British spelling here.", [], "ledger_only"),
        ("Use the project event naming convention.", [], "project_convention"),
        ("Please simplify this helper.", [{"theme": "simplify helper"}], "repeated_theme"),
    ],
)
def test_learning_policy_promotes_only_objective_or_scoped_or_repeated_feedback(body, prior_events, expected):
    mod = load_module(MODULE_PATH, "moss_review_feedback_learning")

    rule = mod.learn_prevention_rule({"body": body, "repo": "owner/project"}, prior_events)

    assert rule["kind"] == expected
    if expected == "project_convention":
        assert rule["scope"] == "owner/project"


def test_guarded_executor_needs_explicit_runner_and_allow_mutation_and_never_runs_forbidden_work():
    mod = load_module(MODULE_PATH, "moss_review_feedback_executor")
    routine = {"action": "routine_patch", "key": "inline:r#1:1", "test_evidence": ["pytest tests/test_x.py -q"]}

    assert mod.execute_action(routine, allow_mutation=False)["status"] == "denied"
    assert mod.execute_action(routine, allow_mutation=True)["status"] == "denied"

    invoked = []
    def runner(command, **kwargs):
        invoked.append(command)
        return "ok"

    unsafe = {"action": "routine_patch", "key": "x", "risk": "security", "test_evidence": ["pytest x"]}
    assert mod.execute_action(unsafe, runner=runner, allow_mutation=True)["status"] == "rejected"
    assert invoked == []


def test_guarded_executor_runs_named_tests_before_safe_push_and_replies_to_inline_thread():
    mod = load_module(MODULE_PATH, "moss_review_feedback_executor_order")
    invoked = []
    def runner(command, **kwargs):
        invoked.append(command)
        return "ok"

    result = mod.execute_action({
        "action": "routine_patch", "repo": "owner/repo", "pr_number": 1,
        "source_item_id": "99", "surface": "inline", "author": "alice",
        "test_evidence": ["pytest tests/test_x.py -q"],
        "commands": [["git", "push", "origin", "HEAD"]],
        "reply": "Fixed with a regression test.",
    }, runner=runner, allow_mutation=True)

    assert result["status"] == "executed"
    assert invoked == [
        ["pytest", "tests/test_x.py", "-q"], ["git", "push", "origin", "HEAD"],
        ["gh", "api", "/repos/owner/repo/pulls/1/comments/99/replies", "-f", "body=Fixed with a regression test."],
    ]
    forbidden = {
        "action": "routine_patch", "repo": "owner/repo", "pr_number": 1,
        "source_item_id": "100", "surface": "inline", "author": "alice", "reply": "No force push.",
        "test_evidence": ["pytest tests/test_x.py -q"], "commands": [["git", "push", "--force"]],
    }
    assert mod.execute_action(forbidden, runner=runner, allow_mutation=True)["status"] == "rejected"
    assert len(invoked) == 3


@pytest.mark.parametrize(
    ("action_name", "content_field", "content"),
    [
        ("reply_only", "reply", "Thanks."),
        ("clarification", "question", "Which caller owns this invariant?"),
    ],
)
def test_reply_actions_require_an_explicit_runner_and_mutation_permission(action_name, content_field, content):
    mod = load_module(MODULE_PATH, f"moss_review_feedback_{action_name}_guard")
    action = {
        "action": action_name, "repo": "owner/repo", "pr_number": 1,
        "source_item_id": "5", "surface": "conversation", "author": "alice", content_field: content,
    }

    assert mod.execute_action(action)["status"] == "denied"
    assert mod.execute_action(action, runner=lambda command: None)["status"] == "denied"


def test_reply_only_and_clarification_reply_to_original_surface_without_git_actions():
    mod = load_module(MODULE_PATH, "moss_review_feedback_safe_replies")
    invoked = []
    runner = lambda command: invoked.append(command) or "ok"

    reply = mod.execute_action({
        "action": "reply_only", "repo": "owner/repo", "pr_number": 1,
        "source_item_id": "5", "surface": "conversation", "author": "alice", "reply": "Thanks for the review.",
    }, runner=runner, allow_mutation=True)
    question = mod.execute_action({
        "action": "clarification", "repo": "owner/repo", "pr_number": 1,
        "source_item_id": "6", "surface": "review", "author": "bob", "question": "Which caller owns this invariant?",
    }, runner=runner, allow_mutation=True)

    assert reply["status"] == question["status"] == "executed"
    assert invoked == [
        ["gh", "api", "/repos/owner/repo/issues/1/comments", "-f", "body=@alice Thanks for the review."],
        ["gh", "api", "/repos/owner/repo/issues/1/comments", "-f", "body=@bob Which caller owns this invariant?"],
    ]
    assert all(command[0] != "git" for command in invoked)


def test_clarification_uses_the_exact_supplied_question_for_inline_reply():
    mod = load_module(MODULE_PATH, "moss_review_feedback_inline_question")
    invoked = []
    question = "Which caller owns this invariant?"

    result = mod.execute_action({
        "action": "clarification", "repo": "owner/repo", "pr_number": 1,
        "source_item_id": "7", "surface": "inline", "author": "alice", "question": question,
    }, runner=lambda command: invoked.append(command) or "ok", allow_mutation=True)

    assert result["status"] == "executed"
    assert invoked == [["gh", "api", "/repos/owner/repo/pulls/1/comments/7/replies", "-f", f"body={question}"]]


def test_sahil_escalation_never_executes():
    mod = load_module(MODULE_PATH, "moss_review_feedback_escalation")
    invoked = []

    result = mod.execute_action({"action": "sahil_escalation"}, runner=lambda command: invoked.append(command), allow_mutation=True)

    assert result["status"] == "rejected"
    assert invoked == []
def test_teknium1_login_receives_maintainer_priority_and_safety_classification():
    """Regression: the exact login ``teknium1`` (added to MAINTAINER_AUTHORS)
    must receive ``maintainer`` priority through the real ``normalise_feedback``
    path while remaining subject to the same safety classification as any
    other reviewer.

    This guards two invariants of the maintainer-priority change:
      1. ``teknium1`` is recognised as a maintainer by ``normalise_feedback``
         (priority assignment lives in the real normalisation path, not in a
         separate lookup that could drift).
      2. Maintainer priority never bypasses ``classify_feedback``: a
         safety-laden body still escalates to ``sahil_escalation``.
    """
    mod = load_module(MODULE_PATH, "moss_review_feedback_teknium1")

    payload = {
        "repo": "NousResearch/hermes-agent",
        "pr": {"number": 7, "title": "Tighten boundary", "url": "https://example.test/pr/7"},
        "reviews": [
            # ``teknium1`` must NOT be excluded (not self/bot) and must get priority.
            {"id": "R-t1", "author": {"login": "teknium1"},
             "body": "This is a security issue in the credential path.", "state": "CHANGES_REQUESTED"},
        ],
        "review_comments": [],
        "comments": [],
    }

    records = mod.normalise_feedback(payload)
    assert len(records) == 1
    record = records[0]

    # Invariant 1: priority is maintainer via the real normalisation path.
    assert record["author"] == "teknium1"
    assert record["priority"] == "maintainer"
    assert record["surface"] == "review"
    assert record["pr_number"] == 7
    assert record["repo"] == "NousResearch/hermes-agent"

    # Invariant 2: safety classification is independent of maintainer priority.
    classification = mod.classify_feedback(record)
    assert classification == "sahil_escalation"

    # Belt-and-braces: a benign teknium1 comment stays a routine patch —
    # maintainer priority does not itself force escalation either.
    benign = {**record, "body": "nit: rename this helper for clarity"}
    assert mod.classify_feedback(benign) == "routine_patch"
