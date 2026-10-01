"""Required admission is a transaction boundary, not an optional observer or tool guard."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd
from hermes_cli import plugins


@pytest.fixture
def board(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    plugins._reset_plugin_managers_for_tests()
    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    yield home
    plugins._reset_plugin_managers_for_tests()


def _plugin(home: Path, body: str = "return {'allow': True}", *, drift: bool = False) -> None:
    directory = home / "plugins" / "gate"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "plugin.yaml").write_text("name: gate\nversion: '0.1.0'\n")
    registration = (
        "def register(context):\n    global ctx\n    ctx = context\n    ctx.register_kanban_transition_admission(admit)\n"
        if drift else "def register(ctx):\n    ctx.register_kanban_transition_admission(admit)\n"
    )
    (directory / "__init__.py").write_text(
        "CALLS = []\n"
        "def admit(**kwargs):\n    " + body + "\n" + registration
    )
    (home / "config.yaml").write_text(
        "plugins:\n  enabled: [gate]\nkanban:\n  required_transition_admission_plugin: gate\n"
    )


def _snapshot(conn, tid):
    task = kb.get_task(conn, tid)
    return (task.status, task.assignee, task.current_run_id, task.completion_contract,
            [(r.id, r.status, r.outcome) for r in kb.list_runs(conn, tid)],
            [(e.id, e.kind) for e in kb.list_events(conn, tid)])


@pytest.mark.parametrize("phase", ["review", "running", "blocked", "classified_blocked", "promoted"])
def test_required_admission_denies_review_and_reviewer_completion_without_mutation(board, phase):
    _plugin(board, "return {'allow': False}")
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="review", assignee="builder", completion_contract="acme/repo")
        run = kb.claim_task(conn, tid)
        before = _snapshot(conn, tid)
        assert not kb.request_review(conn, tid, summary="ready", expected_run_id=run.current_run_id)
        assert _snapshot(conn, tid) == before
        # A review card that predates the opt-in is also protected, including force.
        (board / "config.yaml").write_text("plugins:\n  enabled: [gate]\n")
        assert kb.request_review(conn, tid, summary="ready", expected_run_id=run.current_run_id)
        review = kb.claim_review_task(conn, tid) if phase != "review" else None
        if phase in ("blocked", "promoted"):
            assert kb.block_task(conn, tid, reason="review paused", kind="needs_input",
                                 expected_run_id=review.current_run_id)
            if phase == "promoted":
                # Operator promotion preserves the suspended review phase
                # (same landing rule as unblock_task / recompute_ready).
                assert kb.promote_task(conn, tid, actor="operator")[0]
                assert kb.get_task(conn, tid).status == "review"
        elif phase == "classified_blocked":
            assert kbd._record_task_failure(
                conn, tid, "review unavailable", outcome="spawn_failed", failure_limit=1,
                release_claim=True, end_run=True)
            assert kb.block_task(conn, tid, reason="operator needed", kind="needs_input")
        (board / "config.yaml").write_text(
            "plugins:\n  enabled: [gate]\nkanban:\n  required_transition_admission_plugin: gate\n"
        )
        before = _snapshot(conn, tid)
        assert not kb.complete_task(
            conn, tid, summary="approved", expected_run_id=review.current_run_id if phase == "running" else None,
            created_cards=["t_deadbeef"], metadata={"published_pr": "https://github.com/acme/repo/pull/1"},
            force=True,
        )
        assert _snapshot(conn, tid) == before


@pytest.mark.parametrize("body", ["raise RuntimeError('private data')", "return None", "return {'allow': 1}", "import time; time.sleep(5); return {'allow': True}"])
def test_required_admission_failure_modes_do_not_mutate(board, body):
    _plugin(board, body)
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="failure", assignee="builder")
        run = kb.claim_task(conn, tid)
        before = _snapshot(conn, tid)
        assert not kb.request_review(conn, tid, expected_run_id=run.current_run_id, force=True)
        assert _snapshot(conn, tid) == before


@pytest.mark.parametrize("phase", ["running", "blocked", "classified_blocked"])
def test_required_admission_allows_both_transitions_once(board, phase):
    _plugin(board, "CALLS.append(kwargs['action']); return {'allow': True}")
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="approved", assignee="builder")
        run = kb.claim_task(conn, tid)
        assert kb.request_review(conn, tid, summary="ready", expected_run_id=run.current_run_id)
        review = kb.claim_review_task(conn, tid)
        if phase == "blocked":
            assert kb.block_task(conn, tid, reason="review paused", kind="needs_input",
                                 expected_run_id=review.current_run_id)
        elif phase == "classified_blocked":
            assert kbd._record_task_failure(
                conn, tid, "review unavailable", outcome="spawn_failed", failure_limit=1,
                release_claim=True, end_run=True)
            assert kb.block_task(conn, tid, reason="operator needed", kind="needs_input")
        assert kb.complete_task(
            conn, tid, summary="approved",
            expected_run_id=review.current_run_id if phase == "running" else None)
        assert kb.get_task(conn, tid).status == "done"
    assert plugins.get_plugin_manager()._plugins["gate"].module.CALLS == [
        "request_review", "complete_review",
    ]


def test_required_admission_rejects_registration_drift(board):
    _plugin(board, "ctx.register_kanban_transition_admission(admit); return {'allow': True}", drift=True)
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="drift", assignee="builder")
        run = kb.claim_task(conn, tid)
        before = _snapshot(conn, tid)
        assert not kb.request_review(conn, tid, expected_run_id=run.current_run_id)
        assert _snapshot(conn, tid) == before


@pytest.mark.parametrize("bad_config", ["[]", "null", "''", "{wrong: gate}"])
def test_malformed_required_provider_configuration_denies(board, bad_config):
    (board / "config.yaml").write_text(
        f"kanban:\n  required_transition_admission_plugin: {bad_config}\n"
    )
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="config", assignee="builder")
        run = kb.claim_task(conn, tid)
        before = _snapshot(conn, tid)
        assert not kb.request_review(conn, tid, expected_run_id=run.current_run_id)
        assert _snapshot(conn, tid) == before


def test_disabling_loaded_required_provider_denies_without_reload(board):
    _plugin(board)
    with kbc.connect() as conn:
        first = kb.create_task(conn, title="initial", assignee="builder")
        first_run = kb.claim_task(conn, first)
        assert kb.request_review(conn, first, expected_run_id=first_run.current_run_id)
        tid = kb.create_task(conn, title="disabled", assignee="builder")
        run = kb.claim_task(conn, tid)
        before = _snapshot(conn, tid)
        (board / "config.yaml").write_text(
            "plugins:\n  enabled: []\n  disabled: [gate]\n"
            "kanban:\n  required_transition_admission_plugin: gate\n"
        )
        assert not kb.request_review(conn, tid, expected_run_id=run.current_run_id)
        assert _snapshot(conn, tid) == before


def test_required_provider_load_failure_denies(board):
    _plugin(board)
    (board / "plugins" / "gate" / "__init__.py").write_text("def register(ctx):\n  this is invalid syntax\n")
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="load", assignee="builder")
        run = kb.claim_task(conn, tid)
        before = _snapshot(conn, tid)
        assert not kb.request_review(conn, tid, expected_run_id=run.current_run_id)
        assert _snapshot(conn, tid) == before


def test_required_policy_is_scoped_to_active_home(board, tmp_path, monkeypatch):
    _plugin(board, "return {'allow': False}")
    other = tmp_path / "other-profile"
    other.mkdir()
    (other / "config.yaml").write_text("{}\n")
    with kbc.connect() as conn:
        denied = kb.create_task(conn, title="home A", assignee="builder")
        allowed = kb.create_task(conn, title="home B", assignee="builder")
        denied_again = kb.create_task(conn, title="home A again", assignee="builder")
        first_run = kb.claim_task(conn, denied)
        second_run = kb.claim_task(conn, allowed)
        third_run = kb.claim_task(conn, denied_again)
        assert not kb.request_review(conn, denied, expected_run_id=first_run.current_run_id)
        monkeypatch.setenv("HERMES_HOME", str(other))
        assert kb.request_review(conn, allowed, expected_run_id=second_run.current_run_id)
        monkeypatch.setenv("HERMES_HOME", str(board))
        assert not kb.request_review(conn, denied_again, expected_run_id=third_run.current_run_id)


def test_required_admission_missing_provider_denies_tool_and_cli(board, monkeypatch):
    (board / "config.yaml").write_text("kanban:\n  required_transition_admission_plugin: absent\n")
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="tool", assignee="builder")
        run = kb.claim_task(conn, tid)
        before = _snapshot(conn, tid)
    monkeypatch.setenv("HERMES_KANBAN_TASK", tid)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(run.current_run_id))
    from tools import kanban_tools
    from hermes_cli import kanban
    assert "error" in json.loads(kanban_tools._handle_request_review({"summary": "ready"}))
    assert "cannot request review" in kanban.run_slash(f"request-review {tid} --summary ready").lower()
    with kbc.connect() as conn:
        assert _snapshot(conn, tid) == before

    (board / "config.yaml").write_text("{}\n")
    with kbc.connect() as conn:
        assert kb.request_review(conn, tid, expected_run_id=run.current_run_id)
        reviewer = kb.claim_review_task(conn, tid)
        assert reviewer is not None
        prior = _snapshot(conn, tid)
    (board / "config.yaml").write_text("kanban:\n  required_transition_admission_plugin: absent\n")
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(reviewer.current_run_id))
    assert "error" in json.loads(kanban_tools._handle_complete({"summary": "approved"}))
    assert "cannot complete" in kanban.run_slash(f"complete {tid} --summary approved --force").lower()
    with kbc.connect() as conn:
        assert _snapshot(conn, tid) == prior


@pytest.mark.parametrize("phase", ["blocked", "classified_blocked", "changes_requested", "review_reopened"])
def test_maker_completion_does_not_inherit_an_old_review_phase(board, phase):
    _plugin(board, "CALLS.append(kwargs['action']); return {'allow': False}")
    (board / "config.yaml").write_text("plugins:\n  enabled: [gate]\n")
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="maker phase", assignee="builder")
        run = kb.claim_task(conn, tid)
        if phase == "blocked":
            assert kb.block_task(conn, tid, reason="maker paused", kind="needs_input",
                                 expected_run_id=run.current_run_id)
        elif phase == "classified_blocked":
            assert kbd._record_task_failure(
                conn, tid, "maker unavailable", outcome="spawn_failed", failure_limit=1,
                release_claim=True, end_run=True)
            assert kb.block_task(conn, tid, reason="operator needed", kind="needs_input")
        else:
            assert kb.request_review(conn, tid, summary="ready", reviewer="checker",
                                     expected_run_id=run.current_run_id)
            if phase == "changes_requested":
                review = kb.claim_review_task(conn, tid)
                assert kb.request_changes(conn, tid, reason="rework",
                                          expected_run_id=review.current_run_id)[0]
            else:
                assert kb.reopen_review_task(conn, tid)
        (board / "config.yaml").write_text(
            "plugins:\n  enabled: [gate]\nkanban:\n  required_transition_admission_plugin: gate\n"
        )
        assert kb.complete_task(conn, tid, summary="maker finished")
        assert kb.get_task(conn, tid).status == "done"


def test_completion_rechecks_suspended_review_phase_inside_transaction(board, monkeypatch):
    from hermes_cli import kanban_pr_acceptance_store
    _plugin(board, "CALLS.append(kwargs['action']); return {'allow': kwargs['action'] == 'request_review'}")
    prepare = kanban_pr_acceptance_store.prepare_acceptance
    after_transition = []
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="phase changes before commit", assignee="builder")

        def suspend_review(*args):
            result = prepare(*args)
            assert kb.request_review(conn, tid, summary="ready", reviewer="checker")
            review = kb.claim_review_task(conn, tid)
            assert kb.block_task(conn, tid, reason="review paused", kind="needs_input",
                                 expected_run_id=review.current_run_id)
            after_transition.append(_snapshot(conn, tid))
            return result

        monkeypatch.setattr(kanban_pr_acceptance_store, "prepare_acceptance", suspend_review)
        assert not kb.complete_task(conn, tid, summary="must not approve", force=True)
        assert _snapshot(conn, tid) == after_transition[0]
    assert plugins.get_plugin_manager()._plugins["gate"].module.CALLS == [
        "request_review", "complete_review",
    ]


def test_review_phase_survives_promote_claim_and_reblock(board):
    """promote_task must not strand a suspended review phase on the maker lane:
    block(review) -> promote -> review (not ready), so the reviewer-lane claim
    keeps source_status=review and a re-block stamps review provenance again."""
    _plugin(board, "CALLS.append(kwargs['action']); return {'allow': False}")
    (board / "config.yaml").write_text("plugins:\n  enabled: [gate]\n")
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="promoted review", assignee="builder")
        run = kb.claim_task(conn, tid)
        assert kb.request_review(conn, tid, summary="ready", expected_run_id=run.current_run_id)
        review = kb.claim_review_task(conn, tid)
        assert kb.block_task(conn, tid, reason="review paused", kind="needs_input",
                             expected_run_id=review.current_run_id)
        assert kb.promote_task(conn, tid, actor="operator")[0]
        assert kb.get_task(conn, tid).status == "review"
        # The maker lane cannot claim a review-suspended card.
        assert kb.claim_task(conn, tid) is None
        resumed = kb.claim_review_task(conn, tid)
        assert resumed is not None
        assert kb._retry_status_for_run(conn, tid, resumed.current_run_id) == "review"
        # A re-block after the promoted claim still stamps review provenance.
        assert kb.block_task(conn, tid, reason="paused again", kind="needs_input",
                             expected_run_id=resumed.current_run_id)
        (board / "config.yaml").write_text(
            "plugins:\n  enabled: [gate]\nkanban:\n  required_transition_admission_plugin: gate\n"
        )
        before = _snapshot(conn, tid)
        assert not kb.complete_task(conn, tid, summary="approved", force=True)
        assert _snapshot(conn, tid) == before


def test_maker_promotion_stays_on_the_maker_lane(board):
    """Ordinary maker promotion lands ``ready`` and a maker completion is not
    a reviewer-lane admission, even with a denying provider configured."""
    _plugin(board, "return {'allow': False}")
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="maker promote", assignee="builder")
        run = kb.claim_task(conn, tid)
        assert kb.block_task(conn, tid, reason="maker paused", kind="needs_input",
                             expected_run_id=run.current_run_id)
        assert kb.promote_task(conn, tid, actor="operator")[0]
        assert kb.get_task(conn, tid).status == "ready"
        rerun = kb.claim_task(conn, tid)
        assert rerun is not None
        assert kb.complete_task(conn, tid, summary="maker finished",
                                expected_run_id=rerun.current_run_id)
        assert kb.get_task(conn, tid).status == "done"
