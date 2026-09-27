"""Cross-surface regressions for the complete Kanban review lifecycle."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hermes_cli import kanban as kc
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc


@pytest.fixture
def review_worker(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> str:
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_PROFILE", "builder")
    monkeypatch.delenv("HERMES_DELEGATED_CHILD_CONTEXT", raising=False)
    # kanban_request_review now rejects reviewers that are not installed profiles (#106163).
    (home / "profiles" / "reviewer").mkdir(parents=True)
    (home / "profiles" / "reviewer" / "config.yaml").write_text("{}\n")  # identity marker
    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    with kbc.connect() as conn:
        task_id = kb.create_task(conn, title="Review tool contract", assignee="builder")
        task = kb.claim_task(conn, task_id, claimer="builder:1")
        assert task is not None
    monkeypatch.setenv("HERMES_KANBAN_TASK", task_id)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(task.current_run_id))
    return task_id


def test_review_tools_redact_handoff_and_route_changes(
    review_worker: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tools import kanban_tools as tools

    secret = "ghp_" + "A" * 40
    requested = json.loads(
        tools._handle_request_review({
            "summary": f"Ready; temporary token was {secret}",
            "metadata": {"token": secret, "tests_run": 7},
            "reviewer": "reviewer",
        })
    )
    assert requested["ok"] is True

    with kbc.connect() as conn:
        task = kb.get_task(conn, review_worker)
        assert task is not None
        assert task.status == "review"
        assert task.assignee == "reviewer"
        handoff = kb.latest_run(conn, review_worker)
        assert handoff is not None
        assert secret not in (handoff.summary or "")
        assert secret not in json.dumps(handoff.metadata)
        review = kb.claim_review_task(conn, review_worker, claimer="reviewer:1")
        assert review is not None

    monkeypatch.setenv("HERMES_PROFILE", "reviewer")
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(review.current_run_id))
    change_secret = "sk-" + "B" * 32
    changed = json.loads(
        tools._handle_request_changes({
            "reason": f"Add a boundary assertion; leaked={change_secret}",
        })
    )
    assert changed["ok"] is True
    assert changed["implementer"] == "builder"

    with kbc.connect() as conn:
        task = kb.get_task(conn, review_worker)
        assert task is not None
        assert task.status == "ready"
        assert task.assignee == "builder"
        event = [
            item
            for item in kb.list_events(conn, review_worker)
            if item.kind == "changes_requested"
        ][-1]
        assert event.payload is not None
        assert change_secret not in event.payload["reason"]
        assert event.payload["reason"] != (
            "Add a boundary assertion; leaked=" + change_secret
        )


def test_review_tools_are_gated_and_visible_to_kanban_workers(
    review_worker: str,
) -> None:
    import tools.kanban_tools  # noqa: F401 - registers the tools
    from tools.registry import invalidate_check_fn_cache, registry
    from toolsets import resolve_toolset

    invalidate_check_fn_cache()
    definitions = registry.get_definitions(
        set(resolve_toolset("hermes-cli")), quiet=True
    )
    names = {
        definition["function"]["name"]
        for definition in definitions
        if "function" in definition
    }
    assert "kanban_request_review" in names
    assert "kanban_request_changes" in names

    from agent.transports.hermes_tools_mcp_server import EXPOSED_TOOLS

    assert "kanban_request_changes" in EXPOSED_TOOLS
    assert "kanban_request_changes" in resolve_toolset("kanban")




def test_review_cli_round_trip_preserves_handoff(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb._INITIALIZED_PATHS.clear()
    kb.init_db()

    with kbc.connect() as conn:
        task_id = kb.create_task(conn, title="CLI review", assignee="builder")
        implementation = kb.claim_task(conn, task_id, claimer="builder:1")
        assert implementation is not None
    monkeypatch.setenv("HERMES_KANBAN_TASK", task_id)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(implementation.current_run_id))

    output = kc.run_slash(
        f"request-review {task_id} --summary 'ready for review' "
        "--reviewer reviewer --metadata '{\"tests_run\": 3}'"
    )
    assert "Requested review" in output

    with kbc.connect() as conn:
        task = kb.get_task(conn, task_id)
        assert task is not None
        assert task.assignee == "reviewer"
        handoff = kb.latest_run(conn, task_id)
        assert handoff is not None
        assert handoff.metadata == {"tests_run": 3}
        review = kb.claim_review_task(conn, task_id, claimer="reviewer:1")
        assert review is not None
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(review.current_run_id))

    output = kc.run_slash(
        f"request-changes {task_id} 'cover the malformed payload case'"
    )
    assert "Requested changes" in output
    with kbc.connect() as conn:
        task = kb.get_task(conn, task_id)
        assert task is not None
        assert task.status == "ready"
        assert task.assignee == "builder"


def test_domain_and_cli_review_handoffs_redact_before_persistence(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    secret = "ghp_" + "R" * 40

    with kbc.connect() as conn:
        direct_id = kb.create_task(conn, title="direct redaction", assignee="builder")
        direct_run = kb.claim_task(conn, direct_id)
        assert direct_run is not None
        assert kb.request_review(
            conn,
            direct_id,
            summary=f"direct {secret}",
            metadata={"nested": [secret]},
            expected_run_id=direct_run.current_run_id,
        )
        run = kb.latest_run(conn, direct_id)
        event = [
            item for item in kb.list_events(conn, direct_id)
            if item.kind == "review_requested"
        ][-1]
        assert run is not None
        assert secret not in str(run.summary)
        assert secret not in json.dumps(run.metadata)
        assert secret not in json.dumps(event.payload)

        review = kb.claim_review_task(conn, direct_id)
        assert review is not None
        assert kb.request_changes(
            conn,
            direct_id,
            reason=f"change {secret}",
            expected_run_id=review.current_run_id,
        ) == (True, "builder")
        run = kb.latest_run(conn, direct_id)
        event = [
            item for item in kb.list_events(conn, direct_id)
            if item.kind == "changes_requested"
        ][-1]
        assert run is not None
        assert secret not in str(run.summary)
        assert secret not in json.dumps(event.payload)

        cli_id = kb.create_task(conn, title="CLI redaction", assignee="builder")
    cli_output = kc.run_slash(
        f'request-review {cli_id} --summary "cli {secret}" '
        f"--metadata '{{\"token\":\"{secret}\"}}'"
    )
    assert "Requested review" in cli_output
    assert secret not in cli_output
    with kbc.connect() as conn:
        run = kb.latest_run(conn, cli_id)
        event = [
            item for item in kb.list_events(conn, cli_id)
            if item.kind == "review_requested"
        ][-1]
        assert run is not None
        assert secret not in str(run.summary)
        assert secret not in json.dumps(run.metadata)
        assert secret not in json.dumps(event.payload)




def test_cli_reopen_review_is_transition_first_and_redacts_reason(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    secret = "ghp_" + "Q" * 40
    with kbc.connect() as conn:
        invalid_id = kb.create_task(conn, title="not review", assignee="builder")
        review_id = kb.create_task(conn, title="review", assignee="builder")
        assert kb.request_review(conn, review_id, summary="ready")

    invalid_output = kc.run_slash(
        f'reopen-review {invalid_id} --reason "invalid {secret}"'
    )
    assert "cannot reopen" in invalid_output
    with kbc.connect() as conn:
        assert kb.list_comments(conn, invalid_id) == []

    success_output = kc.run_slash(
        f'reopen-review {review_id} --reason "revise {secret}"'
    )
    assert "Reopened" in success_output
    assert secret not in success_output
    with kbc.connect() as conn:
        task = kb.get_task(conn, review_id)
        assert task is not None
        assert task.status == "ready"
        comments = kb.list_comments(conn, review_id)
        assert len(comments) == 1
        assert secret not in comments[0].body


def test_goal_judge_preserves_card_budget_and_surface_parity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import SimpleNamespace
    from typing import cast

    from tools import kanban_tools as tools

    title = "Boundary card"
    marker = "IMPLEMENTATION CRITERION: preserve this late requirement"
    card_prefix = f"{title}\n\n"
    body = "x" * (1734 - len(card_prefix)) + marker
    body += "y" * (1831 - len(card_prefix) - len(body))
    card = f"{card_prefix}{body}"
    assert len(card) == 1831
    assert card.index(marker) == 1734

    task = cast(kb.Task, SimpleNamespace(
        id="t_boundary",
        title=title,
        body=body,
        goal_mode=True,
    ))
    prompts: list[str] = []

    def capture_call_llm(**kwargs):
        prompts.append(kwargs["messages"][1]["content"])
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(
                content='{"done": true, "reason": "ready"}'
            ))]
        )

    monkeypatch.setattr(tools, "_goal_judge_available", lambda: True)
    monkeypatch.setattr("agent.auxiliary_client.call_llm", capture_call_llm)
    monkeypatch.setattr(
        "agent.auxiliary_client.get_text_auxiliary_client",
        lambda purpose: (object(), "judge-model"),
    )

    tools._goal_gate("kanban_complete", task, task.id, "verified")
    assert kc._goal_mode_handoff_rejection(task, "verified") == ("done", None)
    tools._goal_gate("kanban_request_review", task, task.id, "verified")
    assert kc._goal_mode_handoff_rejection(
        task, "verified", review_handoff=True
    ) == ("done", None)

    tool_completion, cli_completion, tool_review, cli_review = prompts
    assert tool_completion == cli_completion
    assert tool_review == cli_review
    assert all(marker in prompt for prompt in prompts)
    assert "Review-handoff decision" not in tool_completion
    assert "Review-handoff decision" in tool_review


def test_goal_mode_review_handoff_judges_implementation_readiness_on_both_surfaces(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_PROFILE", "builder")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb._INITIALIZED_PATHS.clear()
    kb.init_db()

    body = (
        "Acceptance: implementation and regression tests are complete. "
        "After handoff, Gurney must independently review the candidate and record a verdict."
    )

    def create_claimed(title: str) -> tuple[str, int]:
        with kbc.connect() as conn:
            task_id = kb.create_task(
                conn, title=title, body=body, assignee="builder", goal_mode=True
            )
            claimed = kb.claim_task(conn, task_id, claimer=f"builder:{title}")
            assert claimed is not None and claimed.current_run_id is not None
            return task_id, claimed.current_run_id

    def readiness_judge(*args, **kwargs):
        evidence = kwargs.get("last_response", args[1] if len(args) > 1 else "")
        evidence_lower = evidence.lower()
        review_scope = kwargs.get("review_handoff", False)
        if "external repository is unavailable" in evidence_lower:
            return "blocked", "external repository is unavailable", False, None, False
        if review_scope and "implementation and regression tests pass" in evidence_lower:
            return "done", "implementation is ready for independent review", False, None, False
        return "continue", "independent review verdict is not recorded", False, None, False

    from tools import kanban_tools as tools
    from hermes_cli import goals

    monkeypatch.setattr(tools, "_goal_judge_available", lambda: True)
    monkeypatch.setattr(tools, "judge_goal", readiness_judge)
    monkeypatch.setattr(
        "agent.auxiliary_client.get_text_auxiliary_client",
        lambda purpose: (object(), "judge-model"),
    )
    monkeypatch.setattr(goals, "judge_goal", readiness_judge)

    tool_task, tool_run = create_claimed("Tool review handoff")
    monkeypatch.setenv("HERMES_KANBAN_TASK", tool_task)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(tool_run))
    admitted = json.loads(tools._handle_request_review({
        "summary": "Implementation and regression tests pass; candidate is ready.",
    }))
    assert admitted["ok"] is True

    cli_task, cli_run = create_claimed("CLI review handoff")
    monkeypatch.setenv("HERMES_KANBAN_TASK", cli_task)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(cli_run))
    output = kc.run_slash(
        f"request-review {cli_task} --summary "
        "'Implementation and regression tests pass; candidate is ready.'"
    )
    assert "Requested review" in output

    completion_task, completion_run = create_claimed("Completion still needs review")
    monkeypatch.setenv("HERMES_KANBAN_TASK", completion_task)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(completion_run))
    rejected = json.loads(tools._handle_complete({
        "summary": "Implementation and regression tests pass; candidate is ready.",
    }))
    assert "rejected by judge" in rejected["error"]
    with kbc.connect() as conn:
        completion_after = kb.get_task(conn, completion_task)
        assert completion_after is not None and completion_after.status == "running"

    cli_completion_task, cli_completion_run = create_claimed("CLI completion still needs review")
    monkeypatch.setenv("HERMES_KANBAN_TASK", cli_completion_task)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(cli_completion_run))
    completion_output = kc.run_slash(
        f"complete {cli_completion_task} --summary "
        "'Implementation and regression tests pass; candidate is ready.'"
    )
    assert "rejected by judge" in completion_output
    with kbc.connect() as conn:
        cli_completion_after = kb.get_task(conn, cli_completion_task)
        assert cli_completion_after is not None
        assert cli_completion_after.status == "running"

    blocked_task, blocked_run = create_claimed("Blocked review handoff")
    monkeypatch.setenv("HERMES_KANBAN_TASK", blocked_task)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(blocked_run))
    blocked = json.loads(tools._handle_request_review({
        "summary": "The external repository is unavailable.",
    }))
    assert "kanban_block" in blocked["error"]
    with kbc.connect() as conn:
        blocked_after = kb.get_task(conn, blocked_task)
        assert blocked_after is not None and blocked_after.status == "running"


def test_multi_stage_card_admits_review_on_stage_one_but_complete_needs_live_evidence(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Regression (t_1e330720): a card whose acceptance continues past review
    (independent review, release, live acceptance) must admit the review handoff
    on stage-1 evidence, while completion keeps full terminal-goal judging.

    The fake model sits below ``judge_goal`` so the real prompt construction and
    both real gate surfaces (tool + CLI) are exercised without provider calls.
    """
    from types import SimpleNamespace

    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_PROFILE", "builder")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb._INITIALIZED_PATHS.clear()
    kb.init_db()

    body = (
        "Stage 1: add the Blog link to the site header and verify it on a preview build.\n"
        "Stage 2: Gurney independently reviews the candidate.\n"
        "Stage 3: the release owner deploys to production.\n"
        "Stage 4: the domain owner confirms the link live on the production site.\n"
        "Done when the link is live and accepted."
    )
    stage_one = "Candidate 947f9cf: header Blog link implemented; preview build verified, tests pass."
    prompts: list[str] = []

    def model(**kwargs):
        prompt = kwargs["messages"][1]["content"]
        prompts.append(prompt)
        if "Review-handoff decision" in prompt and "947f9cf" in prompt:
            reply = {"verdict": "done", "reason": "candidate is verified and ready for review"}
        elif "Review-handoff decision" in prompt:
            reply = {"verdict": "continue", "reason": "no candidate identified"}
        else:
            reply = {"verdict": "continue", "reason": "no live production acceptance evidence"}
        return SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(content=json.dumps(reply)))])

    from tools import kanban_tools as tools

    monkeypatch.setattr(tools, "_goal_judge_available", lambda: True)
    monkeypatch.setattr("agent.auxiliary_client.call_llm", model)
    monkeypatch.setattr(
        "agent.auxiliary_client.get_text_auxiliary_client",
        lambda purpose: (object(), "judge-model"),
    )

    def claimed(title: str) -> str:
        with kbc.connect() as conn:
            task_id = kb.create_task(
                conn, title=title, body=body, assignee="builder", goal_mode=True)
            run = kb.claim_task(conn, task_id, claimer=f"builder:{title}")
            assert run is not None and run.current_run_id is not None
        monkeypatch.setenv("HERMES_KANBAN_TASK", task_id)
        monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(run.current_run_id))
        return task_id

    def status(task_id: str) -> str:
        with kbc.connect() as conn:
            return kb.get_task(conn, task_id).status

    # Review is admitted on stage-1 evidence (tool + CLI).
    t = claimed("Tool multi-stage review")
    assert json.loads(tools._handle_request_review({"summary": stage_one}))["ok"] is True
    assert status(t) == "review"
    t = claimed("CLI multi-stage review")
    assert "Requested review" in kc.run_slash(f"request-review {t} --summary '{stage_one}'")
    assert status(t) == "review"

    # Review still needs an identified, verified candidate.
    t = claimed("Tool multi-stage review without candidate")
    refused = json.loads(tools._handle_request_review({"summary": "Working on it."}))
    assert "rejected by judge" in refused["error"] and status(t) == "running"

    # Completion keeps terminal-goal judging: stage-1 evidence is not enough (tool + CLI).
    t = claimed("Tool multi-stage complete")
    refused = json.loads(tools._handle_complete({"summary": stage_one}))
    assert "live production acceptance" in refused["error"] and status(t) == "running"
    t = claimed("CLI multi-stage complete")
    assert "rejected by judge" in kc.run_slash(f"complete {t} --summary '{stage_one}'")
    assert status(t) == "running"

    review_prompts = [p for p in prompts if "Review-handoff decision" in p]
    assert len(review_prompts) == 3 and len(prompts) == 5
    assert all("Stage 4" in p for p in prompts)


def test_goal_mode_review_handoff_cannot_bypass_judge(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb._INITIALIZED_PATHS.clear()
    kb.init_db()

    with kbc.connect() as conn:
        tool_task = kb.create_task(
            conn,
            title="Goal-mode tool task",
            assignee="builder",
            goal_mode=True,
        )
        claimed = kb.claim_task(conn, tool_task, claimer="builder:1")
        assert claimed is not None
    monkeypatch.setenv("HERMES_KANBAN_TASK", tool_task)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(claimed.current_run_id))

    from tools import kanban_tools as tools

    monkeypatch.setattr(tools, "_goal_judge_available", lambda: True)
    monkeypatch.setattr(
        tools,
        "judge_goal",
        lambda *args, **kwargs: (
            "continue",
            "acceptance evidence is missing",
            False,
            None,
            False,
        ),
    )
    rejected = json.loads(tools._handle_request_review({"summary": "Looks ready."}))
    assert "error" in rejected
    assert "rejected by judge" in rejected["error"]
    with kbc.connect() as conn:
        tool_after = kb.get_task(conn, tool_task)
        assert tool_after is not None
        assert tool_after.status == "running"

    # The shell/CLI path applies the same gate and must not bypass the tool.
    with kbc.connect() as conn:
        cli_task = kb.create_task(
            conn,
            title="Goal-mode CLI task",
            assignee="builder",
            goal_mode=True,
        )
        cli_claimed = kb.claim_task(conn, cli_task, claimer="builder:2")
        assert cli_claimed is not None
    monkeypatch.setenv("HERMES_KANBAN_TASK", cli_task)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(cli_claimed.current_run_id))

    import agent.auxiliary_client as auxiliary_client
    from hermes_cli import goals

    monkeypatch.setattr(
        auxiliary_client,
        "get_text_auxiliary_client",
        lambda purpose: (object(), "judge-model"),
    )
    monkeypatch.setattr(
        goals,
        "judge_goal",
        lambda *args, **kwargs: ("continue", "tests are missing", False, None, False),
    )
    output = kc.run_slash(f"request-review {cli_task} --summary 'Looks ready.'")
    assert "rejected by judge" in output
    with kbc.connect() as conn:
        cli_after = kb.get_task(conn, cli_task)
        assert cli_after is not None
        assert cli_after.status == "running"


def test_goal_loop_stops_after_reviewer_requests_changes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from hermes_cli import goals

    monkeypatch.setattr(
        goals,
        "judge_goal",
        lambda *args, **kwargs: pytest.fail(
            "a terminal review verdict must not be judged"
        ),
    )
    result = goals.run_kanban_goal_loop(
        task_id="t_review",
        goal_text="review the change",
        run_turn=lambda prompt: pytest.fail("must not run another reviewer turn"),
        task_status_fn=lambda: "changes_requested",
        block_fn=lambda reason: pytest.fail("must not block"),
        first_response="Changes requested.",
    )
    assert result["outcome"] == "changes_requested_by_reviewer"
    assert result["turns_used"] == 1


def test_cli_and_dashboard_receive_graph_aware_deadlock_diagnostic(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb._INITIALIZED_PATHS.clear()
    kb.init_db()

    with kbc.connect() as conn:
        parent_id = kb.create_task(conn, title="Implementation", assignee="builder")
        child_id = kb.create_task(
            conn,
            title="Review",
            assignee="reviewer",
            parents=[parent_id],
        )
        parent = kb.claim_task(conn, parent_id, claimer="builder:1")
        assert parent is not None
        assert kb.block_task(
            conn,
            parent_id,
            reason="review-required: ready",
            expected_run_id=parent.current_run_id,
        )

    payload = json.loads(kc.run_slash(f"diagnostics --task {parent_id} --json"))
    assert any(
        item["kind"] == "review_dependency_deadlock"
        for item in payload[0]["diagnostics"]
    )

    from plugins.kanban.dashboard.plugin_api import _compute_task_diagnostics

    with kbc.connect() as conn:
        dashboard = _compute_task_diagnostics(conn, task_ids=[parent_id])
    assert dashboard[parent_id][0]["kind"] == "review_dependency_deadlock"
    assert dashboard[parent_id][0]["data"]["waiting_child_ids"] == [child_id]
