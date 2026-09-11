from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

from hermes_cli import kanban_db as kb

SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "github_ci_kanban_digest.py"
spec = importlib.util.spec_from_file_location("github_ci_kanban_digest", SCRIPT_PATH)
assert spec and spec.loader
digest = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = digest
spec.loader.exec_module(digest)


def fake_runner(args: list[str]) -> str:
    if args[:2] == ["pr", "view"]:
        return """
        {
          "number": 167,
          "url": "https://github.com/NousResearch/hermes-agent/pull/167",
          "headRefName": "fix/pr-167",
          "headRefOid": "abcdef1234567890",
          "baseRefName": "main",
          "statusCheckRollup": [
            {"name":"web unit / vitest", "conclusion":"FAILURE", "detailsUrl":"https://github.com/NousResearch/hermes-agent/actions/runs/111/jobs/222"},
            {"name":"biome lint", "conclusion":"SUCCESS", "detailsUrl":"https://github.com/NousResearch/hermes-agent/actions/runs/112/jobs/223"}
          ]
        }
        """
    if args[:2] == ["run", "view"]:
        assert "--log-failed" in args
        return """
        web unit / vitest\tRun tests\tapps/web/src/chat/session.test.tsx:42:13 AssertionError: expected duplicate listener count to equal 1
        web unit / vitest\tRun tests\tpackages/river-client/src/stream.ts:88 Error: listener called twice
        """
    if args[:2] == ["run", "list"]:
        return '[{"name":"unrelated nightly","conclusion":"success","status":"completed"}]'
    raise AssertionError(args)


def baseline_red_runner(args: list[str]) -> str:
    if args[:2] == ["run", "list"]:
        return '[{"name":"web unit / vitest","conclusion":"failure","status":"completed"}]'
    return fake_runner(args)


def baseline_red_workflow_name_runner(args: list[str]) -> str:
    if args[:2] == ["run", "list"]:
        return '[{"name":"CI","conclusion":"failure","status":"completed"}]'
    return fake_runner(args)


def pr_169_style_runner(args: list[str]) -> str:
    if args[:2] == ["pr", "view"]:
        return """
        {
          "number": 169,
          "url": "https://github.com/NousResearch/hermes-agent/pull/169",
          "headRefName": "fix/pr-169",
          "headRefOid": "123456abcdef7890",
          "baseRefName": "main",
          "statusCheckRollup": [
            {"name":"python tests", "conclusion":"FAILURE", "detailsUrl":"https://github.com/NousResearch/hermes-agent/actions/runs/333/jobs/444"}
          ]
        }
        """
    if args[:2] == ["run", "view"]:
        assert "--log-failed" in args
        return """
        python tests\tRun pytest\ttests/tools/test_terminal_tool.py:117 AssertionError: expected command output to include cwd
        """
    if args[:2] == ["run", "list"]:
        return "[]"
    raise AssertionError(args)


def test_extracts_concise_failures_from_pr_167_style_log():
    log = fake_runner(["run", "view", "111", "--repo", "NousResearch/hermes-agent", "--log-failed"])
    failures = digest.concise_failures(log)
    assert failures == [
        "apps/web/src/chat/session.test.tsx:42 — apps/web/src/chat/session.test.tsx:42:13 AssertionError: expected duplicate listener count to equal 1",
        "packages/river-client/src/stream.ts:88 — packages/river-client/src/stream.ts:88 Error: listener called twice",
    ]


def test_formats_pr_169_style_digest_without_raw_log_dump():
    info = digest.pr_status("NousResearch/hermes-agent", 167, fake_runner)
    red = [c for c in info["checks"] if c.is_red]
    green = [c for c in info["checks"] if c.is_green]
    failures = digest.concise_failures(fake_runner(["run", "view", "111", "--repo", "NousResearch/hermes-agent", "--log-failed"]))
    body = digest.format_digest(
        info,
        red,
        green,
        failures,
        digest.infer_repro_commands(red, failures),
        "branch-specific likely: red checks are not currently red on the base branch",
    )
    assert "CI digest for PR #167" in body
    assert "fix/pr-167 @ abcdef123456" in body
    assert "Run id(s): 111" in body
    assert "Red checks: web unit / vitest" in body
    assert "Green checks: biome lint" in body
    assert "apps/web/src/chat/session.test.tsx:42" in body
    assert "pnpm --filter ./apps/web test src/chat/session.test.tsx" in body
    assert "duplicate listener count" in body
    # It is a digest, not the raw GitHub log table.
    assert "web unit / vitest\tRun tests" not in body


def test_infers_frontend_repro_command_scoped_to_extracted_file_and_package():
    info = digest.pr_status("NousResearch/hermes-agent", 167, fake_runner)
    red = [c for c in info["checks"] if c.is_red]
    failures = digest.concise_failures(fake_runner(["run", "view", "111", "--repo", "NousResearch/hermes-agent", "--log-failed"]))

    commands = digest.infer_repro_commands(red, failures)

    assert "pnpm --filter ./apps/web test src/chat/session.test.tsx" in commands
    assert "pnpm test" not in commands


def test_infers_pytest_repro_command_scoped_to_extracted_file():
    info = digest.pr_status("NousResearch/hermes-agent", 169, pr_169_style_runner)
    red = [c for c in info["checks"] if c.is_red]
    failures = digest.concise_failures(
        pr_169_style_runner(["run", "view", "333", "--repo", "NousResearch/hermes-agent", "--log-failed"])
    )

    commands = digest.infer_repro_commands(red, failures)

    assert "python -m pytest -o 'addopts=' -q tests/tools/test_terminal_tool.py" in commands
    assert "python -m pytest -o 'addopts=' -q" not in commands


def test_posts_once_and_then_is_silent_for_same_failure(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(tmp_path))
    with kb.connect_closing() as conn:
        task_id = kb.create_task(
            conn,
            title="PR-gate: unblock PR #167",
            body="Keep this mergeable.",
            assignee="dev",
            initial_status="blocked",
        )
        card = digest.GateCard(task_id, "PR-gate: unblock PR #167", "", 167)
        state: dict[str, str] = {}
        first = digest.maybe_post_digest(conn, card, "NousResearch/hermes-agent", state, fake_runner)
        second = digest.maybe_post_digest(conn, card, "NousResearch/hermes-agent", state, fake_runner)
        comments = kb.list_comments(conn, task_id)

    assert first is not None
    assert second is None
    assert len(comments) == 1
    assert comments[0].author == "ci-digest"
    assert "CI digest for PR #167" in comments[0].body


def test_classifies_baseline_stack_order_debt_when_base_has_same_red_check():
    info = digest.pr_status("NousResearch/hermes-agent", 167, fake_runner)
    red = [c for c in info["checks"] if c.is_red]
    base_red = digest.latest_base_red_check_names("NousResearch/hermes-agent", "main", baseline_red_runner)
    assert digest.classify_failure(red, base_red).startswith("baseline/stack-order debt likely")


def test_classification_is_uncertain_when_base_red_workflow_name_differs_from_pr_job():
    info = digest.pr_status("NousResearch/hermes-agent", 167, fake_runner)
    red = [c for c in info["checks"] if c.is_red]
    base_red = digest.latest_base_red_check_names("NousResearch/hermes-agent", "main", baseline_red_workflow_name_runner)

    assessment = digest.classify_failure(red, base_red)

    assert assessment.startswith("uncertain:")
    assert "branch-specific likely" not in assessment


def test_discovers_active_pr_gate_cards_only(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(tmp_path))
    with kb.connect_closing() as conn:
        wanted = kb.create_task(conn, title="PR-gate: PR #169", body="", assignee="dev", initial_status="blocked")
        kb.create_task(conn, title="research PR #170", body="not a gate", assignee="dev", initial_status="blocked")
        done_id = kb.create_task(conn, title="PR-gate: PR #171", body="done", assignee="dev", initial_status="blocked")
        kb.complete_task(conn, done_id, summary="done")
        cards = digest.active_pr_gate_cards(conn)

    assert [(c.task_id, c.pr_number) for c in cards] == [(wanted, 169)]


def test_main_pr_dry_run_does_not_open_kanban(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(tmp_path))
    monkeypatch.setattr(
        digest,
        "load_state",
        lambda _path: (_ for _ in ()).throw(AssertionError("dry-run read dedupe state")),
    )
    monkeypatch.setattr(
        digest.kb,
        "connect_closing",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("dry-run opened Kanban")),
    )
    selected = []
    monkeypatch.setattr(
        digest,
        "maybe_post_digest",
        lambda _conn, card, *_args, **_kwargs: selected.append(card),
    )

    before = set(tmp_path.rglob("*"))
    assert digest.main(["--repo", "NousResearch/hermes-agent", "--pr", "169", "--dry-run"]) == 0
    assert [(card.task_id, card.pr_number) for card in selected] == [("dry-run", 169)]
    assert set(tmp_path.rglob("*")) == before


def test_main_explicit_pr_overrides_selected_task_pr(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(tmp_path))
    with kb.connect_closing() as conn:
        task_id = kb.create_task(
            conn,
            title="PR-gate: PR #167",
            body="Keep this mergeable.",
            assignee="dev",
            initial_status="blocked",
        )

    monkeypatch.setattr(
        digest.kb,
        "connect_closing",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("dry-run opened Kanban")),
    )
    selected = []
    monkeypatch.setattr(
        digest,
        "maybe_post_digest",
        lambda _conn, card, *_args, **_kwargs: selected.append(card),
    )

    before = set(tmp_path.rglob("*"))
    assert (
        digest.main(
            [
                "--repo",
                "NousResearch/hermes-agent",
                "--task-id",
                task_id,
                "--pr",
                "169",
                "--dry-run",
            ]
        )
        == 0
    )
    assert [(card.task_id, card.pr_number) for card in selected] == [(task_id, 169)]
    assert set(tmp_path.rglob("*")) == before


def test_main_explicit_pr_overrides_pr_on_loaded_card(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(tmp_path))
    with kb.connect_closing() as conn:
        task_id = kb.create_task(
            conn,
            title="PR-gate: PR #167",
            body="Keep this mergeable.",
            assignee="dev",
            initial_status="blocked",
        )

    selected = []
    monkeypatch.setattr(
        digest,
        "maybe_post_digest",
        lambda _conn, card, *_args, **_kwargs: selected.append(card),
    )

    assert (
        digest.main(
            [
                "--repo",
                "NousResearch/hermes-agent",
                "--task-id",
                task_id,
                "--pr",
                "169",
            ]
        )
        == 0
    )
    assert [(card.task_id, card.pr_number) for card in selected] == [(task_id, 169)]


def test_main_explicit_pr_dry_run_keeps_all_requested_task_ids(monkeypatch):
    monkeypatch.setattr(
        digest.kb,
        "connect_closing",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("dry-run opened Kanban")),
    )
    selected = []
    monkeypatch.setattr(
        digest,
        "maybe_post_digest",
        lambda _conn, card, *_args, **_kwargs: selected.append(card),
    )

    assert (
        digest.main(
            [
                "--repo",
                "NousResearch/hermes-agent",
                "--task-id",
                "t_first",
                "--task-id",
                "t_second",
                "--pr",
                "169",
                "--dry-run",
            ]
        )
        == 0
    )
    assert [(card.task_id, card.pr_number) for card in selected] == [
        ("t_first", 169),
        ("t_second", 169),
    ]
