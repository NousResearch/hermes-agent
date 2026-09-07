from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import signal
import shutil
import subprocess
import sys
import threading
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

from github_pr_feedback.ci_runner import CIAuditIdentity, CIAuditReceipt
from github_pr_feedback.cli import (
    _ci_audit_comment,
    _factual_reply_is_missing,
    _retrigger_codex_review,
)
from github_pr_feedback.controller import KanbanTask
from github_pr_feedback.github_client import CheckState, Feedback
from github_pr_feedback.ledger import FeedbackLedger
from github_pr_feedback.merge_controller import MergeDecision
from github_pr_feedback.policy import (
    CODEX_REVIEW_TRIGGER,
    FeedbackReceipt,
    PullRequest,
    Reviewer,
    codex_review_trigger_comment,
)
from github_pr_feedback.repair_controller import pr_repair_attribution_line


def test_grouped_audit_opens_sqlite_ledger_in_worker_thread(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from github_pr_feedback.ci_runner import CIAuditIdentity, CIAuditReceipt
    from github_pr_feedback.cli import _run_grouped_exact_head_audit

    worktree = tmp_path / "worktree"
    manifest = worktree / "tests" / "manifests" / "test_lanes.toml"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(
        "[lanes.fast]\nci_status = 'required'\nargv = ['pytest']\n",
        encoding="utf-8",
    )
    identity = CIAuditIdentity("acme/widgets", 17, "b" * 40, "a" * 40)
    receipt = CIAuditReceipt(
        receipt_id="f" * 64,
        identity=identity,
        manifest_digest=hashlib.sha256(manifest.read_bytes()).hexdigest(),
        status="failed",
        started_at=datetime(2026, 8, 25, 12, 0, tzinfo=UTC),
        completed_at=datetime(2026, 8, 25, 12, 1, tzinfo=UTC),
        actions_state=CheckState(False, True, 0),
        commands=(),
    )
    caller_thread = threading.get_ident()
    opened_threads: list[int] = []
    closed_threads: list[int] = []

    class OuterLedger:
        def latest_ci_receipt_for_head(self, *_args: object) -> None:
            return None

    class WorkerLedger:
        def close(self) -> None:
            closed_threads.append(threading.get_ident())

    class Runner:
        def __init__(self, _github: object, _ledger: WorkerLedger) -> None:
            assert opened_threads[-1] == threading.get_ident()

        def run(self, _identity: CIAuditIdentity, _worktree: Path) -> CIAuditReceipt:
            return receipt

    def open_worker_ledger() -> WorkerLedger:
        opened_threads.append(threading.get_ident())
        return WorkerLedger()

    monkeypatch.setattr(
        "github_pr_feedback.cli.FeedbackLedger.for_current_profile",
        open_worker_ledger,
    )
    monkeypatch.setattr("github_pr_feedback.cli.LocalCIRunner", Runner)

    result = _run_grouped_exact_head_audit(
        object(),  # type: ignore[arg-type]
        OuterLedger(),  # type: ignore[arg-type]
        identity,
        worktree,
        force_fresh=True,
    )

    assert result == receipt
    assert opened_threads and opened_threads[0] != caller_thread
    assert closed_threads == opened_threads


def _plugin_module():
    root = Path(__file__).resolve().parents[1]
    name = "test_github_pr_feedback_plugin"
    spec = importlib.util.spec_from_file_location(
        name, root / "__init__.py", submodule_search_locations=[str(root)]
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class RecordingContext:
    def __init__(self, settings: dict[str, object] | None = None) -> None:
        self.calls: list[dict[str, object]] = []
        self.settings = settings or {}
        self.config_reads: list[str] = []

    def register_cli_command(self, **kwargs: object) -> None:
        self.calls.append(kwargs)

    def get_config(self, key: str, default: object = None) -> object:
        self.config_reads.append(key)
        return self.settings.get(key, default)


def test_register_exposes_the_github_feedback_cli_command() -> None:
    context = RecordingContext()

    _plugin_module().register(context)

    assert len(context.calls) == 1
    command = context.calls[0]
    assert command["name"] == "github-pr-feedback"
    assert command["help"] == "Scan governed GitHub pull-request feedback"
    assert callable(command["setup_fn"])
    assert callable(command["handler_fn"])
    parser = argparse.ArgumentParser()
    command["setup_fn"](parser)
    assert parser.parse_args(["scan"]).github_pr_feedback_action == "scan"


def test_scan_prioritizes_feedback_before_degraded_repair_maintenance(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from github_pr_feedback.cli import _scan
    from github_pr_feedback.controller import ScanResult

    order: list[str] = []

    class Lock:
        def __enter__(self) -> bool:
            return True

        def __exit__(self, *_args: object) -> None:
            return None

    class Ledger:
        @classmethod
        def for_current_profile(cls):
            return cls()

        def close(self) -> None:
            pass

    class Policy:
        repair_steward = object()
        merge_maintainer = None
        release_maintenance = None

    class Repair:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        def scan(self) -> ScanResult:
            order.append("repair")
            return ScanResult(0, {"github_state_unavailable": 1}, degraded=True)

    class Feedback:
        def scan(self, *, apply_labels: bool) -> ScanResult:
            assert apply_labels is False
            order.append("feedback")
            return ScanResult(0, {})

        def apply_agent_labels(self) -> dict[str, object]:
            return {"status": "ok", "updated": 0, "skipped": {}}

    monkeypatch.setattr("github_pr_feedback.cli._load_policy_from_context", lambda _ctx: Policy())
    monkeypatch.setattr("github_pr_feedback.cli._exclusive_scan_lock", lambda: Lock())
    monkeypatch.setattr("github_pr_feedback.cli.FeedbackLedger", Ledger)
    monkeypatch.setattr("github_pr_feedback.cli.RepairController", Repair)
    monkeypatch.setattr("github_pr_feedback.cli._controller", lambda *_args: Feedback())

    assert _scan(object()) == 1
    assert order == ["feedback", "repair"]
    payload = json.loads(capsys.readouterr().out)
    assert payload["repair"]["status"] == "degraded"


def _run_scan_with_primary_result(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    primary_result: SimpleNamespace,
) -> tuple[int, list[str], dict[str, object]]:
    from github_pr_feedback.cli import _scan

    order: list[str] = []

    class Lock:
        def __enter__(self) -> bool:
            return True

        def __exit__(self, *_args: object) -> None:
            return None

    class Ledger:
        @classmethod
        def for_current_profile(cls):
            return cls()

        def close(self) -> None:
            pass

    class Policy:
        repair_steward = object()
        merge_maintainer = object()
        release_maintenance = object()

    class Primary:
        def scan(self, *, apply_labels: bool):
            assert apply_labels is False
            order.append("primary")
            return primary_result

        def apply_agent_labels(self):
            return {"status": "ok", "updated": 0, "skipped": {}}

    class Repair:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        def scan(self):
            order.append("repair")
            return SimpleNamespace(
                created=0,
                skipped={},
                degraded=False,
                required_local_ci_backlog=0,
            )

    def merge(*_args: object, **_kwargs: object) -> dict[str, object]:
        order.append("merge")
        return {"status": "ok"}

    def release(*_args: object, **_kwargs: object) -> dict[str, object]:
        order.append("release")
        return {"status": "ok"}

    monkeypatch.setattr(
        "github_pr_feedback.cli._load_policy_from_context", lambda _ctx: Policy()
    )
    monkeypatch.setattr("github_pr_feedback.cli._exclusive_scan_lock", lambda: Lock())
    monkeypatch.setattr("github_pr_feedback.cli.FeedbackLedger", Ledger)
    monkeypatch.setattr("github_pr_feedback.cli.RepairController", Repair)
    monkeypatch.setattr(
        "github_pr_feedback.cli._controller", lambda *_args: Primary()
    )
    monkeypatch.setattr("github_pr_feedback.cli._run_merge_scan", merge)
    monkeypatch.setattr(
        "github_pr_feedback.cli._run_release_maintenance_scan", release
    )

    returncode = _scan(object())
    payload = json.loads(capsys.readouterr().out)
    return returncode, order, payload


def test_scan_keeps_merge_maintainer_moving_during_required_ci_backlog(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    result = SimpleNamespace(
        created=1,
        skipped={"local_ci_dispatch_cap": 1},
        degraded=False,
        required_local_ci_backlog=2,
    )

    returncode, order, payload = _run_scan_with_primary_result(
        monkeypatch, capsys, result
    )

    assert returncode == 0
    assert order == ["primary", "merge"]
    assert payload["required_local_ci_backlog"] == 2
    assert payload["deferred"] == ["repair", "release_maintenance"]
    assert payload["merge"]["status"] == "ok"


def test_scan_runs_label_side_lane_after_merge_maintainer(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from github_pr_feedback.cli import _scan

    order: list[str] = []

    class Lock:
        def __enter__(self) -> bool:
            return True

        def __exit__(self, *_args: object) -> None:
            return None

    class Ledger:
        @classmethod
        def for_current_profile(cls):
            return cls()

        def close(self) -> None:
            pass

    class Policy:
        repair_steward = None
        merge_maintainer = object()
        release_maintenance = None

    class Primary:
        def scan(self, *, apply_labels: bool):
            assert apply_labels is False
            order.append("primary")
            return SimpleNamespace(
                created=0,
                skipped={},
                degraded=False,
                required_local_ci_backlog=1,
                local_ci_catalogue_deferred=0,
            )

        def apply_agent_labels(self):
            order.append("labels")
            return {"status": "ok", "updated": 1, "skipped": {}}

    def merge(*_args: object, **_kwargs: object) -> dict[str, object]:
        order.append("merge")
        return {"status": "ok"}

    monkeypatch.setattr("github_pr_feedback.cli._load_policy_from_context", lambda _ctx: Policy())
    monkeypatch.setattr("github_pr_feedback.cli._exclusive_scan_lock", lambda: Lock())
    monkeypatch.setattr("github_pr_feedback.cli.FeedbackLedger", Ledger)
    monkeypatch.setattr("github_pr_feedback.cli._controller", lambda *_args: Primary())
    monkeypatch.setattr("github_pr_feedback.cli._run_merge_scan", merge)

    assert _scan(object()) == 0
    payload = json.loads(capsys.readouterr().out)
    assert order == ["primary", "merge", "labels"]
    assert payload["labels"]["status"] == "ok"


def test_scan_does_not_defer_secondary_fanout_for_read_cap_without_ci_backlog(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    result = SimpleNamespace(
        created=0,
        skipped={"local_ci_open_pr_scan_cap": 1},
        degraded=False,
        required_local_ci_backlog=0,
    )

    returncode, order, payload = _run_scan_with_primary_result(
        monkeypatch, capsys, result
    )

    assert returncode == 0
    assert order == ["primary", "repair", "merge", "release"]
    assert "required_local_ci_backlog" not in payload
    assert "deferred" not in payload


def test_scan_payload_reports_catalogue_deferred_separately_from_skips() -> None:
    from github_pr_feedback.controller import ScanResult
    from github_pr_feedback.cli import _scan_payload

    payload = _scan_payload(
        ScanResult(
            created=1,
            skipped={},
            local_ci_catalogue_deferred=199,
        )
    )

    assert payload["local_ci_catalogue_deferred"] == 199
    assert "local_ci_open_pr_scan_cap" not in payload["skipped"]


class RecordingKanbanRunner:
    def __init__(self, stdout: str, returncode: int = 0, stderr: str = "") -> None:
        self.stdout = stdout
        self.returncode = returncode
        self.stderr = stderr
        self.calls: list[list[str]] = []

    def run(self, argv: list[str]):
        from github_pr_feedback.cli import KanbanCommandResult

        self.calls.append(argv)
        return KanbanCommandResult(self.returncode, self.stdout, self.stderr)


class RecordingDoctorRunner:
    def __init__(self, responses: dict[tuple[str, ...], tuple[int, str]]) -> None:
        self.responses = responses
        self.calls: list[list[str]] = []

    def which(self, executable: str) -> str | None:
        return {
            "gh": "/opt/tools/gh",
            "hermes": "/opt/tools/hermes",
            "git": "/opt/tools/git",
        }.get(executable)

    def run(self, argv: list[str]):
        from github_pr_feedback.cli import DoctorCommandResult

        self.calls.append(argv)
        returncode, stdout = self.responses[tuple(argv)]
        return DoctorCommandResult(returncode, stdout)


def kanban_task() -> KanbanTask:
    hostile = "--skill github-auth; gh pr merge 17; $(printenv GH_TOKEN)"
    return KanbanTask(
        title="GitHub PR feedback: acme/widgets#17",
        instructions="GitHub push/reply/merge require operator approval.",
        board="repairs",
        assignee="repair-agent",
        repository_path=Path("/repositories/widgets"),
        head_sha="a" * 40,
        branch="hermes/github-pr-feedback/receipt-branch",
        idempotency_key="github-pr-feedback:key",
        evidence={"untrusted": True, "body": hostile},
    )


def test_kanban_client_creates_only_a_blocked_card_with_inert_hostile_evidence() -> (
    None
):
    from github_pr_feedback.cli import KanbanSubprocessClient

    runner = RecordingKanbanRunner('{"id": "task-123"}')

    task_id = KanbanSubprocessClient(runner).create_or_get_task(kanban_task())

    assert task_id == "task-123"
    argv = runner.calls == [
        [
            "hermes",
            "kanban",
            "--board",
            "repairs",
            "create",
            "GitHub PR feedback: acme/widgets#17",
            "--body",
            runner.calls[0][7],
            "--assignee",
            "repair-agent",
            "--workspace",
            "dir:/repositories/widgets",
            "--idempotency-key",
            "github-pr-feedback:key",
            "--max-retries",
            "1",
            "--initial-status",
            "blocked",
            "--json",
        ]
    ]
    assert argv
    assert "operator approval" in runner.calls[0][7]
    assert '"untrusted": true' in runner.calls[0][7]
    assert (
        "--skill github-auth; gh pr merge 17; $(printenv GH_TOKEN)"
        in runner.calls[0][7]
    )
    assert "--skill" not in runner.calls[0]
    assert "github-code-review" not in runner.calls[0]


def test_kanban_client_dispatches_an_opted_in_repair_as_ready() -> None:
    from github_pr_feedback.cli import KanbanSubprocessClient

    runner = RecordingKanbanRunner('{"id": "task-123"}')
    task = replace(
        kanban_task(), initial_status="running", max_retries=3, max_runtime_seconds=1200
    )

    task_id = KanbanSubprocessClient(runner).create_or_get_task(task)

    assert task_id == "task-123"
    status_index = runner.calls[0].index("--initial-status") + 1
    assert runner.calls[0][status_index] == "running"
    retries_index = runner.calls[0].index("--max-retries") + 1
    assert runner.calls[0][retries_index] == "3"
    runtime_index = runner.calls[0].index("--max-runtime") + 1
    assert runner.calls[0][runtime_index] == "1200"


def test_kanban_client_reads_archived_task_status_for_orphan_reconciliation() -> None:
    from github_pr_feedback.cli import KanbanSubprocessClient

    runner = RecordingKanbanRunner('{"task": {"id": "task-123", "status": "archived"}}')

    assert KanbanSubprocessClient(runner).task_status("repairs", "task-123") == "archived"
    assert runner.calls == [
        ["hermes", "kanban", "--board", "repairs", "show", "task-123", "--json"]
    ]


def test_kanban_client_reads_missing_task_error_from_stderr() -> None:
    from github_pr_feedback.cli import KanbanSubprocessClient

    client = KanbanSubprocessClient(
        RecordingKanbanRunner(
            "", returncode=1, stderr="no such task: task-missing\n"
        )
    )

    assert client.task_status("repairs", "task-missing") is None


def test_subprocess_kanban_runner_captures_bounded_stderr(monkeypatch) -> None:
    from types import SimpleNamespace

    from github_pr_feedback.cli import SubprocessKanbanRunner

    monkeypatch.setattr(
        "github_pr_feedback.cli.subprocess.run",
        lambda *_args, **kwargs: (
            SimpleNamespace(returncode=1, stdout="", stderr="no such task: task-missing\n")
            if kwargs["stderr"] is subprocess.PIPE
            else (_ for _ in ()).throw(AssertionError("Kanban stderr must be captured"))
        ),
    )

    result = SubprocessKanbanRunner().run(["hermes", "kanban", "show", "task-missing"])

    assert result.stderr == "no such task: task-missing\n"


def test_kanban_client_uses_the_task_specific_evidence_heading() -> None:
    from github_pr_feedback.cli import KanbanSubprocessClient

    runner = RecordingKanbanRunner('{"id": "task-123"}')
    task = replace(
        kanban_task(),
        evidence_heading="Canonical PR audit receipt (JSON)",
        evidence={"repository": "acme/widgets", "pr_number": 17},
    )

    KanbanSubprocessClient(runner).create_or_get_task(task)

    assert "Canonical PR audit receipt (JSON):" in runner.calls[0][7]
    assert "Untrusted evidence" not in runner.calls[0][7]


def test_auto_dispatch_argv_is_accepted_by_the_real_kanban_parser() -> None:
    from github_pr_feedback.cli import _kanban_create_argv
    from hermes_cli.kanban import build_parser

    root = argparse.ArgumentParser()
    subcommands = root.add_subparsers(dest="command", required=True)
    build_parser(subcommands)
    task = replace(kanban_task(), initial_status="running", max_retries=3)

    parsed = root.parse_args(_kanban_create_argv(task)[1:])

    assert parsed.command == "kanban"
    assert parsed.kanban_action == "create"
    assert parsed.initial_status == "running"
    assert parsed.max_retries == 3


def test_kanban_client_persists_explicit_worker_route() -> None:
    from github_pr_feedback.cli import _kanban_create_argv
    from hermes_cli.kanban import build_parser

    root = argparse.ArgumentParser()
    subcommands = root.add_subparsers(dest="command", required=True)
    build_parser(subcommands)
    task = replace(
        kanban_task(),
        model_override="qwen3.5:4b",
        provider_override="ollama-launch",
        reasoning_effort="none",
    )

    parsed = root.parse_args(_kanban_create_argv(task)[1:])

    assert parsed.model_override == "qwen3.5:4b"
    assert parsed.provider_override == "ollama-launch"
    assert parsed.reasoning_effort == "none"


@pytest.mark.parametrize("stdout", ["{}", "[]", '{"id": ""}', '{"id": 17}', "not json"])
def test_kanban_client_fails_closed_on_an_invalid_create_response(stdout: str) -> None:
    from github_pr_feedback.cli import KanbanSubprocessClient

    with pytest.raises(RuntimeError, match="Kanban task creation failed"):
        KanbanSubprocessClient(RecordingKanbanRunner(stdout)).create_or_get_task(
            kanban_task()
        )


def test_ledger_status_counts_each_receipt_state(tmp_path: Path) -> None:
    ledger = FeedbackLedger(tmp_path / "ledger.sqlite3")
    claimed = FeedbackReceipt("acme/widgets", 1, "issue_comment", "claimed", "a" * 40)
    completed = FeedbackReceipt("acme/widgets", 2, "review", "completed", "b" * 40)
    failed = FeedbackReceipt("acme/widgets", 3, "review_comment", "failed", "c" * 40)
    claimed_lease = _claim(ledger, claimed, owner="claimed")
    completed_lease = _claim(ledger, completed, owner="completed")
    assert claimed_lease is not None and completed_lease is not None
    ledger.finalize(completed, "kanban-2", completed_lease)
    failed_lease = _claim(ledger, failed, owner="failed")
    assert failed_lease is not None
    ledger.fail(failed, "Kanban unavailable", failed_lease)

    assert ledger.status_counts() == {"claimed": 1, "completed": 1, "failed": 1}

    ledger.close()


def test_scan_reads_disabled_policy_through_the_plugin_config_context(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    context = RecordingContext({"enabled": False})
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    _plugin_module().register(context)
    command = context.calls[0]
    parser = argparse.ArgumentParser()
    command["setup_fn"](parser)

    exit_code = command["handler_fn"](parser.parse_args(["scan"]))

    assert exit_code == 0
    assert json.loads(capsys.readouterr().out) == {
        "created": 0,
        "skipped": {},
        "status": "ok",
    }
    assert context.config_reads == ["enabled"]


def test_scan_lock_rejects_a_concurrent_scan_for_the_same_control_home(
    tmp_path: Path,
) -> None:
    from github_pr_feedback.cli import _exclusive_scan_lock

    with _exclusive_scan_lock(tmp_path) as first:
        with _exclusive_scan_lock(tmp_path) as second:
            assert first is True
            assert second is False

    with _exclusive_scan_lock(tmp_path) as after_release:
        assert after_release is True


def test_cli_exposes_status_doctor_and_an_exact_immutable_retry_identity() -> None:
    context = RecordingContext()
    _plugin_module().register(context)
    parser = argparse.ArgumentParser()
    context.calls[0]["setup_fn"](parser)

    assert parser.parse_args(["status"]).github_pr_feedback_action == "status"
    assert parser.parse_args(["doctor"]).github_pr_feedback_action == "doctor"
    audit = parser.parse_args(
        [
            "audit-pr",
            "--repository",
            "acme/widgets",
            "--pr-number",
            "17",
            "--head-sha",
            "a" * 40,
            "--worktree",
            "/repositories/widgets",
        ]
    )
    assert audit.github_pr_feedback_action == "audit-pr"
    fresh = parser.parse_args(
        [
            "audit-pr",
            "--repository",
            "acme/widgets",
            "--pr-number",
            "17",
            "--head-sha",
            "a" * 40,
            "--worktree",
            "/repositories/widgets",
            "--fresh",
        ]
    )
    assert fresh.fresh is True
    assert parser.parse_args(["merge-scan"]).github_pr_feedback_action == "merge-scan"
    assert (
        parser.parse_args(["merge-status"]).github_pr_feedback_action == "merge-status"
    )
    completed = parser.parse_args(
        [
            "complete-feedback",
            "--repository",
            "acme/widgets",
            "--pr-number",
            "17",
            "--feedback-kind",
            "review_comment",
            "--feedback-id",
            "120",
            "--receipt-head-sha",
            "a" * 40,
            "--resolved-head-sha",
            "b" * 40,
        ]
    )
    assert completed.github_pr_feedback_action == "complete-feedback"
    retry = parser.parse_args(
        [
            "retry",
            "--repository",
            "acme/widgets",
            "--pr-number",
            "17",
            "--feedback-kind",
            "review_comment",
            "--feedback-id",
            "120",
            "--head-sha",
            "a" * 40,
        ]
    )
    assert retry.github_pr_feedback_action == "retry"
    assert retry.repository == "acme/widgets"
    with pytest.raises(SystemExit):
        parser.parse_args(["retry", "--repository", "acme/widgets"])


def test_status_reports_durable_receipt_counts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    ledger = FeedbackLedger.for_current_profile()
    failed = FeedbackReceipt("acme/widgets", 1, "issue_comment", "failed", "a" * 40)
    failed_lease = _claim(ledger, failed)
    assert failed_lease is not None
    ledger.fail(failed, "Kanban unavailable", failed_lease)
    ledger.close()
    context = RecordingContext()
    _plugin_module().register(context)
    parser = argparse.ArgumentParser()
    context.calls[0]["setup_fn"](parser)

    exit_code = context.calls[0]["handler_fn"](parser.parse_args(["status"]))

    assert exit_code == 0
    assert json.loads(capsys.readouterr().out) == {
        "claimed": 0,
        "completed": 0,
        "failed": 1,
    }


def test_doctor_reports_a_disabled_plugin_without_scanning(
    capsys: pytest.CaptureFixture[str],
) -> None:
    context = RecordingContext({"enabled": False})
    _plugin_module().register(context)
    parser = argparse.ArgumentParser()
    context.calls[0]["setup_fn"](parser)

    exit_code = context.calls[0]["handler_fn"](parser.parse_args(["doctor"]))

    assert exit_code == 0
    assert json.loads(capsys.readouterr().out) == {"status": "disabled"}
    assert context.config_reads == ["enabled"]


def test_namespaced_context_loads_assignee_rules_for_runtime_routing(
    tmp_path: Path,
) -> None:
    from github_pr_feedback.cli import _load_policy_from_context

    repository = tmp_path / "repository"
    subprocess.run(["git", "init", "--quiet", str(repository)], check=True)
    settings = enabled_settings(repository)
    settings["assignee_rules"] = [
        {"assignee": "performance-patch-steward", "match_any": ["latency"]}
    ]

    policy = _load_policy_from_context(RecordingContext(settings))

    assert policy.assignee_for("Reduce latency") == "performance-patch-steward"


def test_inspect_pr_emits_canonical_identity_from_the_shared_github_client(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from github_pr_feedback.cli import _inspect_pr

    repository = tmp_path / "repository"
    subprocess.run(["git", "init", "--quiet", str(repository)], check=True)
    settings = enabled_settings(repository)
    expected = PullRequest(
        17,
        "OPEN",
        "acme/widgets",
        "acme/widgets",
        "owner",
        "codex/fix",
        "a" * 40,
        base_branch="main",
        base_sha="b" * 40,
    )

    class FakeGitHub:
        def get_pull_request(self, repository: str, number: int) -> PullRequest:
            assert repository == "acme/widgets"
            assert number == 17
            return expected

    monkeypatch.setattr("github_pr_feedback.cli.GitHubClient", FakeGitHub)

    exit_code = _inspect_pr(
        RecordingContext(settings),
        argparse.Namespace(repository="acme/widgets", pr_number=17),
    )

    assert exit_code == 0
    assert json.loads(capsys.readouterr().out) == {
        "base_branch": "main",
        "base_sha": "b" * 40,
        "head_ref_name": "codex/fix",
        "head_repository": "acme/widgets",
        "head_sha": "a" * 40,
        "number": 17,
        "repository": "acme/widgets",
    }


def test_namespaced_context_preserves_auto_dispatch_and_local_ci_audit_settings(
    tmp_path: Path,
) -> None:
    from github_pr_feedback.cli import _load_policy_from_context

    repository = tmp_path / "repository"
    subprocess.run(["git", "init", "--quiet", str(repository)], check=True)
    settings = enabled_settings(repository)
    settings["auto_dispatch"] = True
    settings["local_ci_audit"] = {
        "enabled": True,
        "assignee": "pr-local-ci-auditor",
        "post_results": True,
    }

    policy = _load_policy_from_context(RecordingContext(settings))

    assert policy.auto_dispatch is True
    assert policy.local_ci_audit is not None
    assert policy.local_ci_audit.assignee == "pr-local-ci-auditor"


def test_namespaced_context_preserves_agent_label_settings(tmp_path: Path) -> None:
    from github_pr_feedback.cli import _load_policy_from_context

    repository = tmp_path / "repository"
    subprocess.run(["git", "init", "--quiet", str(repository)], check=True)
    settings = enabled_settings(repository)
    settings["agent_labels"] = {
        "enabled": True,
        "max_updates_per_scan": 25,
        "create_missing": True,
        "mappings": [
            {
                "branch_prefix": "codex/",
                "label": "codex",
                "color": "1f6feb",
                "description": "PR authored by Codex",
            }
        ],
    }

    policy = _load_policy_from_context(RecordingContext(settings))

    assert policy.agent_labels is not None
    assert policy.agent_labels.max_updates_per_scan == 25
    assert policy.agent_labels.label_for_branch("codex/fix") == "codex"


def test_namespaced_context_preserves_strict_merge_maintainer_settings(
    tmp_path: Path,
) -> None:
    from github_pr_feedback.cli import _load_policy_from_context

    repository = tmp_path / "repository"
    deployment = tmp_path / "deployment"
    subprocess.run(["git", "init", "--quiet", str(repository)], check=True)
    subprocess.run(["git", "init", "--quiet", str(deployment)], check=True)
    settings = enabled_settings(repository)
    settings["merge_maintainer"] = {
        "enabled": True,
        "assignee": "pr-merge-maintainer",
        "repository": "acme/widgets",
        "author_login": "owner",
        "base_branch": "stable",
        "merge_methods": ["squash", "rebase", "merge"],
        "receipt_max_age_seconds": 3600,
        "report_only": True,
        "post_merge": {"enabled": False},
    }

    loaded = _load_policy_from_context(RecordingContext(settings))

    assert loaded.merge_maintainer is not None
    assert loaded.merge_maintainer.assignee == "pr-merge-maintainer"
    assert loaded.merge_maintainer.merge_methods == ("squash", "rebase", "merge")
    assert loaded.merge_maintainer.report_only is True


def test_complete_maintenance_cli_records_only_a_configured_exact_head_lane(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    repository = tmp_path / "repository"
    subprocess.run(["git", "init", "--quiet", str(repository)], check=True)
    settings = enabled_settings(repository)
    settings["release_maintenance"] = {
        "enabled": True,
        "assignee": "release-maintenance-steward",
        "repository": "acme/widgets",
        "base_branch": "stable",
        "quiet_period_seconds": 900,
        "max_runtime_seconds": 7200,
        "lanes": [
            {
                "name": "unit-tests",
                "assignee": "test-contract-steward",
                "command": ["python3", "-m", "pytest", "-q"],
            }
        ],
    }
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "profile"))
    context = RecordingContext(settings)
    parser = argparse.ArgumentParser()
    from github_pr_feedback.cli import handle_cli_with_context, setup_cli

    setup_cli(context, parser)
    args = parser.parse_args(
        [
            "complete-maintenance",
            "--repository",
            "acme/widgets",
            "--head-sha",
            "a" * 40,
            "--lane",
            "unit-tests",
            "--status",
            "passed",
            "--summary",
            "220 tests passed",
            "--command-evidence-json",
            json.dumps(
                [
                    {
                        "argv": ["python3", "-m", "pytest", "-q"],
                        "cwd": str(repository),
                        "returncode": 0,
                        "duration_ms": 125,
                        "timed_out": False,
                        "stdout_sha256": "a" * 64,
                        "stderr_sha256": "b" * 64,
                    }
                ]
            ),
        ]
    )

    assert handle_cli_with_context(context, args) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "recorded"
    ledger = FeedbackLedger.for_current_profile()
    try:
        assert (
            ledger.maintenance_receipts("acme/widgets", "a" * 40)["unit-tests"].status
            == "passed"
        )
    finally:
        ledger.close()


def test_release_maintenance_scan_is_part_of_the_governed_scan_surface(
    tmp_path: Path,
) -> None:
    from github_pr_feedback.cli import (
        _load_policy_from_context,
        _run_release_maintenance_scan,
    )

    repository = tmp_path / "repository"
    subprocess.run(["git", "init", "--quiet", str(repository)], check=True)
    settings = enabled_settings(repository)
    settings["release_maintenance"] = {
        "enabled": True,
        "assignee": "release-maintenance-steward",
        "repository": "acme/widgets",
        "base_branch": "stable",
        "quiet_period_seconds": 900,
        "max_runtime_seconds": 7200,
        "lanes": [
            {
                "name": "unit-tests",
                "assignee": "test-contract-steward",
                "command": ["python3", "-m", "pytest", "-q"],
            }
        ],
    }
    policy = _load_policy_from_context(RecordingContext(settings))

    class GitHub:
        def list_all_open_pull_requests(self, repository: str):
            return (object(),)

        def get_branch_head(self, repository: str, branch: str):
            raise AssertionError(
                "open PRs must stop the scan before reading the base head"
            )

    payload = _run_release_maintenance_scan(
        policy,
        FeedbackLedger(tmp_path / "ledger.sqlite3"),
        github=GitHub(),
        kanban=object(),
        workspaces=object(),
        now=lambda: datetime(2026, 8, 25, tzinfo=UTC),
        control_home=tmp_path / "control",
    )

    assert payload == {
        "status": "waiting_open_prs",
        "head_sha": None,
        "tasks_created": 0,
        "blockers": ["open_prs"],
    }


def test_merge_maintainer_task_has_no_model_merge_authority(tmp_path: Path) -> None:
    from github_pr_feedback.cli import _load_policy_from_context, _merge_maintainer_task

    repository = tmp_path / "repository"
    subprocess.run(["git", "init", "--quiet", str(repository)], check=True)
    settings = enabled_settings(repository)
    settings["merge_maintainer"] = {
        "enabled": True,
        "assignee": "pr-merge-maintainer",
        "repository": "acme/widgets",
        "author_login": "owner",
        "base_branch": "stable",
        "merge_methods": ["squash", "rebase", "merge"],
        "receipt_max_age_seconds": 3600,
        "report_only": True,
        "post_merge": {"enabled": False},
    }
    loaded = _load_policy_from_context(RecordingContext(settings))
    pull = PullRequest(
        17,
        "OPEN",
        "acme/widgets",
        "acme/widgets",
        "owner",
        "codex/fix",
        "a" * 40,
    )
    decision = MergeDecision(False, ("ci_receipt_missing",), None, "d" * 64)

    task = _merge_maintainer_task(loaded, pull, decision)

    assert task.assignee == "pr-merge-maintainer"
    assert task.initial_status == "running"
    assert task.evidence["blockers"] == ["ci_receipt_missing"]
    assert "no repository or GitHub mutation authority" in task.instructions
    assert "Model output cannot waive" in task.instructions
    assert "supplied deterministic evidence" in task.instructions
    assert "not a blocker for this observability card" in task.instructions
    assert "immediately call kanban_complete" in task.instructions
    assert "kanban_block only if kanban_complete" in task.instructions


def test_merge_scan_skips_expensive_github_reads_without_exact_head_ci_receipt(
    tmp_path: Path,
) -> None:
    from github_pr_feedback.cli import _load_policy_from_context, _run_merge_scan

    repository = tmp_path / "repository"
    subprocess.run(["git", "init", "--quiet", str(repository)], check=True)
    manifest = repository / "tests" / "manifests" / "test_lanes.toml"
    manifest.parent.mkdir(parents=True)
    manifest.write_text("version = 1\n", encoding="utf-8")
    settings = enabled_settings(repository)
    settings["merge_maintainer"] = {
        "enabled": True,
        "assignee": "pr-merge-maintainer",
        "repository": "acme/widgets",
        "author_login": "owner",
        "base_branch": "stable",
        "merge_methods": ["squash"],
        "receipt_max_age_seconds": 3600,
        "report_only": False,
        "post_merge": {"enabled": False},
    }
    policy = _load_policy_from_context(RecordingContext(settings))

    class ListingOnlyGitHub:
        def list_open_pull_requests(self, repository: str, owner_login: str):
            return (
                PullRequest(
                    17,
                    "OPEN",
                    "acme/widgets",
                    "acme/widgets",
                    "owner",
                    "codex/fix",
                    "a" * 40,
                ),
            )

        def __getattr__(self, name: str):
            raise AssertionError(f"unexpected expensive GitHub read: {name}")

    class NoTasks:
        def create_or_get_task(self, task):
            raise AssertionError(
                "missing CI receipt must not create an observability task"
            )

    ledger = FeedbackLedger(tmp_path / "ledger.sqlite3")
    try:
        result = _run_merge_scan(
            policy,
            ledger,
            github=ListingOnlyGitHub(),
            kanban=NoTasks(),
        )
    finally:
        ledger.close()

    assert result["status"] == "ok"
    assert result["blocked"] == {"17": ["ci_receipt_missing"]}
    assert result["maintainer_tasks_created"] == 0


def test_merge_scan_reports_failed_exact_head_receipt_as_not_passing(
    tmp_path: Path,
) -> None:
    from github_pr_feedback.cli import _load_policy_from_context, _run_merge_scan

    repository = tmp_path / "repository"
    subprocess.run(["git", "init", "--quiet", str(repository)], check=True)
    manifest = repository / "tests" / "manifests" / "test_lanes.toml"
    manifest.parent.mkdir(parents=True)
    manifest.write_text("version = 1\n", encoding="utf-8")
    settings = enabled_settings(repository)
    settings["merge_maintainer"] = {
        "enabled": True,
        "assignee": "pr-merge-maintainer",
        "repository": "acme/widgets",
        "author_login": "owner",
        "base_branch": "stable",
        "merge_methods": ["squash"],
        "receipt_max_age_seconds": 3600,
        "report_only": False,
        "post_merge": {"enabled": False},
    }
    policy = _load_policy_from_context(RecordingContext(settings))

    class ListingOnlyGitHub:
        def list_open_pull_requests(self, repository: str, owner_login: str):
            return (
                PullRequest(
                    17,
                    "OPEN",
                    "acme/widgets",
                    "acme/widgets",
                    "owner",
                    "codex/fix",
                    "a" * 40,
                ),
            )

        def __getattr__(self, name: str):
            raise AssertionError(f"unexpected expensive GitHub read: {name}")

    class FailedReceipt:
        status = "failed"

    class FailedLedger:
        def latest_ci_receipt(self, *args, **kwargs):
            return FailedReceipt()

    class NoTasks:
        def create_or_get_task(self, task):
            raise AssertionError(
                "failed CI receipt must not create an observability task"
            )

    result = _run_merge_scan(
        policy,
        FailedLedger(),
        github=ListingOnlyGitHub(),
        kanban=NoTasks(),
    )

    assert result["status"] == "ok"
    assert result["blocked"] == {"17": ["ci_receipt_not_passing"]}
    assert result["maintainer_tasks_created"] == 0


def test_merge_scan_does_not_hide_failed_receipt_behind_manifest_mismatch(
    tmp_path: Path,
) -> None:
    from github_pr_feedback.cli import _load_policy_from_context, _run_merge_scan

    repository = tmp_path / "repository"
    subprocess.run(["git", "init", "--quiet", str(repository)], check=True)
    manifest = repository / "tests" / "manifests" / "test_lanes.toml"
    manifest.parent.mkdir(parents=True)
    manifest.write_text("version = 2\n", encoding="utf-8")
    settings = enabled_settings(repository)
    settings["merge_maintainer"] = {
        "enabled": True,
        "assignee": "pr-merge-maintainer",
        "repository": "acme/widgets",
        "author_login": "owner",
        "base_branch": "stable",
        "merge_methods": ["squash"],
        "receipt_max_age_seconds": 3600,
        "report_only": False,
        "post_merge": {"enabled": False},
    }
    policy = _load_policy_from_context(RecordingContext(settings))

    class ListingOnlyGitHub:
        def list_open_pull_requests(self, repository: str, owner_login: str):
            return (
                PullRequest(
                    17,
                    "OPEN",
                    "acme/widgets",
                    "acme/widgets",
                    "owner",
                    "codex/fix",
                    "a" * 40,
                ),
            )

        def __getattr__(self, name: str):
            raise AssertionError(f"unexpected expensive GitHub read: {name}")

    class FailedReceipt:
        status = "failed"

    class MismatchedLedger:
        def latest_ci_receipt(self, *args, **kwargs):
            return None

        def latest_ci_receipt_for_head(self, *args, **kwargs):
            return FailedReceipt()

    class NoTasks:
        def create_or_get_task(self, task):
            raise AssertionError(
                "failed CI receipt must not create an observability task"
            )

    result = _run_merge_scan(
        policy,
        MismatchedLedger(),
        github=ListingOnlyGitHub(),
        kanban=NoTasks(),
    )

    assert result["status"] == "ok"
    assert result["blocked"] == {"17": ["ci_receipt_not_passing"]}
    assert result["maintainer_tasks_created"] == 0


def test_merge_scan_reconciles_verification_required_pr_that_is_no_longer_open(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from github_pr_feedback.cli import _load_policy_from_context, _run_merge_scan
    from github_pr_feedback.merge_controller import (
        MergeDecision,
        MergeReceipt,
        MergeRunResult,
    )

    repository = tmp_path / "repository"
    subprocess.run(["git", "init", "--quiet", str(repository)], check=True)
    manifest = repository / "tests" / "manifests" / "test_lanes.toml"
    manifest.parent.mkdir(parents=True)
    manifest.write_text("version = 1\n", encoding="utf-8")
    settings = enabled_settings(repository)
    settings["merge_maintainer"] = {
        "enabled": True,
        "assignee": "pr-merge-maintainer",
        "repository": "acme/widgets",
        "author_login": "owner",
        "base_branch": "stable",
        "merge_methods": ["squash"],
        "receipt_max_age_seconds": 3600,
        "report_only": False,
        "post_merge": {"enabled": False},
    }
    policy = _load_policy_from_context(RecordingContext(settings))
    receipt = MergeReceipt(
        repository="acme/widgets",
        pr_number=149,
        author_login="owner",
        base_branch="stable",
        tested_head_sha="a" * 40,
        ci_receipt_id="d" * 64,
        snapshot_digest="e" * 64,
        method="squash",
        merge_commit_oid="c" * 40,
        merged_at=datetime(2026, 8, 25, 12, 0, tzinfo=UTC),
        executor="test",
    )

    class GitHub:
        def list_open_pull_requests(self, repository: str, owner_login: str):
            return ()

    class Ledger:
        def verification_required_merge_numbers(self, repository: str):
            return (149,)

    class Controller:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def run(self, number: int) -> MergeRunResult:
            assert number == 149
            return MergeRunResult(
                MergeDecision(True, (), "squash", "e" * 64), receipt
            )

    monkeypatch.setattr(
        "github_pr_feedback.cli.CanonicalMergeEvidenceSource", lambda *args: object()
    )
    monkeypatch.setattr("github_pr_feedback.cli.MergeController", Controller)

    result = _run_merge_scan(
        policy,
        Ledger(),
        github=GitHub(),
        kanban=object(),
    )

    assert result["processed"] == 1
    assert result["merged"] == [
        {
            "pr_number": 149,
            "head_sha": "a" * 40,
            "method": "squash",
            "merge_commit_oid": "c" * 40,
        }
    ]


def test_doctor_read_only_verifies_every_runtime_dependency(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from github_pr_feedback.cli import DoctorProbe, _doctor

    repository = tmp_path / "repository"
    subprocess.run(["git", "init", "--quiet", str(repository)], check=True)
    profile_root = tmp_path / "profile"
    (profile_root / "kanban" / "boards" / "repairs").mkdir(parents=True)
    (profile_root / "kanban" / "boards" / "repairs" / "board.json").write_text(
        '{"slug":"repairs"}\n', encoding="utf-8"
    )
    (profile_root / "profiles" / "repair-agent").mkdir(parents=True)
    (profile_root / "profiles" / "repair-agent" / "config.yaml").write_text(
        "profile: repair-agent\n", encoding="utf-8"
    )
    (profile_root / "profiles" / "pr-local-ci-auditor").mkdir(parents=True)
    (profile_root / "profiles" / "pr-local-ci-auditor" / "config.yaml").write_text(
        "profile: pr-local-ci-auditor\n", encoding="utf-8"
    )
    ledger_path = profile_root / "github-pr-feedback" / "ledger.sqlite3"
    settings = enabled_settings(repository)
    settings["local_ci_audit"] = {
        "enabled": True,
        "assignee": "pr-local-ci-auditor",
        "post_results": True,
    }
    context = RecordingContext(settings)
    responses = {
        ("/opt/tools/gh", "auth", "status", "--hostname", "github.com"): (0, ""),
        ("/opt/tools/hermes", "--version"): (0, "Hermes Agent test"),
        ("/opt/tools/git", "-C", str(repository), "rev-parse", "--show-toplevel"): (
            0,
            f"{repository}\n",
        ),
        ("/opt/tools/git", "-C", str(repository), "rev-parse", "--git-common-dir"): (
            0,
            ".git\n",
        ),
        ("/opt/tools/git", "-C", str(repository), "worktree", "list", "--porcelain"): (
            0,
            f"worktree {repository}\n",
        ),
    }
    runner = RecordingDoctorRunner(responses)
    before = profile_snapshot(profile_root)

    exit_code = _doctor(
        context,
        probe=DoctorProbe(profile_root, runner),
        ledger_path=ledger_path,
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "ready"
    assert payload["checks"] == {
        "assignee": "ok",
        "board": "ok",
        "gh_auth": "ok",
        "gh_executable": "ok",
        "hermes_executable": "ok",
        "ledger_access": "ok",
        "repository_worktree": "ok",
    }
    assert runner.calls == [
        ["/opt/tools/gh", "auth", "status", "--hostname", "github.com"],
        ["/opt/tools/hermes", "--version"],
        ["/opt/tools/git", "-C", str(repository), "rev-parse", "--show-toplevel"],
        ["/opt/tools/git", "-C", str(repository), "rev-parse", "--git-common-dir"],
        ["/opt/tools/git", "-C", str(repository), "worktree", "list", "--porcelain"],
    ]
    assert profile_snapshot(profile_root) == before


def test_doctor_reports_degraded_but_still_runs_all_read_only_checks(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from github_pr_feedback.cli import DoctorProbe, _doctor

    repository = tmp_path / "repository"
    subprocess.run(["git", "init", "--quiet", str(repository)], check=True)
    profile_root = tmp_path / "profile"
    profile_root.mkdir()
    ledger_path = profile_root / "github-pr-feedback" / "ledger.sqlite3"
    responses = {
        ("/opt/tools/gh", "auth", "status", "--hostname", "github.com"): (1, ""),
        ("/opt/tools/hermes", "--version"): (0, "Hermes Agent test"),
        ("/opt/tools/git", "-C", str(repository), "rev-parse", "--show-toplevel"): (
            0,
            f"{repository}\n",
        ),
        ("/opt/tools/git", "-C", str(repository), "rev-parse", "--git-common-dir"): (
            0,
            ".git\n",
        ),
        ("/opt/tools/git", "-C", str(repository), "worktree", "list", "--porcelain"): (
            0,
            f"worktree {repository}\n",
        ),
    }
    runner = RecordingDoctorRunner(responses)

    exit_code = _doctor(
        RecordingContext(enabled_settings(repository)),
        probe=DoctorProbe(profile_root, runner),
        ledger_path=ledger_path,
    )

    assert exit_code == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "degraded"
    assert payload["checks"]["gh_auth"] == "failed"
    assert set(payload["checks"]) == {
        "assignee",
        "board",
        "gh_auth",
        "gh_executable",
        "hermes_executable",
        "ledger_access",
        "repository_worktree",
    }
    assert len(runner.calls) == 5


def test_doctor_requires_the_configured_local_ci_auditor_profile(
    tmp_path: Path,
) -> None:
    from github_pr_feedback.cli import DoctorProbe
    from github_pr_feedback.policy import load_policy

    repository = tmp_path / "repository"
    subprocess.run(["git", "init", "--quiet", str(repository)], check=True)
    profile_root = tmp_path / "profile"
    (profile_root / "profiles" / "repair-agent").mkdir(parents=True)
    (profile_root / "profiles" / "repair-agent" / "config.yaml").write_text(
        "profile: repair-agent\n", encoding="utf-8"
    )
    settings = enabled_settings(repository)
    settings["local_ci_audit"] = {
        "enabled": True,
        "assignee": "pr-local-ci-auditor",
        "post_results": True,
    }
    policy = load_policy(settings)
    responses = {
        ("/opt/tools/gh", "auth", "status", "--hostname", "github.com"): (0, ""),
        ("/opt/tools/hermes", "--version"): (0, "Hermes Agent test"),
        ("/opt/tools/git", "-C", str(repository), "rev-parse", "--show-toplevel"): (
            0,
            f"{repository}\n",
        ),
        ("/opt/tools/git", "-C", str(repository), "rev-parse", "--git-common-dir"): (
            0,
            ".git\n",
        ),
        ("/opt/tools/git", "-C", str(repository), "worktree", "list", "--porcelain"): (
            0,
            f"worktree {repository}\n",
        ),
    }
    probe = DoctorProbe(profile_root, RecordingDoctorRunner(responses))

    checks = probe.checks(policy, profile_root / "ledger.sqlite3")

    assert checks["assignee"] == "failed"


def test_retry_passes_the_exact_immutable_receipt_to_controller_revalidation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from github_pr_feedback import cli
    from github_pr_feedback.controller import ScanResult

    seen: list[FeedbackReceipt] = []

    class RevalidatingController:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        def retry_failed(self, receipt: FeedbackReceipt) -> ScanResult:
            seen.append(receipt)
            return ScanResult(1, {})

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    monkeypatch.setattr(cli, "ScanController", RevalidatingController)
    context = RecordingContext({"enabled": False})
    parser = argparse.ArgumentParser()
    cli.setup_cli(context, parser)

    exit_code = cli.handle_cli_with_context(
        context,
        parser.parse_args(
            [
                "retry",
                "--repository",
                "acme/widgets",
                "--pr-number",
                "17",
                "--feedback-kind",
                "review_comment",
                "--feedback-id",
                "120",
                "--head-sha",
                "a" * 40,
            ]
        ),
    )

    assert exit_code == 0
    assert seen == [
        FeedbackReceipt("acme/widgets", 17, "review_comment", "120", "a" * 40)
    ]
    assert json.loads(capsys.readouterr().out) == {
        "created": 1,
        "skipped": {},
        "status": "ok",
    }


@pytest.mark.parametrize("action", ["scan", "retry"])
def test_scan_and_retry_exit_nonzero_and_report_degraded_on_incomplete_work(
    action: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from github_pr_feedback import cli
    from github_pr_feedback.controller import ScanResult

    class DegradedController:
        def scan(self, *, apply_labels: bool) -> ScanResult:
            assert apply_labels is False
            return ScanResult(0, {"github_error": 1}, degraded=True)

        def apply_agent_labels(self):
            return {"status": "ok", "updated": 0, "skipped": {}}

        def retry_failed(self, _receipt: FeedbackReceipt) -> ScanResult:
            return ScanResult(0, {"dispatch_failed": 1}, degraded=True)

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    monkeypatch.setattr(
        cli, "_controller", lambda _policy, _ledger: DegradedController()
    )
    parser = argparse.ArgumentParser()
    cli.setup_cli(RecordingContext({"enabled": False}), parser)
    argv = [action]
    if action == "retry":
        argv.extend(
            [
                "--repository",
                "acme/widgets",
                "--pr-number",
                "17",
                "--feedback-kind",
                "review_comment",
                "--feedback-id",
                "120",
                "--head-sha",
                "a" * 40,
            ]
        )

    exit_code = cli.handle_cli_with_context(
        RecordingContext({"enabled": False}),
        parser.parse_args(argv),
    )

    assert exit_code == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "degraded"
    assert payload["created"] == 0


def test_doctor_fails_closed_for_an_incomplete_enabled_configuration(
    capsys: pytest.CaptureFixture[str],
) -> None:
    context = RecordingContext({"enabled": True})
    _plugin_module().register(context)
    parser = argparse.ArgumentParser()
    context.calls[0]["setup_fn"](parser)

    exit_code = context.calls[0]["handler_fn"](parser.parse_args(["doctor"]))

    assert exit_code == 1
    assert json.loads(capsys.readouterr().out) == {"status": "invalid_configuration"}
    assert context.config_reads == [
        "enabled",
        "repositories",
        "reviewer_logins",
        "reviewer_associations",
        "include_self_feedback",
        "include_bot_feedback",
        "auto_dispatch",
        "assignee_rules",
        "routing_rules",
        "local_ci_audit",
        "agent_labels",
        "merge_maintainer",
        "repair_steward",
        "release_maintenance",
        "not_before",
        "assignee",
        "board",
    ]


def test_scan_fails_closed_for_an_incomplete_enabled_configuration(
    capsys: pytest.CaptureFixture[str],
) -> None:
    context = RecordingContext({"enabled": True})
    _plugin_module().register(context)
    parser = argparse.ArgumentParser()
    context.calls[0]["setup_fn"](parser)

    exit_code = context.calls[0]["handler_fn"](parser.parse_args(["scan"]))

    assert exit_code == 1
    assert json.loads(capsys.readouterr().out) == {"status": "invalid_configuration"}


def test_real_hermes_discovers_temp_profile_plugin_and_dry_scan_never_invokes_gh(
    tmp_path: Path,
) -> None:
    hermes = shutil.which("hermes")
    if hermes is None:
        pytest.skip("Hermes executable is not installed on this host")
    plugin_root = Path(__file__).resolve().parents[1]
    profile = tmp_path / "profile"
    installed = profile / "plugins" / "github-pr-feedback"
    shutil.copytree(
        plugin_root,
        installed,
        ignore=shutil.ignore_patterns(
            ".git",
            ".superpowers",
            ".pytest_cache",
            ".ruff_cache",
            "__pycache__",
            "tests",
        ),
    )
    profile.mkdir(exist_ok=True)
    (profile / "config.yaml").write_text(
        "plugins:\n"
        "  enabled:\n"
        "    - github-pr-feedback\n"
        "  entries:\n"
        "    github-pr-feedback:\n"
        "      settings:\n"
        "        enabled: false\n",
        encoding="utf-8",
    )
    fake_bin = tmp_path / "fake-bin"
    fake_bin.mkdir()
    marker = tmp_path / "gh-was-invoked"
    fake_gh = fake_bin / "gh"
    fake_gh.write_text(
        f"#!/bin/sh\nprintf invoked > {marker}\nexit 97\n",
        encoding="utf-8",
    )
    fake_gh.chmod(0o755)
    environment = os.environ.copy()
    environment.update(
        {
            "HERMES_HOME": str(profile),
            "HOME": str(tmp_path / "home"),
            "PATH": f"{fake_bin}:{environment.get('PATH', '')}",
        }
    )
    for name in ("GH_TOKEN", "GITHUB_TOKEN", "HERMES_PROFILE"):
        environment.pop(name, None)

    completed = subprocess.run(
        [hermes, "github-pr-feedback", "scan"],
        check=False,
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout) == {"created": 0, "skipped": {}, "status": "ok"}
    assert marker.exists() is False


def test_real_hermes_registers_the_fixed_card_as_blocked_dir_workspace_without_skills(
    tmp_path: Path,
) -> None:
    from github_pr_feedback.cli import _kanban_create_argv

    hermes = shutil.which("hermes")
    if hermes is None:
        pytest.skip("Hermes executable is not installed on this host")
    profile = tmp_path / "profile"
    workspace = tmp_path / "exact-head-worktree"
    workspace.mkdir()
    environment = os.environ.copy()
    environment.update({"HERMES_HOME": str(profile), "HOME": str(tmp_path / "home")})
    for name in (
        "GH_TOKEN",
        "GITHUB_TOKEN",
        "HERMES_PROFILE",
        "HERMES_KANBAN_DB",
        "HERMES_KANBAN_BOARD",
        "HERMES_KANBAN_WORKSPACES_ROOT",
    ):
        environment.pop(name, None)
    board = subprocess.run(
        [hermes, "kanban", "boards", "create", "repairs"],
        check=False,
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert board.returncode == 0, board.stderr
    task = replace(kanban_task(), repository_path=workspace)
    argv = _kanban_create_argv(task)

    created = subprocess.run(
        [hermes, *argv[1:]],
        check=False,
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert created.returncode == 0, created.stderr
    payload = json.loads(created.stdout)
    assert payload["status"] == "blocked"
    assert payload["workspace_kind"] == "dir"
    assert payload["workspace_path"] == str(workspace)
    assert payload["skills"] == []
    assert payload["started_at"] is None
    assert payload["session_id"] is None


def test_cron_wrapper_invokes_only_the_fixed_scan_argv_with_an_absolute_hermes_executable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    script_path = (
        Path(__file__).resolve().parents[1] / "scripts" / "github-pr-feedback-scan.py"
    )
    spec = importlib.util.spec_from_file_location(
        "github_pr_feedback_cron", script_path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    calls: list[tuple[list[str], bool]] = []

    class Completed:
        returncode = 0
        stdout = '{"status":"ok"}\n'
        stderr = ""

    def run(argv: list[str], *, check: bool, capture_output: bool, text: bool) -> Completed:
        assert capture_output is True
        assert text is True
        calls.append((argv, check))
        return Completed()

    monkeypatch.setattr(module.subprocess, "run", run)
    executable = tmp_path / "bin" / "hermes"
    executable.parent.mkdir()
    executable.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    executable.chmod(0o755)
    monkeypatch.setenv("HERMES_EXECUTABLE", str(executable))

    assert module.main() == 0
    assert calls == [([str(executable), "github-pr-feedback", "scan"], False)]


def test_cron_wrapper_reports_missing_child_stdout_as_a_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    script_path = (
        Path(__file__).resolve().parents[1] / "scripts" / "github-pr-feedback-scan.py"
    )
    spec = importlib.util.spec_from_file_location(
        "github_pr_feedback_cron_missing_output", script_path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    class Completed:
        returncode = 0
        stdout = ""
        stderr = "child exited without scan JSON\n"

    monkeypatch.setattr(module.subprocess, "run", lambda *args, **kwargs: Completed())
    executable = tmp_path / "bin" / "hermes"
    executable.parent.mkdir()
    executable.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    executable.chmod(0o755)
    monkeypatch.setenv("HERMES_EXECUTABLE", str(executable))

    assert module.main() == 1
    assert json.loads(capsys.readouterr().out) == {
        "child_returncode": 0,
        "status": "worker_output_missing",
        "stderr_present": True,
    }


@pytest.mark.parametrize(
    ("payload", "process_returncode", "expected"),
    [
        (
            {
                "status": "ok",
                "repair": {"status": "degraded"},
                "merge": {"status": "ok"},
                "release_maintenance": {"status": "waiting_open_prs"},
            },
            1,
            0,
        ),
        (
            {
                "status": "degraded",
                "repair": {"status": "degraded"},
                "merge": {
                    "status": "degraded",
                    "merged": [
                        {
                            "pr_number": 149,
                            "head_sha": "abc123",
                            "method": "squash",
                            "merge_commit_oid": "def456",
                        }
                    ],
                    "blocked": {"133": ["ci_receipt_not_passing"]},
                },
                "release_maintenance": {"status": "degraded"},
            },
            1,
            0,
        ),
        ({"status": "degraded", "repair": {"status": "ok"}}, 1, 1),
        ({"status": "ok", "merge": {"status": "degraded"}}, 1, 1),
        ({"status": "degraded", "merge": {"merged": []}}, 1, 1),
        ({"status": "degraded", "merge": {"merged": "149"}}, 1, 1),
        ({"status": "degraded", "merge": {"merged": [True]}}, 1, 1),
        ({"status": "degraded", "merge": {"merged": [{}]}}, 1, 1),
    ],
)
def test_cron_wrapper_classifies_successful_merges_as_partial_success(
    payload: dict[str, object],
    process_returncode: int,
    expected: int,
) -> None:
    script_path = (
        Path(__file__).resolve().parents[1] / "scripts" / "github-pr-feedback-scan.py"
    )
    spec = importlib.util.spec_from_file_location(
        "github_pr_feedback_cron_status", script_path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert module._cron_exit_code(json.dumps(payload), process_returncode) == expected


def test_cron_wrapper_keeps_completed_scan_green_for_ci_metadata_gap() -> None:
    script_path = (
        Path(__file__).resolve().parents[1] / "scripts" / "github-pr-feedback-scan.py"
    )
    spec = importlib.util.spec_from_file_location(
        "github_pr_feedback_cron_status_soft_gap", script_path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    payload = {
        "status": "degraded",
        "skipped": {"duplicate": 189, "github_ci_state_unavailable": 28},
        "repair": {"status": "ok"},
        "merge": {"status": "ok"},
    }
    assert module._cron_exit_code(json.dumps(payload), 1) == 0


@pytest.mark.parametrize("configured", [None, "hermes", "/missing/hermes"])
def test_cron_wrapper_fails_cleanly_without_an_executable_absolute_hermes_path(
    configured: str | None,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    script_path = (
        Path(__file__).resolve().parents[1] / "scripts" / "github-pr-feedback-scan.py"
    )
    spec = importlib.util.spec_from_file_location(
        "github_pr_feedback_cron_invalid", script_path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if configured is None:
        monkeypatch.delenv("HERMES_EXECUTABLE", raising=False)
    else:
        monkeypatch.setenv("HERMES_EXECUTABLE", configured)
    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda *_args, **_kwargs: pytest.fail(
            "invalid scheduler configuration invoked Hermes"
        ),
    )

    assert module.main() == 127
    assert (
        "HERMES_EXECUTABLE must name an executable absolute path"
        in capsys.readouterr().err
    )


def _claim(
    ledger: FeedbackLedger, receipt: FeedbackReceipt, *, owner: str = "test-scanner"
):
    now = datetime(2026, 8, 24, 12, 0, tzinfo=UTC)
    return ledger.claim(
        receipt,
        owner=owner,
        claimed_at=now,
        stale_before=now - timedelta(minutes=5),
    )


def enabled_settings(repository: Path) -> dict[str, object]:
    return {
        "enabled": True,
        "repositories": [
            {
                "base_repository": "acme/widgets",
                "head_repository": "acme/widgets",
                "local_path": str(repository),
                "owner_login": "owner",
                "branch_prefixes": ["codex/"],
            }
        ],
        "reviewer_logins": ["reviewer"],
        "reviewer_associations": [],
        "not_before": "2026-08-24T00:00:00Z",
        "assignee": "repair-agent",
        "board": "repairs",
    }


def profile_snapshot(root: Path) -> dict[str, tuple[int, int]]:
    return {
        str(path.relative_to(root)): (path.stat().st_size, path.stat().st_mtime_ns)
        for path in root.rglob("*")
        if path.is_file()
    }


def test_ci_audit_handoff_completes_current_task_without_waiting_for_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from github_pr_feedback.ci_runner import (
        CIAuditIdentity,
        CIAuditReceipt,
        CheckState,
    )
    from github_pr_feedback.cli import _complete_current_ci_task

    receipt = CIAuditReceipt(
        receipt_id="r" * 64,
        identity=CIAuditIdentity("acme/widgets", 17, "a" * 40, "b" * 40),
        manifest_digest="m" * 64,
        status="failed",
        started_at=datetime(2026, 8, 25, tzinfo=UTC),
        completed_at=datetime(2026, 8, 25, tzinfo=UTC),
        actions_state=CheckState(False, True, 0),
        commands=(),
    )
    calls: list[tuple[list[str], dict[str, object]]] = []

    class Completed:
        returncode = 0

    def run(argv: list[str], **kwargs: object) -> Completed:
        calls.append((argv, kwargs))
        return Completed()

    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_exact")
    monkeypatch.setenv("HERMES_KANBAN_BOARD", "repairs")
    monkeypatch.setattr("github_pr_feedback.cli.subprocess.run", run)

    _complete_current_ci_task(receipt)

    assert calls[0][0] == [
        sys.executable,
        "-m",
        "hermes_cli.main",
        "kanban",
        "--board",
        "repairs",
        "complete",
        "t_exact",
        "--result",
        f"Exact-head local CI receipt {'r' * 64}: failed.",
    ]


def test_ci_audit_handoff_classifies_missing_hermes_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from github_pr_feedback.ci_runner import CIAuditIdentity, CIAuditReceipt, CheckState
    from github_pr_feedback.cli import _complete_current_ci_task

    receipt = CIAuditReceipt(
        receipt_id="r" * 64,
        identity=CIAuditIdentity("acme/widgets", 17, "a" * 40, "b" * 40),
        manifest_digest="m" * 64,
        status="failed",
        started_at=datetime(2026, 8, 25, tzinfo=UTC),
        completed_at=datetime(2026, 8, 25, tzinfo=UTC),
        actions_state=CheckState(False, True, 0),
        commands=(),
    )
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_exact")
    monkeypatch.setattr(
        "github_pr_feedback.cli.subprocess.run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            FileNotFoundError("missing runtime")
        ),
    )

    with pytest.raises(RuntimeError, match="Hermes runtime unavailable"):
        _complete_current_ci_task(receipt)


def test_failed_audit_handoff_dispatches_the_typed_receipt_before_completion(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from github_pr_feedback.ci_runner import (
        CIAuditIdentity,
        CIAuditReceipt,
        CommandEvidence,
    )
    from github_pr_feedback.cli import _audit_pr
    from github_pr_feedback.github_client import CheckState, PullRequestMergeState

    head_sha = "a" * 40
    base_sha = "b" * 40
    repository = tmp_path / "repository"
    repository.mkdir()
    manifest = repository / "tests" / "manifests" / "test_lanes.toml"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(
        "[lanes.fast]\nci_status = 'required'\nargv = ['pytest']\n",
        encoding="utf-8",
    )
    settings = enabled_settings(Path(__file__).resolve().parents[3])
    settings["auto_dispatch"] = True
    settings["routing_rules"] = [
        {
            "assignee": "ci-static-fixer",
            "precedence": 150,
            "match_any": ["static lane"],
            "match_labels_any": ["ci/static"],
            "tags": ["type/ci", "ci/static"],
            "priority": "P2",
            "blast_radius": "contained",
            "risks": [],
            "requires_review": False,
        }
    ]
    settings["local_ci_audit"] = {
        "enabled": True,
        "assignee": "pr-local-ci-auditor",
        "post_results": True,
    }
    receipt = CIAuditReceipt(
        receipt_id="f" * 64,
        identity=CIAuditIdentity("acme/widgets", 17, base_sha, head_sha),
            manifest_digest=hashlib.sha256(manifest.read_bytes()).hexdigest(),
        status="failed",
        started_at=datetime(2026, 8, 25, 12, 0, tzinfo=UTC),
        completed_at=datetime(2026, 8, 25, 12, 1, tzinfo=UTC),
        actions_state=CheckState(False, True, 0),
        commands=(
            CommandEvidence(
                argv=(".venv/bin/python", "scripts/run_static_lane.py"),
                cwd=".",
                returncode=1,
                duration_ms=1,
                timed_out=False,
                stdout_sha256="0" * 64,
                stderr_sha256="0" * 64,
                classification="logic-regression",
            ),
        ),
    )

    class GitHub:
        def get_merge_state(self, repository: str, pr_number: int):
            return PullRequestMergeState(
                repository=repository,
                number=pr_number,
                state="OPEN",
                is_draft=False,
                merged=False,
                mergeable=True,
                merge_state_status="CLEAN",
                base_branch="main",
                base_sha=base_sha,
                head_repository=repository,
                author_login="owner",
                head_ref_name="codex/fix",
                head_sha=head_sha,
                merge_commit_oid=None,
            )

        def list_feedback(self, _repository: str, _pr_number: int):
            return ()

        def post_issue_comment(self, _repository: str, _pr_number: int, _body: str):
            return None

    class Runner:
        def __init__(self, _github: object, _ledger: object) -> None:
            pass

        def run(self, _identity: object, _worktree: Path) -> CIAuditReceipt:
            return receipt

    class Ledger:
        def close(self) -> None:
            pass

    dispatched: list[CIAuditReceipt] = []

    class Controller:
        def dispatch_ci_failure(self, audit: CIAuditReceipt) -> str:
            dispatched.append(audit)
            return "rejected"

    monkeypatch.setattr("github_pr_feedback.cli.GitHubClient", GitHub)
    monkeypatch.setattr("github_pr_feedback.cli.LocalCIRunner", Runner)
    monkeypatch.setattr(
        "github_pr_feedback.cli.FeedbackLedger.for_current_profile", lambda: Ledger()
    )
    monkeypatch.setattr(
        "github_pr_feedback.cli._controller", lambda _policy, _ledger: Controller()
    )
    monkeypatch.setattr("github_pr_feedback.cli._complete_current_ci_task", lambda _receipt: None)
    monkeypatch.setattr("github_pr_feedback.cli._terminate_current_ci_worker", lambda: None)

    result = _audit_pr(
        RecordingContext(settings),
        argparse.Namespace(
            repository="acme/widgets",
            pr_number=17,
            head_sha=head_sha,
            worktree=str(repository),
        ),
    )

    assert result == 1
    assert dispatched == [receipt]
    rendered = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert rendered == [
        {
            "command_count": 1,
            "head_sha": head_sha,
            "manifest_digest": receipt.manifest_digest,
            "pr_number": 17,
            "receipt_id": receipt.receipt_id,
            "repository": "acme/widgets",
            "status": "failed",
            "handoff_status": "pending",
        },
        {
            "handoff_reason": "rejected",
            "repair_status": "rejected",
            "receipt_id": receipt.receipt_id,
            "retryable": True,
            "status": "audit_handoff_retryable",
        }
    ]


def test_audit_handoff_exception_renders_retryable_reason(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from github_pr_feedback.ci_runner import CIAuditIdentity, CIAuditReceipt
    from github_pr_feedback.cli import _audit_pr
    from github_pr_feedback.github_client import CheckState, PullRequestMergeState

    head_sha = "a" * 40
    base_sha = "b" * 40
    repository = tmp_path / "repository"
    repository.mkdir()
    settings = enabled_settings(Path(__file__).resolve().parents[3])
    settings["local_ci_audit"] = {
        "enabled": True,
        "assignee": "pr-local-ci-auditor",
        "post_results": True,
    }
    receipt = CIAuditReceipt(
        receipt_id="f" * 64,
        identity=CIAuditIdentity("acme/widgets", 17, base_sha, head_sha),
        manifest_digest="e" * 64,
        status="failed",
        started_at=datetime(2026, 8, 25, 12, 0, tzinfo=UTC),
        completed_at=datetime(2026, 8, 25, 12, 1, tzinfo=UTC),
        actions_state=CheckState(False, True, 0),
        commands=(),
    )

    class GitHub:
        def get_merge_state(self, repository: str, pr_number: int):
            return PullRequestMergeState(
                repository=repository,
                number=pr_number,
                state="OPEN",
                is_draft=False,
                mergeable=True,
                merge_state_status="CLEAN",
                base_branch="main",
                base_sha=base_sha,
                head_repository=repository,
                author_login="owner",
                head_ref_name="codex/fix",
                head_sha=head_sha,
                merged=False,
                merge_commit_oid=None,
            )

        def list_feedback(self, _repository: str, _pr_number: int):
            return ()

        def post_issue_comment(self, _repository: str, _pr_number: int, _body: str):
            return None

    class Ledger:
        def close(self) -> None:
            pass

    class Controller:
        def dispatch_ci_failure(self, _audit: CIAuditReceipt) -> str:
            raise RuntimeError("repair dispatch crashed")

    monkeypatch.setattr("github_pr_feedback.cli.GitHubClient", GitHub)
    monkeypatch.setattr(
        "github_pr_feedback.cli.FeedbackLedger.for_current_profile", lambda: Ledger()
    )
    monkeypatch.setattr(
        "github_pr_feedback.cli._controller", lambda _policy, _ledger: Controller()
    )
    monkeypatch.setattr(
        "github_pr_feedback.cli._run_grouped_exact_head_audit",
        lambda *_args, **_kwargs: receipt,
    )
    monkeypatch.setattr("github_pr_feedback.cli._block_current_ci_task", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("github_pr_feedback.cli._terminate_current_ci_worker", lambda: None)

    result = _audit_pr(
        RecordingContext(settings),
        argparse.Namespace(
            repository="acme/widgets",
            pr_number=17,
            head_sha=head_sha,
            worktree=str(repository),
        ),
    )

    assert result == 1
    rendered = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert rendered[-1] == {
        "handoff_reason": "repair dispatch crashed",
        "receipt_id": receipt.receipt_id,
        "retryable": True,
        "status": "audit_handoff_retryable",
    }


def test_blocked_merge_handoff_blocks_task_with_exact_blockers(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from github_pr_feedback.ci_runner import CIAuditIdentity, CIAuditReceipt
    from github_pr_feedback.cli import _audit_pr
    from github_pr_feedback.github_client import CheckState, PullRequestMergeState

    head_sha = "a" * 40
    base_sha = "b" * 40
    repository = tmp_path / "repository"
    repository.mkdir()
    subprocess.run(["git", "init", "--quiet", str(repository)], check=True)
    manifest = repository / "tests" / "manifests" / "test_lanes.toml"
    manifest.parent.mkdir(parents=True)
    manifest.write_text("[lanes.fast]\nargv = ['pytest']\n", encoding="utf-8")
    settings = enabled_settings(repository)
    settings["local_ci_audit"] = {
        "enabled": True,
        "assignee": "pr-local-ci-auditor",
        "post_results": False,
    }
    receipt = CIAuditReceipt(
        receipt_id="p" * 64,
        identity=CIAuditIdentity("acme/widgets", 17, base_sha, head_sha),
        manifest_digest=hashlib.sha256(manifest.read_bytes()).hexdigest(),
        status="passed",
        started_at=datetime(2026, 8, 25, 12, 0, tzinfo=UTC),
        completed_at=datetime(2026, 8, 25, 12, 1, tzinfo=UTC),
        actions_state=CheckState(False, True, 0),
        commands=(),
    )

    class GitHub:
        def get_merge_state(self, _repository: str, _pr_number: int):
            return PullRequestMergeState(
                repository="acme/widgets",
                number=17,
                state="OPEN",
                is_draft=False,
                merged=False,
                mergeable=True,
                merge_state_status="CLEAN",
                base_branch="main",
                base_sha=base_sha,
                head_repository="acme/widgets",
                author_login="owner",
                head_ref_name="codex/fix",
                head_sha=head_sha,
                merge_commit_oid=None,
            )

    class Ledger:
        def close(self) -> None:
            pass

    blocked: list[tuple[CIAuditReceipt, list[str]]] = []
    completed: list[CIAuditReceipt] = []
    monkeypatch.setattr("github_pr_feedback.cli.GitHubClient", GitHub)
    monkeypatch.setattr(
        "github_pr_feedback.cli.FeedbackLedger.for_current_profile", lambda: Ledger()
    )
    monkeypatch.setattr(
        "github_pr_feedback.cli._run_grouped_exact_head_audit",
        lambda *_args, **_kwargs: receipt,
    )
    monkeypatch.setattr(
        "github_pr_feedback.cli._run_single_pr_merge_handoff",
        lambda *_args, **_kwargs: {
            "status": "blocked",
            "blockers": ["ci_receipt_not_passing", "review_required"],
        },
    )
    monkeypatch.setattr(
        "github_pr_feedback.cli._block_current_ci_task",
        lambda audit, blockers: blocked.append((audit, blockers)),
    )
    monkeypatch.setattr(
        "github_pr_feedback.cli._complete_current_ci_task",
        lambda audit: completed.append(audit),
    )

    result = _audit_pr(
        RecordingContext(settings),
        argparse.Namespace(
            repository="acme/widgets",
            pr_number=17,
            head_sha=head_sha,
            worktree=str(repository),
        ),
    )

    assert result == 1
    assert blocked == [(receipt, ["ci_receipt_not_passing", "review_required"])]
    assert completed == []
    rendered = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert rendered[-1] == {
        "blockers": ["ci_receipt_not_passing", "review_required"],
        "receipt_id": receipt.receipt_id,
        "retryable": False,
        "status": "audit_handoff_blocked",
    }


def test_audit_reuses_exact_head_manifest_receipt_without_rerunning_lane(
    tmp_path: Path,
) -> None:
    from github_pr_feedback.ci_runner import CIAuditIdentity, CIAuditReceipt
    from github_pr_feedback.cli import _reusable_ci_receipt
    from github_pr_feedback.github_client import CheckState

    worktree = tmp_path / "repository"
    manifest = worktree / "tests/manifests/test_lanes.toml"
    manifest.parent.mkdir(parents=True)
    manifest.write_text("[lane.fast]\nargv = ['pytest']\n", encoding="utf-8")
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    identity = CIAuditIdentity("acme/widgets", 17, "b" * 40, "a" * 40)
    receipt = CIAuditReceipt(
        receipt_id="f" * 64,
        identity=identity,
        manifest_digest=digest,
        status="failed",
        started_at=datetime(2026, 8, 25, 12, 0, tzinfo=UTC),
        completed_at=datetime(2026, 8, 25, 12, 1, tzinfo=UTC),
        actions_state=CheckState(False, True, 0),
        commands=(),
    )

    class Ledger:
        def latest_ci_receipt_for_head(
            self, repository: str, pr_number: int, head_sha: str
        ) -> CIAuditReceipt:
            assert (repository, pr_number, head_sha) == (
                identity.repository,
                identity.pr_number,
                identity.head_sha,
            )
            return receipt

    assert _reusable_ci_receipt(Ledger(), identity, worktree) is receipt
    assert _reusable_ci_receipt(Ledger(), identity, worktree, allow_reuse=False) is None

    manifest.write_text("[lane.changed]\nargv = ['pytest']\n", encoding="utf-8")
    assert _reusable_ci_receipt(Ledger(), identity, worktree) is None


def test_audit_reruns_when_canonical_base_advances_with_same_head(
    tmp_path: Path,
) -> None:
    from github_pr_feedback.ci_runner import CIAuditIdentity, CIAuditReceipt
    from github_pr_feedback.cli import _reusable_ci_receipt
    from github_pr_feedback.github_client import CheckState

    worktree = tmp_path / "repository"
    manifest = worktree / "tests/manifests/test_lanes.toml"
    manifest.parent.mkdir(parents=True)
    manifest.write_text("[lane.fast]\nargv = ['pytest']\n", encoding="utf-8")
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    current = CIAuditIdentity("acme/widgets", 17, "c" * 40, "a" * 40)
    stale = CIAuditReceipt(
        receipt_id="f" * 64,
        identity=CIAuditIdentity("acme/widgets", 17, "b" * 40, "a" * 40),
        manifest_digest=digest,
        status="passed",
        started_at=datetime(2026, 8, 25, 12, 0, tzinfo=UTC),
        completed_at=datetime(2026, 8, 25, 12, 1, tzinfo=UTC),
        actions_state=CheckState(False, True, 0),
        commands=(),
    )

    class Ledger:
        def latest_ci_receipt_for_head(self, *args):
            return stale

    assert _reusable_ci_receipt(Ledger(), current, worktree) is None


def test_ci_audit_handoff_terminates_only_a_task_scoped_parent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from github_pr_feedback.cli import _terminate_current_ci_worker

    signals: list[tuple[int, object]] = []
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    monkeypatch.setattr("github_pr_feedback.cli.os.getppid", lambda: 4321)
    monkeypatch.setattr(
        "github_pr_feedback.cli.os.kill", lambda pid, sig: signals.append((pid, sig))
    )

    _terminate_current_ci_worker()
    assert signals == []

    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_exact")
    _terminate_current_ci_worker()
    assert signals == [(4321, signal.SIGTERM)]


def _feedback_comment(body: str) -> Feedback:
    return Feedback(
        kind="issue_comment",
        feedback_id="1",
        reviewer=Reviewer("pr-repair-steward"),
        body=body,
        created_at=datetime(2026, 8, 28, 12, 0, tzinfo=UTC),
        is_bot=True,
    )


class _FakeGitHubComments:
    def __init__(self, bodies: list[str]) -> None:
        self._bodies = bodies
        self.posted: list[tuple[str, int, str]] = []

    def list_feedback(self, repository: str, number: int):
        return tuple(_feedback_comment(body) for body in self._bodies)

    def post_issue_comment(self, repository: str, number: int, body: str) -> None:
        self.posted.append((repository, number, body))


class _FakeGitHubCommentsUnavailable(_FakeGitHubComments):
    def list_feedback(self, repository: str, number: int):
        from github_pr_feedback.github_client import GitHubClientError

        raise GitHubClientError("boom")


def _repair_receipt(repository: str = "mrkillbob/luna-bot") -> FeedbackReceipt:
    return FeedbackReceipt(repository, 17, "pr_repair", "repair:actions_not_green", "a" * 40)


def _feedback_receipt(
    feedback_kind: str, repository: str = "mrkillbob/luna-bot"
) -> FeedbackReceipt:
    return FeedbackReceipt(repository, 17, feedback_kind, "123456", "a" * 40)


def test_factual_reply_is_required_for_ordinary_admitted_feedback_kinds_too() -> None:
    """Same gate as pr_repair: this used to only cover repair receipts, letting an

    ordinary review-comment-driven push complete with no trace a reply was skipped.
    """

    github = _FakeGitHubComments(["looks good to me"])

    for kind in ("issue_comment", "review_comment", "review"):
        assert _factual_reply_is_missing(
            github, _feedback_receipt(kind), resolved_head_sha="a" * 40
        )


def test_factual_reply_for_ordinary_feedback_is_satisfied_by_its_own_marker_kind() -> None:
    reply = (
        f"{pr_repair_attribution_line('runtime-correctness-steward')}\n"
        "Fixed the reported issue; focused tests pass.\n"
        f"<!-- pr-maintenance-receipt:v1 status=completed kind=review_comment head={'a' * 40} -->"
    )
    github = _FakeGitHubComments([reply])

    assert not _factual_reply_is_missing(
        github, _feedback_receipt("review_comment"), resolved_head_sha="a" * 40
    )


def test_factual_reply_for_ordinary_feedback_is_satisfied_by_an_earlier_receipts_marker() -> (
    None
):
    """A second admitted feedback item resolving to the same already-fixed head

    is satisfied by the earlier reply -- no duplicate comment required.
    """

    reply = (
        f"{pr_repair_attribution_line('runtime-correctness-steward')}\n"
        "Already fixed by an earlier receipt.\n"
        f"<!-- pr-maintenance-receipt:v1 status=completed kind=issue_comment head={'a' * 40} -->"
    )
    github = _FakeGitHubComments([reply])

    assert not _factual_reply_is_missing(
        github, _feedback_receipt("review_comment"), resolved_head_sha="a" * 40
    )


def test_factual_reply_is_missing_when_no_comment_carries_the_receipt_marker() -> None:
    github = _FakeGitHubComments(["looks good to me"])

    assert _factual_reply_is_missing(
        github, _repair_receipt(), resolved_head_sha="a" * 40
    )


def test_factual_reply_is_missing_when_the_marker_head_does_not_match() -> None:
    other_head_marker = (
        "Fixed it.\n"
        "<!-- pr-maintenance-receipt:v1 status=completed kind=pr_repair "
        f"head={'b' * 40} -->"
    )
    github = _FakeGitHubComments([other_head_marker])

    assert _factual_reply_is_missing(
        github, _repair_receipt(), resolved_head_sha="a" * 40
    )


def test_factual_reply_is_missing_on_our_repo_without_the_attribution_line() -> None:
    """Our repo requires self-identification -- the marker alone is not enough."""

    marker_only = (
        "Fixed the conflict.\n"
        f"<!-- pr-maintenance-receipt:v1 status=completed kind=pr_repair head={'a' * 40} -->"
    )
    github = _FakeGitHubComments([marker_only])

    assert _factual_reply_is_missing(
        github, _repair_receipt("mrkillbob/luna-bot"), resolved_head_sha="a" * 40
    )


def test_pr_repair_reply_is_satisfied_by_a_marker_and_attribution_on_our_repo() -> None:
    reply = (
        f"{pr_repair_attribution_line('pr-repair-steward')}\n"
        "Fixed the conflict; focused tests pass.\n"
        f"<!-- pr-maintenance-receipt:v1 status=completed kind=pr_repair head={'a' * 40} -->"
    )
    github = _FakeGitHubComments([reply])

    assert not _factual_reply_is_missing(
        github, _repair_receipt("mrkillbob/luna-bot"), resolved_head_sha="a" * 40
    )


def test_pr_repair_reply_on_the_upstream_repo_does_not_require_attribution() -> None:
    """NousResearch/hermes-agent stays brand-neutral -- the marker alone suffices."""

    marker_only = (
        "Fixed the conflict.\n"
        f"<!-- pr-maintenance-receipt:v1 status=completed kind=pr_repair head={'a' * 40} -->"
    )
    github = _FakeGitHubComments([marker_only])

    assert not _factual_reply_is_missing(
        github,
        _repair_receipt("NousResearch/hermes-agent"),
        resolved_head_sha="a" * 40,
    )


def test_pr_repair_reply_accepts_the_ci_failure_typed_fixers_own_marker_kind() -> None:
    """_ci_failure_task's own instructions require `kind=ci_repair`, not `kind=pr_repair`,

    even though its receipt's feedback_kind is 'pr_repair' like every other repair
    dispatch. Both must satisfy the same completion gate.
    """

    reply = (
        f"{pr_repair_attribution_line('ci-static-fixer')}\n"
        "Fixed the static-lane failure; focused tests pass.\n"
        f"<!-- pr-maintenance-receipt:v1 status=completed kind=ci_repair head={'a' * 40} -->"
    )
    github = _FakeGitHubComments([reply])

    assert not _factual_reply_is_missing(
        github, _repair_receipt("mrkillbob/luna-bot"), resolved_head_sha="a" * 40
    )


def _passed_ci_receipt(repository: str) -> CIAuditReceipt:
    return CIAuditReceipt(
        receipt_id="d" * 64,
        identity=CIAuditIdentity(repository, 17, "b" * 40, "a" * 40),
        manifest_digest="e" * 64,
        status="passed",
        started_at=datetime(2026, 8, 25, 12, 0, tzinfo=UTC),
        completed_at=datetime(2026, 8, 25, 12, 1, tzinfo=UTC),
        actions_state=CheckState(False, True, 0),
        commands=(),
    )


def test_ci_audit_comment_self_identifies_as_hermes_on_our_own_repository() -> None:
    body = _ci_audit_comment(_passed_ci_receipt("mrkillbob/luna-bot"))

    assert body.startswith("Hermes automated CI audit (pr-local-ci-auditor)")
    assert (
        f"<!-- pr-ci-receipt:v1 status=passed id={'d' * 64} head={'a' * 40} -->"
        in body
    )


def test_ci_audit_comment_stays_brand_neutral_on_the_upstream_repository() -> None:
    body = _ci_audit_comment(_passed_ci_receipt("NousResearch/hermes-agent"))

    assert "Hermes automated" not in body
    assert body.startswith("Addressed local CI audit")


def _codex_summary_body(status: str, sha: str) -> str:
    return (
        "<!-- codex-pull-request-review-summary -->\n\n"
        "| Review | Status | Commit | Review trigger |\n| --- | --- | --- | --- |\n"
        f"| Code Review | {status} "
        '<relative-time datetime="2026-08-25T20:00:00Z"></relative-time> | '
        f"`{sha}` | PR opened |"
    )


def _codex_feedback(body: str) -> Feedback:
    return Feedback(
        "issue_comment",
        "1",
        Reviewer("chatgpt-codex-connector[bot]", None),
        body,
        datetime(2026, 8, 25, 12, 0, tzinfo=UTC),
        True,
    )


class _FakeGitHubCodex(_FakeGitHubComments):
    def __init__(self, codex_feedback: tuple[Feedback, ...]) -> None:
        super().__init__([])
        self._codex_feedback = codex_feedback

    def list_feedback(self, repository: str, number: int):
        return self._codex_feedback


def test_retrigger_codex_review_mentions_codex_when_its_review_is_stale() -> None:
    head = "a" * 40
    github = _FakeGitHubCodex(
        (_codex_feedback(_codex_summary_body("Completed", ("f" * 40)[:7])),)
    )

    status = _retrigger_codex_review(github, "mrkillbob/luna-bot", 17, head)

    assert status == "triggered"
    assert github.posted == [
        ("mrkillbob/luna-bot", 17, codex_review_trigger_comment(head))
    ]


def test_retrigger_codex_review_is_a_noop_while_same_head_request_is_pending() -> None:
    head = "a" * 40
    github = _FakeGitHubCodex(
        (_codex_feedback(codex_review_trigger_comment(head)),)
    )

    status = _retrigger_codex_review(github, "mrkillbob/luna-bot", 17, head)

    assert status == "already_requested"
    assert github.posted == []


def test_retrigger_codex_review_is_a_noop_when_codex_already_reviewed_this_head() -> None:
    head = "a" * 40
    github = _FakeGitHubCodex((_codex_feedback(_codex_summary_body("Completed", head[:7])),))

    status = _retrigger_codex_review(github, "mrkillbob/luna-bot", 17, head)

    assert status == "already_current"
    assert github.posted == []


def test_retrigger_codex_review_reports_unavailable_without_raising() -> None:
    github = _FakeGitHubCommentsUnavailable([])

    status = _retrigger_codex_review(github, "mrkillbob/luna-bot", 17, "a" * 40)

    assert status == "unavailable"
    assert github.posted == []
