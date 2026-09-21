"""Apply completion contracts before durable Kanban transitions."""

from __future__ import annotations

import os
import sqlite3
from contextlib import closing
from pathlib import Path


class CompletionPolicyError(ValueError):
    """A registered completion contract could not be satisfied."""


def _load_bundled_github_pr_feedback_guard():
    """Import ``github_pr_feedback.repair_completion_policy`` from the bundled
    plugin source, not the worker's own (possibly plugin-disabled) sys.path.

    A dispatched worker profile that doesn't enable ``github-pr-feedback`` never
    puts it on sys.path, so a bare ``import github_pr_feedback...`` here raises
    ModuleNotFoundError even though its control-plane receipt still needs this
    guard enforced. Load it directly from the bundled plugin directory (part of
    this same trusted Hermes source tree, unlike a profile-local override
    manifest) via its real package name so its relative imports resolve.
    """
    import sys

    module = sys.modules.get("github_pr_feedback")
    if module is None:
        import importlib.util

        from hermes_cli._startup_fast import project_root_str

        plugin_dir = Path(project_root_str()) / "plugins" / "github-pr-feedback" / "github_pr_feedback"
        spec = importlib.util.spec_from_file_location(
            "github_pr_feedback", plugin_dir / "__init__.py",
            submodule_search_locations=[str(plugin_dir)],
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load bundled github_pr_feedback from {plugin_dir}")
        module = importlib.util.module_from_spec(spec)
        sys.modules["github_pr_feedback"] = module
        spec.loader.exec_module(module)
    import importlib

    return importlib.import_module("github_pr_feedback.repair_completion_policy").guard_repair_completion


def enforce_completion_policies(*, task_id, board, assignee, summary):
    from hermes_cli.plugins import invoke_hook

    results = list(invoke_hook(
        "pre_kanban_complete", task_id=task_id, board=board,
        assignee=assignee, summary=summary,
    ))
    results.extend(_control_plane_github_feedback_results(task_id=task_id))
    _raise_on_policy_results(results)


def enforce_review_policies(*, task_id, board, assignee, summary):
    """Apply feedback contracts before handing a task to human review."""
    from hermes_cli.plugins import invoke_hook

    results = list(invoke_hook(
        "pre_kanban_review", task_id=task_id, board=board,
        assignee=assignee, summary=summary,
    ))
    results.extend(_control_plane_github_feedback_results(task_id=task_id))
    _raise_on_policy_results(results)


def _raise_on_policy_results(results):
    for result in results:
        if result is None:
            continue
        if not isinstance(result, dict) or result.get("action") not in {"block", "allow", "approve"}:
            raise CompletionPolicyError("Kanban completion policy returned an invalid decision")
        if result["action"] == "block":
            raise CompletionPolicyError(result.get("message") or "Kanban completion policy rejected this transition")


def _control_plane_github_feedback_results(*, task_id):
    """Run control-plane PR-feedback policy for dispatched workers with a task binding.

    Repair workers may run under a profile that intentionally does not enable the
    optional github-pr-feedback plugin.  Their durable dispatch receipt still
    lives in ``HERMES_CONTROL_HOME``; when that receipt binds this task, enforce
    the control-plane acknowledgement contract before allowing completion.
    """
    control_home = os.environ.get("HERMES_CONTROL_HOME", "").strip()
    worker_task = os.environ.get("HERMES_KANBAN_TASK", "").strip()
    if not control_home or not worker_task:
        return []
    ledger = Path(control_home) / "github-pr-feedback" / "ledger.sqlite3"
    if not ledger.exists():
        return []
    try:
        with closing(sqlite3.connect(ledger.resolve().as_uri() + "?mode=ro", uri=True, timeout=1)) as connection:
            table = connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='feedback_receipts'"
            ).fetchone()
            if not table:
                return []
            bound = connection.execute(
                "SELECT 1 FROM feedback_receipts WHERE task_id = ? AND feedback_kind IN "
                "('review_comment', 'issue_comment', 'review', 'pr_repair') "
                "AND NOT (feedback_kind = 'pr_repair' AND feedback_id LIKE 'report:%') LIMIT 1",
                (task_id,),
            ).fetchone()
            if not bound:
                return []
    except (OSError, sqlite3.Error, ValueError):
        return [{"action": "block", "message": "Kanban completion policy could not verify the control-plane GitHub PR feedback binding"}]
    try:
        guard_repair_completion = _load_bundled_github_pr_feedback_guard()
    except ImportError:
        return [{"action": "block", "message": "Kanban completion policy could not load the control-plane GitHub PR feedback guard"}]

    ctx = type("ControlPlaneFeedbackContext", (), {"get_config": staticmethod(lambda key, default=None: True if key == "enabled" else default)})()
    return [guard_repair_completion(ctx, task_id=task_id)]
