"""Regression test for t_04509175: a task with ``model_override`` but no
``provider_override`` must not spawn its worker with a bare ``-m`` and let
the model resolve against whatever provider the process environment happens
to default to.

Two real board tasks (t_e0885210, t_2cce4e0f) burned 10 worker runs that
died before their first model turn this way: ``set-model claude-sonnet-5``
with no ``--provider`` produced a worker pinned to ``claude-sonnet-5`` on
``openai-codex`` (the environment's ambient default), which 400s
immediately. ``_worker_argv`` now falls back to a static-catalog guess of
the model's owning provider when ``provider_override`` is empty.
"""
from __future__ import annotations

from unittest.mock import patch

from hermes_cli.kanban_db import Task
from hermes_cli.kanban_db_dispatch import _worker_argv


def _make_task(model_override, provider_override=None):
    return Task(
        id="t_test0001",
        title="test task",
        body=None,
        assignee="default",
        status="running",
        priority=0,
        created_by=None,
        created_at=0,
        started_at=None,
        completed_at=None,
        workspace_kind="scratch",
        workspace_path=None,
        claim_lock=None,
        claim_expires=None,
        tenant=None,
        model_override=model_override,
        provider_override=provider_override,
    )


def _build_argv(task):
    with patch(
        "hermes_cli.kanban_db_dispatch._resolve_hermes_argv",
        return_value=["hermes"],
    ), patch(
        "hermes_cli.kanban_db_dispatch._resolve_worker_cli_toolsets",
        return_value=None,
    ):
        return _worker_argv(task, "default", None)


def test_model_override_without_provider_resolves_static_catalog_provider():
    """The historical failure mode: `set-model claude-sonnet-5` with no
    `--provider` must not spawn a bare `-m` that inherits the ambient
    provider (observed: openai-codex, HTTP 400 before first turn)."""
    task = _make_task("claude-sonnet-5")
    argv = _build_argv(task)
    assert "-m" in argv and argv[argv.index("-m") + 1] == "claude-sonnet-5"
    assert "--provider" in argv
    assert argv[argv.index("--provider") + 1] == "anthropic"


def test_explicit_provider_override_still_wins():
    """An explicit provider_override is never replaced by the static guess."""
    task = _make_task("glm-5.3", provider_override="opencode-zen")
    argv = _build_argv(task)
    assert argv[argv.index("--provider") + 1] == "opencode-zen"


def test_unrecognized_model_leaves_no_provider_guess():
    """A model absent from every static catalog gets no fabricated
    --provider — old bare-`-m` behavior, so the worker's own startup guard
    reports a clear error instead of a confidently wrong guess."""
    task = _make_task("totally-unrecognized-model-xyz")
    argv = _build_argv(task)
    assert "-m" in argv
    assert "--provider" not in argv


def test_no_model_override_is_unaffected():
    """A task with neither override gets neither flag (unrelated to this bug)."""
    task = _make_task(None)
    argv = _build_argv(task)
    assert "-m" not in argv
    assert "--provider" not in argv


def test_detection_failure_falls_back_to_bare_dash_m():
    """If detect_static_provider_for_model raises for any reason, the worker
    still spawns (old behavior) instead of crashing the dispatcher."""
    task = _make_task("claude-sonnet-5")
    with patch(
        "hermes_cli.kanban_db_dispatch._resolve_hermes_argv",
        return_value=["hermes"],
    ), patch(
        "hermes_cli.kanban_db_dispatch._resolve_worker_cli_toolsets",
        return_value=None,
    ), patch(
        "hermes_cli.models.detect_static_provider_for_model",
        side_effect=RuntimeError("boom"),
    ):
        argv = _worker_argv(task, "default", None)
    assert "-m" in argv
    assert "--provider" not in argv
