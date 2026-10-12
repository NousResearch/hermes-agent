"""A bare ``claude-*`` model override on an aggregator provider must reach the worker prefixed.

Live failure (Oct 2026): a card created with ``model_override="claude-sonnet-5.5"`` +
``provider_override="nous"`` spawned ``-m claude-sonnet-5.5 --provider nous``; nous serves only
``anthropic/claude-sonnet-5.5`` and answered HTTP 404 on the first turn. The CLI ``-m`` path does
not repair aggregator ids, so the kanban layer that pins the provider must.
"""

from __future__ import annotations

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_dispatch as dispatch
from hermes_cli.kanban_db import Task


def _task(**overrides) -> Task:
    base = dict(
        id="t_pfx1", title="x", body=None, assignee="worker", status="ready", priority=0,
        created_by=None, created_at=1, started_at=None, completed_at=None,
        workspace_kind="scratch", workspace_path=None, claim_lock=None, claim_expires=None,
        tenant=None,
    )
    base.update(overrides)
    return Task(**base)


def test_validate_prefixes_bare_id_for_aggregator():
    assert kb._validate_model_override("claude-sonnet-5.5", "nous") == (
        "anthropic/claude-sonnet-5.5", "nous")


def test_validate_leaves_prefixed_and_direct_provider_ids_alone():
    assert kb._validate_model_override("anthropic/claude-sonnet-5.5", "nous")[0] == (
        "anthropic/claude-sonnet-5.5")
    assert kb._validate_model_override("claude-sonnet-5", "anthropic")[0] == "claude-sonnet-5"
    assert kb._validate_model_override("claude-sonnet-5.5", None)[0] == "claude-sonnet-5.5"


def test_worker_argv_repairs_a_row_stored_before_the_fix(monkeypatch):
    monkeypatch.setattr(dispatch, "_resolve_hermes_argv", lambda: ["hermes"])
    monkeypatch.setattr(dispatch, "_resolve_worker_cli_toolsets", lambda home: None)
    argv = dispatch._worker_argv(
        _task(model_override="claude-sonnet-5.5", provider_override="nous"), "worker", None)
    assert argv[argv.index("-m") + 1] == "anthropic/claude-sonnet-5.5"
    assert argv[argv.index("--provider") + 1] == "nous"
