"""Operator-declared required checks for repositories whose plan hides GitHub's rules.

GitHub answers the branch-rules read of a private repository on GitHub Free with an exact
plan-boundary 403 while the PR, its head and its check runs stay readable. "Every observed
check must pass" is wrong there (path/condition-gated jobs legitimately skip), so the operator
declares the required set per repository::

    kanban:
      pr_required_checks:
        acme/private: [test, {context: build, app_id: 15368}]

Consulted only on that exact boundary when classic protection names nothing; a GitHub-derived
set always wins. Read from the config of the profile whose ``gh`` login reads the repository,
so the verdict cannot depend on which process completes the card. A malformed declaration
fails the gate instead of being skipped.
"""
from __future__ import annotations

from pathlib import Path

import hermes_yaml as yaml
from hermes_cli.kanban_pr_acceptance import _REPO, _PolicyError


def normalize_required_checks(value) -> dict[str, frozenset[tuple[str, int | None]]]:
    """``kanban.pr_required_checks`` as ``{OWNER/REPO: {(context, app_id), ...}}``.
    ``None`` = no declarations; anything malformed raises ``ValueError`` naming it."""
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError("it must map OWNER/REPO to a list of checks")
    declared = {}
    for repo, entries in value.items():
        if not isinstance(repo, str) or not _REPO.fullmatch(repo):
            raise ValueError(f"key {repo!r} is not OWNER/REPO")
        if not isinstance(entries, list) or not entries:
            raise ValueError(f"{repo} must list at least one check")
        declared[repo] = frozenset(_check(repo, entry) for entry in entries)
    return declared


def _check(repo: str, entry) -> tuple[str, int | None]:
    context, app_id = (entry, None) if isinstance(entry, str) else (None, None)
    if isinstance(entry, dict) and set(entry) <= {"context", "app_id"}:
        context, app_id = entry.get("context"), entry.get("app_id")
    if not isinstance(context, str) or not context.strip():
        raise ValueError(f"{repo}: {entry!r} is not a check name or {{context, app_id}}")
    # type() rather than isinstance(): YAML `true` is a bool, and bool is an int.
    if app_id is not None and (type(app_id) is not int or app_id <= 0):
        raise ValueError(f"{repo}: app_id {app_id!r} is not a GitHub App id")
    return context.strip(), app_id


def declared_required(repo: str, profile_home: str | None) -> set[tuple[str, int | None]]:
    """The declared required set for *repo*, or :class:`_PolicyError` when none is usable."""
    from hermes_cli.config import get_config_path
    from hermes_cli.config_effective import load_user_config_effective
    path = Path(profile_home) / "config.yaml" if profile_home else get_config_path()
    try:
        config = load_user_config_effective(path, fail_closed=True)
    except (OSError, ValueError, yaml.YAMLError):
        # Never echo the parse error: it quotes config lines into a persisted receipt.
        raise _PolicyError("config.yaml cannot be read or parsed") from None
    kanban = config.get("kanban")
    try:
        if kanban is not None and not isinstance(kanban, dict):
            raise ValueError("kanban is not a mapping")
        declared = normalize_required_checks((kanban or {}).get("pr_required_checks"))
    except ValueError as exc:
        raise _PolicyError(f"kanban.pr_required_checks is malformed ({exc})") from None
    if repo not in declared:
        raise _PolicyError(f"kanban.pr_required_checks declares nothing for {repo}")
    return set(declared[repo])
