"""Pure invocation normalization and read-only local recipe resolution."""
from __future__ import annotations

import copy
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import NoReturn

from hermes_cli.kanban_recipes import (
    MAX_PLAN_BYTES, RecipeError, _json_safe, _object, _pointer, canonical,
)


def _invalid(path, message) -> NoReturn:
    raise RecipeError("BINDING_INVALID", path, message)


def normalize_bindings(prepared, bindings) -> dict:
    """Canonicalize explicit choices, without consulting installed state/defaults.

    In particular project spelling is request identity, not its resolved slug.
    Omitted project/tenant stay omitted so a retry can reuse frozen defaults.
    """
    try:
        _json_safe(bindings, "/bindings")
        _object(bindings, "/bindings", {"profiles", "project", "tenant"})
        aliases = bindings.get("profiles", {})
        _object(aliases, "/bindings/profiles")
    except RecipeError as exc:
        _invalid(exc.path, exc.message)
    references = {node["assignee"] for node in prepared["nodes"]}
    result: dict = {"profiles": {}}
    for alias, value in aliases.items():
        path = _pointer("/bindings/profiles", alias)
        if alias not in references:
            _invalid(path, "Unused profile alias")
        result["profiles"][alias] = _profile_name(value, path)
    for field in ("project", "tenant"):
        if field in bindings:
            value = bindings[field]
            if type(value) is not str or not value.strip():
                _invalid("/bindings/" + field, "Binding must be a nonempty string")
            result[field] = value
    return result


def _profile_name(value, path):
    from hermes_cli.profiles import normalize_profile_name, validate_profile_name

    if type(value) is not str:
        _invalid(path, "Profile must be a string")
    try:
        profile = normalize_profile_name(value)
        validate_profile_name(profile)
    except ValueError:
        _invalid(path, "Invalid profile name")
    return profile


def resolve_assignees(prepared, normalized_bindings):
    """Aliases override exact references; other values name installed profiles."""
    from hermes_cli.profiles import profile_exists

    aliases = normalized_bindings["profiles"]
    assignees = {}
    for index, node in enumerate(prepared["nodes"]):
        reference = node["assignee"]
        path = (_pointer("/bindings/profiles", reference) if reference in aliases
                else f"/nodes/{index}/assignee")
        profile = _profile_name(aliases.get(reference, reference), path)
        if not profile_exists(profile):
            _invalid(path, "Assigned profile is not installed")
        assignees[node["key"]] = profile
    return assignees


def _resolve_project(reference):
    from hermes_cli import projects_db

    # Native connect performs migrations even on an existing registry. Validation
    # must not call it, including when the registry is absent or legacy-shaped.
    uri = projects_db.projects_db_path().resolve().as_uri() + "?mode=ro"
    try:
        with closing(sqlite3.connect(uri, uri=True)) as conn:
            conn.row_factory = sqlite3.Row
            project = projects_db.get_project(conn, reference)
    except sqlite3.Error:
        _invalid("/bindings/project", "Project registry is unavailable")
    if project is None or project.archived:
        _invalid("/bindings/project", "Project binding does not resolve to an active project")
    if not project.primary_path or not Path(project.primary_path).is_absolute():
        _invalid("/bindings/project", "Project requires an absolute primary repository path")
    return {"id": project.id, "slug": project.slug, "repo": project.primary_path}


def resolve_plan(prepared, normalized_bindings, board) -> dict:
    """Freeze native creation choices; never write or materialize a workspace."""
    from hermes_cli import kanban_db
    from hermes_cli.kanban_pr_acceptance import validate_contract

    assignees = resolve_assignees(prepared, normalized_bindings)
    metadata = kanban_db.read_board_metadata(board)
    reference = normalized_bindings.get("project", metadata.get("project_id"))
    project = _resolve_project(reference) if reference else None
    nodes = copy.deepcopy(prepared["nodes"])
    for index, node in enumerate(nodes):
        task = node["task"]
        task["workspace_kind"] = task.get("workspace_kind", "scratch")
        if project:
            # Native project-linked scratch tasks become fresh worktrees.
            task["workspace_kind"] = "worktree"
        if task["workspace_kind"] == "worktree" and project is None:
            _invalid(f"/nodes/{index}/task/workspace_kind", "Worktree requires a project binding")
        if task["goal_mode"] and "goal_max_turns" not in task:
            from hermes_cli.goals import DEFAULT_MAX_TURNS

            if type(DEFAULT_MAX_TURNS) is not int or not 1 <= DEFAULT_MAX_TURNS <= 1000:
                _invalid(f"/nodes/{index}/task/goal_max_turns", "Native goal default is outside the recipe range")
            task["goal_max_turns"] = DEFAULT_MAX_TURNS
        # These NULLs describe native unset semantics, not dispatcher policy.
        for field in ("skills", "max_runtime_seconds", "max_retries", "model_override",
                      "provider_override", "reasoning_effort", "goal_max_turns", "completion_contract"):
            task.setdefault(field, None)
        node["title"] = node["title"].strip()
        task["completion_contract"] = validate_contract(task["completion_contract"])
        node["assignee"] = assignees[node["key"]]
        node["tenant"] = normalized_bindings.get("tenant")
    plan = {"nodes": nodes, "project": project}
    if len(canonical(plan).encode("utf-8")) > MAX_PLAN_BYTES:
        _invalid("/nodes", "Effective plan exceeds 2 MiB")
    return plan
